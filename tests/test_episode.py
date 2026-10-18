"""Episode (internal session) tests.

Episode is one "discussion" the SDK carves out by write-time gaps (it may span several writes). It has two uses:

1. Read side: use `episode_id` to **scope the candidates of one discussion** and weight them -- this covers relations like
   "the park, the bike ride and lunch were written together, but the query only matches lunch" that vector search misses.
2. Write side: summarize once when the episode closes, replacing rolling summary's "rewrite every N turns and feed the old summary back to the LLM".
"""

import asyncio
import tempfile
from datetime import datetime, timedelta

import pytest

from hy_memory.config import CacheConfig, MemoryConfig
from hy_memory.data.cache_sqlite import SqliteCache
from hy_memory.models.memory import MemoryLayer, MemoryNode


# ======================================================================
# Episode state storage
# ======================================================================

class TestEpisodeState:
    """Episode state in the cache: keeps only (episode_id, last write time, count accumulated so far)."""

    @pytest.mark.asyncio
    async def test_roundtrip_and_absent(self):
        cfg = MemoryConfig()
        cfg.cache = CacheConfig(
            backend="sqlite", db_path=tempfile.mkdtemp() + "/c.db"
        )
        cache = SqliteCache(cfg)
        await cache.initialize()
        try:
            assert await cache.get_episode_state("u::a::s") is None

            await cache.upsert_episode_state(
                "u::a::s", user_id="u", agent_id="a", session_id="s",
                episode_id="ep1", last_write_ts=100.0, memory_count=3,
            )
            got = await cache.get_episode_state("u::a::s")
            assert got["episode_id"] == "ep1"
            assert got["last_write_ts"] == 100.0
            assert got["memory_count"] == 3

            # Writing the same key again updates the row instead of inserting a new one
            await cache.upsert_episode_state(
                "u::a::s", user_id="u", agent_id="a", session_id="s",
                episode_id="ep1", last_write_ts=200.0, memory_count=7,
            )
            got = await cache.get_episode_state("u::a::s")
            assert got["memory_count"] == 7 and got["last_write_ts"] == 200.0
        finally:
            await cache.close()


# ======================================================================
# Episode partition rules (write side, no LLM)
# ======================================================================

class _FakeCache:
    """Fake cache implementing just the two episode state methods."""

    def __init__(self):
        self.store = {}

    async def get_episode_state(self, key):
        return self.store.get(key)

    async def upsert_episode_state(self, key, **kw):
        self.store[key] = {
            "episode_id": kw["episode_id"],
            "last_write_ts": kw["last_write_ts"],
            "memory_count": kw["memory_count"],
        }
        return True


def _writer(cache, *, gap=1800.0, cap=50):
    """Build a writer minimal enough to run _resolve_episode_id / _schedule_episode_summary."""
    from hy_memory.pipelines.writer import MemoryWriter

    w = MemoryWriter.__new__(MemoryWriter)
    cfg = MemoryConfig()
    cfg.recall.episode_gap_seconds = gap
    cfg.recall.episode_max_memories = cap
    w.config = cfg
    w._cache = cache
    return w


def _req(ts, *, user="u", session="s"):
    from hy_memory.pipelines.base import WriteRequest

    return WriteRequest(
        user_id=user, agent_id="a", session_id=session,
        messages=[], memory_at=ts,
    )


class TestEpisodePartition:
    """Writes inside the gap count as one discussion, past it a new one starts -- purely deterministic, no LLM."""

    @pytest.mark.asyncio
    async def test_writes_within_gap_share_episode(self):
        """One topic spanning several writes: short gaps -> the same episode_id."""
        w = _writer(_FakeCache(), gap=1800.0)
        t0 = datetime(2026, 9, 13, 10, 0, 0)

        ep1 = await w._resolve_episode_id(_req(t0), 3)
        ep2 = await w._resolve_episode_id(_req(t0 + timedelta(minutes=5)), 2)
        ep3 = await w._resolve_episode_id(_req(t0 + timedelta(minutes=20)), 2)

        assert ep1 and ep1 == ep2 == ep3, "consecutive writes within the gap belong to one episode"

    @pytest.mark.asyncio
    async def test_gap_exceeded_starts_new_episode(self):
        w = _writer(_FakeCache(), gap=1800.0)
        t0 = datetime(2026, 9, 13, 10, 0, 0)

        ep1 = await w._resolve_episode_id(_req(t0), 3)
        ep2 = await w._resolve_episode_id(_req(t0 + timedelta(hours=3)), 3)

        assert ep1 != ep2, "exceeding the gap opens a new episode"

    @pytest.mark.asyncio
    async def test_cap_forces_rollover(self):
        """A session that never closes must not turn into one giant episode."""
        w = _writer(_FakeCache(), gap=1800.0, cap=5)
        t0 = datetime(2026, 9, 13, 10, 0, 0)

        ep1 = await w._resolve_episode_id(_req(t0), 4)
        ep2 = await w._resolve_episode_id(_req(t0 + timedelta(minutes=1)), 4)  # ->8 exceeds cap
        ep3 = await w._resolve_episode_id(_req(t0 + timedelta(minutes=2)), 1)

        assert ep1 == ep2, "the running total is still under cap at the second write"
        assert ep3 != ep2, "the running total past cap must force a rollover"

    @pytest.mark.asyncio
    async def test_different_users_and_sessions_isolated(self):
        w = _writer(_FakeCache())
        t0 = datetime(2026, 9, 13, 10, 0, 0)

        a = await w._resolve_episode_id(_req(t0, user="u1"), 1)
        b = await w._resolve_episode_id(_req(t0, user="u2"), 1)
        c = await w._resolve_episode_id(_req(t0, user="u1", session="other"), 1)

        assert a != b and a != c, "different user / session do not share an episode"

    @pytest.mark.asyncio
    async def test_no_cache_degrades_gracefully(self):
        """With the cache unavailable it returns an empty string; writes go through and episode degrades gracefully."""
        w = _writer(None)
        assert await w._resolve_episode_id(_req(datetime.now()), 3) == ""


class TestEpisodeCloseTriggersSummary:
    """Closing (rolling over) an episode is the moment to summarize it -- triggered only then, and only once."""

    @pytest.mark.asyncio
    async def test_summary_scheduled_only_on_rollover(self):
        w = _writer(_FakeCache(), gap=1800.0)
        scheduled = []
        w._schedule_episode_summary = lambda req, ep: scheduled.append(ep)
        t0 = datetime(2026, 9, 13, 10, 0, 0)

        ep1 = await w._resolve_episode_id(_req(t0), 3)
        await w._resolve_episode_id(_req(t0 + timedelta(minutes=5)), 3)
        assert scheduled == [], "continued writes inside the same episode must not trigger a summary"

        await w._resolve_episode_id(_req(t0 + timedelta(hours=3)), 3)
        assert scheduled == [ep1], "fires exactly once, on the episode that has just closed"


# ======================================================================
# Episode summary (write side, token saving)
# ======================================================================

class _FakeLLM:
    def __init__(self, text="The user discussed the plan to go to the park this weekend."):
        self.text = text
        self.calls = []

    async def complete(self, prompt, **kw):
        self.calls.append(prompt)
        return type("R", (), {"content": self.text})()


class _FakeEmbed:
    async def embed_queued(self, text):
        return [0.1] * 8


class _FakeVDB:
    def __init__(self, nodes):
        self._nodes = nodes
        self.upserted = []

    async def list_by_episode(self, episode_ids, **kw):
        want = set(episode_ids)
        return [n for n in self._nodes if (n.episode_id or "") in want]

    async def upsert(self, node):
        self.upserted.append(node)
        return node.node_id


def _mem(content, ep, *, tags=None, at=None):
    return MemoryNode(
        user_id="u", agent_id="a", session_id="s", episode_id=ep,
        layer=MemoryLayer.L2_FACT, content=content,
        tags=tags or [], memory_at=at or datetime(2026, 9, 13, 10, 0, 0),
    )


class TestEpisodeSummary:
    @pytest.mark.asyncio
    async def test_summarizes_episode_once_without_prev_summary(self):
        """Key difference: the input holds **no** previous summary -- nothing to merge or rewrite, hence fewer tokens."""
        from hy_memory.agent.episode_summary import (
            SUMMARY_KIND_EPISODE, summarize_closed_episode,
        )

        nodes = [
            _mem("The user went to the park on Saturday", "ep1", tags=["outing"]),
            _mem("The user biked to the park", "ep1", tags=["outing"]),
            _mem("The user had lunch near the park", "ep1", tags=["food"]),
        ]
        vdb, llm = _FakeVDB(nodes), _FakeLLM()

        nid = await summarize_closed_episode(
            vector_store=vdb, embed_service=_FakeEmbed(), llm_provider=llm,
            episode_id="ep1", user_id="u", agent_id="a",
        )

        assert nid and len(llm.calls) == 1, "one episode costs one LLM call"
        assert "prev" not in llm.calls[0].lower()
        assert "park" in llm.calls[0] and "bike" in llm.calls[0]

        node = vdb.upserted[0]
        assert node.layer == MemoryLayer.L3_SUMMARY
        assert node.custom["summary_kind"] == SUMMARY_KIND_EPISODE
        assert node.custom["episode_id"] == "ep1"
        assert node.custom["covered_count"] == 3
        assert node.episode_id == "ep1"
        # tags carry over, so topical words also retrieve the summary
        assert "outing" in node.tags and "food" in node.tags

    @pytest.mark.asyncio
    async def test_tiny_episode_skipped(self):
        """An episode of one or two memories is not worth a separate summary -- the originals are already retrievable."""
        from hy_memory.agent.episode_summary import summarize_closed_episode

        vdb, llm = _FakeVDB([_mem("The user likes cats", "ep1")]), _FakeLLM()
        nid = await summarize_closed_episode(
            vector_store=vdb, embed_service=_FakeEmbed(), llm_provider=llm,
            episode_id="ep1", user_id="u", min_memories=3,
        )
        assert nid is None and llm.calls == [], "should be skipped without calling the LLM"

    @pytest.mark.asyncio
    async def test_memory_at_follows_episode_not_now(self):
        """The summary's event time is the last memory in the episode, so time filters land on when it happened."""
        from hy_memory.agent.episode_summary import summarize_closed_episode

        last = datetime(2026, 9, 10, 15, 0, 0)
        nodes = [
            _mem("a", "ep1", at=datetime(2026, 9, 10, 9, 0, 0)),
            _mem("b", "ep1", at=last),
            _mem("c", "ep1", at=datetime(2026, 9, 10, 12, 0, 0)),
        ]
        vdb = _FakeVDB(nodes)
        await summarize_closed_episode(
            vector_store=vdb, embed_service=_FakeEmbed(), llm_provider=_FakeLLM(),
            episode_id="ep1", user_id="u",
        )
        assert vdb.upserted[0].memory_at == last

    @pytest.mark.asyncio
    async def test_other_episodes_not_included(self):
        from hy_memory.agent.episode_summary import summarize_closed_episode

        nodes = [
            _mem("park matter 1", "ep1"), _mem("park matter 2", "ep1"),
            _mem("park matter 3", "ep1"), _mem("completely unrelated work", "ep2"),
        ]
        vdb, llm = _FakeVDB(nodes), _FakeLLM()
        await summarize_closed_episode(
            vector_store=vdb, embed_service=_FakeEmbed(), llm_provider=llm,
            episode_id="ep1", user_id="u",
        )
        assert "completely unrelated work" not in llm.calls[0]

    @pytest.mark.asyncio
    async def test_token_cost_vs_rolling(self):
        """For one long discussion across 3 segments: rolling pays 3 calls and feeds back the old summary, episode pays 1."""
        from hy_memory.agent.episode_summary import summarize_closed_episode

        nodes = [_mem(f"memory {i}", "ep1") for i in range(12)]
        llm = _FakeLLM()
        await summarize_closed_episode(
            vector_store=_FakeVDB(nodes), embed_service=_FakeEmbed(),
            llm_provider=llm, episode_id="ep1", user_id="u",
        )
        assert len(llm.calls) == 1, (
            "an episode of 12 memories should pay for one LLM call; rolling summary's window-based triggering pays several"
        )


# ======================================================================
# Read side: use episodes to scope candidates
# ======================================================================

class TestEpisodeSupport:
    """Episode co-membership becomes structural support, shaped like reachable_within so it enters the same funnel."""

    def _hit(self, nid, score, ep):
        return {
            "node_id": nid, "score": score,
            "node": MemoryNode(
                user_id="u", node_id=nid, content=nid,
                episode_id=ep, layer=MemoryLayer.L2_FACT,
            ),
        }

    def test_members_of_hit_episode_become_candidates(self):
        """The user example: the query hits "lunch", so "the park" written in the same batch should be scoped in."""
        from hy_memory.pipelines._graph.rerank import episode_support

        hits = [self._hit("eat", 0.8, "ep1")]
        out = episode_support(
            hits, {"ep1": ["eat", "park", "bike"]}, seed_floor=0.6,
        )
        assert set(out) == {"park", "bike"}, "a hit is not re-added to the pool alongside itself"
        sup, via, etype, hops = out["park"]
        assert sup > 0 and via == "eat" and etype == "episode" and hops == 1

    def test_weak_hits_do_not_seed(self):
        """A low-score hit must not seed, otherwise a pile of candidates gets scoped out of noise."""
        from hy_memory.pipelines._graph.rerank import episode_support

        hits = [self._hit("eat", 0.3, "ep1")]
        assert episode_support(hits, {"ep1": ["eat", "park"]}, seed_floor=0.6) == {}

    def test_large_episode_yields_weaker_support(self):
        """The larger the episode, the less a single co-membership distinguishes anything (same as tag hubs)."""
        from hy_memory.pipelines._graph.rerank import episode_support

        hits = [self._hit("eat", 0.8, "ep1")]
        small = episode_support(hits, {"ep1": ["eat", "park", "bike"]})
        big = episode_support(
            hits, {"ep1": ["eat"] + [f"m{i}" for i in range(40)]}
        )
        assert small["park"][0] > big["m0"][0]

    def test_multi_seed_corroboration(self):
        """Two high-score hits in one episode -> that episode's members are more credible."""
        from hy_memory.pipelines._graph.rerank import episode_support

        one = episode_support(
            [self._hit("eat", 0.8, "ep1")], {"ep1": ["eat", "park", "bike"]}
        )
        two = episode_support(
            [self._hit("eat", 0.8, "ep1"), self._hit("bike", 0.75, "ep1")],
            {"ep1": ["eat", "park", "bike"]},
        )
        assert two["park"][0] > one["park"][0]

    def test_unrelated_episode_not_expanded(self):
        from hy_memory.pipelines._graph.rerank import episode_support

        hits = [self._hit("eat", 0.8, "ep1")]
        out = episode_support(hits, {"ep1": ["eat", "park"], "ep2": ["work", "code"]})
        assert "work" not in out and "code" not in out

    def test_empty_inputs(self):
        from hy_memory.pipelines._graph.rerank import episode_support

        assert episode_support([], {"ep1": ["a"]}) == {}
        assert episode_support([self._hit("a", 0.9, "ep1")], {}) == {}

    def test_episode_is_relational_not_temporal(self):
        """An episode is "co_occurs stretched across batches" (a narrative relation), not a temporal edge."""
        from hy_memory.pipelines._graph.rerank import TYPE_PRIOR, edge_group

        assert edge_group("episode") == "relational"
        # Aligned with co_occurs, and above semantic_neighbor.
        assert TYPE_PRIOR["episode"] == TYPE_PRIOR["co_occurs"]
        assert TYPE_PRIOR["episode"] > TYPE_PRIOR["semantic_neighbor"]


class TestEpisodeIdPlumbing:
    """episode_id has to survive the payload roundtrip, otherwise the read side cannot scope anything out."""

    def test_node_dict_roundtrip(self):
        node = _mem("The user went to the park", "ep_abc")
        assert MemoryNode.from_dict(node.to_dict()).episode_id == "ep_abc"

    def test_payload_roundtrip(self):
        from hy_memory.data.vector_store_base import VectorStoreBase

        payload = VectorStoreBase._node_to_payload(_mem("x", "ep_xyz"))
        assert payload["episode_id"] == "ep_xyz"
        assert VectorStoreBase._payload_to_node(payload).episode_id == "ep_xyz"

    def test_episode_id_is_not_isolation_key(self):
        """episode_id is not part of isolation -- it is an internal granularity, not a business isolation key."""
        a = _mem("x", "ep1")
        b = _mem("x", "ep2")
        assert a.get_isolation_key() == b.get_isolation_key()


class TestEpisodeRecallWithoutGraph:
    """Episode scoping needs only the episode_id field plus one VDB filter -- it must work with the graph turned off."""

    def _reader(self, *, graph_store, episode_on, graph_rerank_on):
        from hy_memory.pipelines.reader_hybrid import HybridReadPipeline

        r = HybridReadPipeline.__new__(HybridReadPipeline)
        cfg = MemoryConfig()
        cfg.recall.episode_recall_enabled = episode_on
        cfg.graph_store.graph_rerank_enabled = graph_rerank_on
        r.config = cfg
        r._graph_store = graph_store
        return r

    def test_expansion_runs_with_episode_only(self):
        r = self._reader(graph_store=None, episode_on=True, graph_rerank_on=False)
        assert r._episode_recall_on() is True
        assert r._graph_rerank_on() is False
        assert r._expansion_on() is True, "episode scoping must still run when the graph is off"

    def test_expansion_off_when_both_off(self):
        r = self._reader(graph_store=None, episode_on=False, graph_rerank_on=False)
        assert r._expansion_on() is False

    def test_apply_graph_rerank_accepts_request(self):
        """Regression: without the request parameter _apply_graph_rerank raises NameError, swallowed by the outer except
        -> the whole expansion fails silently. Pinned here at the signature level."""
        import inspect

        from hy_memory.pipelines.reader_hybrid import HybridReadPipeline

        sig = inspect.signature(HybridReadPipeline._apply_graph_rerank)
        assert "request" in sig.parameters

    def test_episode_candidates_gated_by_config(self):
        """With the switch off, no VDB query should be issued."""
        class _Boom:
            async def list_by_episode(self, *a, **kw):
                raise AssertionError("the VDB must not be queried while the switch is off")

        r = self._reader(graph_store=None, episode_on=False, graph_rerank_on=False)
        from hy_memory.pipelines.base import ReadRequest

        out = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            r._episode_candidates(
                _Boom(), [], {}, ReadRequest(query="q", user_id="u"), 0.6, 0.5,
            )
        )
        assert out == {}


class TestEpisodeCountsUserSideOnly:
    """Episode counting and summary input both look at the user side only -- agent-side memories only burn tokens.

    Same convention as rolling summary counting only pending_user_count.
    """

    def test_count_helper_ignores_agent_and_defaults_to_user(self):
        from hy_memory.pipelines.writer import _count_user_owned

        metas = [
            {"content": "a", "owner": "user"},
            {"content": "b", "owner": "agent"},
            {"content": "c"},              # owner absent -> treated as user
            {"content": "d", "owner": None},
        ]
        assert _count_user_owned(metas) == 3
        assert _count_user_owned([]) == 0

    @pytest.mark.asyncio
    async def test_agent_memories_do_not_burn_episode_cap(self):
        """Counting agent-side memories would push the episode to its cap early and split it too finely."""
        w = _writer(_FakeCache(), gap=1800.0, cap=4)
        t0 = datetime(2026, 9, 13, 10, 0, 0)

        # Each write adds 3 user + 5 agent memories; counting only user, two writes reach 6 > 4
        ep1 = await w._resolve_episode_id(_req(t0), 3)
        ep2 = await w._resolve_episode_id(_req(t0 + timedelta(minutes=1)), 3)
        assert ep1 == ep2, "6 user-side memories in total over two writes; the running total of 3 is still under cap"

    @pytest.mark.asyncio
    async def test_summary_input_excludes_agent_memories(self):
        from hy_memory.agent.episode_summary import summarize_closed_episode

        def _owned(content, owner):
            n = _mem(content, "ep1")
            n.owner = owner
            return n

        nodes = [
            _owned("The user went to the park on Saturday", "user"),
            _owned("The assistant recommended three restaurants", "agent"),
            _owned("The user biked there", "user"),
            _owned("The assistant explained the cycling route", "agent"),
            _owned("The user had lunch nearby", "user"),
        ]
        vdb, llm = _FakeVDB(nodes), _FakeLLM()

        await summarize_closed_episode(
            vector_store=vdb, embed_service=_FakeEmbed(), llm_provider=llm,
            episode_id="ep1", user_id="u",
        )

        prompt = llm.calls[0]
        assert "The user went to the park on Saturday" in prompt and "The user biked there" in prompt
        assert "The assistant recommended three restaurants" not in prompt, "agent-side memories must not enter the summary input"
        assert "The assistant explained the cycling route" not in prompt
        # covered_count / source_memory_ids also count the user side only
        assert vdb.upserted[0].custom["covered_count"] == 3
        assert len(vdb.upserted[0].custom["source_memory_ids"]) == 3

    @pytest.mark.asyncio
    async def test_agent_only_episode_is_skipped_entirely(self):
        """The whole episode is agent-side -> not one LLM call should happen."""
        from hy_memory.agent.episode_summary import summarize_closed_episode

        nodes = []
        for i in range(5):
            n = _mem(f"The assistant said sentence {i}", "ep1")
            n.owner = "agent"
            nodes.append(n)

        llm = _FakeLLM()
        nid = await summarize_closed_episode(
            vector_store=_FakeVDB(nodes), embed_service=_FakeEmbed(),
            llm_provider=llm, episode_id="ep1", user_id="u",
        )
        assert nid is None and llm.calls == []

    @pytest.mark.asyncio
    async def test_missing_owner_treated_as_user(self):
        """A missing owner must be treated as user, otherwise legacy data gets skipped wholesale."""
        from hy_memory.agent.episode_summary import summarize_closed_episode

        nodes = [_mem(f"user matter {i}", "ep1") for i in range(3)]
        for n in nodes:
            n.owner = None

        llm = _FakeLLM()
        nid = await summarize_closed_episode(
            vector_store=_FakeVDB(nodes), embed_service=_FakeEmbed(),
            llm_provider=llm, episode_id="ep1", user_id="u",
        )
        assert nid is not None and len(llm.calls) == 1


# ======================================================================
# Instrumentation (pipeline log)
# ======================================================================

class TestInstrumentation:
    """New stages must ship instrumentation: LLM calls have to be traceable to prompt and output, and async graph projection has to confirm whether it built anything.

    These stages are all fire-and-forget and only warn on failure, so without instrumentation production is unobservable --
    the usual reason the read side "shows no graph effect" is that the write side never projected successfully at all.
    """

    def test_episode_summary_reports_prompt_and_tokens(self):
        """_generate must hand back the actual prompt and token usage, otherwise instrumentation records only results, never inputs."""
        import inspect

        from hy_memory.agent import episode_summary

        src = inspect.getsource(episode_summary._generate)
        assert "return \"\", \"\", {}" in src or "prompt, tokens" in src
        assert "trace" in inspect.signature(
            episode_summary.summarize_closed_episode
        ).parameters

    def test_topic_summary_accepts_trace(self):
        """topic summary has been calling the LLM since dev4, but never had instrumentation."""
        import inspect

        from hy_memory.pipelines._graph import topic_summary

        assert "trace" in inspect.signature(
            topic_summary.maybe_summarize_topics
        ).parameters

    @pytest.mark.asyncio
    async def test_episode_summary_emits_trace_with_prompt(self):
        from hy_memory.agent.episode_summary import summarize_closed_episode

        seen = []

        async def _trace(**kw):
            seen.append(kw)

        nodes = [_mem(f"user matter {i}", "ep1") for i in range(3)]
        await summarize_closed_episode(
            vector_store=_FakeVDB(nodes), embed_service=_FakeEmbed(),
            llm_provider=_FakeLLM("The user discussed three things this time."),
            episode_id="ep1", user_id="u", trace=_trace,
        )

        assert len(seen) == 1
        rec = seen[0]
        assert "user matter 0" in rec["prompt"], "the prompt must be recorded in full, otherwise badcases cannot be reviewed"
        assert rec["response"] == "The user discussed three things this time."
        assert rec["parsed"]["episode_id"] == "ep1"
        assert rec["parsed"]["covered_count"] == 3
        assert len(rec["memory_ids"]) == 3
        assert rec["elapsed_ms"] >= 0

    @pytest.mark.asyncio
    async def test_no_trace_sink_is_harmless(self):
        """trace=None (tracing off) must not affect the main flow."""
        from hy_memory.agent.episode_summary import summarize_closed_episode

        nodes = [_mem(f"user matter {i}", "ep1") for i in range(3)]
        nid = await summarize_closed_episode(
            vector_store=_FakeVDB(nodes), embed_service=_FakeEmbed(),
            llm_provider=_FakeLLM(), episode_id="ep1", user_id="u", trace=None,
        )
        assert nid is not None

    def test_read_side_step_exists_and_funnel_is_observable(self):
        """The stats of READ_GRAPH_RERANK must separate "the graph helped" from "the pool just grew"."""
        import inspect

        from hy_memory.pipelines._retrieval.trace import (
            STEP_READ_GRAPH_RERANK, ReadTraceLogger,
        )

        assert STEP_READ_GRAPH_RERANK == "READ_GRAPH_RERANK"
        params = inspect.signature(ReadTraceLogger.log_graph_rerank).parameters
        # pool_size is the key to attribution: without it you cannot tell whether a change is just a bigger pool
        for k in ("stats", "pool_size", "admitted_ids", "elapsed_ms"):
            assert k in params

    def test_writer_has_trace_sink_and_new_steps(self):
        import inspect

        from hy_memory.pipelines.writer import MemoryWriter

        assert hasattr(MemoryWriter, "_make_trace_sink")
        assert hasattr(MemoryWriter, "_log_episode")

        src = inspect.getsource(MemoryWriter)
        for step in ("EPISODE", "EPISODE_SUMMARY", "TOPIC_SUMMARY", "GRAPH_PROJECT"):
            assert f'"{step}"' in src, f"missing instrumentation for step={step}"

    @pytest.mark.asyncio
    async def test_episode_partition_decision_is_logged(self):
        """The partition decision (reused or new, idle time since the last write, running count) must be reviewable."""
        w = _writer(_FakeCache(), gap=1800.0)
        logged = []

        async def _log(request, **kw):
            logged.append(kw)

        w._log_episode = _log
        t0 = datetime(2026, 9, 13, 10, 0, 0)

        await w._resolve_episode_id(_req(t0), 3)
        await w._resolve_episode_id(_req(t0 + timedelta(minutes=5)), 2)
        await w._resolve_episode_id(_req(t0 + timedelta(hours=5)), 1)

        assert [r["reused"] for r in logged] == [False, True, False]
        assert logged[1]["idle_s"] == 300.0
        assert logged[1]["count"] == 5, "the running count has to be recorded"
        assert logged[2]["reused"] is False
        assert all(r["gap"] == 1800.0 for r in logged)
