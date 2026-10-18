# -*- coding: utf-8 -*-
"""
Tests for the memory deduplication mechanism.

Coverage:
- Pure dedup helper functions: 4 deletion rules / deterministic retention / threshold boundary / no duplicate no delete;
- execute_dedup orchestration: store deletion + DEDUP log (including delete_from_store=False in the extractor);
- Backend get_embeddings: chroma returns non-empty vectors;
- End-to-end dedup of reconcile candidates (chroma + fake LLM): a duplicate chain head is removed from the store and from the candidates.
"""

import asyncio
import random
import uuid
from datetime import datetime, timedelta

import pytest

from hy_memory.config import MemoryConfig
from hy_memory.models.memory import MemoryNode, MemoryLayer, MemoryStatus
from hy_memory.pipelines._retrieval.dedup import (
    DedupItem, plan_dedup, execute_dedup, get_dedup_threshold,
    group_items_by_owner,
)


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _emb(dims=32):
    v = [random.gauss(0, 1) for _ in range(dims)]
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


_SAME = [1.0, 0.0, 0.0, 0.0]
_SAME2 = [0.999, 0.0010, 0.0, 0.0]
_DIFF = [0.0, 1.0, 0.0, 0.0]


class TestPlanDedupRules:
    def test_all_non_chain_keep_earliest(self):
        # Rule 2: all off-chain, two near-identical -> keep the earliest gmt, delete the other; unrelated items untouched
        r = plan_dedup([
            DedupItem("a", _SAME, content="Playing basketball", gmt_created=100),
            DedupItem("b", _SAME2, content="Playing basketball", gmt_created=200),
            DedupItem("c", _DIFF, content="Drinking tea", gmt_created=50),
        ])
        assert r["delete_ids"] == ["b"]
        assert r["kept_ids"] == ["a"]

    def test_one_head_keeps_head(self):
        # Rule 3: 1 chain head + 1 off-chain duplicate -> keep the head, delete the off-chain one (even with a later gmt)
        r = plan_dedup([
            DedupItem("h", _SAME, content="Lives in Yokohama", is_chain_head=True,
                      gmt_created=300, chain_node_ids=["h", "old1"]),
            DedupItem("p", _SAME2, content="Lives in Yokohama", gmt_created=100),
        ])
        assert r["delete_ids"] == ["p"]
        assert r["kept_ids"] == ["h"]

    def test_two_heads_cascade_delete_chain(self):
        # Rule 4: two duplicate heads -> keep the head with the earliest gmt, delete the other head plus its history chain
        r = plan_dedup([
            DedupItem("h1", _SAME, content="Lives in Yokohama", is_chain_head=True,
                      gmt_created=100, chain_node_ids=["h1", "b1"]),
            DedupItem("h2", _SAME2, content="Lives in Yokohama", is_chain_head=True,
                      gmt_created=200, chain_node_ids=["h2", "b2", "b3"]),
        ])
        assert r["kept_ids"] == ["h1"]
        assert sorted(r["delete_ids"]) == ["b2", "b3", "h2"]
        g = r["groups"][0]
        assert g["kept"]["reason"] == "earliest_chain_head"
        d = [x for x in g["deleted"] if x["node_id"] == "h2"][0]
        assert sorted(d["also_deleted_chain"]) == ["b2", "b3"]

    def test_no_duplicate_no_delete(self):
        r = plan_dedup([
            DedupItem("x", _SAME, gmt_created=1),
            DedupItem("y", _DIFF, gmt_created=2),
        ])
        assert r["delete_ids"] == []
        assert r["groups"] == []

    def test_chain_body_ignored(self):
        # is_latest=False (chain body) does not take part in dedup
        r = plan_dedup([
            DedupItem("a", _SAME, is_latest=True, gmt_created=1),
            DedupItem("body", _SAME, is_latest=False, gmt_created=2),
        ])
        assert r["delete_ids"] == []  # Only 1 is_latest left, so no duplicate

    def test_threshold_env(self, monkeypatch):
        monkeypatch.setenv("MEMORY_DEDUP_THRESHOLD", "0.5")
        assert abs(get_dedup_threshold() - 0.5) < 1e-9
        monkeypatch.delenv("MEMORY_DEDUP_THRESHOLD", raising=False)
        assert abs(get_dedup_threshold() - 0.95) < 1e-9


class _CaptureCache:
    def __init__(self):
        self.logs = []

    async def store_pipeline_log(self, **kw):
        self.logs.append(kw)


class _FakeStore:
    """Records the writes dedup performs on the store.

    Now that dedup deletes logically it sets SHADOW through update_payload instead of
    calling delete(); delete() is kept to assert that nothing is ever hard-deleted.
    """

    def __init__(self):
        self.deleted = []
        self.payload_updates = []  # [(node_id, updates_dict), ...]

    async def delete(self, nid):
        self.deleted.append(nid)
        return True

    async def update_payload(self, node_id, updates):
        self.payload_updates.append((node_id, dict(updates)))
        return True

    @property
    def shadowed_ids(self):
        return [
            nid for nid, up in self.payload_updates
            if up.get("status") == "shadow"
        ]


class TestExecuteDedup:
    def test_shadows_and_logs(self):
        cache = _CaptureCache()
        store = _FakeStore()
        items = [
            DedupItem("a", _SAME, content="x", gmt_created=1),
            DedupItem("b", _SAME2, content="x", gmt_created=2),
        ]
        plan = run(execute_dedup(
            items, vector_store=store, cache=cache, trigger="reconcile",
            request_id="rq", user_id="u1", agent_id="a1", delete_from_store=True,
        ))
        assert plan["delete_ids"] == ["b"]
        # Logical deletion: set SHADOW, never hard-delete
        assert store.deleted == [], "dedup must not hard-delete (unrecoverable)"
        assert store.shadowed_ids == ["b"]
        _nid, _up = store.payload_updates[0]
        assert _up["status"] == "shadow"
        assert _up["is_latest"] is False
        # Audit fields: why it disappeared + which record replaced it
        assert _up["shadow_reason"] == "dedup:reconcile"
        assert _up["dedup_kept_id"] == "a"

        dedup_logs = [l for l in cache.logs if l["step"] == "DEDUP"]
        assert len(dedup_logs) == 1
        import json
        parsed = json.loads(dedup_logs[0]["parsed"])
        assert parsed["trigger"] == "reconcile"
        assert parsed["deleted_count"] == 1

    def test_extractor_no_store_delete(self):
        # delete_from_store=False in the extractor stage: no store write, but the log is still recorded
        cache = _CaptureCache()
        store = _FakeStore()
        items = [
            DedupItem("0", _SAME, content="x", gmt_created=0),
            DedupItem("1", _SAME2, content="x", gmt_created=1),
        ]
        plan = run(execute_dedup(
            items, vector_store=store, cache=cache, trigger="extractor",
            request_id="rq", user_id="u1", agent_id="a1", delete_from_store=False,
        ))
        assert plan["delete_ids"] == ["1"]
        assert store.deleted == []  # No store deletion
        assert store.payload_updates == []  # Also no SHADOW flag
        assert any(l["step"] == "DEDUP" for l in cache.logs)


class TestGroupItemsByOwner:
    """The isolation dimension of dedup is decided by owner, not a blanket split by agent_id.

    owner='user' (including empty values): the memory belongs to the user side. The same
      fact said by one user under agent_a / agent_b is one fact and must be judged in the same
      batch, otherwise every agent keeps its own copy that can never be merged (reconcile
      candidate retrieval used to hardcode agent_ids=[agent_id], so the two never shared a batch).
    owner='agent': what different assistants said are independent memories that must not delete each other even when semantically close.
    """

    def _item(self, nid, *, owner=None, uid="u1", aid="a1", source="default"):
        return DedupItem(
            nid, _SAME, content="x", owner=owner, user_id=uid,
            agent_id=aid, source=source,
        )

    def test_owner_user_merges_across_agents(self):
        items = [
            self._item("n1", owner="user", aid="agent_a"),
            self._item("n2", owner="user", aid="agent_b"),
        ]
        batches = group_items_by_owner(items, default_user_id="u1", default_agent_id="agent_b")
        assert len(batches) == 1, "owner=user must merge across agents into a single batch"
        assert batches[0]["owner"] == "user"
        assert sorted(i.node_id for i in batches[0]["items"]) == ["n1", "n2"]
        assert batches[0]["user_id"] == "u1"

    def test_owner_agent_isolated_per_agent(self):
        items = [
            self._item("n3", owner="agent", aid="agent_a"),
            self._item("n4", owner="agent", aid="agent_b"),
        ]
        batches = group_items_by_owner(items, default_user_id="u1", default_agent_id="agent_b")
        # One item each -> both below 2 items, nothing to compare
        assert batches == [], "owner=agent items from different agents must not share a batch"

    def test_owner_agent_same_agent_batched(self):
        items = [
            self._item("n5", owner="agent", aid="agent_a"),
            self._item("n6", owner="agent", aid="agent_a"),
        ]
        batches = group_items_by_owner(items, default_user_id="u1", default_agent_id="agent_a")
        assert len(batches) == 1
        assert batches[0]["owner"] == "agent"
        assert batches[0]["agent_id"] == "agent_a"

    @pytest.mark.parametrize("owner", [None, "", "USER", "unknown-value"])
    def test_missing_or_unknown_owner_falls_back_to_user(self, owner):
        """owner was only introduced on 2026-06-16, so old nodes have it empty; the LLM may also omit or misfill it.

        The convention across the pipeline is `owner or "user"` (extractor prompt / reconciler
        existing_mem_list included), so this must stay consistent, otherwise legacy rows land
        in the agent bucket and lose cross-agent dedup.
        """
        items = [
            self._item("n7", owner=owner, aid="agent_a"),
            self._item("n8", owner=owner, aid="agent_b"),
        ]
        batches = group_items_by_owner(items, default_user_id="u1", default_agent_id="agent_b")
        assert len(batches) == 1
        assert batches[0]["owner"] == "user"

    def test_mixed_owners_split_into_separate_batches(self):
        items = [
            self._item("u_a", owner="user", aid="agent_a"),
            self._item("u_b", owner="user", aid="agent_b"),
            self._item("g_a1", owner="agent", aid="agent_a"),
            self._item("g_a2", owner="agent", aid="agent_a"),
        ]
        batches = group_items_by_owner(items, default_user_id="u1", default_agent_id="agent_a")
        assert len(batches) == 2
        by_owner = {b["owner"]: b for b in batches}
        assert sorted(i.node_id for i in by_owner["user"]["items"]) == ["u_a", "u_b"]
        assert sorted(i.node_id for i in by_owner["agent"]["items"]) == ["g_a1", "g_a2"]

    def test_different_users_never_mixed(self):
        """With multi-user retrieval (user_ids=[u1,u2]) dedup must never cross users."""
        items = [
            self._item("x1", owner="user", uid="u1"),
            self._item("x2", owner="user", uid="u2"),
        ]
        batches = group_items_by_owner(items, default_user_id="u1", default_agent_id="a1")
        assert batches == []

    def test_different_sources_never_mixed(self):
        items = [
            self._item("cursor", owner="user", source="cursor"),
            self._item("codex", owner="user", source="codex"),
        ]
        batches = group_items_by_owner(
            items, default_user_id="u1", default_agent_id="a1",
        )
        assert batches == []

    def test_same_source_still_batches(self):
        items = [
            self._item("codex-1", owner="user", source="Codex"),
            self._item("codex-2", owner="user", source=" codex "),
        ]
        batches = group_items_by_owner(
            items, default_user_id="u1", default_agent_id="a1",
        )
        assert len(batches) == 1
        assert batches[0]["source"] == "codex"

    def test_batch_carries_accurate_attribution(self):
        """DEDUP log attribution: callers used to pass effective_*_ids[0], which is wrong across users."""
        items = [
            self._item("y1", owner="user", uid="u2"),
            self._item("y2", owner="user", uid="u2"),
        ]
        batches = group_items_by_owner(items, default_user_id="u1", default_agent_id="a1")
        assert len(batches) == 1
        assert batches[0]["user_id"] == "u2", "the batch must carry its actual owner user, not the caller's first one"

    def test_single_item_batch_dropped(self):
        items = [self._item("z1", owner="user")]
        assert group_items_by_owner(items, default_user_id="u1", default_agent_id="a1") == []


@pytest.fixture
def config(tmp_path):
    return MemoryConfig.from_dict({
        "vector_store": {
            "provider": "chroma",
            "collection_name": f"test_dedup_{uuid.uuid4().hex[:8]}",
            "persist_directory": str(tmp_path / "chroma_data"),
            "embedding_dims": 32,
        },
    })


@pytest.fixture
def store(config):
    from hy_memory.data.vector_store_chroma import ChromaVectorStore
    vs = ChromaVectorStore(config)
    run(vs.initialize())
    yield vs
    run(vs.close())


class TestBackendGetEmbeddings:
    def test_chroma_get_embeddings(self, store):
        e1, e2 = _emb(), _emb()
        for nid, e in [("a", e1), ("b", e2)]:
            n = MemoryNode(node_id=nid, user_id="u1", agent_id="a1", session_id="s1",
                           content="x", layer=MemoryLayer.L2_FACT,
                           status=MemoryStatus.ACTIVE, is_latest=True)
            n.embedding = e
            run(store.upsert(n))
        embs = run(store.get_embeddings(["a", "b", "missing"]))
        assert set(embs.keys()) == {"a", "b"}
        assert len(embs["a"]) == 32


class _CaptureLLM:
    def __init__(self):
        self.last_prompt = ""

    async def complete(self, prompt, **kw):
        self.last_prompt = prompt
        class R:
            content = "[]"
            prompt_tokens = 1
            completion_tokens = 1
            tokens_used = 2
        return R()


class _Embed:
    def __init__(self, vec):
        self._vec = vec

    async def embed(self, t):
        return list(self._vec)

    async def embed_batch(self, ts):
        return [list(self._vec) for _ in ts]

    async def embed_queued(self, t, use_cache=True):
        return list(self._vec)


class TestReconcileDedupE2E:
    def test_duplicate_candidates_shadowed(self, store, config):
        from hy_memory.agent.reconciler import MemoryReconciler
        shared = _emb()
        # Two near-identical ACTIVE L2 nodes (non-chain); both get recalled as candidates
        for nid, days in [("m1", 30), ("m2", 5)]:
            n = MemoryNode(node_id=nid, user_id="u1", agent_id="default_agent",
                           session_id="s1", content="The user likes playing basketball",
                           layer=MemoryLayer.L2_FACT, status=MemoryStatus.ACTIVE,
                           is_latest=True)
            n.memory_at = datetime.now() - timedelta(days=days)
            n.gmt_created = datetime.now() - timedelta(days=days)
            n.embedding = list(shared)
            run(store.upsert(n))

        recon = MemoryReconciler(config)
        recon._get_llm = lambda: _CaptureLLM()
        recon._enable_search_query = False
        run(recon.reconcile(
            new_memories=["The user loves playing basketball"],
            user_id="u1", agent_id="default_agent",
            vector_store=store, embed_service=_Embed(shared),
            request_id="rq1",
        ))
        # Dedup is logical deletion: both nodes are still in the store, only the earliest m1 stays ACTIVE
        m1 = run(store.get_by_id("m1"))
        m2 = run(store.get_by_id("m2"))
        assert m1 is not None and m2 is not None, "dedup must not hard-delete nodes"
        assert m1.status == MemoryStatus.ACTIVE
        assert m1.is_latest is True
        assert m2.status == MemoryStatus.SHADOW, f"m2 should be logically deleted, got {m2.status}"
        assert m2.is_latest is False

    def test_non_l2_l4_candidates_not_deduped(self, store, config):
        # Duplicate L3_SUMMARY candidates are out of dedup scope (L2+L4 only) and must survive
        from hy_memory.agent.reconciler import MemoryReconciler
        shared = _emb()
        for nid, days in [("s1n", 30), ("s2n", 5)]:
            n = MemoryNode(node_id=nid, user_id="u1", agent_id="default_agent",
                           session_id="s1", content="Conversation summary content",
                           layer=MemoryLayer.L3_SUMMARY, status=MemoryStatus.ACTIVE,
                           is_latest=True)
            n.memory_at = datetime.now() - timedelta(days=days)
            n.gmt_created = datetime.now() - timedelta(days=days)
            n.embedding = list(shared)
            run(store.upsert(n))

        recon = MemoryReconciler(config)
        recon._get_llm = lambda: _CaptureLLM()
        recon._enable_search_query = False
        run(recon.reconcile(
            new_memories=["New summary"],
            user_id="u1", agent_id="default_agent",
            vector_store=store, embed_service=_Embed(shared),
            layers=[MemoryLayer.L3_SUMMARY],
            request_id="rq2",
        ))
        # L3 takes no part in dedup -> both rows remain and both stay ACTIVE.
        # status must be asserted: after logical deletion a deduped node still comes back from
        # get_by_id, so asserting "is not None" alone no longer proves it was left alone.
        remain = [run(store.get_by_id("s1n")), run(store.get_by_id("s2n"))]
        assert all(n is not None for n in remain)
        assert all(n.status == MemoryStatus.ACTIVE for n in remain), (
            "L3_SUMMARY is out of dedup scope and must not be flagged SHADOW"
        )
