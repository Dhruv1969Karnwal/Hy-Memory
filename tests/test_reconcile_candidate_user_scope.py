# -*- coding: utf-8 -*-
"""Candidate retrieval for reconcile is scoped to the user, never narrowed by agent.

Background: candidate retrieval used to hardcode `agent_ids=[agent_id]`, while the agent_id the writer
passes down is never empty (`request.agent_id or "default_agent"`). So reconcile only ever saw the current
agent's history -- which breaks more than deduplication, it breaks reconcile entirely:

  The user says "I am a vegetarian" on agent_a, later says "I eat fish now" on agent_b
  -> this search on agent_b misses the record on agent_a -> the LLM gets no candidates -> ADD instead of SUPERSEDE
  -> two contradictory user memories sit in the store ACTIVE at the same time

owner and agent_id are two orthogonal dimensions: an owner='user' memory is still stamped with the agent_id
it was written under, so "isolating by agent" is redundant and harmful for user-side facts.

Backward compatibility rests not on "keeping the agent filter" but on two-way evolution-chain tracing: after a
cross-agent SUPERSEDE the new node belongs to agent_b, yet a user search on agent_a still hits the old node left
on agent_a (status=SUPERSEDED is also within recall range), and _trace_full_chain then follows superseded_by to
the chain head on agent_b and returns the whole chain -- see test_cross_agent_chain_still_recallable.
"""

import asyncio
import random
import uuid
from datetime import datetime, timedelta

import pytest

from hy_memory.config import MemoryConfig
from hy_memory.models.memory import MemoryNode, MemoryLayer, MemoryStatus


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


@pytest.fixture
def config(tmp_path):
    return MemoryConfig.from_dict({
        "vector_store": {
            "provider": "chroma",
            "collection_name": f"test_xagent_{uuid.uuid4().hex[:8]}",
            "persist_directory": str(tmp_path / "chroma"),
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


class _CaptureLLM:
    """Captures the reconcile prompt so assertions can check whether a candidate made it into the prompt."""

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
        self._v = vec

    async def embed_batch(self, texts):
        return [list(self._v) for _ in texts]

    async def embed_queued(self, text, use_cache=True):
        return list(self._v)


def _l2(store, *, content, agent_id, owner="user", days=10, emb=None,
        status=MemoryStatus.ACTIVE, is_latest=True, supersedes=None):
    n = MemoryNode(
        user_id="u1", agent_id=agent_id, session_id="s1",
        layer=MemoryLayer.L2_FACT, content=content, owner=owner,
        status=status, is_latest=is_latest, supersedes=supersedes,
    )
    n.memory_at = datetime.now() - timedelta(days=days)
    n.gmt_created = datetime.now() - timedelta(days=days)
    n.embedding = emb or _emb()
    return run(store.upsert(n))


def _reconcile(config, store, *, agent_id, new_memories, emb):
    from hy_memory.agent.reconciler import MemoryReconciler
    recon = MemoryReconciler(config)
    llm = _CaptureLLM()
    recon._get_llm = lambda: llm
    recon._enable_search_query = False
    run(recon.reconcile(
        new_memories=new_memories,
        user_id="u1", agent_id=agent_id,
        vector_store=store, embed_service=_Embed(emb),
        request_id="rq",
    ))
    return llm


class TestCandidateScopeIsUser:
    def test_other_agent_user_memory_enters_candidates(self, config, store):
        """An owner=user memory written on agent_a must show up in agent_b's reconcile candidates."""
        shared = _emb()
        _l2(store, content="The user is a vegetarian", agent_id="agent_a", emb=shared)

        llm = _reconcile(
            config, store, agent_id="agent_b",
            new_memories=["The user has started eating fish"], emb=shared,
        )
        assert "vegetarian" in llm.last_prompt, (
            "the cross-agent owner=user candidate never reached the prompt -- reconcile is still isolated by agent"
        )

    def test_other_agent_agent_owned_memory_also_recalled(self, config, store):
        """Cross-agent memories owned by the agent also become candidates (no filtering on the candidate side).

        The cross-owner constraint is guaranteed by RECONCILE_PROMPT's
        "NEVER merge or supersede across different owners";
        on the deduplication side, group_items_by_owner isolates per agent.
        The candidate side does no owner filtering -- a deliberate design choice, pinned here against "drive-by optimization".
        """
        shared = _emb()
        _l2(store, content="The assistant recommended a ryokan in Kyoto to the user",
            agent_id="agent_a", owner="agent", emb=shared)

        llm = _reconcile(
            config, store, agent_id="agent_b",
            new_memories=["The assistant recommended a ryokan in Osaka to the user"], emb=shared,
        )
        assert "Kyoto" in llm.last_prompt

    def test_other_user_memory_never_enters_candidates(self, config, store):
        """What gets relaxed is the agent dimension; user isolation must stay strict."""
        shared = _emb()
        n = MemoryNode(
            user_id="u_other", agent_id="agent_a", session_id="s1",
            layer=MemoryLayer.L2_FACT, content="Another user's secret",
            owner="user", status=MemoryStatus.ACTIVE, is_latest=True,
        )
        n.memory_at = datetime.now()
        n.gmt_created = datetime.now()
        n.embedding = list(shared)
        run(store.upsert(n))

        llm = _reconcile(
            config, store, agent_id="agent_b",
            new_memories=["The user likes playing basketball"], emb=shared,
        )
        assert "Another user's secret" not in llm.last_prompt, "candidates must never be recalled across users"

    def test_same_agent_candidate_still_works(self, config, store):
        """Candidates from the same agent are still recalled (the relaxation must not break the original path)."""
        shared = _emb()
        _l2(store, content="The user likes playing basketball", agent_id="agent_b", emb=shared)

        llm = _reconcile(
            config, store, agent_id="agent_b",
            new_memories=["The user loves playing basketball"], emb=shared,
        )
        assert "basketball" in llm.last_prompt


class TestBackwardCompatibility:
    def test_cross_agent_chain_still_recallable(self, store):
        """After a cross-agent SUPERSEDE, searching as agent_a still yields the evolved latest value.

        This is the compatibility basis for the decision "new nodes record the current agent in agent_id":
          - The VDB filter on agent_a only picks up the old node left on agent_a (SUPERSEDED)
          - _trace_full_chain traces both ways along superseded_by to the chain head on agent_b
          - The chain head is returned as representative, plus the full evolution_chain
        """
        from hy_memory.pipelines._retrieval.evolution import expand_evolution_chains

        shared = _emb()
        old_id = _l2(store, content="The user is a vegetarian", agent_id="agent_a",
                     emb=shared, days=30,
                     status=MemoryStatus.SUPERSEDED, is_latest=False)
        new_id = _l2(store, content="The user now eats fish", agent_id="agent_b",
                     emb=shared, days=1, supersedes=[old_id])
        run(store.update_payload(old_id, {"superseded_by": [new_id]}))

        # Search as agent_a (the caller explicitly passes agent_ids=['agent_a'])
        hits = run(store.search(
            query_embedding=shared, user_id="u1", agent_ids=["agent_a"],
            layers=[MemoryLayer.L2_FACT], limit=10, score_threshold=0.0,
            status_filter=[MemoryStatus.ACTIVE, MemoryStatus.SUPERSEDED],
            only_latest=False,
        ))
        assert [h["node"].node_id for h in hits] == [old_id], \
            "the agent_a filter should only hit nodes belonging to agent_a"

        expanded = run(expand_evolution_chains(
            store, [h for h in hits if h.get("node")]
        ))
        assert len(expanded) == 1
        rep = expanded[0]["node"]
        # The representative node is the chain head on agent_b -- the evolved, correct value
        assert rep.node_id == new_id
        assert rep.agent_id == "agent_b"
        assert rep.content == "The user now eats fish"
        # The whole chain comes back, so history stays visible
        contents = [c["content"] for c in expanded[0]["evolution_chain"]]
        assert contents == ["The user now eats fish", "The user is a vegetarian"]
