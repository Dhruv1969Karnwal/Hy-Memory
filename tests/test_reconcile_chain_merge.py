# -*- coding: utf-8 -*-
"""
Evolution-chain merging for reconcile candidates (chroma backend + fake LLM).

Regression coverage (master fix):
- candidate search now includes SUPERSEDED nodes (previously ACTIVE only);
- a hit on any node of the chain (head or body) makes _trace_full_chain complete the chain and
  deduplicate it into a single candidate keyed on the head;
- existing_memories exposes only the chain head (memory_id + content),
  **never history_versions** (so the LLM cannot act on stale nodes).
"""

import asyncio
import json
import random
import uuid
from datetime import datetime, timedelta

import pytest

from hy_memory.config import MemoryConfig
from hy_memory.models.memory import MemoryNode, MemoryLayer, MemoryStatus
from hy_memory.agent.reconciler import MemoryReconciler
from hy_memory.pipelines._retrieval.evolution import trace_full_chains


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
            "collection_name": f"test_recon_{uuid.uuid4().hex[:8]}",
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


class _CaptureLLM:
    """Records the real reconcile prompt; returns an empty op array (candidate construction is unaffected)."""
    def __init__(self):
        self.last_prompt = ""

    async def complete(self, prompt, **kw):
        self.last_prompt = prompt
        class R:
            content = "[]"
            prompt_tokens = 10
            completion_tokens = 2
            tokens_used = 12
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


class _CountingChainStore:
    def __init__(self, nodes):
        self.nodes = {node.node_id: node for node in nodes}
        self.calls = []

    async def get_by_ids(self, ids):
        self.calls.append(list(ids))
        return [self.nodes[node_id] for node_id in ids if node_id in self.nodes]


@pytest.mark.asyncio
async def test_batch_chain_trace_fetches_each_depth_once():
    now = datetime.now()
    old = MemoryNode(
        node_id="old", user_id="u1", agent_id="a", session_id="s",
        content="old", layer=MemoryLayer.L2_FACT,
        status=MemoryStatus.SUPERSEDED, is_latest=False,
        superseded_by=["middle"], memory_at=now - timedelta(days=2),
    )
    middle = MemoryNode(
        node_id="middle", user_id="u1", agent_id="a", session_id="s",
        content="middle", layer=MemoryLayer.L2_FACT,
        status=MemoryStatus.SUPERSEDED, is_latest=False,
        supersedes=["old"], superseded_by=["head"],
        memory_at=now - timedelta(days=1),
    )
    head = MemoryNode(
        node_id="head", user_id="u1", agent_id="a", session_id="s",
        content="head", layer=MemoryLayer.L2_FACT,
        status=MemoryStatus.ACTIVE, is_latest=True,
        supersedes=["middle"], memory_at=now,
    )
    store = _CountingChainStore([old, middle, head])

    chains = await trace_full_chains(store, [old, middle])

    assert [node.node_id for node in chains["old"]] == ["head", "middle", "old"]
    assert [node.node_id for node in chains["middle"]] == ["head", "middle", "old"]
    assert store.calls == [["head"]]


def _seed_chain(store, shared):
    """E0 <- E1 <- E2 (E2 = ACTIVE chain head, E0/E1 = SUPERSEDED), same vector."""
    def mk(nid, content, status, is_latest, sup=None, sb=None, days=0):
        n = MemoryNode(
            node_id=nid, user_id="u1", agent_id="default_agent", session_id="s1",
            content=content, layer=MemoryLayer.L2_FACT, status=status,
            is_latest=is_latest, supersedes=sup, superseded_by=sb,
        )
        n.memory_at = datetime.now() - timedelta(days=days)
        n.embedding = list(shared)
        run(store.upsert(n))

    mk("E0", "The user lives in an apartment in Shibuya", MemoryStatus.SUPERSEDED, False, sb=["E1"], days=30)
    mk("E1", "The user moved to Shinjuku", MemoryStatus.SUPERSEDED, False, sup=["E0"], sb=["E2"], days=15)
    mk("E2", "The user now lives in Yokohama", MemoryStatus.ACTIVE, True, sup=["E1"], days=1)


def _run_reconcile(store, config, shared, new_mem):
    llm = _CaptureLLM()
    recon = MemoryReconciler(config)
    recon._get_llm = lambda: llm
    recon._enable_search_query = False
    res = run(recon.reconcile(
        new_memories=[new_mem],
        user_id="u1", agent_id="default_agent",
        vector_store=store, embed_service=_Embed(shared),
        request_id="rq1",
    ))
    return llm, res


class TestReconcileChainMerge:
    def test_chain_dedups_to_head_in_prompt(self, store, config):
        shared = _emb()
        _seed_chain(store, shared)
        llm, res = _run_reconcile(store, config, shared, "The user lives in Osaka")
        assert res.success

        # existing_memories in the prompt should show only the chain head E2, never the body nodes E0/E1
        assert "E2" in llm.last_prompt
        assert "E0" not in llm.last_prompt
        assert "E1" not in llm.last_prompt

    def test_prompt_has_no_history_versions(self, store, config):
        shared = _emb()
        _seed_chain(store, shared)
        llm, res = _run_reconcile(store, config, shared, "The user lives in Osaka")
        assert res.success
        # no history version field is exposed
        assert "history_versions" not in llm.last_prompt
        # content of the old nodes must not leak into the prompt either
        assert "Shibuya" not in llm.last_prompt
        assert "Shinjuku" not in llm.last_prompt
        # the chain head's content is present
        assert "Yokohama" in llm.last_prompt
