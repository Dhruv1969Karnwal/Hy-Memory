# -*- coding: utf-8 -*-
"""
SUPERSEDE evolution chain recall (chroma backend).

Regression protection for two core behaviors:
- Nodes with status=SUPERSEDED also enter the recall pool (not only the ACTIVE chain head);
- Hitting any node on the chain makes expand_evolution_chains expand the whole chain in both
  directions and return the chain head (is_latest=True) as the representative;
- Several nodes of the same chain hitting at once are deduplicated into one entry (no repeated output).

Also protected: the reconcile candidate search still searches ACTIVE only (it must not pull SUPERSEDED
nodes in as candidates to supersede a second time).
"""

import asyncio
import random
import uuid
from datetime import datetime, timedelta

import pytest

from hy_memory.config import MemoryConfig
from hy_memory.models.memory import MemoryNode, MemoryLayer, MemoryStatus
from hy_memory.pipelines.reader_hybrid import HybridReadPipeline
from hy_memory.pipelines.base import ReadRequest


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
            "collection_name": f"test_chain_{uuid.uuid4().hex[:8]}",
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


def _node(node_id, content, *, status, is_latest, supersedes=None,
          superseded_by=None, emb=None, days_ago=0):
    n = MemoryNode(
        node_id=node_id, user_id="u1", agent_id="default_agent", session_id="s1",
        content=content, layer=MemoryLayer.L2_FACT, status=status,
        is_latest=is_latest,
        supersedes=supersedes, superseded_by=superseded_by,
    )
    n.memory_at = datetime.now() - timedelta(days=days_ago)
    n.embedding = emb or _emb()
    return n


def _seed_chain(store, shared_emb):
    """Seed a 3-node chain: E0 <- E1 <- E2 (E2 is the ACTIVE head, E0/E1 are SUPERSEDED).

    All three nodes share one vector so the query hits them all at once (verifying multi-hit dedup).
    """
    e0 = _node("E0", "The user lives in an apartment in Shibuya", status=MemoryStatus.SUPERSEDED,
               is_latest=False, superseded_by=["E1"], emb=list(shared_emb), days_ago=30)
    e1 = _node("E1", "The user moved to Shinjuku", status=MemoryStatus.SUPERSEDED,
               is_latest=False, supersedes=["E0"], superseded_by=["E2"],
               emb=list(shared_emb), days_ago=15)
    e2 = _node("E2", "The user now lives in Yokohama", status=MemoryStatus.ACTIVE,
               is_latest=True, supersedes=["E1"], emb=list(shared_emb), days_ago=1)
    for n in (e0, e1, e2):
        run(store.upsert(n))
    return e0, e1, e2


class TestSupersededRecall:
    def test_superseded_node_enters_recall_pool(self, store, config):
        """search defaults to ACTIVE only and would miss SUPERSEDED; hybrid must include SUPERSEDED."""
        shared = _emb()
        _seed_chain(store, shared)

        # Directly verifies the status_filter the reader uses: with ACTIVE only, just E2 is recalled
        only_active = run(store.search(
            query_embedding=list(shared), user_id="u1",
            layers=[MemoryLayer.L2_FACT], limit=10,
            status_filter=[MemoryStatus.ACTIVE], only_latest=False,
        ))
        assert {h["node_id"] for h in only_active} == {"E2"}

        # Once SUPERSEDED is included, E0/E1/E2 can all enter the pool
        with_super = run(store.search(
            query_embedding=list(shared), user_id="u1",
            layers=[MemoryLayer.L2_FACT], limit=10,
            status_filter=[MemoryStatus.ACTIVE, MemoryStatus.SUPERSEDED],
            only_latest=False,
        ))
        assert {h["node_id"] for h in with_super} == {"E0", "E1", "E2"}

    def test_reader_recalls_chain_and_dedups(self, store, config):
        """hybrid: several chain nodes hit at once -> expanded into one entry (the chain head) plus the full chain."""
        shared = _emb()
        _seed_chain(store, shared)

        reader = HybridReadPipeline(config, vector_store=store)
        run(reader.initialize())
        req = ReadRequest(query="Where does the user live", user_id="u1",
                          query_embedding=list(shared), min_score=0.0, limit=10)
        resp = run(reader.read(req))
        assert resp.success

        # The 3 nodes of one chain all hit, and dedup reduces them to a single entry
        l2 = [m for m in resp.memories if m.get("layer") == MemoryLayer.L2_FACT.value]
        assert len(l2) == 1, f"chain should dedup to 1, got {[m['memory_id'] for m in l2]}"
        # Representative node = chain head E2 (is_latest)
        assert l2[0]["memory_id"] == "E2"

    def test_hit_old_node_expands_full_chain(self, store, config):
        """Hitting an old node (not the chain head) still expands the full chain in both directions."""
        from hy_memory.pipelines._retrieval.evolution import expand_evolution_chains

        shared = _emb()
        e0, e1, e2 = _seed_chain(store, shared)

        # Simulate hitting only the middle old node E1 (with the node object attached)
        e1_node = run(store.get_by_id("E1"))
        hits = [{"node_id": "E1", "score": 0.9, "node": e1_node}]
        out = run(expand_evolution_chains(store, hits))

        assert len(out) == 1
        exp = out[0]
        # After two-way expansion the chain head E2 is the representative
        assert exp["node_id"] == "E2"
        assert exp.get("is_evolved") is True
        chain_ids = {c["node_id"] for c in exp["evolution_chain"]}
        assert chain_ids == {"E0", "E1", "E2"}


class TestReconcileCandidatesUnaffected:
    def test_reconcile_search_still_active_only(self, store, config):
        """The reconcile candidate search passes an explicit status_filter=[ACTIVE] and must not recall SUPERSEDED old nodes."""
        shared = _emb()
        _seed_chain(store, shared)
        # Candidate search semantics of reconcile: ACTIVE only + only_latest=False
        hits = run(store.search(
            query_embedding=list(shared), user_id="u1",
            layers=[MemoryLayer.L2_FACT], limit=10,
            status_filter=[MemoryStatus.ACTIVE], only_latest=False,
        ))
        assert {h["node_id"] for h in hits} == {"E2"}
