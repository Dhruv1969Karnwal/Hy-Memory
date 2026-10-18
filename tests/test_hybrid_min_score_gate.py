# -*- coding: utf-8 -*-
"""
Regression tests for the hybrid output threshold and the per-item additive scoring.

New formula: final = 𝟙[bm25>0]*w_bm25*bm25 + 𝟙[tag]*w_tag*tag_cos
Memories without BM25 are no longer multiplied by the global has_bm25 factor 0.6.
"""

import asyncio
import math
import uuid

import pytest

from hy_memory.config import MemoryConfig
from hy_memory.models.memory import MemoryNode, MemoryLayer, MemoryStatus
from hy_memory.pipelines.base import ReadRequest


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _unit(vec):
    n = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / n for x in vec]


def _q():
    return _unit([1.0] + [0.0] * 31)


def _vec_a():
    return _unit([0.8, 0.6] + [0.0] * 30)


def _vec_b():
    s = 0.25
    return _unit([s, math.sqrt(1 - s * s)] + [0.0] * 30)


@pytest.fixture
def config(tmp_path):
    return MemoryConfig.from_dict({"vector_store": {
        "provider": "chroma",
        "collection_name": f"t_minscore_{uuid.uuid4().hex[:8]}",
        "persist_directory": str(tmp_path / "chroma"),
        "embedding_dims": 32,
    }})


def _mk(node_id, content, vec, tags=None):
    n = MemoryNode(
        node_id=node_id, user_id="u1", agent_id="a1", session_id="s1",
        content=content, layer=MemoryLayer.L2_FACT,
        status=MemoryStatus.ACTIVE, is_latest=True,
        tags=list(tags or []),
    )
    n.embedding = vec
    return n


class TestHybridAdditiveScore:
    def test_no_bm25_keeps_full_sem_above_min_score(self, config, monkeypatch):
        """B has no BM25: the old logic final=sem*0.6 dropped below 0.2; with final=sem it must be kept."""
        from hy_memory.data.vector_store_chroma import ChromaVectorStore
        from hy_memory.pipelines.reader_hybrid import HybridReadPipeline

        monkeypatch.setenv("MEMORY_HYBRID_V2_VDB_WEIGHT_BM25", "0.2")
        monkeypatch.setenv("MEMORY_HYBRID_V2_VDB_WEIGHT_TAG", "0.3")

        store = ChromaVectorStore(config)
        run(store.initialize())
        try:
            run(store.upsert(_mk("A", "The user's favorite spot is Nanshan Reservoir", _vec_a())))
            run(store.upsert(_mk("B", "The user plans to build an exercise habit starting with yoga", _vec_b())))

            reader = HybridReadPipeline(config, vector_store=store)
            run(reader.initialize())

            async def _fake_kw(query, top_k=10, user_id=None, agent_ids=None,
                               layers=None, status_filter=None, only_latest=True, **kw):
                return [{"node_id": "A", "score": 0.5, "node": None}]

            store.keyword_search = _fake_kw

            req = ReadRequest(
                query="Which is my favorite outing destination",
                query_embedding=_q(),
                user_ids=["u1"], agent_ids=["a1"],
                limit=10, min_score=0.2, schema_min_score=0.2,
            )
            resp = run(reader.read(req))

            ids = {m["memory_id"]: m["score"] for m in resp.memories}
            assert "A" in ids, f"A should be kept: {ids}"
            assert "B" in ids, (
                f"B has no BM25 but sem≈0.25 must still be kept (no more x0.6), actual scores {ids}"
            )
            assert ids["B"] >= 0.2
            for m in resp.memories:
                if m.get("source") in ("vdb", "vdb_keyword_only"):
                    assert m["score"] >= 0.2
        finally:
            run(store.close())

    def test_below_min_score_still_filtered(self, config, monkeypatch):
        """Items whose sem alone is below min_score must still be filtered out."""
        from hy_memory.data.vector_store_chroma import ChromaVectorStore
        from hy_memory.pipelines.reader_hybrid import HybridReadPipeline

        store = ChromaVectorStore(config)
        run(store.initialize())
        try:
            # cos≈0.1 < min_score 0.2
            low = _unit([0.1, math.sqrt(1 - 0.1 * 0.1)] + [0.0] * 30)
            run(store.upsert(_mk("L", "completely unrelated content xyz", low)))

            reader = HybridReadPipeline(config, vector_store=store)
            run(reader.initialize())

            async def _fake_kw(**kw):
                return []

            store.keyword_search = _fake_kw

            req = ReadRequest(
                query="Which is my favorite outing destination",
                query_embedding=_q(),
                user_ids=["u1"], agent_ids=["a1"],
                limit=10, min_score=0.2, schema_min_score=0.2,
            )
            resp = run(reader.read(req))
            ids = {m["memory_id"] for m in resp.memories}
            assert "L" not in ids
        finally:
            run(store.close())
