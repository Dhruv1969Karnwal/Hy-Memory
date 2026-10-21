# -*- coding: utf-8 -*-
"""
UPDATE updates the old node in place (chroma backend, driving writer._reconcile_and_store).

UPDATE semantics = merge and refine the same topic, essentially one continuing memory. After switching to in-place updates:
- node_id is unchanged (no more delete-old-create-new);
- content + embedding are replaced;
- access_count / last_accessed_at are preserved (not reset to cold start);
- no SHADOW garbage nodes are produced.
"""

import asyncio
import random
import uuid
from datetime import datetime, timedelta

import pytest

from hy_memory.config import MemoryConfig
from hy_memory.models.memory import MemoryNode, MemoryLayer, MemoryStatus
from hy_memory.agent.reconciler import ReconcileResult, ReconcileOp


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
    return MemoryConfig.from_dict({"vector_store": {
        "provider": "chroma",
        "collection_name": f"test_upd_{uuid.uuid4().hex[:8]}",
        "persist_directory": str(tmp_path / "chroma"),
        "embedding_dims": 32,
    }})


@pytest.fixture
def store(config):
    from hy_memory.data.vector_store_chroma import ChromaVectorStore
    vs = ChromaVectorStore(config)
    run(vs.initialize())
    yield vs
    run(vs.close())


class TestUpdateInPlace:
    def test_update_keeps_node_id_and_access_count(self, store, config):
        from hy_memory.pipelines.writer import MemoryWriter
        from hy_memory.pipelines.base import WriteRequest

        shared = _emb()
        new_vec = _emb()
        # old node: has accumulated access_count=7
        old = MemoryNode(node_id="m1", user_id="u1", agent_id="default_agent",
                         session_id="s1", content="The user is learning guitar",
                         layer=MemoryLayer.L2_FACT, status=MemoryStatus.ACTIVE,
                         is_latest=True)
        old.embedding = list(shared)
        old.access_count = 7
        old.last_accessed_at = datetime.now() - timedelta(days=2)
        old.gmt_created = datetime.now() - timedelta(days=30)
        run(store.upsert(old))

        writer = MemoryWriter(config, vector_store=store)
        run(writer.initialize())

        class _FakeEmbed:
            async def embed_batch(self, texts):
                return [list(new_vec) for _ in texts]
            async def embed_queued(self, text, use_cache=True):
                return list(new_vec)
        writer._embed_service = _FakeEmbed()

        new_content = "The user can play guitar and performs at open mics"
        async def _fake_reconcile(*args, **kwargs):
            return ReconcileResult(ops=[
                ReconcileOp(op="UPDATE", memory_id="m1", content=new_content,
                            layer="L2_FACT", owner="user", tags=["music"])
            ], success=True)
        writer._reconciler.reconcile = _fake_reconcile

        req = WriteRequest(content="x", user_id="u1", agent_id="default_agent", session_id="s1")
        stored_ids, err, ops, tok = run(writer._reconcile_and_store(
            new_memory_texts=[new_content],
            new_memories_meta=[{"content": new_content, "layer": "L2_FACT", "owner": "user", "tags": ["music"]}],
            request=req, vector_store=store, req_id="rq1",
        ))
        assert err is None
        assert stored_ids == ["m1"], "in-place update should reuse m1 and produce no new node_id"
        assert ops["update"] == 1
        assert ops["total"] == ops["add"] + ops["supersede"] + ops["update"]
        assert set(tok) == {"prompt", "completion", "total"}

        # m1 still present, content/embedding replaced, access_count preserved, still ACTIVE+is_latest
        node = run(store.get_by_id("m1"))
        assert node is not None
        assert node.content == new_content
        assert node.status == MemoryStatus.ACTIVE
        assert node.is_latest is True
        assert node.access_count == 7, "access_count must be preserved (not reset to cold start)"
        # embedding replaced with new_vec (dot product with new_vec ~1)
        embs = run(store.get_embeddings(["m1"]))
        dot = sum(a * b for a, b in zip(embs["m1"], new_vec))
        assert dot > 0.99, f"embedding should have been updated to new_vec, dot={dot}"

        # no SHADOW garbage: ACTIVE still only has m1
        active = run(store.search(query_embedding=list(new_vec), user_id="u1",
                                  layers=[MemoryLayer.L2_FACT], limit=10,
                                  status_filter=[MemoryStatus.ACTIVE]))
        assert {h["node_id"] for h in active} == {"m1"}
