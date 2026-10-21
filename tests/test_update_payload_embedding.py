# -*- coding: utf-8 -*-
"""
Backend verification that update_payload also updates the embedding (chroma / qdrant / faiss).

Now that UPDATE updates in place, update_payload(node_id, {"content":..., "embedding": new_vec})
must really replace the stored vector too. For each backend this test:
  1. upsert a node whose vector is v_old;
  2. update_payload swaps the embedding to v_new (near-orthogonal to v_old);
  3. querying with v_new hits that node at cosine about 1; querying with v_old no longer scores high;
  4. the vector returned by get_embeddings == v_new.
"""

import asyncio
import uuid

import pytest

from hy_memory.config import MemoryConfig
from hy_memory.models.memory import MemoryNode, MemoryLayer, MemoryStatus


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _unit(vec):
    n = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / n for x in vec]


# 32 dims: v_old points at axis 0, v_new points at axis 1 (orthogonal)
def _v_old():
    return _unit([1.0] + [0.0] * 31)


def _v_new():
    return _unit([0.0, 1.0] + [0.0] * 30)


def _mk_node(emb):
    n = MemoryNode(
        node_id="n1", user_id="u1", agent_id="a1", session_id="s1",
        content="Original content", layer=MemoryLayer.L2_FACT,
        status=MemoryStatus.ACTIVE, is_latest=True,
    )
    n.embedding = emb
    return n


def _check_update_with_embedding(store):
    """Shared assertions: once update_payload carries an embedding, the vector really is replaced."""
    run(store.upsert(_mk_node(_v_old())))

    # in-place update: new content + new vector
    ok = run(store.update_payload("n1", {
        "content": "Updated content",
        "embedding": _v_new(),
    }))
    assert ok is True

    # 1) get_embeddings should return v_new
    embs = run(store.get_embeddings(["n1"]))
    assert "n1" in embs
    got = embs["n1"]
    # dot product with v_new is about 1, with v_old about 0
    dot_new = sum(a * b for a, b in zip(got, _v_new()))
    dot_old = sum(a * b for a, b in zip(got, _v_old()))
    assert dot_new > 0.99, f"stored vector should match v_new, dot={dot_new}"
    assert abs(dot_old) < 0.01, f"stored vector should NOT match v_old, dot={dot_old}"

    # 2) querying with v_new must hit n1 (high score)
    hits_new = run(store.search(query_embedding=_v_new(), user_id="u1", limit=5))
    ids_new = {h["node_id"]: h["score"] for h in hits_new}
    assert "n1" in ids_new
    assert ids_new["n1"] > 0.99

    # 3) content was updated too
    node = run(store.get_by_id("n1"))
    assert node.content == "Updated content"


class TestChromaUpdateEmbedding:
    def test_update_payload_with_embedding(self, tmp_path):
        from hy_memory.data.vector_store_chroma import ChromaVectorStore
        cfg = MemoryConfig.from_dict({"vector_store": {
            "provider": "chroma",
            "collection_name": f"t_upd_{uuid.uuid4().hex[:8]}",
            "persist_directory": str(tmp_path / "chroma"),
            "embedding_dims": 32,
        }})
        vs = ChromaVectorStore(cfg)
        run(vs.initialize())
        try:
            _check_update_with_embedding(vs)
        finally:
            run(vs.close())


def _qdrant_ok():
    try:
        import qdrant_client  # noqa
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _qdrant_ok(), reason="qdrant-client not installed")
class TestQdrantUpdateEmbedding:
    def test_update_payload_with_embedding(self, tmp_path):
        from hy_memory.data.vector_store_qdrant import QdrantVectorStore
        cfg = MemoryConfig.from_dict({"vector_store": {
            "provider": "qdrant",
            "collection_name": f"t_upd_{uuid.uuid4().hex[:8]}",
            "persist_directory": str(tmp_path / "qdrant"),
            "embedding_dims": 32,
        }})
        vs = QdrantVectorStore(cfg)
        run(vs.initialize())
        try:
            _check_update_with_embedding(vs)
        finally:
            run(vs.close())


def _faiss_ok():
    try:
        import faiss  # noqa
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _faiss_ok(), reason="faiss not installed")
class TestFaissUpdateEmbedding:
    def test_update_payload_with_embedding(self, tmp_path):
        from hy_memory.data.vector_store_faiss import FaissVectorStore
        cfg = MemoryConfig.from_dict({"vector_store": {
            "provider": "faiss",
            "collection_name": f"t_upd_{uuid.uuid4().hex[:8]}",
            "persist_directory": str(tmp_path / "faiss"),
            "embedding_dims": 32,
        }})
        vs = FaissVectorStore(cfg)
        run(vs.initialize())
        try:
            _check_update_with_embedding(vs)
        finally:
            run(vs.close())
