# -*- coding: utf-8 -*-
"""
clone_user regression on a BM25 sparse qdrant collection.

Since 0.1.6.4, once qdrant enables fastembed BM25, the point.vector returned by scroll/retrieve is a
named multi-vector dict ({"": dense, "bm25": SparseVector}). clone reads src nodes through list_by_user,
and the old code assigned that whole dict to node.embedding; after the clone is upserted again,
_build_vector_field wraps the dict once more as if it were List[float] -> illegal nesting -> PointStruct
ValidationError → rollback → cloned_count=0。

This test, on a qdrant collection with sparse enabled:
  1. the embedding returned by list_by_user must be List[float] (not a dict);
  2. clone_user succeeds with cloned_count == the number of src nodes;
  3. dst can be read back by list_by_user with a matching count.
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


def _emb(dims=32):
    import random
    v = [random.gauss(0, 1) for _ in range(dims)]
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


def _qdrant_sparse_ok():
    try:
        import qdrant_client  # noqa
        from hy_memory.pipelines._retrieval import bm25_fastembed
        return bm25_fastembed.is_available()
    except Exception:
        return False


def _mk(node_id, content, uid="src_u", aid="a1"):
    n = MemoryNode(
        node_id=node_id, user_id=uid, agent_id=aid, session_id="s1",
        content=content, layer=MemoryLayer.L2_FACT,
        status=MemoryStatus.ACTIVE, is_latest=True,
    )
    n.embedding = _emb()
    return n


@pytest.mark.skipif(not _qdrant_sparse_ok(),
                    reason="qdrant-client + fastembed (BM25 sparse) required")
class TestCloneQdrantSparse:
    def test_clone_on_bm25_sparse_collection(self, tmp_path):
        from hy_memory.data.vector_store_qdrant import QdrantVectorStore

        cfg = MemoryConfig.from_dict({"vector_store": {
            "provider": "qdrant",
            "collection_name": f"t_clone_{uuid.uuid4().hex[:8]}",
            "persist_directory": str(tmp_path / "qdrant"),
            "embedding_dims": 32,
        }})
        vs = QdrantVectorStore(cfg)
        run(vs.initialize())
        try:
            # this collection must have BM25 sparse enabled (otherwise the regression is not exercised)
            assert vs._supports_fulltext, "expected BM25 sparse collection"

            src_uid = "src_user_clone"
            nodes = [_mk(f"n{i}", f"The user likes sport {i}", uid=src_uid) for i in range(5)]
            run(vs.upsert_batch(nodes))

            # 1) the embedding from list_by_user must be List[float], never a multi-vector dict
            listed = run(vs.list_by_user(src_uid, agent_id="a1"))
            assert len(listed) == 5
            for nd in listed:
                assert isinstance(nd.embedding, list), \
                    f"embedding should be List[float], got {type(nd.embedding)}"
                assert nd.embedding and not isinstance(nd.embedding[0], dict)

            # 2) simulate clone: copy the nodes read by list_by_user to a new user and upsert them
            #    (equivalent to the core data flow of async_clone_user, reproducing "read -> copy -> write")
            cloned = []
            for nd in listed:
                c = MemoryNode(
                    node_id=str(uuid.uuid4()), user_id="dst_user_clone",
                    agent_id="a1", session_id="s1", content=nd.content,
                    layer=nd.layer, status=MemoryStatus.ACTIVE, is_latest=True,
                )
                c.embedding = nd.embedding  # copy the embedding produced by the read path
                cloned.append(c)
            written = run(vs.upsert_batch(cloned))
            assert len(written) == 5, f"clone upsert should write 5 nodes, got {len(written)}"

            # 3) dst reads back with a matching count
            dst = run(vs.list_by_user("dst_user_clone", agent_id="a1"))
            assert len(dst) == 5
        finally:
            run(vs.close())
