# -*- coding: utf-8 -*-
"""
Real end-to-end test: Chroma VDB + recall_intentions dedup

No VDB mock; a chromadb PersistentClient (temp directory) is used to fully verify:
  1. Upsert 3 intention nodes with identical content + memory_at + valid_until into Chroma
  2. Call recall_intentions to deduplicate
  3. Verify:
     - Only 1 result comes back (the copies were merged)
     - 2 nodes were deleted from the VDB asynchronously (delete_calls once the background create_task finishes)

The **only purpose** of this test is to expose the problem a mocked VDB hides:
  Chroma search defaults to include=["metadatas", "documents", "distances"], which excludes
  embeddings, so hit.node.embedding is None and the cosine dedup in _partition_duplicates
  silently degrades to "no dedup" because it cannot get the embedding.
"""

import asyncio
import shutil
import tempfile
import uuid
from datetime import datetime

import pytest

from hy_memory.config import MemoryConfig
from hy_memory.data.vector_store_chroma import ChromaVectorStore
from hy_memory.models.memory import MemoryLayer, MemoryNode, MemoryStatus
from hy_memory.pipelines._retrieval.intention import recall_intentions


def _one_hot(idx: int, dims: int = 32):
    v = [0.0] * dims
    v[idx] = 1.0
    return v


def _make_node(content, embedding, memory_at, valid_until,
               user_id="u_e2e", agent_id="a_e2e"):
    return MemoryNode(
        node_id=uuid.uuid4().hex,
        user_id=user_id,
        agent_id=agent_id,
        session_id="s_e2e",
        layer=MemoryLayer.L7_INTENTION,
        content=content,
        embedding=embedding,
        status=MemoryStatus.ACTIVE,
        is_latest=True,
        memory_at=memory_at,
        valid_until=valid_until,
    )


@pytest.mark.asyncio
async def test_recall_intentions_e2e_with_real_chroma():
    """Real Chroma backend: 3 identical intentions -> only 1 returned + 2 deleted asynchronously."""
    tmp = tempfile.mkdtemp(prefix="hymem_e2e_")
    try:
        cfg = MemoryConfig.from_env()
        cfg.vector_store.backend = "chroma"
        cfg.vector_store.persist_directory = tmp
        # Use a separate collection name to avoid leftovers
        vs = ChromaVectorStore(cfg)
        # Give the collection a unique name
        vs._collection_name = f"e2e_test_{uuid.uuid4().hex[:8]}"
        await vs.initialize()

        ma = datetime(2026, 8, 6, 10, 0)  # System time is 2026-08-06; a future valid_until avoids expiry
        vu = datetime(2026, 8, 31, 23, 59)
        emb = _one_hot(3)

        # 3 identical end-of-month product launches (same memory_at + same valid_until + same embedding)
        p_ids = []
        for i in range(3):
            n = _make_node("The user is launching a new product at the end of the month", emb, ma, vu)
            await vs.upsert(n)
            p_ids.append(n.node_id)

        # 1 different parents' meeting intention
        m = _make_node("The user has a parents' meeting next Friday", _one_hot(7), ma, datetime(2026, 8, 15))
        await vs.upsert(m)

        # Query with the product-launch vector; the top 3 by score should be product launches
        result = await recall_intentions(
            vs, emb,
            user_ids=["u_e2e"], agent_ids=["a_e2e"],
            limit=3,
            now=datetime(2026, 8, 6, 12, 0),
        )

        contents = [h["node"].content for h in result]
        print(f"\n[E2E] returned contents: {contents}")
        print(f"[E2E] returned count: {len(result)}")

        # Expected: 3 copies collapse into 1 + 1 parents' meeting = 2 distinct intentions
        # Assert: the product-launch intention appears exactly once in the results
        assert contents.count("The user is launching a new product at the end of the month") == 1, \
            f"E2E FAIL: expected 1 copy of 'product launch' after dedup, got {contents.count('The user is launching a new product at the end of the month')}"

        # Wait for the async delete tasks to finish
        pending = [t for t in asyncio.all_tasks() if not t.done() and t is not asyncio.current_task()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        # Verify the VDB really lost 2 nodes
        remaining_new_products = 0
        for pid in p_ids:
            got = await vs.get_by_id(pid)
            if got is not None:
                remaining_new_products += 1
        print(f"[E2E] VDB remaining new-product nodes: {remaining_new_products} (expected 1)")
        assert remaining_new_products == 1, \
            f"E2E FAIL: expected 1 new-product node left after async delete, got {remaining_new_products}"

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.asyncio
async def test_different_valid_until_e2e_not_deduped():
    """Real Chroma: same content + same memory_at + different valid_until -> both nodes are kept.

    The core case you discussed with a colleague:
      "meeting this Friday, said in the morning" vs "meeting next Friday, said in the afternoon"
      -> Same content and memory_at but a different valid_until: two different events, so they cannot merge.
    """
    tmp = tempfile.mkdtemp(prefix="hymem_e2e_")
    try:
        cfg = MemoryConfig.from_env()
        cfg.vector_store.backend = "chroma"
        cfg.vector_store.persist_directory = tmp
        vs = ChromaVectorStore(cfg)
        vs._collection_name = f"e2e_test_{uuid.uuid4().hex[:8]}"
        await vs.initialize()

        ma = datetime(2026, 8, 6, 10, 0)
        emb = _one_hot(5)

        # Two intentions with identical content and embedding but different valid_until (two separate meetings)
        n1 = _make_node("There is a meeting on Friday", emb, ma, datetime(2026, 8, 8))    # this Friday
        n2 = _make_node("There is a meeting on Friday", emb, ma, datetime(2026, 8, 15))   # next Friday
        await vs.upsert(n1)
        await vs.upsert(n2)

        result = await recall_intentions(
            vs, emb,
            user_ids=["u_e2e"], agent_ids=["a_e2e"],
            limit=10,
            now=datetime(2026, 8, 6, 12, 0),
        )

        assert len(result) == 2, \
            f"E2E FAIL: different valid_until should NOT be deduped, got {len(result)} results"

        # Wait for async work (there should be no delete task, but just in case)
        pending = [t for t in asyncio.all_tasks() if not t.done() and t is not asyncio.current_task()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        # Both nodes should still be in the VDB
        assert await vs.get_by_id(n1.node_id) is not None
        assert await vs.get_by_id(n2.node_id) is not None
        print("\n[E2E-different-vu] PASSED")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.asyncio
async def test_writer_batch_dedup_e2e():
    """Real Chroma: writer._store_intentions end to end, 3 identical intentions dedup to 1 stored in the batch."""
    from hy_memory.pipelines.base import WriteRequest
    from hy_memory.pipelines.writer import MemoryWriter

    class _FakeEmbedByHash:
        """Returns a different one-hot vector for different text."""
        def __init__(self, dims=32):
            self.dims = dims

        def _vec(self, text):
            v = [0.0] * self.dims
            v[abs(hash(text)) % self.dims] = 1.0
            return v

        async def embed_batch(self, texts):
            return [self._vec(t) for t in texts]

        async def embed_queued(self, text):
            return self._vec(text)

    tmp = tempfile.mkdtemp(prefix="hymem_e2e_")
    try:
        cfg = MemoryConfig.from_env()
        cfg.vector_store.backend = "chroma"
        cfg.vector_store.persist_directory = tmp
        vs = ChromaVectorStore(cfg)
        vs._collection_name = f"e2e_test_{uuid.uuid4().hex[:8]}"
        await vs.initialize()

        writer = MemoryWriter(
            cfg, embed_service=_FakeEmbedByHash(dims=32),
            vector_store=vs, cache=None,
        )
        req = WriteRequest(
            content="", messages=[], user_id="u_e2e", agent_id="a_e2e",
            session_id="s_e2e", request_id=uuid.uuid4().hex,
            memory_at=datetime(2026, 8, 6, 10, 0),
        )
        vu = datetime(2026, 8, 8)
        intentions = [
            {"content": "Launch the new product at month end", "valid_until": vu, "tags": ["a"]},
            {"content": "Launch the new product at month end", "valid_until": vu, "tags": ["b"]},
            {"content": "Launch the new product at month end", "valid_until": vu, "tags": ["c"]},
            {"content": "Parents' meeting on Friday", "valid_until": vu, "tags": []},
        ]
        ids = await writer._store_intentions(intentions, req, vs, req.request_id)
        print(f"\n[E2E-writer] stored ids: {len(ids)} (expected 2)")

        assert len(ids) == 2, \
            f"E2E FAIL: 3 duplicate intentions should collapse to 1, got {len(ids)} stored"

        # Confirm again from the VDB side
        for nid in ids:
            got = await vs.get_by_id(nid)
            assert got is not None

        print("[E2E-writer] PASSED")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(test_recall_intentions_e2e_with_real_chroma())
    print("\n[E2E-recall] PASSED")
    asyncio.run(test_different_valid_until_e2e_not_deduped())
    asyncio.run(test_writer_batch_dedup_e2e())
    print("\n[E2E ALL PASSED]")
