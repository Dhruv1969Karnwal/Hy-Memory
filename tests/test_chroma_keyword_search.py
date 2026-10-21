# -*- coding: utf-8 -*-
"""
End-to-end tests for Chroma keyword_search.

Note: local embedded Chroma (PersistentClient) does not support native BM25/FTS retrieval --
the new collection.search() Search expression engine exists only on Chroma Server / Cloud (the Rust
distributed backend), so a local call raises NotImplementedError("Search is not implemented
for Local Chroma"). The old $contains substring match can only yield a binary 1.0 score, which the
reader-side normalize_bm25 sigmoid squashes to ~0.039: it adds almost nothing to fusion while
diluting strong semantic hits, so keyword_search deliberately returns [] on the chroma backend (an honest degrade to pure vector recall).

-> These assertions about "having hits" do not hold for local chroma, so the whole file is skipped.
  Use a backend with sparse/BM25 support (qdrant) for real keyword retrieval. chromadb>=1.5.9
  already has BM25 in server mode, so this can be re-enabled once chroma server is in place.
"""

import asyncio
import random
import uuid
from typing import List

import pytest

from hy_memory.config import MemoryConfig
from hy_memory.models.memory import MemoryNode, MemoryLayer, MemoryStatus

pytestmark = pytest.mark.skip(
    reason="Local Chroma has no native BM25/FTS (the Search engine only exists on server/cloud), so "
           "keyword_search deliberately returns []. Real keyword retrieval uses the qdrant backend."
)


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_node(
    *,
    content: str,
    user_id: str = "u1",
    agent_id: str = "a1",
    session_id: str = "s1",
    layer: MemoryLayer = MemoryLayer.L2_FACT,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    is_latest: bool = True,
    dims: int = 32,
) -> MemoryNode:
    node = MemoryNode(
        node_id=str(uuid.uuid4()),
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        content=content,
        layer=layer,
        status=status,
        is_latest=is_latest,
    )
    vec = [random.gauss(0, 1) for _ in range(dims)]
    norm = sum(v * v for v in vec) ** 0.5
    node.embedding = [v / norm for v in vec]
    return node


@pytest.fixture
def store(tmp_path):
    config = MemoryConfig.from_dict({
        "vector_store": {
            "provider": "chroma",
            "collection_name": f"test_kw_{uuid.uuid4().hex[:8]}",
            "persist_directory": str(tmp_path / "chroma_data"),
            "embedding_dims": 32,
        },
    })
    from hy_memory.data.vector_store_chroma import ChromaVectorStore
    vs = ChromaVectorStore(config)
    run(vs.initialize())
    yield vs
    run(vs.close())


class TestChromaKeywordSearchChinese:

    def test_chinese_query_hits_chinese_content(self, store):
        nodes = [
            _make_node(content="The user prefers pragmatic solutions and actively looks for updates", user_id="u1"),
            _make_node(content="The weather is nice today, with many people walking in the park", user_id="u1"),
            _make_node(content="The user likes spicy food; Sichuan dishes and hotpot are both fine", user_id="u1"),
        ]
        run(store.upsert_batch(nodes))

        hits = run(store.keyword_search(query="Users tend to accept pragmatic solutions", user_id="u1", top_k=5))
        ids = {h["node_id"] for h in hits}
        # Hits entry 1 (contains "user/tend/accept/pragmatic") and may also hit entry 3 (contains "user")
        assert nodes[0].node_id in ids
        # Entry 2 must not hit (it has none of "user/tend/accept/pragmatic")
        assert nodes[1].node_id not in ids

    def test_chinese_no_overlap_returns_empty(self, store):
        nodes = [
            _make_node(content="The weather is nice today, with many people in the park", user_id="u1"),
        ]
        run(store.upsert_batch(nodes))

        hits = run(store.keyword_search(query="a completely unrelated query", user_id="u1", top_k=5))
        assert hits == []

    def test_score_is_binary_one(self, store):
        """Aligned with Qdrant behavior: every hit scores 1.0"""
        nodes = [_make_node(content="Users tend to accept pragmatic solutions", user_id="u1")]
        run(store.upsert_batch(nodes))

        hits = run(store.keyword_search(query="user", user_id="u1", top_k=5))
        assert len(hits) == 1
        assert hits[0]["score"] == 1.0


class TestChromaKeywordSearchEnglish:

    def test_english_query_hits(self, store):
        nodes = [
            _make_node(content="How to deploy to production with twine upload", user_id="u1"),
            _make_node(content="User likes coffee in the morning", user_id="u1"),
        ]
        run(store.upsert_batch(nodes))

        hits = run(store.keyword_search(query="deploy production", user_id="u1", top_k=5))
        ids = {h["node_id"] for h in hits}
        assert nodes[0].node_id in ids
        assert nodes[1].node_id not in ids


class TestChromaKeywordSearchFilters:

    def test_user_id_filter(self, store):
        nodes = [
            _make_node(content="Users tend to accept pragmatic solutions", user_id="u1"),
            _make_node(content="Users tend to accept pragmatic solutions", user_id="u2"),
        ]
        run(store.upsert_batch(nodes))

        hits = run(store.keyword_search(query="users tend", user_id="u1", top_k=10))
        ids = {h["node_id"] for h in hits}
        assert nodes[0].node_id in ids
        assert nodes[1].node_id not in ids

    def test_agent_id_filter(self, store):
        nodes = [
            _make_node(content="Users tend to accept pragmatic solutions", user_id="u1", agent_id="a1"),
            _make_node(content="Users tend to accept pragmatic solutions", user_id="u1", agent_id="a2"),
        ]
        run(store.upsert_batch(nodes))

        hits = run(store.keyword_search(
            query="users tend", user_id="u1", agent_ids=["a1"], top_k=10,
        ))
        ids = {h["node_id"] for h in hits}
        assert nodes[0].node_id in ids
        assert nodes[1].node_id not in ids

    def test_layer_filter(self, store):
        nodes = [
            _make_node(content="Users tend to accept pragmatic solutions", user_id="u1", layer=MemoryLayer.L2_FACT),
            _make_node(content="Users tend to accept pragmatic solutions", user_id="u1", layer=MemoryLayer.L4_IDENTITY),
        ]
        run(store.upsert_batch(nodes))

        hits = run(store.keyword_search(
            query="users tend", user_id="u1", layers=[MemoryLayer.L4_IDENTITY], top_k=10,
        ))
        ids = {h["node_id"] for h in hits}
        assert nodes[1].node_id in ids
        assert nodes[0].node_id not in ids

    def test_status_filter_excludes_non_active(self, store):
        nodes = [
            _make_node(content="Users tend to accept pragmatic solutions", user_id="u1", status=MemoryStatus.ACTIVE),
            _make_node(content="Users tend to accept pragmatic solutions", user_id="u1", status=MemoryStatus.SUPERSEDED),
        ]
        run(store.upsert_batch(nodes))

        # Only active is searched by default
        hits = run(store.keyword_search(query="users tend", user_id="u1", top_k=10))
        ids = {h["node_id"] for h in hits}
        assert nodes[0].node_id in ids
        assert nodes[1].node_id not in ids

    def test_only_latest(self, store):
        nodes = [
            _make_node(content="Users tend to accept pragmatic solutions", user_id="u1", is_latest=True),
            _make_node(content="Users tend to accept pragmatic solutions", user_id="u1", is_latest=False),
        ]
        run(store.upsert_batch(nodes))

        hits = run(store.keyword_search(
            query="users tend", user_id="u1", only_latest=True, top_k=10,
        ))
        ids = {h["node_id"] for h in hits}
        assert nodes[0].node_id in ids
        assert nodes[1].node_id not in ids


class TestChromaKeywordSearchEdgeCases:

    def test_empty_query_returns_empty(self, store):
        hits = run(store.keyword_search(query="", user_id="u1"))
        assert hits == []

    def test_whitespace_only_query_returns_empty(self, store):
        hits = run(store.keyword_search(query="   ", user_id="u1"))
        assert hits == []

    def test_top_k_caps_results(self, store):
        nodes = [
            _make_node(content=f"Users tend candidate {i}", user_id="u1") for i in range(5)
        ]
        run(store.upsert_batch(nodes))

        hits = run(store.keyword_search(query="users tend", user_id="u1", top_k=3))
        assert len(hits) <= 3
