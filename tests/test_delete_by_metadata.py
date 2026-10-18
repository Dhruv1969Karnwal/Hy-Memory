# -*- coding: utf-8 -*-
"""
Tests for delete_by_metadata -- verifying deletion by combinations of metadata fields

Coverage:
- delete_by_metadata on VectorStore (FAISS in-memory, Chroma embedded)
- Four deletion granularities: user / user+agent / user+agent+session / single row
- Verifies that search really finds nothing after deletion
- Edge cases: deletion on an empty store, nonexistent user_id, etc.
- delete_by_isolation_key regression (Chroma used to return -1)

How to run:
    pytest tests/test_delete_by_metadata.py -v
"""

import asyncio
import uuid
from typing import List

import pytest

from hy_memory.config import MemoryConfig
from hy_memory.models.memory import MemoryNode, MemoryLayer, MemoryStatus


# ==================== Helpers ====================

def run(coro):
    """Run a coroutine synchronously"""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_node(
    user_id: str,
    agent_id: str = "agent_default",
    session_id: str = "sess_default",
    content: str = "test content",
    dims: int = 32,
) -> MemoryNode:
    """Create a MemoryNode with an embedding"""
    import random
    node = MemoryNode(
        node_id=str(uuid.uuid4()),
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        content=content,
        layer=MemoryLayer.L2_FACT,
        status=MemoryStatus.ACTIVE,
    )
    # Random embedding, normalized
    vec = [random.gauss(0, 1) for _ in range(dims)]
    norm = sum(v * v for v in vec) ** 0.5
    node.embedding = [v / norm for v in vec]
    return node


# ==================== FAISS tests ====================

class TestFaissDeleteByMetadata:
    """delete_by_metadata tests for the FAISS VectorStore"""

    DIMS = 32

    @pytest.fixture
    def store(self, tmp_path):
        """Create and initialize a FAISS VectorStore"""
        config = MemoryConfig.from_dict({
            "vector_store": {
                "provider": "faiss",
                "collection_name": "test_delete",
                "persist_directory": str(tmp_path / "faiss_data"),
                "embedding_dims": self.DIMS,
            },
        })
        from hy_memory.data.vector_store_faiss import FaissVectorStore
        vs = FaissVectorStore(config)
        run(vs.initialize())
        yield vs
        run(vs.close())

    def _insert_nodes(self, store, nodes: List[MemoryNode]):
        """Insert nodes in batch"""
        run(store.upsert_batch(nodes))

    # --- Basic behaviour ---

    def test_delete_by_user_id_only(self, store):
        """Granularity 1: user_id only -> every memory of that user is deleted"""
        n1 = _make_node("u1", "a1", "s1", dims=self.DIMS)
        n2 = _make_node("u1", "a2", "s2", dims=self.DIMS)
        n3 = _make_node("u2", "a1", "s1", dims=self.DIMS)  # Different user
        self._insert_nodes(store, [n1, n2, n3])

        deleted = run(store.delete_by_metadata(user_id="u1"))
        assert deleted == 2

        # Everything of u1 is gone
        assert run(store.get_by_id(n1.node_id)) is None
        assert run(store.get_by_id(n2.node_id)) is None
        # u2 is still there
        assert run(store.get_by_id(n3.node_id)) is not None

    def test_delete_by_user_and_agent(self, store):
        """Granularity 2: user_id + agent_id -> all memories of that agent for the user"""
        n1 = _make_node("u1", "a1", "s1", dims=self.DIMS)
        n2 = _make_node("u1", "a1", "s2", dims=self.DIMS)
        n3 = _make_node("u1", "a2", "s1", dims=self.DIMS)  # Same user, different agent
        self._insert_nodes(store, [n1, n2, n3])

        deleted = run(store.delete_by_metadata(user_id="u1", agent_id="a1"))
        assert deleted == 2

        assert run(store.get_by_id(n1.node_id)) is None
        assert run(store.get_by_id(n2.node_id)) is None
        assert run(store.get_by_id(n3.node_id)) is not None

    def test_delete_by_user_agent_session(self, store):
        """Granularity 3: user_id + agent_id + session_id -> one exact session"""
        n1 = _make_node("u1", "a1", "s1", dims=self.DIMS)
        n2 = _make_node("u1", "a1", "s2", dims=self.DIMS)
        n3 = _make_node("u1", "a1", "s3", dims=self.DIMS)
        self._insert_nodes(store, [n1, n2, n3])

        deleted = run(store.delete_by_metadata(user_id="u1", agent_id="a1", session_id="s2"))
        assert deleted == 1

        assert run(store.get_by_id(n1.node_id)) is not None
        assert run(store.get_by_id(n2.node_id)) is None
        assert run(store.get_by_id(n3.node_id)) is not None

    # --- Edge cases ---

    def test_delete_empty_store(self, store):
        """Deleting from an empty store does not raise and returns 0"""
        deleted = run(store.delete_by_metadata(user_id="nonexistent"))
        assert deleted == 0

    def test_delete_nonexistent_user(self, store):
        """Deleting a nonexistent user returns 0 and leaves existing data untouched"""
        n1 = _make_node("u1", "a1", "s1", dims=self.DIMS)
        self._insert_nodes(store, [n1])

        deleted = run(store.delete_by_metadata(user_id="u_not_exist"))
        assert deleted == 0
        assert run(store.get_by_id(n1.node_id)) is not None

    def test_delete_user_all_then_count_zero(self, store):
        """After deleting all memories of a user its memory count is 0"""
        nodes = [_make_node("u1", f"a{i}", f"s{i}", dims=self.DIMS) for i in range(5)]
        self._insert_nodes(store, nodes)

        deleted = run(store.delete_by_metadata(user_id="u1"))
        assert deleted == 5

        # Confirm nothing is left behind
        for n in nodes:
            assert run(store.get_by_id(n.node_id)) is None

    def test_delete_idempotent(self, store):
        """Deleting the same data twice returns 0 the second time"""
        n1 = _make_node("u1", "a1", "s1", dims=self.DIMS)
        self._insert_nodes(store, [n1])

        d1 = run(store.delete_by_metadata(user_id="u1"))
        assert d1 == 1

        d2 = run(store.delete_by_metadata(user_id="u1"))
        assert d2 == 0

    # --- delete_by_isolation_key regression ---

    def test_delete_by_isolation_key_exact(self, store):
        """delete_by_isolation_key matches exactly, not by prefix"""
        n1 = _make_node("u1", "a1", "s1", dims=self.DIMS)
        n2 = _make_node("u1", "a1", "s2", dims=self.DIMS)
        self._insert_nodes(store, [n1, n2])

        ik1 = n1.get_isolation_key()  # u1::a1::s1
        deleted = run(store.delete_by_isolation_key(ik1))
        assert deleted == 1

        # n1 is deleted, n2 remains
        assert run(store.get_by_id(n1.node_id)) is None
        assert run(store.get_by_id(n2.node_id)) is not None

    def test_delete_by_isolation_key_prefix_does_not_match(self, store):
        """Passing the prefix 'u1::' to delete_by_isolation_key must match no data"""
        n1 = _make_node("u1", "a1", "s1", dims=self.DIMS)
        self._insert_nodes(store, [n1])

        # This was exactly the old bug: the client passed "u1::" to delete_by_isolation_key
        deleted = run(store.delete_by_isolation_key("u1::"))
        assert deleted == 0  # The prefix matches nothing

        # The data is still there
        assert run(store.get_by_id(n1.node_id)) is not None


# ==================== Chroma tests ====================

class TestChromaDeleteByMetadata:
    """delete_by_metadata tests for the Chroma VectorStore"""

    DIMS = 32

    @pytest.fixture
    def store(self, tmp_path):
        """Create and initialize a Chroma VectorStore"""
        config = MemoryConfig.from_dict({
            "vector_store": {
                "provider": "chroma",
                "collection_name": "test_delete_chroma",
                "persist_directory": str(tmp_path / "chroma_data"),
                "embedding_dims": self.DIMS,
            },
        })
        from hy_memory.data.vector_store_chroma import ChromaVectorStore
        vs = ChromaVectorStore(config)
        run(vs.initialize())
        yield vs
        run(vs.close())

    def _insert_nodes(self, store, nodes: List[MemoryNode]):
        run(store.upsert_batch(nodes))

    def test_delete_by_user_id_only(self, store):
        """Granularity 1: user_id only"""
        n1 = _make_node("u1", "a1", "s1", dims=self.DIMS)
        n2 = _make_node("u1", "a2", "s2", dims=self.DIMS)
        n3 = _make_node("u2", "a1", "s1", dims=self.DIMS)
        self._insert_nodes(store, [n1, n2, n3])

        deleted = run(store.delete_by_metadata(user_id="u1"))
        assert deleted == 2

        assert run(store.get_by_id(n1.node_id)) is None
        assert run(store.get_by_id(n2.node_id)) is None
        assert run(store.get_by_id(n3.node_id)) is not None

    def test_delete_by_user_and_agent(self, store):
        """Granularity 2: user_id + agent_id"""
        n1 = _make_node("u1", "a1", "s1", dims=self.DIMS)
        n2 = _make_node("u1", "a1", "s2", dims=self.DIMS)
        n3 = _make_node("u1", "a2", "s1", dims=self.DIMS)
        self._insert_nodes(store, [n1, n2, n3])

        deleted = run(store.delete_by_metadata(user_id="u1", agent_id="a1"))
        assert deleted == 2

        assert run(store.get_by_id(n1.node_id)) is None
        assert run(store.get_by_id(n2.node_id)) is None
        assert run(store.get_by_id(n3.node_id)) is not None

    def test_delete_by_user_agent_session(self, store):
        """Granularity 3: user_id + agent_id + session_id"""
        n1 = _make_node("u1", "a1", "s1", dims=self.DIMS)
        n2 = _make_node("u1", "a1", "s2", dims=self.DIMS)
        self._insert_nodes(store, [n1, n2])

        deleted = run(store.delete_by_metadata(user_id="u1", agent_id="a1", session_id="s1"))
        assert deleted == 1

        assert run(store.get_by_id(n1.node_id)) is None
        assert run(store.get_by_id(n2.node_id)) is not None

    def test_delete_empty_store(self, store):
        """Deleting from an empty store returns 0"""
        deleted = run(store.delete_by_metadata(user_id="nonexistent"))
        assert deleted == 0

    def test_delete_by_isolation_key_returns_count(self, store):
        """Chroma delete_by_isolation_key now returns an exact count (no more -1)"""
        n1 = _make_node("u1", "a1", "s1", dims=self.DIMS)
        n2 = _make_node("u1", "a1", "s1", dims=self.DIMS)
        self._insert_nodes(store, [n1, n2])

        ik = n1.get_isolation_key()
        deleted = run(store.delete_by_isolation_key(ik))
        assert deleted == 2  # Exact count, not -1

    def test_delete_by_isolation_key_prefix_does_not_match(self, store):
        """A prefix matches no data"""
        n1 = _make_node("u1", "a1", "s1", dims=self.DIMS)
        self._insert_nodes(store, [n1])

        deleted = run(store.delete_by_isolation_key("u1::"))
        assert deleted == 0
        assert run(store.get_by_id(n1.node_id)) is not None

    def test_delete_idempotent(self, store):
        """Repeated deletion"""
        n1 = _make_node("u1", "a1", "s1", dims=self.DIMS)
        self._insert_nodes(store, [n1])

        d1 = run(store.delete_by_metadata(user_id="u1"))
        assert d1 == 1

        d2 = run(store.delete_by_metadata(user_id="u1"))
        assert d2 == 0


# ==================== Multi-user isolation tests ====================

class TestMultiUserIsolation:
    """Isolation in multi-user scenarios (FAISS)"""

    DIMS = 32

    @pytest.fixture
    def store(self, tmp_path):
        config = MemoryConfig.from_dict({
            "vector_store": {
                "provider": "faiss",
                "collection_name": "test_isolation",
                "persist_directory": str(tmp_path / "faiss_iso"),
                "embedding_dims": self.DIMS,
            },
        })
        from hy_memory.data.vector_store_faiss import FaissVectorStore
        vs = FaissVectorStore(config)
        run(vs.initialize())
        yield vs
        run(vs.close())

    def test_delete_one_user_does_not_affect_others(self, store):
        """Deleting all memories of user_A leaves user_B and user_C untouched"""
        nodes_a = [_make_node("A", f"a{i}", f"s{i}", dims=self.DIMS) for i in range(3)]
        nodes_b = [_make_node("B", f"a{i}", f"s{i}", dims=self.DIMS) for i in range(3)]
        nodes_c = [_make_node("C", f"a{i}", f"s{i}", dims=self.DIMS) for i in range(3)]
        run(store.upsert_batch(nodes_a + nodes_b + nodes_c))

        deleted = run(store.delete_by_metadata(user_id="A"))
        assert deleted == 3

        # B and C are intact
        for n in nodes_b + nodes_c:
            assert run(store.get_by_id(n.node_id)) is not None

    def test_delete_one_agent_under_user(self, store):
        """Deleting one agent of a user leaves the other agents unaffected"""
        n1 = _make_node("u1", "chat", "s1", dims=self.DIMS)
        n2 = _make_node("u1", "chat", "s2", dims=self.DIMS)
        n3 = _make_node("u1", "work", "s1", dims=self.DIMS)
        n4 = _make_node("u1", "work", "s2", dims=self.DIMS)
        run(store.upsert_batch([n1, n2, n3, n4]))

        deleted = run(store.delete_by_metadata(user_id="u1", agent_id="chat"))
        assert deleted == 2

        assert run(store.get_by_id(n1.node_id)) is None
        assert run(store.get_by_id(n2.node_id)) is None
        assert run(store.get_by_id(n3.node_id)) is not None
        assert run(store.get_by_id(n4.node_id)) is not None

    def test_large_batch_delete(self, store):
        """Insert 100 rows in batch, then delete by user"""
        nodes = [_make_node("bulk_user", f"a{i%5}", f"s{i%10}", dims=self.DIMS) for i in range(100)]
        run(store.upsert_batch(nodes))

        deleted = run(store.delete_by_metadata(user_id="bulk_user"))
        assert deleted == 100

        # Everything is deleted
        for n in nodes:
            assert run(store.get_by_id(n.node_id)) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
