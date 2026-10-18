# -*- coding: utf-8 -*-
"""
L1_RAW is invisible to external read paths: the gate is welded onto the VDB's sole query entry point.

Design background (2026-08-31):
Business callers could retrieve raw dialogue via search / list / keyword_search. The previous approach
pulled raw back into the process along with everything else and filtered it out in Python (the old _list_vdb_bucket in `client.py`),
i.e. "post-query filtering": raw had left the database and entered this process's memory.

Now the exclusion condition is folded into the filter and sent down to the vdb with the request, so raw never leaves the database. The gate is welded onto
`_build_filter_expr` -- it is the common mandatory path of the four query methods search / keyword_search /
hybrid_search_native / list_by_user,
so upper layers cannot bypass it, and new endpoints are automatically protected.

Direct fetch by ID uses a primary-key query and does not pass through the filter, so it is blocked separately at `client.async_get`
(the common mandatory path of GET /v1/memories/{id} and /v1/memories/batch).

Internal pipelines that need to read raw dialogue pass allow_raw_layer=True explicitly to opt in:
- `writer._get_recent_dialogue_history`: assembles multi-turn dialogue history to feed the LLM before extraction
- `client.async_clone_user_memories`: full copy including raw
"""

import asyncio
import uuid

import pytest

from hy_memory.config import MemoryConfig
from hy_memory.models.memory import MemoryNode, MemoryLayer, MemoryStatus

RAW = MemoryLayer.L1_RAW.value


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestFaissBackendBlocksRaw:
    """The faiss backend (for tests and local dev) behaves consistently."""

    @pytest.fixture
    def store(self, tmp_path):
        from hy_memory.data.vector_store_faiss import FaissVectorStore
        # persist_directory must point into tmp_path: a wrong key name would write into
        # the real ~/.hy_memory data directory, polluting local data and interfering between cases.
        cfg = MemoryConfig.from_dict({
            "vector_store": {
                "provider": "faiss",
                "collection_name": "test_block_raw",
                "persist_directory": str(tmp_path / "faiss_data"),
                "embedding_dims": 32,
            },
        })
        s = FaissVectorStore(cfg)
        run(s.initialize())
        run(s.upsert_batch([
            self._node("u1", MemoryLayer.L1_RAW, "Raw dialogue"),
            self._node("u1", MemoryLayer.L2_FACT, "Fact"),
        ]))
        yield s

    @staticmethod
    def _node(user_id, layer, content):
        import random
        return MemoryNode(
            node_id=str(uuid.uuid4()),
            user_id=user_id, agent_id="a1", session_id="s1",
            content=content, layer=layer,
            status=MemoryStatus.ACTIVE, is_latest=True,
            embedding=[random.random() for _ in range(32)],
        )

    def test_default_blocks_raw(self, store):
        got = run(store.list_by_user(user_id="u1"))
        assert {n.layer for n in got} == {MemoryLayer.L2_FACT}

    def test_only_raw_returns_empty(self, store):
        got = run(store.list_by_user(user_id="u1", layers=[MemoryLayer.L1_RAW]))
        assert got == []

    def test_allow_raw_returns_it(self, store):
        got = run(store.list_by_user(
            user_id="u1", layers=[MemoryLayer.L1_RAW], allow_raw_layer=True,
        ))
        assert [n.layer for n in got] == [MemoryLayer.L1_RAW]


class TestInternalPipelinesStillReadRaw:
    """Internal opt-in points: missing any one causes functional regression, so each is asserted."""

    def test_writer_dialogue_history_allows_raw(self):
        """Assembling multi-turn dialogue history before extraction must be able to read raw, otherwise extraction quality drops."""
        import inspect
        from hy_memory.pipelines.writer import MemoryWriter
        src = inspect.getsource(MemoryWriter._get_recent_history)
        assert "allow_raw_layer=True" in src, \
            "writer does not opt raw in when assembling dialogue history; it would get no history and hurt extraction quality"

    def test_clone_allows_raw(self):
        """clone's semantics are a full copy including raw; missing it makes the new user lose all dialogue history."""
        import inspect
        from hy_memory.client import HyMemoryClient
        src = inspect.getsource(HyMemoryClient.async_clone_user)
        assert src.count("allow_raw_layer=True") >= 2, \
            "both clone's emptiness check and full listing must opt raw in"


class TestListNoLongerFiltersInProcess:
    """Regression: list must not pull raw back into the process and filter it out in Python.

    Xiang explicitly opposed "post-query filtering". If this Python filter is restored,
    raw would pass through this process's memory again -- a source assertion guards against the regression.
    """

    def test_no_python_side_raw_filter(self):
        import inspect
        from hy_memory.client import HyMemoryClient
        src = inspect.getsource(HyMemoryClient._list_vdb_bucket)
        assert "n.layer != MemoryLayer.L1_RAW" not in src, \
            "an in-process raw filter has reappeared in list; it should be done by the vdb-side filter"


class TestGetByIdPathBlocksRaw:
    """Direct fetch by ID: uses a primary-key query without the filter, so it must be blocked at async_get."""

    def test_async_get_drops_raw_node(self):
        import inspect
        from hy_memory.client import HyMemoryClient
        src = inspect.getsource(HyMemoryClient.async_get)
        assert "MemoryLayer.L1_RAW" in src, \
            "async_get does not block raw; batch and single fetch by ID would leak raw dialogue"

    def test_block_happens_before_return(self):
        """The block must come before both return statements, otherwise it is as good as absent."""
        import inspect
        from hy_memory.client import HyMemoryClient
        lines = inspect.getsource(HyMemoryClient.async_get).splitlines()
        block_at = next(
            i for i, l in enumerate(lines)
            if "MemoryLayer.L1_RAW" in l and "vdb_node" in l
        )
        first_return_at = next(
            i for i, l in enumerate(lines)
            if l.strip().startswith("return {")
        )
        assert block_at < first_return_at, \
            f"raw block is at line {block_at}, first return at line {first_return_at}, wrong order"
