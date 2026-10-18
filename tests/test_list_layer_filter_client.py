# -*- coding: utf-8 -*-
"""
list path server-side layers filtering (R3 requirement: CTX needs to list only certain kinds by layer / category).

The underlying list_by_user has long supported layers; what was missing was the client not passing it through. This file covers the semantics after pass-through:

- When layers is not passed, the result must be byte-identical to before the change (all layers except L1_RAW),
  the established list semantics in production, which must not change because of a new parameter.
- When layers is passed, the caller fully decides, including an explicit "only L1_RAW" request.
- When none of the specified layers live in the graph DB (L6/L7), skip the graph query -- connecting itself has overhead, and in pro mode it is a wasted connection.
"""

import asyncio
import uuid

import pytest

from hy_memory.config import MemoryConfig
from hy_memory.client import HyMemoryClient
from hy_memory.models.memory import MemoryNode, MemoryLayer, MemoryStatus


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


DIMS = 32


def _node(user_id="u1", content="x", layer=MemoryLayer.L2_FACT):
    import random
    n = MemoryNode(
        node_id=str(uuid.uuid4()),
        user_id=user_id,
        agent_id="a1",
        session_id="s1",
        content=content,
        layer=layer,
        status=MemoryStatus.ACTIVE,
        is_latest=True,
    )
    vec = [random.gauss(0, 1) for _ in range(DIMS)]
    norm = sum(v * v for v in vec) ** 0.5
    n.embedding = [v / norm for v in vec]
    return n


@pytest.fixture
def faiss_store(tmp_path):
    config = MemoryConfig.from_dict({
        "vector_store": {
            "provider": "faiss",
            "collection_name": "test_layer_filter",
            "persist_directory": str(tmp_path / "faiss_data"),
            "embedding_dims": DIMS,
        },
    })
    from hy_memory.data.vector_store_faiss import FaissVectorStore
    vs = FaissVectorStore(config)
    run(vs.initialize())
    yield vs
    run(vs.close())


def _client_with(store):
    c = HyMemoryClient.__new__(HyMemoryClient)
    c._vector_store = store
    return c


class TestNormalizeLayers:
    """Input normalization: accepting strings aligns with the search side (List[str])."""

    def test_none_and_empty_return_none(self):
        assert HyMemoryClient._normalize_layers(None) is None
        assert HyMemoryClient._normalize_layers([]) is None

    def test_string_converted_to_enum(self):
        got = HyMemoryClient._normalize_layers(["l2_fact", "l4_identity"])
        assert got == [MemoryLayer.L2_FACT, MemoryLayer.L4_IDENTITY]

    def test_enum_passthrough_and_mixed(self):
        got = HyMemoryClient._normalize_layers([MemoryLayer.L2_FACT, "l3_summary"])
        assert got == [MemoryLayer.L2_FACT, MemoryLayer.L3_SUMMARY]

    def test_legacy_alias_mapped(self):
        """The requirement doc writes the old number l5_schema, but it should actually land on l6_schema."""
        assert HyMemoryClient._normalize_layers(["l5_schema"]) == [MemoryLayer.L6_SCHEMA]

    def test_dedup_preserves_order(self):
        got = HyMemoryClient._normalize_layers(["l2_fact", "l2_fact", "l1_raw"])
        assert got == [MemoryLayer.L2_FACT, MemoryLayer.L1_RAW]

    def test_invalid_raises_not_silently_ignored(self):
        """Silently ignoring invalid values would make the caller think the filter took effect while it actually got everything -- it must raise.

        l3_identity is a typo in the requirement doc (L3 is actually summary), making it a fitting case.
        """
        with pytest.raises(ValueError):
            HyMemoryClient._normalize_layers(["l3_identity"])


class TestVdbBucketLayerFilter:
    def _seed(self, store):
        run(store.upsert_batch([
            _node(content="fact", layer=MemoryLayer.L2_FACT),
            _node(content="summary", layer=MemoryLayer.L3_SUMMARY),
            _node(content="identity", layer=MemoryLayer.L4_IDENTITY),
            _node(content="raw", layer=MemoryLayer.L1_RAW),
        ]))

    def test_no_layers_excludes_raw_as_before(self, faiss_store):
        """When not passed, semantics are unchanged: L1_RAW does not appear in list."""
        self._seed(faiss_store)
        got = run(_client_with(faiss_store)._list_vdb_bucket(
            user_id="u1", agent_id=None, limit=100, offset=0, order="desc",
        ))
        assert {m["content"] for m in got["memories"]} == {"fact", "summary", "identity"}

    def test_single_layer_filters(self, faiss_store):
        self._seed(faiss_store)
        got = run(_client_with(faiss_store)._list_vdb_bucket(
            user_id="u1", agent_id=None, limit=100, offset=0, order="desc",
            layers=[MemoryLayer.L2_FACT],
        ))
        assert [m["content"] for m in got["memories"]] == ["fact"]
        assert got["total"] == 1

    def test_multi_layer_filters(self, faiss_store):
        self._seed(faiss_store)
        got = run(_client_with(faiss_store)._list_vdb_bucket(
            user_id="u1", agent_id=None, limit=100, offset=0, order="desc",
            layers=[MemoryLayer.L2_FACT, MemoryLayer.L3_SUMMARY],
        ))
        assert {m["content"] for m in got["memories"]} == {"fact", "summary"}

    def test_explicit_raw_is_blocked(self, faiss_store):
        """Even explicitly asking only for L1_RAW gets nothing: raw is uniformly invisible to external read paths (a contract change).

        Before: the default exclusion was just a policy, so a caller naming L1_RAW got the caller's way.
        After: the vector store's allow_raw_layer defaults to False and the exclusion is sent down with the request,
        so raw never leaves the database; whatever layers the caller passes, it cannot get raw dialogue.
        Only internal pipelines (writer assembling dialogue history, clone full copy) opt in explicitly.
        """
        self._seed(faiss_store)
        got = run(_client_with(faiss_store)._list_vdb_bucket(
            user_id="u1", agent_id=None, limit=100, offset=0, order="desc",
            layers=[MemoryLayer.L1_RAW],
        ))
        assert got["memories"] == []
        assert got["total"] == 0

    def test_raw_mixed_with_other_layers_only_drops_raw(self, faiss_store):
        """When raw is mixed with other layers, only raw is dropped; the other layers return as usual."""
        self._seed(faiss_store)
        got = run(_client_with(faiss_store)._list_vdb_bucket(
            user_id="u1", agent_id=None, limit=100, offset=0, order="desc",
            layers=[MemoryLayer.L1_RAW, MemoryLayer.L2_FACT],
        ))
        assert [m["content"] for m in got["memories"]] == ["fact"]

    def test_layer_filter_composes_with_multi_user(self, faiss_store):
        """R3 combined with R1: multi-user + layer filtering both take effect."""
        run(faiss_store.upsert_batch([
            _node("u1", content="u1 fact", layer=MemoryLayer.L2_FACT),
            _node("u1", content="u1 summary", layer=MemoryLayer.L3_SUMMARY),
            _node("u2", content="u2 fact", layer=MemoryLayer.L2_FACT),
            _node("u3", content="u3 fact", layer=MemoryLayer.L2_FACT),
        ]))
        got = run(_client_with(faiss_store)._list_vdb_bucket(
            user_id="", agent_id=None, user_ids=["u1", "u2"], agent_ids=None,
            limit=100, offset=0, order="desc", layers=[MemoryLayer.L2_FACT],
        ))
        assert {m["content"] for m in got["memories"]} == {"u1 fact", "u2 fact"}


class TestGraphBucketLayerSkip:
    """Connecting to the graph DB has overhead; when none of the specified layers live in it, don't connect."""

    def _client_recording_graph_access(self):
        c = HyMemoryClient.__new__(HyMemoryClient)
        calls = []

        async def _acquire():
            calls.append(1)
            return None, False

        c._acquire_graph_store_for_purge = _acquire
        return c, calls

    def test_vdb_only_layers_skip_graph_store(self):
        c, calls = self._client_recording_graph_access()
        got = run(c._list_graph_bucket(
            user_id="u1", agent_id=None, limit=10, offset=0, order="desc",
            layers=[MemoryLayer.L2_FACT],
        ))
        assert got is None
        assert calls == [], "when all specified layers are on the VDB side, don't connect to the graph DB"

    def test_graph_layer_still_queries(self):
        c, calls = self._client_recording_graph_access()
        run(c._list_graph_bucket(
            user_id="u1", agent_id=None, limit=10, offset=0, order="desc",
            layers=[MemoryLayer.L7_INTENTION],
        ))
        assert calls == [1], "when a graph layer is specified, the graph DB must be queried"

    def test_no_layers_still_queries(self):
        c, calls = self._client_recording_graph_access()
        run(c._list_graph_bucket(
            user_id="u1", agent_id=None, limit=10, offset=0, order="desc",
        ))
        assert calls == [1], "when layers is not passed, behavior matches the pre-change behavior"
