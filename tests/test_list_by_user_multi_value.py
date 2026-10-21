# -*- coding: utf-8 -*-
"""
Multi-value enumeration in list_by_user (user_ids / agent_ids) and only_latest semantics.

Background (R1 requirement: CTX must fetch memories for several users in one call):
- list_by_user originally accepted a single user_id / agent_id, so N users required N sequential calls.
- Added the user_ids / agent_ids multi-value parameters; the single-value parameters stay for backward compatibility.

The only_latest default is the riskiest part of this change and must be pinned down:
- Default False (returns every version of an evolution chain). clone relies on this semantic to copy
  the complete chain; switching the default to True would silently drop every historical version.
- The list API path (client._list_vdb_bucket) also passes nothing: its filter conditions match the
  pre-change ones byte for byte (status=ACTIVE only); this change only adds multi-value enumeration.

Likewise, omitting status_filter must not fall back to filtering on ACTIVE.
"""

import asyncio
import inspect
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


DIMS = 32


def _node(user_id, agent_id="a1", content="x", is_latest=True,
          status=MemoryStatus.ACTIVE, layer=MemoryLayer.L2_FACT):
    import random
    n = MemoryNode(
        node_id=str(uuid.uuid4()),
        user_id=user_id,
        agent_id=agent_id,
        session_id="s1",
        content=content,
        layer=layer,
        status=status,
        is_latest=is_latest,
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
            "collection_name": "test_multi_value",
            "persist_directory": str(tmp_path / "faiss_data"),
            "embedding_dims": DIMS,
        },
    })
    from hy_memory.data.vector_store_faiss import FaissVectorStore
    vs = FaissVectorStore(config)
    run(vs.initialize())
    yield vs
    run(vs.close())


class TestMultiValueEnumeration:
    def test_user_ids_returns_union(self, faiss_store):
        run(faiss_store.upsert_batch([
            _node("u1", content="a"),
            _node("u2", content="b"),
            _node("u3", content="c"),
        ]))

        both = run(faiss_store.list_by_user(user_ids=["u1", "u2"]))
        assert {n.user_id for n in both} == {"u1", "u2"}, "should return only the two requested users"
        assert len(both) == 2

    def test_agent_ids_multi_value(self, faiss_store):
        run(faiss_store.upsert_batch([
            _node("u1", agent_id="g1"),
            _node("u1", agent_id="g2"),
            _node("u1", agent_id="g3"),
        ]))

        got = run(faiss_store.list_by_user(user_id="u1", agent_ids=["g1", "g3"]))
        assert {n.agent_id for n in got} == {"g1", "g3"}

    def test_single_value_still_works(self, faiss_store):
        """Backward compatibility: all 11 production call sites use the single-value form."""
        run(faiss_store.upsert_batch([
            _node("u1", agent_id="g1"),
            _node("u2", agent_id="g1"),
        ]))

        got = run(faiss_store.list_by_user(user_id="u1", agent_id="g1"))
        assert len(got) == 1
        assert got[0].user_id == "u1"

    def test_user_ids_takes_precedence_over_user_id(self, faiss_store):
        run(faiss_store.upsert_batch([_node("u1"), _node("u2")]))

        got = run(faiss_store.list_by_user(user_id="u1", user_ids=["u2"]))
        assert {n.user_id for n in got} == {"u2"}

    def test_no_user_at_all_returns_empty_not_full_scan(self, faiss_store):
        """Now that user_id went from a required positional argument to an optional one, passing neither must be rejected instead of degrading into a full table scan."""
        run(faiss_store.upsert_batch([_node("u1"), _node("u2")]))

        assert run(faiss_store.list_by_user()) == []
        assert run(faiss_store.list_by_user(user_ids=[])) == []


class TestOnlyLatestDefault:
    """Clone's lifeline: the default must return every version of the evolution chain."""

    def test_default_returns_history_versions(self, faiss_store):
        run(faiss_store.upsert_batch([
            _node("u1", content="v2 head", is_latest=True),
            _node("u1", content="v1 old", is_latest=False,
                  status=MemoryStatus.SUPERSEDED),
        ]))

        got = run(faiss_store.list_by_user(user_id="u1"))
        assert len(got) == 2, (
            "list_by_user must not filter is_latest by default -- clone relies on it to copy the full evolution chain, "
            "and changing the default to True would silently drop historical versions when cloning"
        )
        assert {n.content for n in got} == {"v2 head", "v1 old"}

    def test_default_returns_all_statuses(self, faiss_store):
        """Omitting status_filter must not fall back to filtering ACTIVE (clone depends on this as well)."""
        run(faiss_store.upsert_batch([
            _node("u1", content="active", status=MemoryStatus.ACTIVE),
            _node("u1", content="superseded", is_latest=False,
                  status=MemoryStatus.SUPERSEDED),
            _node("u1", content="shadow", is_latest=False,
                  status=MemoryStatus.SHADOW),
        ]))

        got = run(faiss_store.list_by_user(user_id="u1"))
        assert len(got) == 3
        assert {n.content for n in got} == {"active", "superseded", "shadow"}

    def test_only_latest_true_filters_dirty_data(self, faiss_store):
        """When True is passed explicitly the filter must really take effect (no caller today; the capability is reserved for later enablement).

        Build dirty data with status=ACTIVE but is_latest=False: normal writes never produce it (the
        writer sets both fields as a pair), so only dirty data proves the filter is really wired up --
        filtering by status alone would let this row slip through.
        """
        run(faiss_store.upsert_batch([
            _node("u1", content="head", is_latest=True),
            _node("u1", content="dirty", is_latest=False,
                  status=MemoryStatus.ACTIVE),
        ]))

        got = run(faiss_store.list_by_user(
            user_id="u1", status_filter=[MemoryStatus.ACTIVE], only_latest=True,
        ))
        assert [n.content for n in got] == ["head"]


class TestListVdbBucketWiring:
    """The client list path: whether multi-value arguments really are passed through to the backend."""

    def _client_with(self, store):
        from hy_memory.client import HyMemoryClient
        c = HyMemoryClient.__new__(HyMemoryClient)
        c._vector_store = store
        return c

    def test_multi_user_bucket_merges_and_filters(self, faiss_store):
        run(faiss_store.upsert_batch([
            _node("u1", content="u1 head"),
            _node("u2", content="u2 head"),
            _node("u3", content="u3 head"),
            # Historical version: status is already SUPERSEDED, so list must not show it
            _node("u1", content="u1 old", is_latest=False,
                  status=MemoryStatus.SUPERSEDED),
            # L1_RAW: list excludes it unconditionally
            _node("u1", content="raw", layer=MemoryLayer.L1_RAW),
        ]))

        got = run(self._client_with(faiss_store)._list_vdb_bucket(
            user_id="", agent_id=None, user_ids=["u1", "u2"],
            agent_ids=None, limit=100, offset=0, order="desc",
        ))

        contents = {m["content"] for m in got["memories"]}
        assert contents == {"u1 head", "u2 head"}, f"actual result {contents}"
        assert got["total"] == 2

    def test_filter_semantics_unchanged_from_before(self, faiss_store):
        """The list filter semantics must match the pre-change behavior exactly: it looks only at status, never at is_latest.

        Dirty data with status=ACTIVE but is_latest=False was returned before the change and must still
        be returned now. Layering only_latest on top would drop it, which means changing the response
        contract of an interface that is already live; that is explicitly out of scope for this change.
        """
        run(faiss_store.upsert_batch([
            _node("u1", content="head"),
            _node("u1", content="dirty", is_latest=False,
                  status=MemoryStatus.ACTIVE),
        ]))

        got = run(self._client_with(faiss_store)._list_vdb_bucket(
            user_id="u1", agent_id=None, limit=100, offset=0, order="desc",
        ))

        assert {m["content"] for m in got["memories"]} == {"head", "dirty"}

    def test_each_item_carries_user_id_for_attribution(self, faiss_store):
        """R1 acceptance: after mixed multi-user results come back, every memory must be traceable to its owner."""
        run(faiss_store.upsert_batch([
            _node("u1", content="a"), _node("u2", content="b"),
        ]))

        got = run(self._client_with(faiss_store)._list_vdb_bucket(
            user_id="", agent_id=None, user_ids=["u1", "u2"],
            agent_ids=None, limit=100, offset=0, order="desc",
        ))

        by_content = {m["content"]: m for m in got["memories"]}
        assert by_content["a"]["user_id"] == "u1"
        assert by_content["b"]["user_id"] == "u2"

    def test_pagination_applies_after_merge(self, faiss_store):
        run(faiss_store.upsert_batch(
            [_node("u1", content=f"u1-{i}") for i in range(3)]
            + [_node("u2", content=f"u2-{i}") for i in range(3)]
        ))

        c = self._client_with(faiss_store)
        page = run(c._list_vdb_bucket(
            user_id="", agent_id=None, user_ids=["u1", "u2"],
            agent_ids=None, limit=4, offset=0, order="desc",
        ))
        assert page["total"] == 6, "total is the count after merging"
        assert len(page["memories"]) == 4

        rest = run(c._list_vdb_bucket(
            user_id="", agent_id=None, user_ids=["u1", "u2"],
            agent_ids=None, limit=4, offset=4, order="desc",
        ))
        assert len(rest["memories"]) == 2

    def test_single_user_path_unchanged(self, faiss_store):
        """Existing single-value calls must stay field-for-field unchanged (R1 acceptance criterion)."""
        run(faiss_store.upsert_batch([_node("u1", content="a"), _node("u2", content="b")]))

        got = run(self._client_with(faiss_store)._list_vdb_bucket(
            user_id="u1", agent_id=None, limit=100, offset=0, order="desc",
        ))
        assert [m["content"] for m in got["memories"]] == ["a"]


class TestAllBackendSignatures:
    """The three backends plus the base class must keep matching signatures, otherwise switching providers breaks."""

    def _list_by_user_of(self, module_name, class_name):
        import importlib
        mod = importlib.import_module(f"hy_memory.data.{module_name}")
        return getattr(getattr(mod, class_name), "list_by_user")

    @pytest.mark.parametrize("module_name,class_name", [
        ("vector_store_base", "VectorStoreBase"),
        ("vector_store_qdrant", "QdrantVectorStore"),
        ("vector_store_chroma", "ChromaVectorStore"),
        ("vector_store_faiss", "FaissVectorStore"),
    ])
    def test_signature_has_multi_value_params(self, module_name, class_name):
        params = inspect.signature(
            self._list_by_user_of(module_name, class_name)
        ).parameters

        for name in ("user_ids", "agent_ids", "only_latest", "with_vectors"):
            assert name in params, f"{class_name}.list_by_user is missing {name}"

        assert params["only_latest"].default is False, (
            f"The only_latest default of {class_name}.list_by_user must be False"
        )
        assert params["with_vectors"].default is True, (
            f"The with_vectors default of {class_name}.list_by_user must be True so existing callers are unaffected"
        )
        assert params["user_id"].default is None, (
            f"user_id in {class_name}.list_by_user should be optional (replaced by user_ids)"
        )
