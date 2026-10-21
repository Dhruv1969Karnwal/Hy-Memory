# -*- coding: utf-8 -*-
"""
R2: pass-through of caller-supplied meta, plus append-on-merge.

Requirements (CTX):
- meta is a list; one memory node may carry several, each holding the context "at the moment of that write"
- add (new memory): use the meta passed in for this call; when one write yields several new memories they all carry the same one
- merge (merged into an existing memory): append to the old node's meta list, **keeping the old meta, never overwriting**
- meta is never embedded, takes no part in retrieval, and is never a filter condition

This file locks those semantics and the backward-compatibility contract: "with no meta passed, byte-identical to before".
"""

import asyncio
import inspect

import pytest

from hy_memory.client import HyMemoryClient
from hy_memory.pipelines.base import WriteRequest
from hy_memory.pipelines.writer import MemoryWriter


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _req(meta=None):
    return WriteRequest(content="x", client_meta=meta)


class _Op:
    """Reconcile op stand-in good enough here: _build_custom only touches supersede_reason."""

    def __init__(self, supersede_reason=None):
        self.supersede_reason = supersede_reason


META1 = {"source_platform": "codebuddy", "has_tool_call": True, "scene": "chat"}
META2 = {"source_platform": "workbuddy", "has_tool_call": False, "scene": "review"}


class TestWriteRequestField:
    def test_client_meta_defaults_to_none(self):
        assert WriteRequest(content="x").client_meta is None

    def test_client_meta_is_separate_from_extra(self):
        """extra holds SDK control keys (agent_mode / mode / layer); sharing one dict would let them overwrite each other."""
        req = WriteRequest(content="x", extra={"mode": "pro"}, client_meta={"mode": "biz"})
        assert req.extra == {"mode": "pro"}
        assert req.client_meta == {"mode": "biz"}


class TestAddEntrypoint:
    def test_async_add_exposes_meta(self):
        params = inspect.signature(HyMemoryClient.async_add).parameters
        assert "meta" in params
        assert params["meta"].default is None

    def test_sync_add_exposes_meta(self):
        params = inspect.signature(HyMemoryClient.add).parameters
        assert "meta" in params
        assert params["meta"].default is None


class TestBuildCustomOnNewNode:
    """A new node (ADD / SUPERSEDE / direct write / summary / intention) carries only the meta of this call."""

    def test_meta_wrapped_in_list(self):
        custom = MemoryWriter._build_custom(_Op(), "rid-1", _req(META1))
        assert custom["meta"] == [META1], "even a single meta must be a list, so the read side needs no branching"

    def test_no_meta_key_when_not_provided(self):
        """With no meta passed, custom must be byte-identical to before, with no extra empty keys."""
        custom = MemoryWriter._build_custom(_Op(), "rid-1", _req(None))
        assert custom == {"request_id": "rid-1"}

    def test_no_meta_key_when_request_absent(self):
        assert MemoryWriter._build_custom(_Op(), "rid-1") == {"request_id": "rid-1"}

    def test_empty_meta_treated_as_absent(self):
        assert "meta" not in MemoryWriter._build_custom(_Op(), "rid-1", _req({}))

    def test_coexists_with_internal_keys(self):
        custom = MemoryWriter._build_custom(_Op("contradiction"), "rid-1", _req(META1))
        assert custom == {
            "request_id": "rid-1",
            "supersede_reason": "contradiction",
            "meta": [META1],
        }

    def test_meta_is_copied_not_referenced(self):
        """Mutating the source dict after it lands in custom must not affect what was already persisted."""
        src = dict(META1)
        custom = MemoryWriter._build_custom(_Op(), "rid-1", _req(src))
        src["scene"] = "mutated"
        assert custom["meta"][0]["scene"] == "chat"

    def test_op_may_be_none(self):
        """Paths such as summary / intention have no reconcile op."""
        custom = MemoryWriter._build_custom(None, "rid-1", _req(META1))
        assert custom["meta"] == [META1]


class TestMergeAppendSemantics:
    """UPDATE rewrites the same node in place, so the old meta must survive -- the core contract of R2."""

    def test_appends_to_existing_chain(self):
        merged = MemoryWriter._merge_client_meta({"meta": [META1]}, _req(META2))
        assert merged["meta"] == [META1, META2], "the old meta stays in front, the new one is appended after"

    def test_accumulates_across_multiple_writes(self):
        custom = {"meta": [META1]}
        for i in range(3):
            custom = MemoryWriter._merge_client_meta(custom, _req({"n": i}))
        assert custom["meta"] == [META1, {"n": 0}, {"n": 1}, {"n": 2}]

    def test_first_meta_on_node_without_any(self):
        merged = MemoryWriter._merge_client_meta({"request_id": "old"}, _req(META1))
        assert merged["meta"] == [META1]

    def test_preserves_other_custom_keys(self):
        """custom is written back as a whole replacement, so internal fields must not be washed away by the meta merge."""
        merged = MemoryWriter._merge_client_meta(
            {"request_id": "old-rid", "basic_info_kv": {"name": "ivan"}, "meta": [META1]},
            _req(META2),
        )
        assert merged["request_id"] == "old-rid"
        assert merged["basic_info_kv"] == {"name": "ivan"}
        assert merged["meta"] == [META1, META2]

    def test_returns_none_when_no_meta_passed(self):
        """Returning None is the signal "no read-back or write-back needed", so callers not using this feature pay nothing."""
        assert MemoryWriter._merge_client_meta({"meta": [META1]}, _req(None)) is None

    def test_handles_missing_old_custom(self):
        assert MemoryWriter._merge_client_meta(None, _req(META1))["meta"] == [META1]

    def test_non_list_legacy_meta_is_not_dropped(self):
        """In legacy data meta may be a bare dict; degrade it to the first element instead of dropping it silently."""
        merged = MemoryWriter._merge_client_meta({"meta": META1}, _req(META2))
        assert merged["meta"] == [META1, META2]

    def test_does_not_mutate_input(self):
        old = {"meta": [META1]}
        MemoryWriter._merge_client_meta(old, _req(META2))
        assert old == {"meta": [META1]}, "the old custom read back must not be rewritten in place"


class TestUpdateAppendsThroughStore:
    """One end-to-end pass through the store: the UPDATE branch must really write the appended meta back to the node.

    The cases above only cover the pure function; they prove neither that the writer calls it nor that custom reaches
    update_payload. That is precisely R2's core risk -- a missing write-back silently loses the old meta.
    """

    DIMS = 32

    def _emb(self):
        import random

        vec = [random.gauss(0, 1) for _ in range(self.DIMS)]
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def _store(self, tmp_path):
        import uuid as _uuid

        from hy_memory.config import MemoryConfig
        from hy_memory.data.vector_store_faiss import FaissVectorStore

        cfg = MemoryConfig.from_dict({
            "vector_store": {
                "provider": "faiss",
                "collection_name": f"test_meta_{_uuid.uuid4().hex[:8]}",
                "persist_directory": str(tmp_path / "faiss_meta"),
                "embedding_dims": self.DIMS,
            },
        })
        store = FaissVectorStore(cfg)
        _run(store.initialize())
        return cfg, store

    def _writer(self, cfg, store, ops):
        from hy_memory.agent.reconciler import ReconcileResult
        from hy_memory.pipelines.writer import MemoryWriter

        writer = MemoryWriter(cfg, vector_store=store)
        _run(writer.initialize())

        outer = self

        class _FakeEmbed:
            async def embed_batch(self, texts):
                return [outer._emb() for _ in texts]

            async def embed_queued(self, text, use_cache=True):
                return outer._emb()

        writer._embed_service = _FakeEmbed()

        async def _fake_reconcile(*args, **kwargs):
            return ReconcileResult(ops=ops)

        writer.reconciler.reconcile = _fake_reconcile
        return writer

    def _seed(self, store, custom):
        import uuid as _uuid

        from hy_memory.models.memory import MemoryNode, MemoryLayer, MemoryStatus

        node = MemoryNode(
            node_id=str(_uuid.uuid4()), user_id="u1", agent_id="default_agent",
            session_id="s1", content="The user uses React.", layer=MemoryLayer.L2_FACT,
            status=MemoryStatus.ACTIVE, is_latest=True, owner="user",
            custom=custom,
        )
        node.embedding = self._emb()
        _run(store.upsert(node))
        return node

    def _update_op(self, node_id, content="The user uses React 18 + TS."):
        from hy_memory.agent.reconciler import ReconcileOp

        return ReconcileOp(
            op="UPDATE", memory_id=node_id, content=content,
            owner="user", tags=[], layer="L2_FACT",
        )

    def _write(self, writer, store, ops, meta):
        req = WriteRequest(content="x", user_id="u1", agent_id="default_agent",
                           client_meta=meta)
        _run(writer._reconcile_and_store(
            new_memory_texts=["The user uses React 18 + TS."],
            new_memories_meta=[{"content": "The user uses React 18 + TS.", "layer": "L2_FACT",
                                "tags": [], "owner": "user"}],
            request=req, vector_store=store, req_id="rid-new",
        ))

    def test_update_appends_to_existing_meta(self, tmp_path):
        cfg, store = self._store(tmp_path)
        node = self._seed(store, {"request_id": "rid-old", "meta": [META1]})
        writer = self._writer(cfg, store, [self._update_op(node.node_id)])

        self._write(writer, store, None, META2)

        got = _run(store.get_by_id(node.node_id))
        assert got.custom.get("meta") == [META1, META2], (
            f"old meta must be kept with the new one appended, got {got.custom.get('meta')}"
        )
        assert got.content == "The user uses React 18 + TS.", "the memory itself must still be updated properly"

    def test_update_without_meta_leaves_custom_untouched(self, tmp_path):
        """With no meta passed, custom must not be touched at all -- the zero-impact guarantee for existing write paths."""
        cfg, store = self._store(tmp_path)
        node = self._seed(store, {"request_id": "rid-old", "meta": [META1]})
        writer = self._writer(cfg, store, [self._update_op(node.node_id)])

        self._write(writer, store, None, None)

        got = _run(store.get_by_id(node.node_id))
        assert got.custom.get("meta") == [META1], "existing meta changed even though no meta was passed"
        assert got.custom.get("request_id") == "rid-old"

    def test_update_on_node_without_meta(self, tmp_path):
        cfg, store = self._store(tmp_path)
        node = self._seed(store, {"request_id": "rid-old"})
        writer = self._writer(cfg, store, [self._update_op(node.node_id)])

        self._write(writer, store, None, META1)

        got = _run(store.get_by_id(node.node_id))
        assert got.custom.get("meta") == [META1]
        assert got.custom.get("request_id") == "rid-old", "internal fields must not be washed away by the merge"

    def test_add_carries_single_meta(self, tmp_path):
        from hy_memory.agent.reconciler import ReconcileOp

        cfg, store = self._store(tmp_path)
        op = ReconcileOp(op="ADD", content="The user uses Vue.", owner="user",
                         tags=[], layer="L2_FACT")
        writer = self._writer(cfg, store, [op])

        self._write(writer, store, None, META1)

        nodes = _run(store.list_by_user(user_id="u1"))
        added = [n for n in nodes if n.content == "The user uses Vue."]
        assert len(added) == 1
        assert added[0].custom.get("meta") == [META1]


class TestReadPathsExposeMeta:
    """list / search / get must all expose meta at the top level, so the read side needs no two implementations."""

    def test_list_item_exposes_meta(self):
        from hy_memory.models.memory import MemoryNode, MemoryLayer

        node = MemoryNode(
            node_id="n1", user_id="u1", layer=MemoryLayer.L2_FACT, content="c",
            custom={"request_id": "rid", "meta": [META1]},
        )
        item = HyMemoryClient._memory_node_to_list_item(node)
        assert item["meta"] == [META1]

    def test_list_item_meta_empty_when_absent(self):
        from hy_memory.models.memory import MemoryNode, MemoryLayer

        node = MemoryNode(node_id="n1", user_id="u1", layer=MemoryLayer.L2_FACT, content="c")
        assert HyMemoryClient._memory_node_to_list_item(node)["meta"] == []

    def test_readers_expose_meta_from_custom(self):
        """The shared main flow of both readers must surface meta."""
        import hy_memory.pipelines.reader_base as reader_base

        src = inspect.getsource(reader_base)
        assert '"meta"' in src and 'get("meta")' in src

    def test_readers_do_not_leak_whole_custom(self):
        """custom holds internal fields such as request_id and must not be dumped into search result items wholesale."""
        import hy_memory.pipelines.reader_base as reader_base

        src = inspect.getsource(reader_base)
        assert '"custom": ' not in src
