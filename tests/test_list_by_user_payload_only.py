# -*- coding: utf-8 -*-
"""
list_by_user payload-only call sites: whether `with_vectors=False` really reaches the backend.

Background: VectorStoreBase.list_by_user gained a `with_vectors` switch (default True for
backward compat); call sites that only need payload/tags/metadata must pass False explicitly,
to avoid pulling embeddings back too; otherwise a single listing may bring back MB-scale vectors.

Call sites covered by this fix (on the fix/list-by-user-without-vectors branch):

client.py:
  - `_list_vdb_bucket`                 (list API, serialized into list items)
  - `async_get_basic_info`             (read-only custom.basic_info_kv + chain)
  - `async_build_entity_store`         (rebuild entity using only node_id/content)
  - `async_clone_user` dst empty check   (only checks length)

writer.py:
  - `_get_recent_history`              (read-only content/node_id/gmt_created)
  - `_collect_existing_tags`           (already covered on the same branch)

basic_profile.py:
  - `_find_latest_l0`                  (read-only layer/custom/is_latest)
  - `_assemble_full_kv`                (read-only custom.basic_info_kv)

Negative assertions (call sites that must retain embeddings):
  - `async_clone_user` src full copy    (embedding is copied via to_dict/from_dict)
"""

from datetime import datetime, timedelta
import asyncio
import uuid

import pytest

from hy_memory.models.memory import (
    MemoryLayer,
    MemoryNode,
    MemoryStatus,
)


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# fake vector store: capture every list_by_user kwargs
# ---------------------------------------------------------------------------

class RecordingVectorStore:
    """Only cares about list_by_user arguments; return values can be preset."""

    def __init__(self, nodes=None):
        self.nodes = nodes or []
        self.calls = []  # kwargs of each list_by_user call
        self.get_calls = []

    async def list_by_user(self, **kwargs):
        self.calls.append(kwargs)
        return list(self.nodes)

    async def get_by_id(self, node_id):
        self.get_calls.append(node_id)
        return None

    # the following methods are referenced on some call paths; provide safe default implementations
    async def delete_entities_for_memory(self, **kwargs):
        return None

    async def upsert_batch(self, nodes):
        return [n.node_id for n in nodes]


def _basic_info_node(kv=None, is_latest=True, gmt_created=None):
    n = MemoryNode(
        node_id=str(uuid.uuid4()),
        user_id="u1",
        agent_id="a1",
        session_id="s1",
        content="basic info",
        layer=MemoryLayer.L0_BASIC_INFO,
        status=MemoryStatus.ACTIVE,
        is_latest=is_latest,
    )
    n.custom = {"basic_info_kv": kv or {"name": "alvin"}}
    n.gmt_created = gmt_created or datetime.utcnow()
    return n


def _summary_node(content, session_id="s1", memory_at=None):
    n = MemoryNode(
        node_id=str(uuid.uuid4()),
        user_id="u1",
        agent_id="a1",
        session_id=session_id,
        content=content,
        layer=MemoryLayer.L3_SUMMARY,
        status=MemoryStatus.ACTIVE,
    )
    n.memory_at = memory_at or datetime.utcnow()
    n.gmt_created = n.memory_at
    return n


def _l1_raw_node(content, gmt=None):
    n = MemoryNode(
        node_id=str(uuid.uuid4()),
        user_id="u1",
        agent_id="a1",
        session_id="s1",
        content=content,
        layer=MemoryLayer.L1_RAW,
        status=MemoryStatus.ACTIVE,
    )
    n.gmt_created = gmt or datetime.utcnow()
    return n


def _l2_fact_node(content):
    n = MemoryNode(
        node_id=str(uuid.uuid4()),
        user_id="u1",
        agent_id="a1",
        session_id="s1",
        content=content,
        layer=MemoryLayer.L2_FACT,
        status=MemoryStatus.ACTIVE,
    )
    return n


# ---------------------------------------------------------------------------
# basic_profile.py：_find_latest_l0 / _assemble_full_kv
# ---------------------------------------------------------------------------

class TestBasicProfilePayloadOnly:
    def test_find_latest_l0_omits_vectors(self):
        from hy_memory.agent.tools.basic_profile import _find_latest_l0
        store = RecordingVectorStore([_basic_info_node()])

        node = run(_find_latest_l0(store, user_id="u1"))

        assert node is not None
        assert len(store.calls) == 1
        assert store.calls[0].get("with_vectors") is False, (
            "_find_latest_l0 only uses layer/custom/is_latest to identify the chain head, no embedding needed"
        )

    def test_assemble_full_kv_omits_vectors(self):
        from hy_memory.agent.tools.basic_profile import _assemble_full_kv
        store = RecordingVectorStore([
            _basic_info_node(kv={"name": "alvin"},
                             gmt_created=datetime.utcnow() - timedelta(days=1)),
            _basic_info_node(kv={"city": "sh"}, gmt_created=datetime.utcnow()),
        ])

        kv = run(_assemble_full_kv(
            store, user_id="u1", allowed_fields=None,
        ))

        assert kv == {"name": "alvin", "city": "sh"}
        assert len(store.calls) == 1
        assert store.calls[0].get("with_vectors") is False, (
            "_assemble_full_kv only reads custom.basic_info_kv, no embedding needed"
        )


# ---------------------------------------------------------------------------
# writer.py：_get_recent_history
# ---------------------------------------------------------------------------

class TestWriterPayloadOnly:
    def _writer(self):
        from hy_memory.pipelines.writer import MemoryWriter
        return MemoryWriter.__new__(MemoryWriter)

    def _request(self):
        from hy_memory.pipelines.writer import WriteRequest
        req = WriteRequest.__new__(WriteRequest)
        req.user_id = "u1"
        req.agent_id = "a1"
        req.session_id = "s1"
        req.memory_at = datetime.utcnow()
        return req


    def test_get_recent_history_omits_vectors(self):
        writer = self._writer()
        base = datetime.utcnow()
        store = RecordingVectorStore([
            _l1_raw_node("[user]: hi\n[assistant]: hello",
                         gmt=base - timedelta(seconds=2)),
            _l1_raw_node("[user]: how are you\n[assistant]: fine",
                         gmt=base - timedelta(seconds=1)),
        ])

        history = run(writer._get_recent_history(
            store, user_id="u1", agent_id="a1", max_turns=10,
        ))

        assert "hi" in history and "hello" in history
        assert len(store.calls) == 1
        assert store.calls[0].get("with_vectors") is False, (
            "_get_recent_history only reads content/node_id/gmt_created, no embedding needed"
        )


# ---------------------------------------------------------------------------
# client.py：_list_vdb_bucket / async_get_basic_info /
#            async_build_entity_store / async_clone_user (dst + src negative assertion)
# ---------------------------------------------------------------------------

class TestClientPayloadOnly:
    def _client_with(self, store):
        """Construct a bare client instance carrying only the minimal self attributes.

        a real client has a pile of runtime deps like _loop_thread / _cache / _graph_store;
        here we only test list_by_user arguments, with all other instrumentation as no-ops.
        """
        from hy_memory.client import HyMemoryClient
        c = HyMemoryClient.__new__(HyMemoryClient)
        c._vector_store = store
        # disable op_log instrumentation (takes the early return)
        c._cache = None
        # disable graph store instrumentation (most methods reference it only when non-None)
        c._graph_store = None
        return c

    def test_list_vdb_bucket_omits_vectors(self):
        store = RecordingVectorStore([])
        client = self._client_with(store)

        run(client._list_vdb_bucket(
            user_id="u1", agent_id="a1",
            limit=10, offset=0, order="desc",
        ))

        # user-level L0 needs an extra query without agent; both paths are payload-only.
        assert len(store.calls) == 2
        assert all(call.get("with_vectors") is False for call in store.calls), (
            "the list API only serializes payload into list items, no embedding needed"
        )
        assert "agent_id" not in store.calls[1] and "agent_ids" not in store.calls[1]

    def test_async_get_basic_info_omits_vectors(self, monkeypatch):
        # avoid the extra IO of evolution expansion: patch it to a pass-through identity
        async def _fake_expand(vs, hits):
            return hits

        monkeypatch.setattr(
            "hy_memory.pipelines._retrieval.evolution.expand_evolution_chains",
            _fake_expand,
        )

        store = RecordingVectorStore([_basic_info_node()])
        client = self._client_with(store)

        # the public method has @_ensure_internal_loop and needs self._loop_thread; here we only verify list_by_user arguments,
        # so bypass the decorator and grab the original function (functools.wraps keeps it in __wrapped__).
        from hy_memory.client import HyMemoryClient
        raw = HyMemoryClient.async_get_user_basic_info.__wrapped__
        result = run(raw(client, user_id="u1"))

        assert result["memories"], "must be able to get at least one basic info"
        assert store.calls, "must really go through list_by_user"
        assert store.calls[0].get("with_vectors") is False, (
            "async_get_user_basic_info only reads custom.basic_info_kv/content/tags/chain"
        )

    def test_async_build_entity_store_omits_vectors(self, monkeypatch):
        # patch the entity indexer: don't really embed, only care whether the list arguments are right
        async def _fake_index(**kwargs):
            return 1

        monkeypatch.setattr(
            "hy_memory.pipelines._retrieval.entity_store.index_memory_entities",
            _fake_index,
        )

        store = RecordingVectorStore([_l2_fact_node("t1"), _l2_fact_node("t2")])
        client = self._client_with(store)
        client._embed_service = None  # index_memory_entities already patched

        from hy_memory.client import HyMemoryClient
        raw = HyMemoryClient.async_build_entity_store.__wrapped__
        result = run(raw(client, user_id="u1", agent_id="a1", rebuild=False))

        assert result["success"] is True
        assert result["memories_scanned"] == 2
        assert store.calls, "must really go through list_by_user"
        assert store.calls[0].get("with_vectors") is False, (
            "entity rebuild only uses node_id/content, re-embedding entity names"
        )

    def test_async_clone_user_dst_check_omits_vectors_src_keeps_them(self):
        """Clone checks agent + user-level L0 on both source and destination.

        Destination checks are payload-only. Source reads must retain vectors:
        the src full copy uses a to_dict/from_dict deep copy; to_dict serializes the embedding
        into the new node too; if the src side passes with_vectors=False, the cloned node's embedding
        is lost, zeroing the new user's later recall hit rate -- a behavior that must never change.
        """
        # call order: dst agent, dst L0, src agent, src L0.
        class _ClonefakeStore(RecordingVectorStore):
            async def list_by_user(self, *args, **kwargs):
                # the dst check's real call is positional list_by_user(dst_user_id, agent_id=..., limit=1, with_vectors=False)
                # normalize into the kwargs record for easier assertions
                if args:
                    kwargs = {**kwargs, "user_id": args[0]}
                self.calls.append(kwargs)
                # both destination-side checks return empty.
                if len(self.calls) <= 2:
                    return []
                # the source side returns one node carrying an embedding.
                node = MemoryNode(
                    node_id=str(uuid.uuid4()),
                    user_id="src",
                    agent_id="a1",
                    session_id="s1",
                    content="src memory",
                    layer=MemoryLayer.L2_FACT,
                    status=MemoryStatus.ACTIVE,
                )
                node.embedding = [0.1] * 16
                return [node]

            async def upsert_batch(self, nodes):
                self._upserted = list(nodes)
                return [n.node_id for n in nodes]

            async def delete_by_metadata(self, **kwargs):
                # rollback fallback, return a no-op
                return 0

        store = _ClonefakeStore()
        client = self._client_with(store)

        from hy_memory.client import HyMemoryClient
        raw = HyMemoryClient.async_clone_user.__wrapped__
        result = run(raw(
            client, src_user_id="src", dst_user_id="dst", agent_id="a1",
        ))

        assert result["success"] is True, result
        assert len(store.calls) == 4

        # dst agent + dst user-level L0: only checks existence, no embedding needed.
        assert all(call.get("with_vectors") is False for call in store.calls[:2])

        # src agent + src user-level L0: must use the default (not passed, equivalent to True).
        assert all(call.get("with_vectors", True) is True for call in store.calls[2:])

        # also confirm the newly written nodes really carry an embedding
        assert hasattr(store, "_upserted") and store._upserted, "src must really upsert"
        assert store._upserted[0].embedding, "cloned nodes must keep the embedding"
