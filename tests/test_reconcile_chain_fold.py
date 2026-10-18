# -*- coding: utf-8 -*-
"""
End-to-end tests for multi-node SUPERSEDE folding (memory_ids) on the chroma backend.

Coverage:
- parser: parsing of memory_ids (multi-node) and memory_id (single node);
- parser: Pass-2 dedup applies to every id on the chain;
- writer: folds two previously unlinked old memories plus the new one into a single 3 node chain
  (a1 <- a2 <- new head): every old node becomes shadow with supersedes/superseded_by links set,
  only the new head keeps is_latest=True, and the head content contains none of the old content.
"""

import asyncio
import random
import uuid

import pytest

from hy_memory.config import MemoryConfig
from hy_memory.models.memory import MemoryNode, MemoryLayer, MemoryStatus
from hy_memory.agent.reconciler import (
    MemoryReconciler,
    ReconcileOp,
    ReconcileResult,
)


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _emb(dims: int = 32):
    vec = [random.gauss(0, 1) for _ in range(dims)]
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


@pytest.fixture
def config(tmp_path):
    return MemoryConfig.from_dict({
        "vector_store": {
            "provider": "chroma",
            "collection_name": f"test_chain_{uuid.uuid4().hex[:8]}",
            "persist_directory": str(tmp_path / "chroma_data"),
            "embedding_dims": 32,
        },
    })


@pytest.fixture
def store(config):
    from hy_memory.data.vector_store_chroma import ChromaVectorStore
    vs = ChromaVectorStore(config)
    run(vs.initialize())
    yield vs
    run(vs.close())


class TestParserMemoryIds:

    def test_multi_node_parses_ordered(self, config):
        import json
        r = MemoryReconciler(config)
        ops = r._parse_ops(json.dumps([
            {"op": "SUPERSEDE", "memory_ids": ["a1", "a2"],
             "content": "head", "owner": "user", "tags": ["music"]}
        ]))
        assert len(ops) == 1
        assert ops[0].memory_ids == ["a1", "a2"]
        assert ops[0].memory_id == "a2"  # latest = chain head target

    def test_single_id_still_works(self, config):
        import json
        r = MemoryReconciler(config)
        ops = r._parse_ops(json.dumps([
            {"op": "SUPERSEDE", "memory_id": "m1", "content": "c", "owner": "user"}
        ]))
        assert ops[0].memory_id == "m1"
        assert ops[0].memory_ids == ["m1"]

    def test_dedup_across_chain_ids(self, config):
        import json
        r = MemoryReconciler(config)
        ops = r._parse_ops(json.dumps([
            {"op": "SUPERSEDE", "memory_ids": ["a1", "a2"], "content": "h", "owner": "user"},
            {"op": "UPDATE", "memory_id": "a2", "content": "dup", "owner": "user"},
        ]))
        assert len(ops) == 1  # second op dropped (a2 already touched)


class TestWriterChainFold:

    def _seed(self, store):
        a1 = MemoryNode(
            node_id=str(uuid.uuid4()), user_id="u1", agent_id="default_agent",
            session_id="s1", content="The user is learning guitar.", layer=MemoryLayer.L2_FACT,
            status=MemoryStatus.ACTIVE, is_latest=True, owner="user",
        )
        a1.embedding = _emb()
        a2 = MemoryNode(
            node_id=str(uuid.uuid4()), user_id="u1", agent_id="default_agent",
            session_id="s1", content="The user can now play several complete guitar pieces.", layer=MemoryLayer.L2_FACT,
            status=MemoryStatus.ACTIVE, is_latest=True, owner="user",
        )
        a2.embedding = _emb()
        run(store.upsert(a1))
        run(store.upsert(a2))
        return a1, a2

    def test_fold_two_into_three_node_chain(self, store, config):
        a1, a2 = self._seed(store)

        # Build the writer with an externally injected store and a stub reconciler returning a multi-node SUPERSEDE
        from hy_memory.pipelines.writer import MemoryWriter
        from hy_memory.pipelines.base import WriteRequest

        writer = MemoryWriter(config, vector_store=store)
        run(writer.initialize())

        # stub embed_service: returns fixed-length random vectors
        class _FakeEmbed:
            async def embed_batch(self, texts):
                return [_emb() for _ in texts]
            async def embed_queued(self, text, use_cache=True):
                return _emb()
        writer._embed_service = _FakeEmbed()

        # stub reconciler.reconcile -> returns one multi-node SUPERSEDE op
        head_content = "The user performed guitar at a local open mic night."
        async def _fake_reconcile(*args, **kwargs):
            return ReconcileResult(ops=[
                ReconcileOp(
                    op="SUPERSEDE",
                    memory_id=a2.node_id,  # parser sets this to memory_ids[-1]
                    memory_ids=[a1.node_id, a2.node_id],
                    content=head_content,
                    owner="user",
                    tags=["music", "hobby"],
                    supersede_reason="Progression: learning -> playing pieces -> performing",
                )
            ], success=True)
        # The op memory_id is overridden to match the real chain tail
        writer._reconciler.reconcile = _fake_reconcile

        req = WriteRequest(content="x", user_id="u1", agent_id="default_agent", session_id="s1")
        stored_ids, err, ops, tok = run(writer._reconcile_and_store(
            new_memory_texts=[head_content],
            new_memories_meta=[{"content": head_content, "layer": "L2_FACT", "owner": "user", "tags": ["music"]}],
            request=req,
            vector_store=store,
            req_id="req_test",
        ))

        assert err is None
        assert len(stored_ids) == 1
        assert ops["supersede"] == 1
        assert ops["total"] == ops["add"] + ops["supersede"] + ops["update"]
        head_id = stored_ids[0]

        # 1. New head: is_latest=True, supersedes=[a2], content excludes the old content
        head = run(store.get_by_id(head_id))
        assert head is not None
        assert head.is_latest is True
        assert head.supersedes == [a2.node_id]
        assert head.content == head_content
        assert "learning" not in head.content and "several" not in head.content

        # 2. a2: joined the evolution chain (SUPERSEDED, still recallable and expandable), superseded_by
        #    contains head, and supersedes contains a1 (chain middle)
        r_a2 = run(store.get_by_id(a2.node_id))
        assert r_a2.is_latest is False
        assert r_a2.status == MemoryStatus.SUPERSEDED
        assert head_id in (r_a2.superseded_by or [])
        assert a1.node_id in (r_a2.supersedes or [])

        # 3. a1: joined the evolution chain (SUPERSEDED), superseded_by contains a2 (chain tail)
        r_a1 = run(store.get_by_id(a1.node_id))
        assert r_a1.is_latest is False
        assert r_a1.status == MemoryStatus.SUPERSEDED
        assert a2.node_id in (r_a1.superseded_by or [])
