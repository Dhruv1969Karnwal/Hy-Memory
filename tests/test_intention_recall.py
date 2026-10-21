# -*- coding: utf-8 -*-
"""
End-to-end tests for intention (L7) recall + lazy expiry conversion (chroma backend).

Coverage:
- Unexpired L7 intentions are recalled by recall_intentions and carry layer l7_intention;
- Expired L7 intentions are not recalled and their stored layer is rewritten to l2_fact (verified by re-reading);
- Intentions with valid_until=None never expire and are always recalled;
- The reader returns no intentions when intention_limit=0 (gate);
- New extractor schema: facts/intentions parsing; the writer no longer emits L4_IDENTITY.
"""

import asyncio
import random
import uuid
from datetime import datetime, timedelta

import pytest

from hy_memory.config import MemoryConfig
from hy_memory.models.memory import MemoryNode, MemoryLayer, MemoryStatus
from hy_memory.pipelines._retrieval.intention import recall_intentions


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


def _make_intention(content, *, user_id="u1", valid_until=None, dims=32):
    node = MemoryNode(
        node_id=str(uuid.uuid4()),
        user_id=user_id,
        agent_id="a1",
        session_id="s1",
        content=content,
        layer=MemoryLayer.L7_INTENTION,
        status=MemoryStatus.ACTIVE,
        is_latest=True,
        valid_until=valid_until,
    )
    node.embedding = _emb(dims)
    return node


@pytest.fixture
def store(tmp_path):
    config = MemoryConfig.from_dict({
        "vector_store": {
            "provider": "chroma",
            "collection_name": f"test_intent_{uuid.uuid4().hex[:8]}",
            "persist_directory": str(tmp_path / "chroma_data"),
            "embedding_dims": 32,
        },
    })
    from hy_memory.data.vector_store_chroma import ChromaVectorStore
    vs = ChromaVectorStore(config)
    run(vs.initialize())
    yield vs
    run(vs.close())


class TestRecallIntentions:

    def test_unexpired_recalled(self, store):
        future = datetime.now() + timedelta(days=7)
        n = _make_intention("The user has a ByteDance interview next Tuesday", valid_until=future)
        run(store.upsert(n))
        q = list(n.embedding)  # Same vector guarantees a cosine=1 hit
        hits = run(recall_intentions(store, q, user_ids=["u1"], limit=5))
        ids = {h["node_id"] for h in hits}
        assert n.node_id in ids
        hit = next(h for h in hits if h["node_id"] == n.node_id)
        assert hit["layer"] == "l7_intention"
        assert hit["retrieval_source"] == "vdb_intention"

    def test_no_expiry_always_recalled(self, store):
        n = _make_intention("The user plans to learn guitar", valid_until=None)
        run(store.upsert(n))
        q = list(n.embedding)
        hits = run(recall_intentions(store, q, user_ids=["u1"], limit=5))
        assert n.node_id in {h["node_id"] for h in hits}

    def test_expired_excluded_and_converted_to_fact(self, store):
        past = datetime.now() - timedelta(days=3)
        n = _make_intention("The user had to hand in a report last week", valid_until=past)
        run(store.upsert(n))
        q = list(n.embedding)
        # Expired -> absent from the intention results
        hits = run(recall_intentions(store, q, user_ids=["u1"], limit=5))
        assert n.node_id not in {h["node_id"] for h in hits}
        # And the stored layer was persisted as l2_fact
        reloaded = run(store.get_by_id(n.node_id))
        assert reloaded is not None
        assert reloaded.layer == MemoryLayer.L2_FACT

    def test_limit_zero_returns_empty(self, store):
        n = _make_intention("The user plans to travel", valid_until=None)
        run(store.upsert(n))
        q = _emb()
        assert run(recall_intentions(store, q, user_ids=["u1"], limit=0)) == []


class TestExtractorSchema:
    """New schema: memory -> L2_FACT, intentions parsed separately; no more L4_IDENTITY."""

    def test_collect_new_memories_memory_bucket_to_l2(self):
        from hy_memory.pipelines.writer import MemoryWriter
        info = {
            "memory": [{"content": "The user likes basketball", "tags": ["sport"]}],
            "intentions": [{"content": "The user has an interview next week", "valid_until": "2099-01-01"}],
        }
        texts, meta = MemoryWriter._collect_new_memories(info)
        assert "The user likes basketball" in texts
        assert all(m["layer"] == "L2_FACT" for m in meta)
        # Intentions never enter normal memories
        assert "The user has an interview next week" not in texts

    def test_collect_new_memories_legacy_facts_to_l2(self):
        """Backward compatibility: the facts field of the legacy extractor still parses."""
        from hy_memory.pipelines.writer import MemoryWriter
        info = {"facts": [{"content": "The user likes basketball", "tags": ["sport"]}]}
        texts, meta = MemoryWriter._collect_new_memories(info)
        assert "The user likes basketball" in texts
        assert all(m["layer"] == "L2_FACT" for m in meta)

    def test_collect_new_memories_legacy_identity_to_l2(self):
        from hy_memory.pipelines.writer import MemoryWriter
        info = {"identity": [{"content": "The user is introverted", "tags": []}]}
        texts, meta = MemoryWriter._collect_new_memories(info)
        assert "The user is introverted" in texts
        assert all(m["layer"] == "L2_FACT" for m in meta)

    def test_collect_intentions_parses_valid_until(self):
        from hy_memory.pipelines.writer import MemoryWriter
        info = {
            "intentions": [
                {"content": "The user has an interview next week", "valid_until": "2099-01-01"},
                {"content": "The user wants to learn guitar", "valid_until": None},
                {"content": "Bad date", "valid_until": "not-a-date"},
            ]
        }
        out = MemoryWriter._collect_intentions(info)
        assert len(out) == 3
        assert out[0]["valid_until"] is not None
        assert out[0]["valid_until"].year == 2099
        assert out[1]["valid_until"] is None
        assert out[2]["valid_until"] is None  # An invalid date degrades to None


class TestOwner:
    """owner field: extractor -> meta, intention, normalization, persistence round-trip."""

    def test_collect_new_memories_owner(self):
        from hy_memory.pipelines.writer import MemoryWriter
        info = {"memory": [
            {"content": "The user likes basketball", "owner": "user"},
            {"content": "The assistant recommended a Sichuan restaurant", "owner": "agent"},
            {"content": "No owner", "tags": []},
            {"content": "Bad owner", "owner": "robot"},
        ]}
        _, meta = MemoryWriter._collect_new_memories(info)
        owners = {m["content"]: m["owner"] for m in meta}
        assert owners["The user likes basketball"] == "user"
        assert owners["The assistant recommended a Sichuan restaurant"] == "agent"
        assert owners["No owner"] is None       # Missing -> None
        assert owners["Bad owner"] is None        # Invalid -> None

    def test_collect_new_memories_owner_assistant_alias(self):
        from hy_memory.pipelines.writer import MemoryWriter
        info = {"memory": [{"content": "x", "owner": "assistant"}]}
        _, meta = MemoryWriter._collect_new_memories(info)
        assert meta[0]["owner"] == "agent"   # mem0-style assistant -> agent

    def test_collect_intentions_owner(self):
        from hy_memory.pipelines.writer import MemoryWriter
        info = {"intentions": [
            {"content": "The user has an interview next week", "owner": "user"},
            {"content": "Remind the user in 8 hours", "owner": "agent"},
        ]}
        out = MemoryWriter._collect_intentions(info)
        assert out[0]["owner"] == "user"
        assert out[1]["owner"] == "agent"

    def test_node_owner_roundtrip(self, store):
        n = MemoryNode(
            node_id=str(uuid.uuid4()), user_id="u1", agent_id="a1", session_id="s1",
            content="The assistant recommended a Kobe documentary", layer=MemoryLayer.L2_FACT,
            status=MemoryStatus.ACTIVE, owner="agent",
        )
        n.embedding = _emb()
        run(store.upsert(n))
        got = run(store.get_by_id(n.node_id))
        assert got is not None
        assert got.owner == "agent"

    def test_node_owner_empty_roundtrip(self, store):
        n = MemoryNode(
            node_id=str(uuid.uuid4()), user_id="u1", agent_id="a1", session_id="s1",
            content="A summary", layer=MemoryLayer.L3_SUMMARY,
            status=MemoryStatus.ACTIVE,  # owner left empty
        )
        n.embedding = _emb()
        run(store.upsert(n))
        got = run(store.get_by_id(n.node_id))
        assert got is not None
        assert got.owner is None
