# -*- coding: utf-8 -*-
"""
Memory Strength (idle-based time decay ranking): unit + integration tests.

Coverage:
- compute_strength formula: full strength when new, cold start falling back to gmt_created, idle decay, access_count
  log frequency weighting, access_count=0 not crashing (log guard).
- apply_strength_to_normal: profile / intention layers pass through undecayed, normal layers get strength folded in.
- bump_access: access_count+1 / last_accessed_at write-back (chroma end to end).
- persistence round-trip: last_accessed_at upsert -> preserved by get_by_id.
- reader integration: a recently hit fact ranks ahead of a stale fact after strength (chroma).
"""

import asyncio
import math
import random
import uuid
from datetime import datetime, timedelta

import pytest

from hy_memory.config import MemoryConfig
from hy_memory.models.memory import MemoryNode, MemoryLayer, MemoryStatus
from hy_memory.pipelines._retrieval.strength import (
    compute_strength,
    apply_strength_to_normal,
    bump_access,
    DEFAULT_TAU,
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


class _FakeNode:
    def __init__(self, *, access_count=0, last_accessed_at=None, gmt_created=None, layer=None):
        self.access_count = access_count
        self.last_accessed_at = last_accessed_at
        self.gmt_created = gmt_created
        self.layer = layer


class TestComputeStrength:
    def test_new_never_accessed_full_strength(self):
        now = datetime.now()
        n = _FakeNode(access_count=0, last_accessed_at=None, gmt_created=now)
        assert abs(compute_strength(n, now=now) - 1.0) < 1e-6

    def test_access_count_zero_no_crash(self):
        now = datetime.now()
        n = _FakeNode(access_count=0, last_accessed_at=now, gmt_created=now)
        # freq=1.0, idle=0 → strength=1.0
        assert abs(compute_strength(n, now=now) - 1.0) < 1e-6

    def test_idle_decay(self):
        now = datetime.now()
        n = _FakeNode(access_count=1, last_accessed_at=now - timedelta(days=365), gmt_created=now)
        expected = (1 + math.log(1)) * math.exp(-365 / DEFAULT_TAU)
        assert abs(compute_strength(n, now=now) - expected) < 1e-6

    def test_frequency_boost(self):
        now = datetime.now()
        n = _FakeNode(access_count=5, last_accessed_at=now - timedelta(days=1), gmt_created=now)
        expected = (1 + math.log(5)) * math.exp(-1 / DEFAULT_TAU)
        assert abs(compute_strength(n, now=now) - expected) < 1e-6
        assert compute_strength(n, now=now) > 1.0  # frequent and recent -> strength > 1

    def test_cold_start_falls_back_to_created(self):
        now = datetime.now()
        # never accessed, but created 300 days ago -> idle uses gmt_created
        n = _FakeNode(access_count=0, last_accessed_at=None, gmt_created=now - timedelta(days=300))
        expected = 1.0 * math.exp(-300 / DEFAULT_TAU)
        assert abs(compute_strength(n, now=now) - expected) < 1e-6

    def test_recent_old_memory_beats_idle_old(self):
        """Created in 2023 but hit yesterday must beat one created recently yet idle for a long time."""
        now = datetime.now()
        kobe = _FakeNode(access_count=3, last_accessed_at=now - timedelta(days=1),
                         gmt_created=now - timedelta(days=1000))
        stale = _FakeNode(access_count=1, last_accessed_at=now - timedelta(days=300),
                          gmt_created=now - timedelta(days=10))
        assert compute_strength(kobe, now=now) > compute_strength(stale, now=now)


class TestApplyStrengthToNormal:
    def test_profile_and_intention_passthrough(self):
        now = datetime.now()
        old = now - timedelta(days=400)
        def mk(layer):
            return {"node": _FakeNode(access_count=1, last_accessed_at=old, gmt_created=old,
                                      layer=MemoryLayer.from_string(layer)),
                    "score": 1.0, "layer": layer}
        hits = [mk("l2_fact"), mk("l4_identity"), mk("l7_intention")]
        apply_strength_to_normal(
            hits,
            profile_layers={"l0_basic_info", "l4_identity"},
            intention_layers={"l7_intention"},
        )
        by_layer = {h["layer"]: h["score"] for h in hits}
        assert by_layer["l4_identity"] == 1.0   # profile passes through
        assert by_layer["l7_intention"] == 1.0   # intention passes through
        assert by_layer["l2_fact"] < 1.0         # normal is decayed


@pytest.fixture
def store(tmp_path):
    config = MemoryConfig.from_dict({
        "vector_store": {
            "provider": "chroma",
            "collection_name": f"test_strength_{uuid.uuid4().hex[:8]}",
            "persist_directory": str(tmp_path / "chroma_data"),
            "embedding_dims": 32,
        },
    })
    from hy_memory.data.vector_store_chroma import ChromaVectorStore
    vs = ChromaVectorStore(config)
    run(vs.initialize())
    yield vs
    run(vs.close())


class TestPersistenceAndBump:
    def test_last_accessed_at_roundtrip(self, store):
        laat = datetime.now() - timedelta(days=2)
        n = MemoryNode(
            node_id=str(uuid.uuid4()), user_id="u1", agent_id="a1", session_id="s1",
            content="The user likes Kobe", layer=MemoryLayer.L2_FACT, status=MemoryStatus.ACTIVE,
            last_accessed_at=laat, access_count=3,
        )
        n.embedding = _emb()
        run(store.upsert(n))
        got = run(store.get_by_id(n.node_id))
        assert got is not None
        assert got.access_count == 3
        assert got.last_accessed_at is not None
        # precision down to the second
        assert abs((got.last_accessed_at - laat).total_seconds()) < 2

    def test_bump_access_increments(self, store):
        n = MemoryNode(
            node_id=str(uuid.uuid4()), user_id="u1", agent_id="a1", session_id="s1",
            content="The user is learning guitar", layer=MemoryLayer.L2_FACT, status=MemoryStatus.ACTIVE,
            access_count=2,
        )
        n.embedding = _emb()
        run(store.upsert(n))
        ok = run(bump_access(store, [(n.node_id, 2)]))
        assert ok == 1
        got = run(store.get_by_id(n.node_id))
        assert got.access_count == 3
        assert got.last_accessed_at is not None


class TestReaderStrengthRanking:
    """A recently and frequently hit fact is ranked ahead of a stale fact after strength.

    Verifies the core reader step directly, "real VDB recall + apply_strength_to_normal": two L2 facts with the
    same embedding (equal semantic scores), so strength is the only differentiator.
    """

    def test_recent_fact_outranks_stale(self, store):
        now = datetime.now()
        shared_emb = _emb()
        recent = MemoryNode(
            node_id=str(uuid.uuid4()), user_id="u1", agent_id="a1", session_id="s1",
            content="The user has been following a series recently", layer=MemoryLayer.L2_FACT, status=MemoryStatus.ACTIVE,
            access_count=10, last_accessed_at=now - timedelta(days=1),
        )
        stale = MemoryNode(
            node_id=str(uuid.uuid4()), user_id="u1", agent_id="a1", session_id="s1",
            content="The user watched a series in the past", layer=MemoryLayer.L2_FACT, status=MemoryStatus.ACTIVE,
            access_count=1, last_accessed_at=now - timedelta(days=400),
        )
        recent.embedding = list(shared_emb)
        stale.embedding = list(shared_emb)
        run(store.upsert(recent))
        run(store.upsert(stale))

        hits = run(store.search(
            query_embedding=list(shared_emb), user_ids=["u1"],
            layers=[MemoryLayer.L2_FACT], limit=10,
            status_filter=[MemoryStatus.ACTIVE], only_latest=True,
        ))
        assert {recent.node_id, stale.node_id} <= {h["node_id"] for h in hits}
        # semantic scores are equal at recall time -> the top two tie; after strength, recent must score higher
        apply_strength_to_normal(hits, score_key="score")
        hits.sort(key=lambda x: x["score"], reverse=True)
        ids = [h["node_id"] for h in hits]
        assert ids.index(recent.node_id) < ids.index(stale.node_id)
