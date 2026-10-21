"""Regression tests for created_after filtering by **event time**.

Bug fixed: the filter field used to be `gmt_created` (persist time). Under bulk ingestion every memory
gets an almost identical gmt_created, which disables time filtering entirely. Filtering now uses `memory_at`
(event time), falling back to `gmt_created` only when `memory_at` is empty.

Coverage:
- payload-level helper (used by Chroma / FAISS post-filtering)
- hit-level helper (reader tail-end fallback)
- Key scenario: bulk ingestion (gmt_created all=today, memory_at spanning years)
"""

from datetime import datetime, timedelta

import pytest

from hy_memory.data.vector_store_base import (
    node_effective_time,
    passes_created_after,
)
from hy_memory.pipelines._retrieval.temporal import (
    filter_hits_by_created_after,
    hit_effective_ts,
)
from hy_memory.models.memory import MemoryNode, MemoryLayer


NOW = datetime(2026, 9, 12, 12, 0, 0)
LAST_YEAR = datetime(2025, 3, 1, 12, 0, 0)
CUTOFF = datetime(2026, 1, 1, 0, 0, 0).timestamp()


# ================================================================
# payload level (Chroma / FAISS)
# ================================================================

class TestNodeEffectiveTime:
    def test_prefers_memory_at(self):
        payload = {"memory_at": 100.0, "gmt_created": 999.0}
        assert node_effective_time(payload) == 100.0

    def test_falls_back_to_gmt_created_when_none(self):
        assert node_effective_time({"memory_at": None, "gmt_created": 999.0}) == 999.0

    def test_falls_back_when_empty_string(self):
        # Chroma rejects None, so _payload_to_chroma_metadata writes empty values as ""
        assert node_effective_time({"memory_at": "", "gmt_created": 999.0}) == 999.0

    def test_none_when_both_missing(self):
        assert node_effective_time({}) is None
        assert node_effective_time({"memory_at": "", "gmt_created": ""}) is None

    def test_ignores_unparseable(self):
        assert node_effective_time({"memory_at": "not-a-number", "gmt_created": 7.0}) == 7.0


class TestPassesCreatedAfter:
    def test_no_filter_passes_everything(self):
        assert passes_created_after({"memory_at": 1.0}, None) is True
        assert passes_created_after({"memory_at": 1.0}, 0) is True

    def test_filters_on_event_time(self):
        old = {"memory_at": LAST_YEAR.timestamp(), "gmt_created": NOW.timestamp()}
        new = {"memory_at": NOW.timestamp(), "gmt_created": NOW.timestamp()}
        assert passes_created_after(old, CUTOFF) is False
        assert passes_created_after(new, CUTOFF) is True

    def test_bulk_ingest_scenario(self):
        """Core regression: under bulk ingestion gmt_created is all today; only memory_at separates the events."""
        ingested_now = NOW.timestamp()
        payloads = [
            {"memory_at": (LAST_YEAR + timedelta(days=d)).timestamp(),
             "gmt_created": ingested_now}
            for d in range(0, 600, 60)
        ]
        kept = [p for p in payloads if passes_created_after(p, CUTOFF)]
        # filtering by gmt_created keeps everything (the old bug); by memory_at only cutoff-and-later survive
        assert 0 < len(kept) < len(payloads)
        assert all(p["memory_at"] >= CUTOFF for p in kept)

    def test_missing_time_is_kept(self):
        """A node with neither time field is kept -- a missing field must not silently drop data."""
        assert passes_created_after({}, CUTOFF) is True

    def test_null_memory_at_uses_gmt_created(self):
        """Bypass writes such as L0_BASIC_INFO have historical rows with empty memory_at; behaviour matches before."""
        old_l0 = {"memory_at": None, "gmt_created": LAST_YEAR.timestamp()}
        new_l0 = {"memory_at": None, "gmt_created": NOW.timestamp()}
        assert passes_created_after(old_l0, CUTOFF) is False
        assert passes_created_after(new_l0, CUTOFF) is True


# ================================================================
# hit level (reader tail-end fallback)
# ================================================================

def _hit(memory_at=None, gmt_created=None, node=True, score=0.9):
    if not node:
        return {"node": None, "score": score,
                "memory_at": memory_at, "gmt_created": gmt_created}
    return {
        "node": MemoryNode(
            node_id="n1", content="c", layer=MemoryLayer.L2_FACT,
            memory_at=memory_at, gmt_created=gmt_created,
        ),
        "score": score,
    }


class TestHitEffectiveTs:
    def test_prefers_node_memory_at(self):
        h = _hit(memory_at=LAST_YEAR, gmt_created=NOW)
        assert hit_effective_ts(h) == LAST_YEAR.timestamp()

    def test_falls_back_to_node_gmt_created(self):
        h = _hit(memory_at=None, gmt_created=NOW)
        assert hit_effective_ts(h) == NOW.timestamp()

    def test_graph_hit_without_node(self):
        # graph-channel results carry node=None, with the fields on the hit dict (possibly unix seconds)
        h = _hit(memory_at=LAST_YEAR.timestamp(), node=False)
        assert hit_effective_ts(h) == LAST_YEAR.timestamp()

    def test_none_when_nothing_available(self):
        assert hit_effective_ts({"node": None, "score": 0.5}) is None


class TestFilterHitsByCreatedAfter:
    def test_no_filter_returns_input(self):
        hits = [_hit(memory_at=LAST_YEAR)]
        assert filter_hits_by_created_after(hits, None) is hits

    def test_empty_input(self):
        assert filter_hits_by_created_after([], CUTOFF) == []

    def test_filters_by_event_time_not_write_time(self):
        hits = [
            _hit(memory_at=LAST_YEAR, gmt_created=NOW),  # old event, recent write -> must be filtered out
            _hit(memory_at=NOW, gmt_created=NOW),
        ]
        out = filter_hits_by_created_after(hits, CUTOFF)
        assert len(out) == 1
        assert out[0]["node"].memory_at == NOW

    def test_keeps_hits_without_any_time(self):
        hits = [{"node": None, "score": 0.5}]
        assert filter_hits_by_created_after(hits, CUTOFF) == hits

    def test_boundary_is_inclusive(self):
        boundary = datetime.fromtimestamp(CUTOFF)
        out = filter_hits_by_created_after([_hit(memory_at=boundary)], CUTOFF)
        assert len(out) == 1


# ================================================================
# Qdrant end to end: nested should + IsNull filter
# ================================================================

@pytest.mark.asyncio
async def test_qdrant_filters_on_event_time_end_to_end(tmp_path):
    """Real pushdown filter behaviour on the Qdrant side (nested should + IsNullCondition).

    A wrong filter would break every created_after query on the main backend, so end-to-end verification is mandatory.
    """
    from hy_memory.config import MemoryConfig, VectorStoreConfig
    from hy_memory.data.vector_store_qdrant import QdrantVectorStore

    cfg = MemoryConfig()
    cfg.vector_store = VectorStoreConfig(
        provider="qdrant",
        persist_directory=str(tmp_path),
        collection_name="test_created_after",
        embedding_dims=4,
    )
    vs = QdrantVectorStore(cfg)
    await vs.initialize()

    def _node(nid, memory_at, gmt_created):
        return MemoryNode(
            node_id=nid, user_id="u1", agent_id="a1", session_id="s1",
            layer=MemoryLayer.L2_FACT, content=f"fact {nid}",
            memory_at=memory_at, gmt_created=gmt_created,
            embedding=[1.0, 0.0, 0.0, 0.0],
        )

    try:
        # Bulk ingestion scenario: gmt_created is all "today", only memory_at separates event times
        await vs.upsert(_node("old_event", LAST_YEAR, NOW))
        await vs.upsert(_node("new_event", NOW, NOW))
        # bypass node with an empty memory_at -> falls back to gmt_created
        await vs.upsert(_node("nullma_old", None, LAST_YEAR))
        await vs.upsert(_node("nullma_new", None, NOW))

        async def _ids(created_after):
            res = await vs.search(
                query_embedding=[1.0, 0.0, 0.0, 0.0], user_id="u1",
                limit=10, score_threshold=0.0, created_after=created_after,
            )
            return sorted(r["node_id"] for r in res)

        assert await _ids(None) == [
            "new_event", "nullma_new", "nullma_old", "old_event",
        ]
        # old_event's gmt_created is "today" -- the old implementation filtering on gmt_created would wrongly admit it
        assert await _ids(CUTOFF) == ["new_event", "nullma_new"]
    finally:
        await vs.close()


# ================================================================
# L0 bypass writes now carry memory_at
# ================================================================

@pytest.mark.asyncio
async def test_basic_profile_node_carries_memory_at(monkeypatch):
    """The L0 node built by upsert_basic_profile must carry memory_at (eliminating a null source)."""
    from hy_memory.agent.tools import basic_profile as bp

    captured = {}

    class _FakeVS:
        async def list_by_user(self, **kw):
            return []
        async def get_by_ids(self, ids):
            return []
        async def upsert(self, node):
            captured["node"] = node
            return node.node_id
        async def update_payload(self, nid, updates):
            return True

    class _FakeEmbed:
        async def embed_queued(self, text):
            return [0.1, 0.2, 0.3]

    res = await bp.upsert_basic_profile(
        user_id="u1",
        kv={"name": "Tim"},
        vector_store=_FakeVS(),
        embed_service=_FakeEmbed(),
        allowed_fields=["name"],
    )

    assert res.success, res.error
    node = captured.get("node")
    assert node is not None, "upsert was never called"
    assert node.layer == MemoryLayer.L0_BASIC_INFO
    assert node.memory_at is not None
