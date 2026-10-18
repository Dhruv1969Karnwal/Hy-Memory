# -*- coding: utf-8 -*-
"""
L7_INTENTION layer dedup (0805 incident fix - v2) unit tests.

Coverage:
  1. Recall dedup (oversampling + joint memory_at/valid_until/cosine dedup + async delete)
  2. Writer batch dedup (batch only, no VDB query)
  3. READ_REQUEST log carries intention_limit (observability)

No dependency on chromadb / venus embedding / numpy; fully in-memory fake VDB.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from hy_memory.models.memory import MemoryLayer, MemoryStatus, MemoryNode
from hy_memory.pipelines._retrieval.intention import (
    _cosine,
    _memory_at_day,
    _valid_until_key,
    _partition_duplicates,
    recall_intentions,
)


# ============================================================================
# Fakes
# ============================================================================


def _run(coro):
    """Run coro once, wait for all background tasks (async delete) to finish naturally, then close the loop."""
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(coro)
        # Give the fire-and-forget create_task a chance to finish
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        return result
    finally:
        loop.close()


class _FakeEmbedByHash:
    """Assign one-hot sparse vectors by text hash: same hash -> cosine=1; different -> cosine=0."""

    def __init__(self, dims: int = 64):
        self.dims = dims

    def vec_for(self, text: str) -> List[float]:
        v = [0.0] * self.dims
        v[abs(hash(text)) % self.dims] = 1.0
        return v

    async def embed_batch(self, texts):
        return [self.vec_for(t) for t in texts]

    async def embed_queued(self, text):
        return self.vec_for(text)


class _FakeVectorStore:
    """Minimal usable in-memory VDB. search returns results sorted by dot-product."""

    def __init__(self):
        self._nodes: Dict[str, MemoryNode] = {}
        self.delete_calls: List[str] = []
        self.update_payload_calls: List[tuple] = []

    async def upsert(self, node: MemoryNode) -> str:
        nid = node.node_id or uuid.uuid4().hex
        node.node_id = nid
        self._nodes[nid] = node
        return nid

    async def delete(self, node_id: str):
        self.delete_calls.append(node_id)
        self._nodes.pop(node_id, None)

    async def update_payload(self, node_id: str, patch: Dict[str, Any]):
        self.update_payload_calls.append((node_id, patch))
        n = self._nodes.get(node_id)
        if not n:
            return
        for k, v in patch.items():
            setattr(n, k, v)

    async def search(
        self,
        *,
        query_embedding=None,
        user_ids=None,
        agent_ids=None,
        layers=None,
        limit: int = 10,
        score_threshold: float = 0.0,
        status_filter=None,
        only_latest: bool = True,
        **kwargs,
    ):
        results = []
        for nid, n in self._nodes.items():
            if layers and n.layer not in layers:
                continue
            if user_ids and n.user_id not in user_ids:
                continue
            if agent_ids and n.agent_id not in agent_ids:
                continue
            if status_filter and n.status not in status_filter:
                continue
            if only_latest and not n.is_latest:
                continue
            emb = n.embedding or []
            score = sum(a * b for a, b in zip(query_embedding or [], emb))
            if score < score_threshold:
                continue
            results.append({"node_id": nid, "node": n, "score": score})
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]


def _make_intention_node(
    content: str,
    *,
    user_id: str = "u1",
    agent_id: str = "a1",
    session_id: str = "s1",
    memory_at: Optional[datetime] = None,
    valid_until: Optional[datetime] = None,
    gmt_created: Optional[datetime] = None,
    tags: Optional[List[str]] = None,
    embedding: Optional[List[float]] = None,
    dims: int = 64,
) -> MemoryNode:
    n = MemoryNode(
        node_id=uuid.uuid4().hex,
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        layer=MemoryLayer.L7_INTENTION,
        content=content,
        embedding=embedding or _FakeEmbedByHash(dims).vec_for(content),
        status=MemoryStatus.ACTIVE,
        is_latest=True,
        tags=list(tags or []),
        memory_at=memory_at,
        valid_until=valid_until,
    )
    if gmt_created is not None:
        n.gmt_created = gmt_created
    return n


# ============================================================================
# 1. Recall dedup (including the new memory_at + valid_until dedup rules)
# ============================================================================


def test_cosine_identical_and_orthogonal():
    """cosine basic robustness: same vector=1, orthogonal=0, empty/zero vector=-inf."""
    v = [1.0, 0.0, 0.0]
    assert _cosine(v, v) == 1.0
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert _cosine([], v) == float("-inf")
    assert _cosine([0.0, 0.0], v) == float("-inf")


def test_memory_at_day_and_valid_until_key():
    """Normalize memory_at / valid_until to a date key."""
    dt = datetime(2025, 8, 6, 15, 30)
    node = _make_intention_node("x", memory_at=dt, valid_until=dt)
    assert _memory_at_day(node) == date(2025, 8, 6)
    assert _valid_until_key(node) == date(2025, 8, 6)

    node2 = _make_intention_node("x")
    assert _memory_at_day(node2) is None
    assert _valid_until_key(node2) is None


def test_partition_duplicates_same_day_same_valid_until_high_cosine_dedup():
    """Same-day memory_at + same valid_until + cosine≈1 -> dedup. Keep the earliest gmt."""
    same_ma = datetime(2025, 8, 6, 10, 0)
    same_vu = datetime(2025, 8, 8, 23, 59, 59)
    n1 = _make_intention_node(
        "User is launching a new product at the end of the month", memory_at=same_ma, valid_until=same_vu,
        gmt_created=datetime(2025, 8, 6, 10, 0),
    )
    n2 = _make_intention_node(
        "User is launching a new product at the end of the month", memory_at=same_ma, valid_until=same_vu,
        gmt_created=datetime(2025, 8, 6, 10, 5),  # later
    )
    n3 = _make_intention_node(
        "User is launching a new product at the end of the month", memory_at=same_ma, valid_until=same_vu,
        gmt_created=datetime(2025, 8, 6, 10, 10),  # latest
    )
    hits = [
        {"node_id": n.node_id, "node": n, "score": 0.68}
        for n in (n2, n1, n3)  # deliberately shuffled, to test keeping the earliest gmt rather than the first
    ]
    kept, delete_ids = _partition_duplicates(hits, threshold=0.95)
    assert len(kept) == 1
    assert kept[0]["node_id"] == n1.node_id  # earliest gmt
    assert set(delete_ids) == {n2.node_id, n3.node_id}


def test_partition_duplicates_different_memory_at_day_keeps_both():
    """Same content + valid_until, but memory_at on a different day -> no dedup."""
    n_this_week = _make_intention_node(
        "User has a meeting on Friday",
        memory_at=datetime(2025, 8, 4, 9, 0),   # Monday
        valid_until=datetime(2025, 8, 8, 23, 59),
    )
    n_next_week = _make_intention_node(
        "User has a meeting on Friday",
        memory_at=datetime(2025, 8, 11, 9, 0),  # next Monday
        valid_until=datetime(2025, 8, 15, 23, 59),
    )
    hits = [
        {"node_id": n_this_week.node_id, "node": n_this_week, "score": 0.7},
        {"node_id": n_next_week.node_id, "node": n_next_week, "score": 0.7},
    ]
    kept, delete_ids = _partition_duplicates(hits, threshold=0.95)
    assert len(kept) == 2
    assert delete_ids == []


def test_partition_duplicates_different_valid_until_keeps_both():
    """Core case: two "meeting on Friday" said the same day, pointing at different meetings (different valid_until) -> keep both.

    "User said this morning the meeting is on 8/8" vs "User said this afternoon the meeting is on 8/15", same content, memory_at
    on the same day, but a different valid_until. Must not be merged.
    """
    same_ma = datetime(2025, 8, 6, 10, 0)
    n_this_fri = _make_intention_node(
        "User has a meeting on Friday", memory_at=same_ma,
        valid_until=datetime(2025, 8, 8, 23, 59),   # this Friday
    )
    n_next_fri = _make_intention_node(
        "User has a meeting on Friday", memory_at=same_ma,
        valid_until=datetime(2025, 8, 15, 23, 59),  # next Friday
    )
    hits = [
        {"node_id": n_this_fri.node_id, "node": n_this_fri, "score": 0.7},
        {"node_id": n_next_fri.node_id, "node": n_next_fri, "score": 0.7},
    ]
    kept, delete_ids = _partition_duplicates(hits, threshold=0.95)
    assert len(kept) == 2, "different valid_until must not be deduped"
    assert delete_ids == []


def test_partition_duplicates_low_cosine_keeps_both():
    """Different content (low cosine) -> no dedup, even with the same memory_at / valid_until."""
    same_ma = datetime(2025, 8, 6)
    same_vu = datetime(2025, 8, 8)
    n1 = _make_intention_node(
        "User has a meeting on Friday", memory_at=same_ma, valid_until=same_vu,
        embedding=[1.0, 0.0],
    )
    n2 = _make_intention_node(
        "User is seeing the dentist on Friday", memory_at=same_ma, valid_until=same_vu,
        embedding=[0.0, 1.0],
    )
    hits = [
        {"node_id": n1.node_id, "node": n1, "score": 0.6},
        {"node_id": n2.node_id, "node": n2, "score": 0.6},
    ]
    kept, delete_ids = _partition_duplicates(hits, threshold=0.95)
    assert len(kept) == 2
    assert delete_ids == []


def test_partition_duplicates_mixed_scenarios():
    """Mixed scenario: 4 end-of-month product launches (same memory_at + same valid_until)
                + 2 parent-teacher meetings (same memory_at + different valid_until)
                + 1 Qiandao Lake (other) -> expected kept=4 (1 product + 2 parent-teacher + 1 Qiandao Lake)."""
    ma = datetime(2025, 8, 6)
    vu_end_month = datetime(2025, 8, 31)
    vu_this_fri = datetime(2025, 8, 8)
    vu_next_fri = datetime(2025, 8, 15)

    p1 = _make_intention_node(
        "Launch new product at end of month", memory_at=ma, valid_until=vu_end_month,
        gmt_created=datetime(2025, 8, 1),
    )
    p2 = _make_intention_node(
        "Launch new product at end of month", memory_at=ma, valid_until=vu_end_month,
        gmt_created=datetime(2025, 8, 2),
    )
    p3 = _make_intention_node(
        "Launch new product at end of month", memory_at=ma, valid_until=vu_end_month,
        gmt_created=datetime(2025, 8, 3),
    )
    p4 = _make_intention_node(
        "Launch new product at end of month", memory_at=ma, valid_until=vu_end_month,
        gmt_created=datetime(2025, 8, 4),
    )
    m1 = _make_intention_node("Parent-teacher meeting", memory_at=ma, valid_until=vu_this_fri)
    m2 = _make_intention_node("Parent-teacher meeting", memory_at=ma, valid_until=vu_next_fri)
    q1 = _make_intention_node("Qiandao Lake", memory_at=ma, valid_until=None)

    hits = [
        {"node_id": n.node_id, "node": n, "score": 0.6}
        for n in (p1, p2, p3, p4, m1, m2, q1)
    ]
    kept, delete_ids = _partition_duplicates(hits, threshold=0.95)

    kept_ids = {h["node_id"] for h in kept}
    # 4 product launches collapse to 1 (p1 with the earliest gmt); both parent-teacher meetings differ in valid_until and are kept; Qiandao Lake kept
    assert p1.node_id in kept_ids
    assert m1.node_id in kept_ids
    assert m2.node_id in kept_ids
    assert q1.node_id in kept_ids
    assert set(delete_ids) == {p2.node_id, p3.node_id, p4.node_id}
    assert len(kept) == 4


# ============================================================================
# recall_intentions end-to-end
# ============================================================================


def test_recall_intentions_dedup_and_async_delete_end_to_end():
    """End-to-end: 3 identical end-of-month product launches + 1 parent-teacher meeting in the VDB; with limit=3 kept=2,
    and the other 2 product launches are asynchronously deleted from the VDB."""
    vs = _FakeVectorStore()
    emb = _FakeEmbedByHash(dims=32)
    fixed_now = datetime(2025, 8, 6, 12, 0)
    ma = datetime(2025, 8, 6)
    vu = datetime(2025, 8, 31, 23, 59)

    p_ids = []
    for i in range(3):
        n = _make_intention_node(
            "User is launching a new product at the end of the month", dims=32,
            memory_at=ma, valid_until=vu,
            gmt_created=datetime(2025, 8, 1 + i),
            embedding=emb.vec_for("User is launching a new product at the end of the month"),
        )
        _run(vs.upsert(n))
        p_ids.append(n.node_id)

    m = _make_intention_node(
        "User has a parent-teacher meeting next Friday", dims=32,
        memory_at=ma, valid_until=datetime(2025, 8, 15),
        embedding=emb.vec_for("User has a parent-teacher meeting next Friday"),
    )
    _run(vs.upsert(m))

    # query uses "end-of-month product launch", dot=1.0 ranks the first 3; parent-teacher meeting dot=0
    q = emb.vec_for("User is launching a new product at the end of the month")
    result = _run(recall_intentions(
        vs, q, user_ids=["u1"], agent_ids=["a1"], limit=3, now=fixed_now,
    ))

    # 4 hits, after dedup kept=2 (1 product + 1 parent-teacher meeting), limited to 3 returns 2
    contents = [h["node"].content for h in result]
    assert contents.count("User is launching a new product at the end of the month") == 1
    # p1 (earliest gmt) should be kept, the other 2 are asynchronously deleted
    assert p_ids[0] in {h["node_id"] for h in result}
    assert set(vs.delete_calls) == {p_ids[1], p_ids[2]}


def test_recall_intentions_different_valid_until_all_returned():
    """Two intentions with the same memory_at but different valid_until -> both recalled, neither deleted."""
    vs = _FakeVectorStore()
    emb = _FakeEmbedByHash(dims=32)
    fixed_now = datetime(2025, 8, 6, 12, 0)
    ma = datetime(2025, 8, 6)

    n_this = _make_intention_node(
        "User has a meeting on Friday", dims=32, memory_at=ma,
        valid_until=datetime(2025, 8, 8, 23, 59),
        embedding=emb.vec_for("User has a meeting on Friday"),
    )
    n_next = _make_intention_node(
        "User has a meeting on Friday", dims=32, memory_at=ma,
        valid_until=datetime(2025, 8, 15, 23, 59),
        embedding=emb.vec_for("User has a meeting on Friday"),
    )
    _run(vs.upsert(n_this))
    _run(vs.upsert(n_next))

    q = emb.vec_for("User has a meeting on Friday")
    result = _run(recall_intentions(vs, q, user_ids=["u1"], limit=10, now=fixed_now))
    assert len(result) == 2
    assert vs.delete_calls == []


def test_recall_intentions_expired_converted_to_l2():
    """valid_until already passed -> lazily update_payload to L2_FACT, not in the results."""
    vs = _FakeVectorStore()
    emb = _FakeEmbedByHash(dims=32)
    yesterday = datetime.now() - timedelta(days=1)

    n = _make_intention_node(
        "Old expired intention", dims=32,
        memory_at=datetime(2025, 1, 1),
        valid_until=yesterday,
        embedding=emb.vec_for("Old expired intention"),
    )
    _run(vs.upsert(n))
    q = emb.vec_for("Old expired intention")
    result = _run(recall_intentions(vs, q, user_ids=["u1"], limit=10))
    assert result == []
    assert vs.update_payload_calls == [(n.node_id, {"layer": MemoryLayer.L2_FACT.value})]


def test_recall_intentions_limit_zero_short_circuit():
    """limit=0 -> return [] immediately, no VDB query."""
    vs = _FakeVectorStore()
    result = _run(recall_intentions(vs, [1.0] * 32, user_ids=["u1"], limit=0))
    assert result == []


def test_recall_intentions_oversample_larger_than_limit():
    """fetch_limit should be much larger than limit; 10 fully orthogonal intentions in the VDB + limit=3
    -> should return 3 distinct ones; since the embeddings are fully orthogonal, dedup should delete none."""
    vs = _FakeVectorStore()
    fixed_now = datetime(2025, 8, 6, 12, 0)
    ma = datetime(2025, 8, 6)
    dims = 128

    def _one_hot(idx: int) -> List[float]:
        v = [0.0] * dims
        v[idx] = 1.0
        return v

    # Use explicit one-hot indices to guarantee the 10 vectors are fully orthogonal (cosine=0)
    for i in range(10):
        n = _make_intention_node(
            f"Intention #{i}", dims=dims, memory_at=ma,
            embedding=_one_hot(i),
        )
        _run(vs.upsert(n))

    # query hits intention #3 (cosine=1), the others cosine=0
    q = _one_hot(3)
    result = _run(recall_intentions(
        vs, q, user_ids=["u1"], limit=3, now=fixed_now,
    ))
    # limit=3 returns at most 3; and no deletes at all (vectors fully orthogonal)
    assert len(result) <= 3
    assert vs.delete_calls == []


# ============================================================================
# 2. Writer batch dedup (batch only, no VDB query)
# ============================================================================


def _make_writer():
    from hy_memory.config import MemoryConfig
    from hy_memory.pipelines.writer import MemoryWriter
    cfg = MemoryConfig.from_env()
    writer = MemoryWriter(
        cfg,
        embed_service=_FakeEmbedByHash(dims=32),
        vector_store=_FakeVectorStore(),
        cache=None,
    )
    return writer


def _make_req(user_id="u1", agent_id="a1"):
    from hy_memory.pipelines.base import WriteRequest
    return WriteRequest(
        content="", messages=[], user_id=user_id, agent_id=agent_id,
        session_id="s1", request_id=uuid.uuid4().hex,
        memory_at=datetime(2025, 8, 6, 10, 0),
    )


def test_writer_batch_dedup_pure_function_same_valid_until():
    """Pure function: 3 entries with the same content + same valid_until -> drop 2."""
    from hy_memory.pipelines.writer import MemoryWriter
    emb = _FakeEmbedByHash(dims=16)
    vu = datetime(2025, 8, 8)
    intentions = [
        {"content": "Launch new product at end of month", "valid_until": vu, "tags": ["work"]},
        {"content": "Launch new product at end of month", "valid_until": vu, "tags": ["deadline"]},
        {"content": "Launch new product at end of month", "valid_until": vu, "tags": ["product"]},
    ]
    embeddings = [emb.vec_for(it["content"]) for it in intentions]
    drop_idx = MemoryWriter._intention_batch_dedup(intentions, embeddings)
    assert drop_idx == [1, 2]


def test_writer_batch_dedup_different_valid_until_keeps_all():
    """Core case: same content, different valid_until -> all kept, no dedup."""
    from hy_memory.pipelines.writer import MemoryWriter
    emb = _FakeEmbedByHash(dims=16)
    intentions = [
        {"content": "Meeting on Friday", "valid_until": datetime(2025, 8, 8)},   # this Friday
        {"content": "Meeting on Friday", "valid_until": datetime(2025, 8, 15)},  # next Friday
    ]
    embeddings = [emb.vec_for(it["content"]) for it in intentions]
    drop_idx = MemoryWriter._intention_batch_dedup(intentions, embeddings)
    assert drop_idx == []


def test_writer_batch_dedup_none_valid_until_all_none_treated_same():
    """valid_until all None and same content -> still dedup (None treated as the same key)."""
    from hy_memory.pipelines.writer import MemoryWriter
    emb = _FakeEmbedByHash(dims=16)
    intentions = [
        {"content": "Boss's empty promises", "valid_until": None},
        {"content": "Boss's empty promises", "valid_until": None},
    ]
    embeddings = [emb.vec_for(it["content"]) for it in intentions]
    drop_idx = MemoryWriter._intention_batch_dedup(intentions, embeddings)
    assert drop_idx == [1]


def test_writer_batch_dedup_none_vs_some_not_deduped():
    """One valid_until=None and one set -> different buckets, no dedup."""
    from hy_memory.pipelines.writer import MemoryWriter
    emb = _FakeEmbedByHash(dims=16)
    intentions = [
        {"content": "Write a novel", "valid_until": None},
        {"content": "Write a novel", "valid_until": datetime(2025, 12, 31)},
    ]
    embeddings = [emb.vec_for(it["content"]) for it in intentions]
    drop_idx = MemoryWriter._intention_batch_dedup(intentions, embeddings)
    assert drop_idx == []


def test_writer_batch_dedup_different_content_keeps_all():
    """Different content -> low cosine, no dedup."""
    from hy_memory.pipelines.writer import MemoryWriter
    # Use explicit one-hot indices to guarantee the 3 embeddings are fully orthogonal (cosine=0)
    def _one_hot(idx: int, dims: int = 64) -> List[float]:
        v = [0.0] * dims
        v[idx] = 1.0
        return v
    intentions = [
        {"content": "Launch new product at end of month", "valid_until": None},
        {"content": "Parent-teacher meeting on Friday", "valid_until": None},
        {"content": "Trip to Qiandao Lake for National Day", "valid_until": None},
    ]
    embeddings = [_one_hot(0), _one_hot(1), _one_hot(2)]
    drop_idx = MemoryWriter._intention_batch_dedup(intentions, embeddings)
    assert drop_idx == []


def test_writer_store_intentions_e2e_batch_dedup():
    """_store_intentions end-to-end: 3 duplicates + 1 new intention -> only 2 persisted in the VDB."""
    vs = _FakeVectorStore()
    from hy_memory.config import MemoryConfig
    from hy_memory.pipelines.writer import MemoryWriter
    cfg = MemoryConfig.from_env()
    writer = MemoryWriter(cfg, embed_service=_FakeEmbedByHash(dims=32),
                          vector_store=vs, cache=None)
    req = _make_req()
    vu = datetime(2025, 8, 8)
    intentions = [
        {"content": "Launch new product at end of month", "valid_until": vu, "tags": ["a"]},
        {"content": "Launch new product at end of month", "valid_until": vu, "tags": ["b"]},
        {"content": "Launch new product at end of month", "valid_until": vu, "tags": ["c"]},
        {"content": "Meeting on Friday", "valid_until": vu, "tags": []},
    ]
    ids = _run(writer._store_intentions(intentions, req, vs, req.request_id))
    assert len(ids) == 2  # 3 product launches collapse to 1 + 1 Friday meeting
    assert len(vs._nodes) == 2


# ============================================================================
# 3. READ_REQUEST log carries intention_limit
# ============================================================================


def test_log_request_records_intention_limit():
    import json
    from hy_memory.pipelines._retrieval.trace import ReadTraceLogger

    captured = []

    class _StubCache:
        async def store_pipeline_log(self, **kwargs):
            captured.append(kwargs)

    async def _main():
        tl = ReadTraceLogger(cache=_StubCache(), request_id="r",
                             user_id="u", agent_id="a", reader_version="v")
        await tl.log_request(
            query="q", limit=5, layers=None, min_score=0.0,
            schema_min_score=None, schema_limit=None,
            user_ids=[], agent_ids=[], session_ids=[],
            intention_limit=20,
        )

    _run(_main())
    parsed = captured[-1]["parsed"]
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    assert parsed["intention_limit"] == 20


def test_log_request_intention_limit_default_none():
    import json
    from hy_memory.pipelines._retrieval.trace import ReadTraceLogger

    captured = []

    class _StubCache:
        async def store_pipeline_log(self, **kwargs):
            captured.append(kwargs)

    async def _main():
        tl = ReadTraceLogger(cache=_StubCache(), request_id="r",
                             user_id="u", agent_id="a", reader_version="v")
        await tl.log_request(
            query="q", limit=5, layers=None, min_score=0.0,
            schema_min_score=None, schema_limit=None,
            user_ids=[], agent_ids=[], session_ids=[],
        )

    _run(_main())
    parsed = captured[-1]["parsed"]
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    assert "intention_limit" in parsed
    assert parsed["intention_limit"] is None
