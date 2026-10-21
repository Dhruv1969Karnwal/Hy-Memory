# -*- coding: utf-8 -*-
"""
Observability tests for intention (L7) writes.

Intentions are upserted straight into the VDB by System 1 (no reconcile), and previously landed in neither
pipeline_logs nor memory_operations. This test verifies that writer._store_intentions now:
  - emits an INTENTION_UPSERT pipeline log (with memory_id plus the read-back verified flag)
  - writes memory_operations (op=INTENTION_UPSERT, with memory_id) so production can be traced by
    request_id / memory_id.

Note: installing observability on the cache captures owner_loop, and dispatching across loops silently drops
the write. Therefore this test keeps "build cache/writer + run _store_intentions + read back" in one event loop.
"""

import asyncio
import json
import os
import shutil
import tempfile
import uuid

from hy_memory.pipelines.base import WriteRequest
from hy_memory.models.memory import MemoryNode


class _FakeEmbed:
    """Return **clearly different** vectors for different texts (each text owns one dimension set to 1.0, the rest 0).
    That way any two texts have cosine=0 and can never hit the 0.95 dedup threshold.
    """

    def __init__(self):
        self._indices = {}

    def _vec_for(self, text: str):
        # 32-dim sparse one-hot: assigned in first-seen order to avoid hash randomization or collisions.
        vec = [0.0] * 32
        idx = self._indices.setdefault(text, len(self._indices)) % 32
        vec[idx] = 1.0
        return vec

    async def embed_batch(self, texts):
        return [self._vec_for(t) for t in texts]

    async def embed_queued(self, text):
        return self._vec_for(text)


class _FakeVectorStore:
    """In-memory VDB: upsert stores into a dict, get_by_id reads it back."""

    def __init__(self):
        self._store = {}

    async def upsert(self, node: MemoryNode) -> str:
        nid = node.node_id or uuid.uuid4().hex
        node.node_id = nid
        self._store[nid] = node
        return nid

    async def get_by_id(self, node_id):
        return self._store.get(node_id)


def test_intention_upsert_emits_pipeline_log_and_operation(monkeypatch):
    d = tempfile.mkdtemp(prefix="intent_obs_")
    monkeypatch.setenv("MEMORY_DEDUP_THRESHOLD", "0.95")
    os.environ["MEMORY_PIPELINE_TRACE_ENABLED"] = "true"
    os.environ["MEMORY_WRITE_APPLY_VERIFY_ENABLED"] = "true"
    os.environ["MEMORY_MEMORY_OPERATIONS_ENABLED"] = "true"  # audit table is off by default, enabled explicitly here
    os.environ["MEMORY_LOG_DIR"] = os.path.join(d, "logs")

    async def _main():
        from hy_memory.config import MemoryConfig
        from hy_memory.data.cache_sqlite import SqliteCache
        from hy_memory.utils.pipeline_observability import (
            install_pipeline_observability, drain_pipeline_log_tasks,
        )
        from hy_memory.pipelines.writer import MemoryWriter

        cfg = MemoryConfig.from_env()
        cfg.cache.db_path = os.path.join(d, "cache.db")
        cache = SqliteCache(cfg)
        await cache.initialize()
        install_pipeline_observability(cache, log_label="test")
        vs = _FakeVectorStore()
        writer = MemoryWriter(cfg, embed_service=_FakeEmbed(), vector_store=vs, cache=cache)

        rid = uuid.uuid4().hex
        req = WriteRequest(
            content="", messages=[], user_id="u1", agent_id="codebuddy",
            session_id="s1", request_id=rid,
        )
        intentions = [
            {"content": "The user has a ByteDance interview next Tuesday", "tags": ["job hunting"], "valid_until": None},
            {"content": "The user plans to learn Rust", "tags": [], "valid_until": None},
        ]
        ids = await writer._store_intentions(intentions, req, vs, rid)
        await drain_pipeline_log_tasks()

        logs = await cache.get_pipeline_logs(request_id=rid)
        ops_by_id = {nid: await cache.get_memory_operations(memory_id=nid) for nid in ids}
        await cache.close()
        return ids, logs, ops_by_id

    try:
        ids, logs, ops_by_id = asyncio.new_event_loop().run_until_complete(_main())
    finally:
        shutil.rmtree(d, ignore_errors=True)

    assert len(ids) == 2

    # 1. pipeline log: two INTENTION_UPSERT entries, each with memory_id + verified=True
    intent_steps = [l for l in logs if l["step"] == "INTENTION_UPSERT"]
    assert len(intent_steps) == 2
    for l in intent_steps:
        assert l.get("memory_ids"), "INTENTION_UPSERT must carry memory_id for tracing"
        parsed = json.loads(l["parsed"]) if isinstance(l["parsed"], str) else l["parsed"]
        assert parsed["op"] == "INTENTION_UPSERT"
        assert parsed["memory_id"] in ids
        assert parsed["layer"] == "l7_intention"
        assert parsed["verified"] is True

    # 2. memory_operations: two op=INTENTION_UPSERT entries, queryable back by memory_id
    for nid in ids:
        assert any(o["op"] == "INTENTION_UPSERT" for o in ops_by_id[nid]), \
            f"memory_operations is missing INTENTION_UPSERT for {nid}"
