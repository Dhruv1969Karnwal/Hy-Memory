# -*- coding: utf-8 -*-
"""
Pipeline log instrumentation for the coding write pipeline.

Verifies that judge / extractor / reconciler write CODING_JUDGE / CODING_EXTRACT /
CODING_RECONCILE as three steps in real time (matching the chat pipeline), and that:
  - prompt / response are written in full (no truncation)
  - a crash loses nothing: when the store insert crashes, the logs of the earlier completed stages are still there

Uses a real SqliteCache (with the observability wrapper) plus mock LLM/embed to avoid external dependencies.

Note: all resources are built inside a single test coroutine (one event loop), so that pytest-asyncio
auto mode does not put the async fixture and the test on different loops and dispatch owner_loop across loops.
"""

import asyncio
import json
import os
import shutil
import tempfile
import uuid

import pytest

from hy_memory.coding.extractor import CodingMemoryExtractor
from hy_memory.coding.judge import classify_messages_is_coding
from hy_memory.coding.reconciler import CodingMemoryReconciler
from hy_memory.coding.store import CodingMemoryStore
from hy_memory.coding.writer import CodingWriter
from hy_memory.pipelines.base import ChatMessage, ToolCall

from tests.coding._helpers import MockLLMProvider, MockEmbedService


def _classify_resp(is_coding, reason="reason"):
    return json.dumps({"is_coding": is_coding, "reason": reason})


def _extract_resp(memories):
    return json.dumps({"memories": memories})


def _reconcile_resp(ops):
    return json.dumps({"ops": ops})


def _coding_messages():
    return [
        ChatMessage(role="user", content="Package this and publish it to the private package index"),
        ChatMessage(role="assistant", content="", tool_calls=[
            ToolCall(id="1", name="Read", arguments={"path": "scripts/publish_package.sh"}),
        ]),
        ChatMessage(role="tool", content="200 lines", tool_call_id="1", tool_name="Read"),
        ChatMessage(role="assistant", content="", tool_calls=[
            ToolCall(id="2", name="Bash", arguments={"command": "twine upload"}),
        ]),
        ChatMessage(role="tool", content="OK uploaded", tool_call_id="2", tool_name="Bash"),
    ]


_DRAFT = {
    "task": "Package and publish to the private package index",
    "search_keys": ["twine upload", "private index publish"],
    "solution": "Run scripts/publish_package.sh and upload to the private index with twine upload",
    "boundary_envs": "twine>=4.0",
    "boundary_scope": "project",
    "confidence": 0.9,
}


async def _make_cache_and_store(d):
    """Builds cache + store on the current (test) event loop so the owner_loop stays consistent."""
    os.environ["MEMORY_PIPELINE_TRACE_ENABLED"] = "true"
    os.environ["MEMORY_LOG_LLM_CONTENT"] = "true"
    os.environ["MEMORY_LOG_DIR"] = os.path.join(d, "logs")
    from hy_memory.config import MemoryConfig
    from hy_memory.data.cache_sqlite import SqliteCache
    from hy_memory.utils.pipeline_observability import install_pipeline_observability

    cfg = MemoryConfig.from_env()
    cfg.cache.db_path = os.path.join(d, "cache.db")
    cfg.coding.db_path = os.path.join(d, "coding_memory.db")
    cfg.vector_store.provider = "chroma"
    cfg.vector_store.persist_directory = os.path.join(d, "chroma")
    cfg.vector_store.collection_name = f"t_{uuid.uuid4().hex[:8]}"
    cfg.vector_store.embedding_dims = 1536

    cache = SqliteCache(cfg)
    await cache.initialize()
    install_pipeline_observability(cache, log_label="test")

    store = CodingMemoryStore(
        cfg,
        embed_service=MockEmbedService(),
        db_path=cfg.coding.db_path,
    )
    await store.initialize()
    return cache, store


def _build_writer(cache, store, llm):
    embed = MockEmbedService()
    extractor = CodingMemoryExtractor(llm, cache=cache)
    reconciler = CodingMemoryReconciler(
        store=store, embed_service=embed, llm_provider=llm, cache=cache,
    )
    return CodingWriter(
        store=store, extractor=extractor, reconciler=reconciler,
        llm_provider=llm, embed_service=embed, cache=cache,
    )


@pytest.mark.asyncio
class TestCodingPipelineLog:

    async def test_judge_emits_coding_judge(self):
        d = tempfile.mkdtemp(prefix="coding_plog_")
        try:
            cache, store = await _make_cache_and_store(d)
            from hy_memory.utils.pipeline_observability import drain_pipeline_log_tasks
            rid = uuid.uuid4().hex
            llm = MockLLMProvider(_classify_resp(True, "concrete code task"))
            result = await classify_messages_is_coding(
                _coding_messages(), llm,
                cache=cache, request_id=rid, user_id="u1", agent_id="codebuddy",
            )
            assert result is True
            await drain_pipeline_log_tasks()
            logs = await cache.get_pipeline_logs(request_id=rid)
            judge = next(l for l in logs if l["step"] == "CODING_JUDGE")
            assert "single-label scene classifier" in judge["prompt"]
            assert "is_coding" in judge["response"]
            await cache.close(); await store.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)

    async def test_judge_logs_even_when_chat(self):
        """A chat verdict still leaves one CODING_JUDGE row (it was judged, conclusion chat)."""
        d = tempfile.mkdtemp(prefix="coding_plog_")
        try:
            cache, store = await _make_cache_and_store(d)
            from hy_memory.utils.pipeline_observability import drain_pipeline_log_tasks
            rid = uuid.uuid4().hex
            llm = MockLLMProvider(_classify_resp(False, "casual"))
            result = await classify_messages_is_coding(
                _coding_messages(), llm,
                cache=cache, request_id=rid, user_id="u1", agent_id="codebuddy",
            )
            assert result is False
            await drain_pipeline_log_tasks()
            steps = [l["step"] for l in await cache.get_pipeline_logs(request_id=rid)]
            assert "CODING_JUDGE" in steps
            await cache.close(); await store.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)

    async def test_extract_and_reconcile_steps(self):
        """Writer end to end: CODING_EXTRACT + CODING_RECONCILE are persisted with complete prompt/response."""
        d = tempfile.mkdtemp(prefix="coding_plog_")
        try:
            cache, store = await _make_cache_and_store(d)
            from hy_memory.utils.pipeline_observability import drain_pipeline_log_tasks
            rid = uuid.uuid4().hex
            llm = MockLLMProvider([
                _extract_resp([_DRAFT]),
                _reconcile_resp([{"draft_idx": 0, "action": "ADD"}]),
            ])
            writer = _build_writer(cache, store, llm)
            resp = await writer.write(
                _coding_messages(), user_id="u1", agent_id="codebuddy",
                workspace_id="github.com/example/project", request_id=rid,
            )
            assert resp["success"] is True
            assert resp["experience_ids"]
            await drain_pipeline_log_tasks()
            logs = await cache.get_pipeline_logs(request_id=rid)
            steps = [l["step"] for l in logs]
            assert "CODING_EXTRACT" in steps
            assert "CODING_RECONCILE" in steps
            ext = next(l for l in logs if l["step"] == "CODING_EXTRACT")
            rec = next(l for l in logs if l["step"] == "CODING_RECONCILE")
            assert "long-lived WORKING memory" in ext["prompt"]
            # response is the raw LLM output (json.dumps defaults to ensure_ascii, so task is kept in full as escapes)
            assert "task" in ext["response"] and "twine" in ext["response"]
            assert rec["prompt"] and rec["response"]
            await cache.close(); await store.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)

    async def test_reconcile_log_survives_store_crash(self):
        """A crash loses nothing: when store.insert raises, CODING_EXTRACT + CODING_RECONCILE
        (instrumented before execute persists) are already in the store."""
        d = tempfile.mkdtemp(prefix="coding_plog_")
        try:
            cache, store = await _make_cache_and_store(d)
            from hy_memory.utils.pipeline_observability import drain_pipeline_log_tasks
            rid = uuid.uuid4().hex
            llm = MockLLMProvider([
                _extract_resp([_DRAFT]),
                _reconcile_resp([{"draft_idx": 0, "action": "ADD"}]),
            ])
            async def _boom(*a, **k):
                raise RuntimeError("mock store crash")
            store.insert = _boom
            writer = _build_writer(cache, store, llm)
            # write must not raise (the try/except inside reconciler execute swallows the ADD failure)
            await writer.write(
                _coding_messages(), user_id="u1", agent_id="codebuddy",
                workspace_id="github.com/example/project", request_id=rid,
            )
            await drain_pipeline_log_tasks()
            steps = [l["step"] for l in await cache.get_pipeline_logs(request_id=rid)]
            assert "CODING_EXTRACT" in steps
            assert "CODING_RECONCILE" in steps
            await cache.close(); await store.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)
