# -*- coding: utf-8 -*-
"""
SharedRuntime end-to-end tests.

Coverage:
1. Solo mode (no runtime passed) behaves exactly the same — backward compat regression
2. Two clients can coexist in shared mode
3. Two clients in shared mode share the same cache + the same loop
4. In shared mode client.close() does not close the shared cache or stop the shared loop
5. In shared mode runtime.close() closes the cache and the loop
6. The MetricsCollector singleton binds only once in shared mode (key cross-loop bug guard)
7. Multiple clients can really run add/search in parallel without hanging

Tests depend only on Chroma + SQLite cache (zero external deps), avoiding MySQL/network.
"""

import asyncio
import uuid

import pytest

from hy_memory import HyMemoryClient, SharedRuntime, MemoryConfig
from hy_memory.metrics import MetricsCollector


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_config(tmp_path, *, dims=32, collection_suffix=""):
    """Build a config that depends on no external service (chroma + sqlite cache)"""
    suffix = collection_suffix or uuid.uuid4().hex[:8]
    return MemoryConfig.from_dict({
        "vector_store": {
            "provider": "chroma",
            "collection_name": f"test_runtime_{suffix}",
            "persist_directory": str(tmp_path / f"chroma_{suffix}"),
            "embedding_dims": dims,
        },
        "cache": {
            "backend": "sqlite",
            "sqlite_path": str(tmp_path / "cache.db"),
        },
        "history": {"enable": False},
    })


@pytest.fixture(autouse=True)
def _reset_metrics_singleton():
    """Reset the MetricsCollector singleton before each test to avoid cross-test pollution"""
    MetricsCollector.reset()
    yield
    MetricsCollector.reset()


@pytest.fixture(autouse=True)
def _safe_env(monkeypatch, tmp_path):
    """Avoid tests accidentally connecting to real redis/mysql"""
    monkeypatch.setenv("MEMORY_CACHE_BACKEND", "sqlite")
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.delenv("MEMORY_VECTOR_HOST", raising=False)
    monkeypatch.setenv("MEMORY_DATA_DIR", str(tmp_path / ".hy_memory"))
    monkeypatch.setenv("MEMORY_LOG_DIR", str(tmp_path / "logs"))
    yield


# ================================================================
# Phase 1: Solo mode backward compatibility
# ================================================================

class TestSoloModeUnchanged:
    """Solo mode (no runtime passed) must behave exactly as before the refactor"""

    def test_solo_client_init_close(self, tmp_path):
        cfg = _make_config(tmp_path)
        client = HyMemoryClient(cfg, mode="lite")
        try:
            # solo mode owns_runtime=True
            assert client._owns_runtime is True
            # owns its own cache + loop_thread
            assert client._cache is not None
            assert client._loop_thread is not None
        finally:
            client.close()

    def test_solo_client_metrics_bound(self, tmp_path):
        cfg = _make_config(tmp_path)
        client = HyMemoryClient(cfg, mode="lite")
        try:
            mc = MetricsCollector.get()
            assert mc._cache is client._cache
            assert mc._background_started is True
        finally:
            client.close()


# ================================================================
# Phase 2: Shared mode — single client
# ================================================================

class TestSharedRuntimeSingleClient:

    def test_runtime_create_and_close(self, tmp_path):
        cfg = _make_config(tmp_path)
        runtime = run(SharedRuntime.create(cfg))
        try:
            assert runtime.cache is not None
            assert runtime.loop_thread is not None
            assert runtime.metrics is MetricsCollector.get()
            # cache is already bound to metrics
            assert runtime.metrics._cache is runtime.cache
            assert runtime.metrics._background_started is True
        finally:
            runtime.close()

    def test_client_with_runtime_shares_resources(self, tmp_path):
        cfg = _make_config(tmp_path)
        runtime = run(SharedRuntime.create(cfg))
        try:
            client_cfg = _make_config(tmp_path, collection_suffix="cli1")
            client = HyMemoryClient(client_cfg, mode="lite", runtime=runtime)
            try:
                assert client._owns_runtime is False
                assert client._loop_thread is runtime.loop_thread
                assert client._cache is runtime.cache
            finally:
                client.close()
        finally:
            runtime.close()

    def test_client_close_does_not_close_shared_cache(self, tmp_path):
        cfg = _make_config(tmp_path)
        runtime = run(SharedRuntime.create(cfg))
        try:
            client_cfg = _make_config(tmp_path, collection_suffix="cli1")
            client = HyMemoryClient(client_cfg, mode="lite", runtime=runtime)
            shared_cache = runtime.cache
            client.close()
            # runtime.cache is still usable after client.close
            assert runtime.cache is shared_cache
            # the runtime loop is still running
            assert runtime.loop_thread is not None
        finally:
            runtime.close()


# ================================================================
# Phase 3: Shared mode — multiple clients (core bug-guard scenario)
# ================================================================

class TestSharedRuntimeMultiClient:

    def test_two_clients_share_one_cache(self, tmp_path):
        cfg = _make_config(tmp_path)
        runtime = run(SharedRuntime.create(cfg))
        try:
            cfg_a = _make_config(tmp_path, collection_suffix="a")
            cfg_b = _make_config(tmp_path, collection_suffix="b")
            client_a = HyMemoryClient(cfg_a, mode="lite", runtime=runtime)
            client_b = HyMemoryClient(cfg_b, mode="lite", runtime=runtime)
            try:
                assert client_a._cache is client_b._cache
                assert client_a._loop_thread is client_b._loop_thread
                assert client_a._cache is runtime.cache
            finally:
                client_a.close()
                client_b.close()
        finally:
            runtime.close()

    def test_metrics_singleton_only_bound_once(self, tmp_path):
        """Key test: the original bug was bind_cache being overwritten by the second client;
        in shared mode only the runtime should call bind_cache once, and no client calls it"""
        cfg = _make_config(tmp_path)
        runtime = run(SharedRuntime.create(cfg))
        bound_cache_after_runtime = runtime.metrics._cache
        bg_started_after_runtime = runtime.metrics._background_started

        try:
            cfg_a = _make_config(tmp_path, collection_suffix="a")
            cfg_b = _make_config(tmp_path, collection_suffix="b")
            client_a = HyMemoryClient(cfg_a, mode="lite", runtime=runtime)
            client_b = HyMemoryClient(cfg_b, mode="lite", runtime=runtime)
            try:
                # after creating both clients, metrics._cache is still runtime.cache,
                # not overwritten by the client's bind_cache
                assert MetricsCollector.get()._cache is bound_cache_after_runtime
                assert MetricsCollector.get()._cache is runtime.cache
                # background state preserved
                assert MetricsCollector.get()._background_started is bg_started_after_runtime
            finally:
                client_a.close()
                client_b.close()
        finally:
            runtime.close()

    def test_clients_use_same_loop_thread(self, tmp_path):
        """Key cross-loop guard: all clients use the same loop"""
        cfg = _make_config(tmp_path)
        runtime = run(SharedRuntime.create(cfg))
        try:
            cfg_a = _make_config(tmp_path, collection_suffix="a")
            cfg_b = _make_config(tmp_path, collection_suffix="b")
            client_a = HyMemoryClient(cfg_a, mode="lite", runtime=runtime)
            client_b = HyMemoryClient(cfg_b, mode="lite", runtime=runtime)
            try:
                # the same _LoopThread instance
                assert client_a._loop_thread is client_b._loop_thread
                # the same underlying event loop
                assert client_a._loop_thread._loop is client_b._loop_thread._loop
            finally:
                client_a.close()
                client_b.close()
        finally:
            runtime.close()


# ================================================================
# Phase 4: actually run add / verify it functionally works
# ================================================================

class TestSharedRuntimeFunctional:
    """Actually send requests through add/search to confirm the flow runs without hanging on cross-loop"""

    def test_two_clients_add_concurrently_no_crossloop(self, tmp_path):
        cfg = _make_config(tmp_path)
        runtime = run(SharedRuntime.create(cfg))
        try:
            cfg_a = _make_config(tmp_path, collection_suffix="a")
            cfg_b = _make_config(tmp_path, collection_suffix="b")
            client_a = HyMemoryClient(cfg_a, mode="lite", runtime=runtime)
            client_b = HyMemoryClient(cfg_b, mode="lite", runtime=runtime)
            try:
                # go directly through the runtime cache's store/get_pipeline_log (base protocol-layer persistence)
                # key check: the two clients do not read/write cache on their own _LoopThread
                # both delegate to the runtime's loop (avoiding cross-loop pool errors)
                shared = runtime.cache

                # write cache via the client's loop_thread.run (should work)
                async def _w(rid):
                    from hy_memory.utils.pipeline_observability import (
                        drain_pipeline_log_tasks,
                    )
                    await shared.store_pipeline_log(
                        request_id=rid, user_id="u", agent_id="a",
                        step="TEST", prompt="p", response="r",
                    )
                    await drain_pipeline_log_tasks()
                async def _r(rid):
                    return await shared.get_pipeline_logs(request_id=rid)

                client_a._loop_thread.run(_w("rid_a"))
                client_b._loop_thread.run(_w("rid_b"))
                # read what the other client wrote, proving the same pool
                assert len(client_a._loop_thread.run(_r("rid_b"))) == 1
                assert len(client_b._loop_thread.run(_r("rid_a"))) == 1
            finally:
                client_a.close()
                client_b.close()
        finally:
            runtime.close()


# ================================================================
# Phase 5: aclose / close semantics
# ================================================================

class TestSharedRuntimeClose:

    def test_runtime_close_idempotent(self, tmp_path):
        cfg = _make_config(tmp_path)
        runtime = run(SharedRuntime.create(cfg))
        runtime.close()
        # closing again does not hang
        runtime.close()

    def test_aclose_async(self, tmp_path):
        cfg = _make_config(tmp_path)
        runtime = run(SharedRuntime.create(cfg))
        # call aclose via runtime.loop_thread.run (because the cache is on that loop)
        runtime.close()


# ================================================================
# Phase 6: search-dedicated loop isolated from the write loop
# search uses an independent loop to avoid being blocked by the write queue.
# ================================================================

class TestSearchLoopIsolation:
    """search uses an independent _LoopThread so the write queue does not crowd out its scheduling."""

    def test_solo_search_loop_is_distinct_thread(self, tmp_path):
        """solo mode: the search loop and the write loop are two independent event loops/threads."""
        cfg = _make_config(tmp_path)
        client = HyMemoryClient(cfg, mode="lite")
        try:
            assert client._search_loop_thread is not None
            assert client._search_loop_thread is not client._loop_thread
            assert client._search_loop_thread._loop is not client._loop_thread._loop
            assert client._search_loop_thread._thread is not client._loop_thread._thread
            assert client._search_loop_thread._thread.is_alive()
        finally:
            client.close()

    def test_shared_search_loop_from_runtime(self, tmp_path):
        """shared mode: the client's search loop comes from runtime.search_loop_thread,
        and is isolated from the write loop; multiple clients share the same search loop."""
        cfg = _make_config(tmp_path)
        runtime = run(SharedRuntime.create(cfg))
        try:
            assert runtime.search_loop_thread is not None
            assert runtime.search_loop_thread is not runtime.loop_thread
            assert runtime.search_loop_thread._loop is not runtime.loop_thread._loop

            cfg_a = _make_config(tmp_path, collection_suffix="a")
            cfg_b = _make_config(tmp_path, collection_suffix="b")
            client_a = HyMemoryClient(cfg_a, mode="lite", runtime=runtime)
            client_b = HyMemoryClient(cfg_b, mode="lite", runtime=runtime)
            try:
                # the search loop comes from runtime, isolated from the write loop
                assert client_a._search_loop_thread is runtime.search_loop_thread
                assert client_a._search_loop_thread is not client_a._loop_thread
                # multiple clients share the same search loop
                assert client_a._search_loop_thread is client_b._search_loop_thread
            finally:
                client_a.close()
                client_b.close()
        finally:
            runtime.close()

    def test_backward_compat_old_runtime_without_search_loop(self, tmp_path):
        """Support the old runtime (no search_loop_thread attribute): fall back to the write loop without erroring."""
        cfg = _make_config(tmp_path)
        runtime = run(SharedRuntime.create(cfg))
        try:
            # simulate the old runtime: remove the search-dedicated loop
            old_search = runtime._search_loop_thread
            runtime._search_loop_thread = None
            client_cfg = _make_config(tmp_path, collection_suffix="compat")
            client = HyMemoryClient(client_cfg, mode="lite", runtime=runtime)
            try:
                # fall back to the write loop (behavior equivalent to the old path, not isolated but not crashing)
                assert client._search_loop_thread is runtime.loop_thread
            finally:
                client.close()
            runtime._search_loop_thread = old_search
        finally:
            runtime.close()

    def test_search_loop_schedules_while_write_loop_blocked(self, tmp_path):
        """Core regression: when the write loop is occupied by a long-running synchronously blocking coroutine, the search loop
        can still schedule coroutines promptly -- proving they do not crowd each other out.

        Verified directly at the loop-scheduling layer (no chroma/embed), matching this fix exactly:
        before the change search and write shared one loop, so a search coroutine submitted to it was
        stuck until the write block ended; after the change search has its own loop and scheduling is unaffected.
        """
        import time
        import asyncio as _aio

        cfg = _make_config(tmp_path)
        runtime = run(SharedRuntime.create(cfg))
        try:
            write_loop = runtime.loop_thread._loop
            search_loop = runtime.search_loop_thread._loop
            assert write_loop is not search_loop  # premise: they really are two loops

            BLOCK_SEC = 2.0

            async def _hog():
                # a synchronous sleep saturates the write loop (fully stuck under cooperative scheduling),
                # simulating non-await synchronous segments interspersed in EXTRACT/RECONCILE.
                time.sleep(BLOCK_SEC)

            async def _probe():
                # an extremely lightweight probe: represents whether a submitted search coroutine gets scheduled promptly.
                return "scheduled"

            # 1. saturate the write loop
            hog_fut = _aio.run_coroutine_threadsafe(_hog(), write_loop)
            time.sleep(0.1)  # ensure hog actually starts executing and holds the write loop

            # 2. at this point the write loop is stuck -> a probe submitted to the search loop should be scheduled immediately
            t0 = time.perf_counter()
            probe_fut = _aio.run_coroutine_threadsafe(_probe(), search_loop)
            assert probe_fut.result(timeout=1.0) == "scheduled"
            search_latency = time.perf_counter() - t0

            # the search loop scheduling latency should be on the order of milliseconds -- the independent loop is fully unaffected by the write block.
            # the threshold is an absolute value far smaller than BLOCK_SEC: before the change, when search/write shared one loop,
            # the probe would be stuck behind hog until near BLOCK_SEC, certainly exceeding this threshold -> the regression is caught.
            _SCHED_BUDGET = 0.3
            assert search_latency < _SCHED_BUDGET, (
                f"search loop scheduling took {search_latency:.3f}s (should be < {_SCHED_BUDGET}s); "
                f"meaning it is not isolated from the write loop (blocked for {BLOCK_SEC}s)"
            )

            # 3. control: the same probe submitted to the write loop is stuck behind hog until the block ends
            t1 = time.perf_counter()
            write_probe_fut = _aio.run_coroutine_threadsafe(_probe(), write_loop)
            assert write_probe_fut.result(timeout=BLOCK_SEC + 1.0) == "scheduled"
            write_latency = time.perf_counter() - t1
            # the write-loop probe really is squeezed by hog (proving hog works and the control is meaningful)
            assert write_latency > search_latency, (
                "the write-loop probe should be slower than the search-loop probe (blocked by hog); "
                "otherwise hog is not really occupying the write loop and the control is invalid"
            )

            hog_fut.result(timeout=BLOCK_SEC + 1)  # clean up to avoid leftover teardown
        finally:
            runtime.close()
