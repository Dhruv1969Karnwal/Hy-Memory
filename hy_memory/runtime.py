# -*- coding: utf-8 -*-
"""
SharedRuntime -- shared infrastructure for multi-client deployments.

## Background

By default every `HyMemoryClient` instance owns:
  - one `_LoopThread` (persistent event loop)
  - one `MysqlCache` / `SqliteCache` pool
  - one pipeline observability hook wrapper

`MetricsCollector`, however, is a process-level singleton. When an app **starts multiple clients**
because embedder dims differ across data domains:

```
Client-A._LoopThread (loop_A)   →  Pool-A bound to loop_A
Client-B._LoopThread (loop_B)   →  Pool-B bound to loop_B

MetricsCollector (singleton)
  ├── bind_cache(Pool-A)  ← during Client-A init
  ├── bind_cache(Pool-B)  ← during Client-B init (overwrites!)
  └── flush coroutine    ← ensure_future on loop_A, but accesses Pool-B
                          → aiomysql across loops → connection pollution → KeyError
```

## Design

"Process-level resources" are pulled out explicitly:

| Owned by SharedRuntime (one per process) | Owned by each HyMemoryClient |
|---|---|
| `_LoopThread` (shared loop)              | `_embed_service` (different dims)      |
| `MysqlCache` / `SqliteCache` pool        | `_vector_store` (different collection) |
| `MetricsCollector` singleton + one bind  | `_graph_store`                         |
| pipeline observability hook       | `_history_store` / `_coding_store`    |
|                                          | `_registry` + all pipelines            |

## Usage

```python
from hy_memory import HyMemoryClient, SharedRuntime, MemoryConfig

# Created once at app startup
base_config = MemoryConfig.from_env()  # provides the cache config
runtime = await SharedRuntime.create(base_config)

# One lightweight client per config domain, all sharing the runtime
clients = {
    "default": HyMemoryClient(default_config, runtime=runtime),
    "research": HyMemoryClient(research_config, runtime=runtime),
}

# App shutdown
for c in clients.values():
    c.close()              # closes only this client's resources
await runtime.aclose()     # closes the shared infrastructure
```

## Compatibility

The legacy usage without `runtime=` is **completely unchanged**: the client holds all resources itself (solo mode).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import MemoryConfig

logger = logging.getLogger(__name__)


class SharedRuntime:
    """
    Infrastructure shared across multiple clients.

    Build it with `await SharedRuntime.create(config)` (not `SharedRuntime(config)` directly,
    because async init is required to create the cache pool and bind metrics).
    """

    def __init__(self, config: "MemoryConfig"):
        # The config is only used to build the cache and pick the loop strategy;
        # vector_store / embedder and the rest are managed by each client.
        self._config = config
        self._loop_thread = None  # type: Optional[object]  # _LoopThread (write/evolve/background)
        # Dedicated search loop: physically isolated from the write loop so the write queue cannot crowd out search scheduling.
        self._search_loop_thread = None  # type: Optional[object]  # _LoopThread
        self._cache = None        # type: Optional[CacheBase]
        self._metrics = None      # type: Optional[object]  # MetricsCollector
        self._observability_installed = False
        self._closed = False

    # ------------------------------------------------------------
    # Factory / lifecycle
    # ------------------------------------------------------------

    @classmethod
    async def create(cls, config: "MemoryConfig") -> "SharedRuntime":
        """
        Create and initialize a SharedRuntime.

        Immediately:
          1. Start a _LoopThread (persistent event loop)
          2. Initialize the cache pool on that loop
          3. Bind the cache to the MetricsCollector singleton
          4. Start metrics background flush + cleanup on that loop
          5. Install the pipeline observability hook on the cache

        Steps 2-5 all run on the same loop (the _LoopThread loop),
        so later metrics flushes always use the pool on that loop and never cross loops.
        """
        rt = cls(config)
        await rt._initialize()
        return rt

    async def _initialize(self) -> None:
        # Note: this method may be called from an external loop (e.g. the app's uvicorn loop),
        # but every operation that "binds to a loop" must complete on the _LoopThread loop.
        # Therefore the real init is delegated there.
        from .client import _LoopThread

        self._loop_thread = _LoopThread()
        # Dedicated search loop (own thread + own event loop), carrying only async_search coroutines.
        # The cache pool stays bound to the write loop (self._loop_thread); pipeline-log DB writes
        # originating on the search loop are dispatched back to the cache's loop by the observability layer.
        self._search_loop_thread = _LoopThread()
        # Do every loop-affinity-dependent init on the _LoopThread loop
        await self._loop_thread.run_async(self._init_on_loop())

    async def _init_on_loop(self) -> None:
        """Initialization executed on the _LoopThread loop."""
        # 1. cache pool (the pool binds to the current loop)
        from .data.cache import create_cache

        self._cache = create_cache(self._config)
        await self._cache.initialize()
        logger.info(
            f"[runtime] Cache ready (backend={self._config.cache.backend})"
        )

        # 2. MetricsCollector singleton: a single bind_cache + start_background_tasks
        #    The flush coroutine is pinned to this loop from now on and never crosses loops.
        from .metrics import MetricsCollector

        self._metrics = MetricsCollector.get()
        self._metrics.bind_cache(self._cache)
        await self._metrics.start_background_tasks()
        logger.info("[runtime] MetricsCollector bound to shared cache")

        # 3. pipeline observability hook — wrapping once is enough, all clients share it
        self._install_pipeline_observability()
        logger.info("[runtime] SharedRuntime ready")

    def _install_pipeline_observability(self) -> None:
        """Equivalent to HyMemoryClient._install_pipeline_observability_hooks,
        but installed only once. Installing it again wraps every store_pipeline_log call in multiple layers."""
        if self._observability_installed:
            return
        self._observability_installed = True

        from .utils.pipeline_observability import install_pipeline_observability

        install_pipeline_observability(self._cache, log_label="runtime")

    # ------------------------------------------------------------
    # Properties exposed to HyMemoryClient (read-only)
    # ------------------------------------------------------------

    @property
    def loop_thread(self):
        """Shared _LoopThread instance (write / evolve / background tasks)."""
        return self._loop_thread

    @property
    def search_loop_thread(self):
        """Dedicated search _LoopThread instance, physically isolated from the write loop."""
        return self._search_loop_thread

    @property
    def cache(self):
        """Shared cache (already initialized)."""
        return self._cache

    @property
    def metrics(self):
        """MetricsCollector singleton (already bind_cache'd to self.cache)."""
        return self._metrics

    @property
    def config(self) -> "MemoryConfig":
        """Base config used when building the runtime (clients need not match it; only the cache config must)."""
        return self._config

    # ------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------

    async def aclose(self) -> None:
        """Asynchronously close shared resources (cache + loop)."""
        if self._closed:
            return
        self._closed = True

        if self._metrics is not None:
            try:
                await self._loop_thread.run_async(
                    self._metrics.stop_background_tasks()
                )
            except Exception as e:
                logger.warning(f"[runtime] metrics close failed: {e}")
            self._metrics = None

        if self._cache is not None:
            try:
                # the cache was created on the _LoopThread loop, so it must also be closed there
                await self._loop_thread.run_async(self._cache.close())
            except Exception as e:
                logger.warning(f"[runtime] cache close failed: {e}")
            self._cache = None

        if self._loop_thread is not None:
            try:
                self._loop_thread.stop()
            except Exception as e:
                logger.warning(f"[runtime] loop_thread stop failed: {e}")
            self._loop_thread = None

        if self._search_loop_thread is not None:
            try:
                self._search_loop_thread.stop()
            except Exception as e:
                logger.warning(f"[runtime] search_loop_thread stop failed: {e}")
            self._search_loop_thread = None

        logger.info("[runtime] SharedRuntime closed")

    def close(self) -> None:
        """Synchronous close, for callers outside an async context."""
        if self._closed:
            return
        if self._loop_thread is not None:
            try:
                # cache.close must run on the loop it is bound to
                self._loop_thread.run(self._aclose_inner())
            except Exception as e:
                logger.warning(f"[runtime] sync close inner failed: {e}")
            try:
                self._loop_thread.stop()
            except Exception as e:
                logger.warning(f"[runtime] loop_thread stop failed: {e}")
        if self._search_loop_thread is not None:
            try:
                self._search_loop_thread.stop()
            except Exception as e:
                logger.warning(f"[runtime] search_loop_thread stop failed: {e}")
        self._loop_thread = None
        self._search_loop_thread = None
        self._cache = None
        self._closed = True
        logger.info("[runtime] SharedRuntime closed")

    async def _aclose_inner(self) -> None:
        """Shutdown sequence run inside the loop (closes the cache only, does not stop the loop -- the caller stops it)."""
        if self._metrics is not None:
            try:
                await self._metrics.stop_background_tasks()
            except Exception as e:
                logger.warning(f"[runtime] metrics close failed: {e}")
            self._metrics = None
        if self._cache is not None:
            try:
                await self._cache.close()
            except Exception as e:
                logger.warning(f"[runtime] cache close failed: {e}")
