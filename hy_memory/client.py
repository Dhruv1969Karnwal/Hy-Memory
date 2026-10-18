# -*- coding: utf-8 -*-
"""
HY Memory Client - user-facing synchronous client (stateless)

All components are initialized in the constructor; failures raise immediately.
The constructor accepts only connection config and binds no business parameters (user_id/agent_id, etc.);
a single client instance can be shared across threads.

    # Minimal usage - configure via env vars
    client = HyMemoryClient()
    client.add("The user likes sci-fi movies", user_id="test_user")
    results = client.search("What does the user like?", user_ids=["test_user"])
    client.close()

    # Custom config (mem0-style)
    client = HyMemoryClient.from_config({
        "embedder": {"provider": "openai", "model": "text-embedding-3-small"},
        "vector_store": {"provider": "qdrant", "collection_name": "my_memories"},
        "graph_store": {
            "provider": "neo4j",
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password",
        },
        "enable_graph": True,
    })
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union, TYPE_CHECKING

from .config import MemoryConfig
from .core.embed_service import EmbedService, embed_endpoint_override
from .data.vector_store import create_vector_store
from .data.history_store import HistoryStore
from .data.cache import create_cache
from .utils.log_setup import setup_logging, set_request_id, request_id_scope
from .utils.async_executor import (
    await_concurrent_future_guarded,
    concurrent_result_guarded,
)

if TYPE_CHECKING:
    from .runtime import SharedRuntime
from .pipelines import (
    ComponentFactory,
    ChatMessage,
    ToolCall,
    WriteRequest,
    WriteResponse,
    ReadRequest,
    ReadResponse,
)
# Productivity memory module (enabled only when tool messages are triaged as working/coding)
from .coding.preproc import has_any_tool_message, strip_tool_messages
from .coding.judge import classify_messages_scene

logger = logging.getLogger(__name__)

# Type of the first argument to add(): a string or a messages list
MessageList = List[Dict[str, str]]
AddInput = Union[str, MessageList]

# Default thresholds for the two layers on the working (coding) path. Deliberately not reusing the chat path's min_score:
# chat matches whole semantic units and 0.4 is a long-standing empirical value; the working fact layer matches single-sentence
# conclusions, whose scores run systematically lower at the same semantic relevance, so 0.4
# would filter out clearly relevant facts. The fact layer is the primary index, and filtering it out is more than one fewer fact: the
# experience path via derived_experience_ids loses its source too, degrading the two-layer structure into a plain
# experience search for this query. So the fact layer errs loose rather than strict, leaving noise to the experience direct-query gate alone.
_WORKING_FACT_DEFAULT_MIN_SCORE = 0.30

# Default bump of the experience direct-query threshold relative to the fact-layer threshold.
# The two layers embed different text forms -- the fact layer is a full-sentence statement, the experience layer is short segments like task + search_keys;
# the cosine-similarity baseline among short segments is inherently higher, so under the same threshold the experience layer is clearly looser.
# The separable relevant/irrelevant bands of the two layers are offset by about 0.1, so a single shared threshold cannot cover both.
_WORKING_EXPERIENCE_MIN_SCORE_DELTA = 0.10

# Maximum number of experience_id values per get_many call. SQLite IN clauses have a bound-variable limit
# (999 by default); this is a safety valve against pathologically long chains, not a return-count control.
_CODING_GET_MANY_MAX_IDS = 500


_WRITE_DIAGNOSTIC_KEYS = (
    "agent_status",
    "agent_error_code",
    "agent_error",
    "l1_rollback",
    "l1_rollback_failed",
    "partial_agent_persist",
    "session_episode",
    "workload",
    "graph_executor",
)


def _attach_write_diagnostics(resp: Dict[str, Any], extra: Dict[str, Any]) -> None:
    """Expose retry-safety details produced by the write pipeline."""
    for key in _WRITE_DIAGNOSTIC_KEYS:
        if key in extra:
            resp[key] = extra[key]


def _search_memory_channel(
    memory: Dict[str, Any],
    *,
    schema_layers: set[str],
    proactive_layers: set[str],
    schema_min_score: float,
) -> str:
    """Return the public search channel and consume private routing markers."""
    if bool(memory.pop("_schema_channel", False)):
        return "schema"
    layer = memory.get("layer", "")
    if layer in schema_layers:
        return "schema" if memory.get("score", 0.0) >= schema_min_score else "normal"
    if layer in proactive_layers:
        return "proactive"
    return "normal"


# File locations for cumulative cgroup CPU throttling time (prefer v2, fall back to v1).
# (path, field name, divisor) -- v1's throttled_time is in nanoseconds, so /1000 converts it to microseconds.
_CGROUP_CPU_STAT_SOURCES = (
    ("/sys/fs/cgroup/cpu.stat", "throttled_usec", 1),
    ("/sys/fs/cgroup/cpu/cpu.stat", "throttled_time", 1000),
)


def _read_throttled_usec() -> int:
    """Read the cumulative microseconds the process has been CPU-throttled by cgroup; return 0 if unreadable.

    Used to attribute process-level stalls: sample once before and once after a block of pure synchronous code, and the difference is the time
    the container forcibly paused it. When the container's CPU quota is exhausted, the kernel suspends the whole process, which shows up as
    "pure synchronous code taking hundreds of milliseconds".

    Reads a small file synchronously (procfs/cgroupfs, no disk IO), microsecond-scale, and does not block the event loop.
    Any exception silently returns 0 -- this is an observability field and must never affect the main flow.

    Note: fields are matched by **exact equality** rather than prefix. cgroup v1's cpu.stat contains both
    `throttled_time` and `throttled_time_bt` (burst stats); prefix matching picks the wrong value depending on the field order within
    the file.
    """
    for path, key, div in _CGROUP_CPU_STAT_SOURCES:
        try:
            with open(path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2 and parts[0] == key:
                        return int(parts[1]) // div
        except Exception:
            continue
    return 0


def _gc_collection_count() -> int:
    """Sum of cumulative collections across GC generations; return 0 on error.

    Used in pairs with _read_throttled_usec: a before/after difference >=1 means GC ran within the block,
    which can explain unexpected stalls in pure synchronous code (at high allocation rates gen2 GC can reach hundreds of milliseconds).
    """
    try:
        import gc
        return sum(s.get("collections", 0) for s in gc.get_stats())
    except Exception:
        return 0


def _ensure_internal_loop(method):
    """
    Decorator for async_xxx() methods on HyMemoryClient.

    Ensures the coroutine runs on _LoopThread's event loop, not the caller's.
    If already on the correct loop, executes directly.
    If on an external loop (e.g., uvicorn), delegates to _LoopThread via run_coroutine_threadsafe.
    """
    import functools

    @functools.wraps(method)
    async def wrapper(self, *args, **kwargs):
        submitted_at = time.perf_counter()

        async def _invoke():
            started_at = time.perf_counter()
            result = await method(self, *args, **kwargs)
            if isinstance(result, dict) and method.__name__ in {
                "async_add", "async_add_extracted", "async_finalize_session",
            }:
                runtime_diag = result.setdefault("runtime", {})
                runtime_diag["submit_wait_ms"] = round(
                    (started_at - submitted_at) * 1000, 3,
                )
                lag_snapshot = getattr(self._loop_thread, "lag_snapshot", None)
                if callable(lag_snapshot):
                    runtime_diag["event_loop"] = lag_snapshot()
            return result

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        internal_loop = self._loop_thread._loop

        if current_loop is internal_loop:
            # Already on _LoopThread's loop — execute directly
            return await _invoke()
        else:
            # On external loop (uvicorn, etc.) — delegate to _LoopThread
            future = asyncio.run_coroutine_threadsafe(
                _invoke(), internal_loop
            )
            return await await_concurrent_future_guarded(future, internal_loop)

    return wrapper


def _ensure_search_loop(method):
    """
    Decorator for async_search() on HyMemoryClient.

    Same shape as _ensure_internal_loop, but binds the coroutine to the **dedicated search loop**
    (self._search_loop_thread) rather than the write loop, so search scheduling is not crowded out by the write
    queue (EXTRACT/RECONCILE/write_queue concurrency).
    search and write use independent loops, so the write queue cannot consume search scheduling.

    In solo mode _search_loop_thread may be the same object as _loop_thread (no isolation),
    behaving like the old path; in shared mode (SharedRuntime) they are two independent threads.
    """
    import functools

    @functools.wraps(method)
    async def wrapper(self, *args, **kwargs):
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        search_loop = self._search_loop_thread._loop

        if current_loop is search_loop:
            # Already on the search loop -- execute directly
            return await method(self, *args, **kwargs)
        else:
            # External loop (uvicorn, etc.) -- delegate to the search loop
            future = asyncio.run_coroutine_threadsafe(
                method(self, *args, **kwargs), search_loop
            )
            return await await_concurrent_future_guarded(future, search_loop)

    return wrapper


class _LoopThread:
    """
    A background thread that holds a persistent event loop.

    Problem solved: local Qdrant (AsyncQdrantLocal) binds to the event loop that created it via a file lock,
    so creating a new loop on every asyncio.run() makes the file locks conflict.
    Running all coroutines on the same loop avoids this.
    """

    def __init__(self):
        import threading
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._lag_lock = threading.Lock()
        self._lag_handle = None
        self._wakeup_guard_handle = None
        self._lag_current_ms = 0.0
        self._lag_max_ms = 0.0
        self._lag_samples = 0
        self._lag_over_100ms = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5):
            raise RuntimeError("background event loop failed to start")

    def _run(self):
        asyncio.set_event_loop(self._loop)
        if os.getenv("MEMORY_EVENT_LOOP_LAG_MONITOR_ENABLED", "false").lower() in (
            "1", "true", "yes", "on",
        ):
            self._schedule_lag_probe()
        # Some runtimes can lose the self-pipe wakeup used by
        # call_soon_threadsafe/run_coroutine_threadsafe.  Keep an internal timer
        # pending so the selector wakes at least every 50ms and processes any
        # cross-thread callbacks already queued.  This timer is scheduled by the
        # loop itself and therefore does not depend on the broken wakeup path.
        self._schedule_wakeup_guard()
        self._loop.call_soon(self._ready.set)
        self._loop.run_forever()

    def _schedule_wakeup_guard(self):
        def _wake():
            self._schedule_wakeup_guard()

        self._wakeup_guard_handle = self._loop.call_later(0.05, _wake)

    def _schedule_lag_probe(self):
        try:
            interval = max(
                0.05,
                float(os.getenv("MEMORY_EVENT_LOOP_LAG_INTERVAL_SEC", "0.5")),
            )
        except (TypeError, ValueError):
            interval = 0.5
        expected = self._loop.time() + interval

        def _probe():
            now = self._loop.time()
            lag_ms = max(0.0, (now - expected) * 1000)
            with self._lag_lock:
                self._lag_current_ms = lag_ms
                self._lag_max_ms = max(self._lag_max_ms, lag_ms)
                self._lag_samples += 1
                if lag_ms >= 100.0:
                    self._lag_over_100ms += 1
            self._schedule_lag_probe()

        self._lag_handle = self._loop.call_at(expected, _probe)

    def lag_snapshot(self) -> Dict[str, Any]:
        with self._lag_lock:
            return {
                "current_ms": round(self._lag_current_ms, 3),
                "max_ms": round(self._lag_max_ms, 3),
                "samples": self._lag_samples,
                "over_100ms": self._lag_over_100ms,
            }

    def run(self, coro, *, include_runtime_diag: bool = False):
        """Run a coroutine on the background loop, waiting for the result synchronously."""
        if not include_runtime_diag:
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            return concurrent_result_guarded(future, self._loop)

        submitted_at = time.perf_counter()

        async def _timed():
            started_at = time.perf_counter()
            result = await coro
            if isinstance(result, dict):
                runtime_diag = result.setdefault("runtime", {})
                runtime_diag["submit_wait_ms"] = round(
                    (started_at - submitted_at) * 1000, 3,
                )
                runtime_diag["event_loop"] = self.lag_snapshot()
            return result

        future = asyncio.run_coroutine_threadsafe(_timed(), self._loop)
        return concurrent_result_guarded(future, self._loop)

    async def run_async(self, coro):
        """
        Run a coroutine on the background loop, awaiting the result asynchronously.

        Safe to await from any event loop -- runs directly if already on the correct loop,
        otherwise delegates to _LoopThread's loop and awaits.
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is self._loop:
            # Already on _LoopThread's loop, execute directly
            return await coro
        else:
            # On an external loop (e.g. uvicorn), delegate to _LoopThread
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            return await await_concurrent_future_guarded(future, self._loop)

    def stop(self):
        def _stop():
            if self._lag_handle is not None:
                self._lag_handle.cancel()
            if self._wakeup_guard_handle is not None:
                self._wakeup_guard_handle.cancel()
            self._loop.stop()

        self._loop.call_soon_threadsafe(_stop)
        self._thread.join(timeout=5)


class HyMemoryClient:
    """
    HY Memory stateless client.

    The constructor accepts only connection config and processing mode, and binds no business parameters.
    All business isolation parameters (user_id/agent_id/session_id, etc.) are passed per call,
    so a single client instance can be shared safely across threads.

    Processing modes (mode):
    - "lite":  raw-only pure-embedding write/retrieval, no LLM call; ignores scene/layers routing
    - "pro":   embedding + LLM agent extraction (facts/entities/intentions), the standard mode
    - "ultra": pro plus explicitly triggerable System 2 cognitive processing (Schema induction), the full mode

    Data isolation model (three levels):
    - user_id:    level-1 key -- a memory store unique to each user
    - agent_id:   level-2 key -- isolation across different Agent scenarios for the same user
    - session_id: level-3 key -- isolation across different sessions for the same Agent

    Usage:
        # lite: pure vector write, no LLM call
        client = HyMemoryClient(mode="lite")

        # pro: standard mode (embedding + agent extraction)
        client = HyMemoryClient(mode="pro")

        # ultra: full mode (pro + System 2 cognitive layer)
        client = HyMemoryClient(mode="ultra")

        client.add("The user likes sci-fi movies", user_id="user_001")
        results = client.search("What does the user like?", user_ids=["user_001"])
        client.close()

    Args:
        config: custom MemoryConfig (loaded from env vars by default)
        mode:   processing mode ("lite" / "pro" / "ultra"), read from the MEMORY_MODE env var by default
    """

    # Valid mode values
    VALID_MODES = ("lite", "pro", "ultra")

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        mode: Optional[str] = None,
        coding_writer: Optional[str] = None,
        runtime: Optional["SharedRuntime"] = None,
    ):
        # Ensure the logging system is configured (idempotent, safe to call repeatedly)
        setup_logging()

        # Config
        self._config = config or MemoryConfig.from_env()

        # An explicit coding_writer overrides config / env (default None → keep config.coding.writer)
        # Allowed values: "legacy" | "agent"
        if coding_writer is not None:
            cw = str(coding_writer).strip().lower()
            if cw in ("legacy", "agent"):
                self._config.coding.writer = cw
            else:
                logger.warning(
                    f"[init] invalid coding_writer={coding_writer!r}; falling back to config default"
                )

        # Resolve mode
        import os
        self._mode = mode or os.getenv("MEMORY_MODE", "pro")
        if self._mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode '{self._mode}'. Must be one of: {self.VALID_MODES}"
            )

        # Graph database toggle: decoupled from mode.
        #   ultra           → always on (System 2's L6_SCHEMA writes depend on it)
        #   pro             → decided by MEMORY_ENABLE_GRAPH (off by default), for the memory relation graph
        #   lite            → always off (lite does no LLM processing, so there is no graph to build)
        # GraphStore init already happens at client startup and is unrelated to evolve()/System 2,
        # so enabling the graph under pro does not require pulling in System 2.
        if self._mode == "ultra":
            self.enable_graph = True
        elif self._mode == "pro":
            from .config import _get_env_bool
            self.enable_graph = _get_env_bool("MEMORY_ENABLE_GRAPH", False)
        else:
            self.enable_graph = False
        self._config.enable_graph = self.enable_graph

        # Process-level init logs force an empty business request_id (sentinel "-") to avoid polluting the App's
        # request-scoped filtering by request_id.
        with request_id_scope():
            logger.info(
                f"Initializing HyMemoryClient: mode={self._mode} "
                f"enable_graph={self.enable_graph}"
            )

            # SharedRuntime hookup (for multi-client deployment, avoids cross-loop bugs)
            # - runtime is None  → solo mode: start its own _LoopThread + own cache pool
            # - runtime is given → shared mode: reuse the runtime's loop / cache / metrics bindings
            self._runtime = runtime
            self._owns_runtime = runtime is None

            # Create/reuse the persistent event loop
            if self._owns_runtime:
                self._loop_thread = _LoopThread()
                # solo mode: search also gets its own loop, isolating the same way shared mode does.
                self._search_loop_thread = _LoopThread()
            else:
                self._loop_thread = runtime.loop_thread
                # shared mode: use the runtime's dedicated search loop (isolated from the write loop).
                # Fall back to the write loop for older runtimes (no search_loop_thread).
                self._search_loop_thread = (
                    getattr(runtime, "search_loop_thread", None) or runtime.loop_thread
                )

            # Initialize all components synchronously
            self._loop_thread.run(self._initialize())

            logger.info("HyMemoryClient ready")

    async def _initialize(self):
        """Initialize all core components."""
        # Process-level init logs must not inherit the outer (e.g. App request context) business request_id,
        # otherwise filtering by request_id picks up one-off init noise. Force it to the sentinel "-".
        with request_id_scope():
            await self._initialize_impl()

    async def _initialize_impl(self):
        """Initialize all core components (actual implementation)."""

        # 0. Enlarge the global default thread pool + each component later uses its own pool
        import concurrent.futures
        loop = asyncio.get_event_loop()
        _max_workers = int(os.getenv("MEMORY_THREAD_POOL_SIZE", "256"))
        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=_max_workers)
        )
        logger.info(f"[init] Default ThreadPoolExecutor max_workers={_max_workers}")

        # 1. EmbedService (config only; it actually connects to the API on first call)
        self._embed_service = EmbedService(self._config)
        logger.debug(
            f"[init] EmbedService created: model={self._config.embedder.model}"
        )

        # 2. VectorStore (backend selected from config)
        self._vector_store = create_vector_store(self._config)
        await self._vector_store.initialize()
        logger.info(
            f"[init] VectorStore ready: provider={self._config.vector_store.provider} "
            f"collection={self._vector_store._collection_name}"
        )

        # 3. GraphStore (optional, controlled by the enable_graph config)
        self._graph_store = None
        if self.enable_graph:
            from .data.graph_store import create_graph_store
            self._graph_store = create_graph_store(self._config)
            await self._graph_store.initialize()
            # Check whether the graph database is truly usable (Kuzu silently degrades to a no-op)
            if hasattr(self._graph_store, '_available') and not self._graph_store._available:
                raise RuntimeError(
                    f"GraphStore ({self._config.graph_store.provider}) init failed. "
                    f"Use mode='pro' instead of mode='ultra' to disable Graph, or check logs for details."
                )
            logger.info(
                f"[init] GraphStore ready: provider={self._config.graph_store.provider}"
            )

        # 4. ComponentFactory (injects the already-initialized shared components)
        self._registry = ComponentFactory(config=self._config)
        self._registry._shared_embed_service = self._embed_service
        self._registry._shared_vector_store = self._vector_store
        self._registry._shared_vector_store_initialized = True
        if self._graph_store is not None:
            self._registry._shared_graph_store = self._graph_store

        # 5. Cache (must be initialized before component warmup; writer creation needs the cache injected)
        # - solo mode: create + initialize its own
        # - shared mode: reuse the runtime's cache (already inited + observability hook installed)
        if self._owns_runtime:
            self._cache = create_cache(self._config)
            await self._cache.initialize()
            logger.info(f"[init] Cache ready (backend={self._config.cache.backend})")
        else:
            self._cache = self._runtime.cache
            logger.info("[init] Cache reused from SharedRuntime")

        self._registry._shared_cache = self._cache

        # 5.1 Pipeline observability hook
        # - solo mode: install it once itself
        # - shared mode: the runtime already installed it on the shared cache, skip to avoid double-wrapping
        if self._owns_runtime:
            self._install_pipeline_observability_hooks()

        # 6. Warm up components (ensures the first add/search need not wait for initialization)
        await self._registry.get_writer()
        await self._registry.get_reader()
        logger.info(f"[init] Writer & Reader ready (mode={self._mode})")

        await self._initialize_local_reranker()

        if self._mode == "ultra":
            try:
                await self._registry.get_system2_writer()
                logger.info("[init] System2Writer ready (ultra mode)")
            except Exception as e:
                logger.warning(f"[init] System2Writer warmup failed: {e}")

        # 7. HistoryStore (optional, controlled by config.history.enable)
        self._history_store = None
        if self._config.history.enable:
            self._history_store = HistoryStore(self._config)
            await self._history_store.initialize()
            logger.info("[init] HistoryStore ready")

        # 8. MetricsCollector
        # - solo mode: bind cache to the metrics singleton + start background flush
        # - shared mode: the runtime already did it, skip (critical: avoids overwriting _cache and triggering cross-loop)
        if self._owns_runtime:
            from .metrics import MetricsCollector
            _metrics = MetricsCollector.get()
            _metrics.bind_cache(self._cache)
            await _metrics.start_background_tasks()
            logger.info("[init] MetricsCollector ready")
        else:
            logger.info("[init] MetricsCollector already bound by SharedRuntime")

        # 9. Coding memory pipeline placeholder (lazy initialization)
        #    Actually initialized only when add(messages_with_tool) first hits is_coding=True.
        self._coding_writer = None       # CodingWriter | None
        self._coding_store = None        # CodingMemoryStore | None
        self._coding_llm_provider = None  # LLMProvider | None
        self._coding_init_lock = asyncio.Lock()
        # ── Cached per variant ────────────────────────────────────────────────────
        # The two variants' stores are **separate** (see the _suffix comment in CodingMemoryStore.__init__),
        # so we cannot keep just one store -- otherwise, after add(scene="working"),
        # search(scene="coding") would read working's store and silently read the wrong one.
        # `_coding_store` / `_coding_writer` stay as "the most recently used one"; the read side uses
        # _store_for_variant() to fetch the correct one.
        self._coding_writers: Dict[str, Any] = {}   # variant → CodingWriter
        self._coding_stores: Dict[str, Any] = {}    # variant → CodingMemoryStore
        # Version-evolution chain layer (unique to scene="coding"). Zero LLM, pure sqlite; see _get_chain_store.
        self._chain_store = None         # HyCodeMem | None
        self._chain_store_failed = False  # Once init fails, never retry
        # Local repo path the chain layer uses to look up real diffs. None = fall back to extracting only old/new from the conversation;
        # it still works without a local repo, except the diffs come from the source text rather than git.
        self._chain_repo_root = os.getenv("MEMORY_CODING_CHAIN_REPO_ROOT") or None

    async def _initialize_local_reranker(self) -> None:
        """Load and warm the configured reranker before the client is ready."""
        if not getattr(self._config.recall, "local_rerank_enabled", False):
            return

        from .pipelines._retrieval.local_rerank import get_local_reranker

        reranker = get_local_reranker(
            getattr(
                self._config.recall,
                "local_rerank_model",
                "nreimers/mmarco-mMiniLMv2-L6-H384-v1",
            ),
            device=getattr(self._config.recall, "local_rerank_device", "auto"),
            max_length=getattr(
                self._config.recall, "local_rerank_max_length", 512,
            ),
            batch_size=getattr(
                self._config.recall, "local_rerank_batch_size", 16,
            ),
        )
        try:
            from .utils.async_executor import to_thread_guarded

            rerank_init = await to_thread_guarded(reranker.initialize)
        except Exception as exc:
            raise RuntimeError(
                "Local rerank is enabled but initialization failed: "
                f"model={reranker.model_name!r}, error={exc}"
            ) from exc
        logger.info(
            "[init] Local reranker ready: model=%s device=%s "
            "load_ms=%s warmup_ms=%s",
            reranker.model_name,
            reranker.device,
            rerank_init.get("load_ms", 0),
            rerank_init.get("warmup_ms", 0),
        )

    def _install_pipeline_observability_hooks(self) -> None:
        """
        Wraps cache.store_pipeline_log:
        - Log: JSONL file, always written (stage-level, independent of the Trace toggle)
        - Trace: SQLite pipeline_logs, controlled by MEMORY_PIPELINE_TRACE_ENABLED (Inspector)
        - Persisting is fire-and-forget and does not block the search/write hot path
        """
        from .utils.pipeline_observability import install_pipeline_observability

        install_pipeline_observability(self._cache, log_label="init")

    @classmethod
    def from_config(
        cls,
        config_dict: Dict[str, Any],
        mode: Optional[str] = None,
    ) -> "HyMemoryClient":
        """
        Create a client from a dict (mem0-style).

        Usage:
            client = HyMemoryClient.from_config({
                "vector_store": {
                    "provider": "qdrant",
                    "collection_name": "my_memories",
                },
                "graph_store": {
                    "provider": "neo4j",
                    "url": "bolt://localhost:7687",
                    "username": "neo4j",
                    "password": "password",
                },
                "enable_graph": True,
                "llm": {
                    "provider": "openai",
                    "model": "deepseek-chat",
                    "api_key": "sk-xxx",
                },
                "embedder": {
                    "provider": "openai",
                    "model": "text-embedding-v3",
                    "api_key": "sk-xxx",
                },
            })
        """
        config = MemoryConfig.from_dict(config_dict)
        resolved_mode = mode or config_dict.get("mode")
        return cls(config=config, mode=resolved_mode)

    # ================================================================
    # Properties
    # ================================================================

    @property
    def mode(self) -> str:
        """The current processing mode (lite/pro/ultra)."""
        return self._mode

    # ================================================================
    # Internal
    # ================================================================

    @staticmethod
    def _build_isolation_key(uid: str, agent_id: str) -> str:
        """
        Build the isolation_key used for history.

        Two-level isolation: "{user_id}:{agent_id}"
        """
        parts = [uid or "default", agent_id or "default"]
        return ":".join(parts)

    @staticmethod
    def _normalize_scene(scene: Optional[str]) -> Optional[str]:
        """
        Normalize the user-supplied scene argument -- resolve only the **route**, not the variant.

        Accepts only three values (case-insensitive):
          - "chat"    → takes the chat pipeline
          - "working" → takes the coding pipeline (general productivity variant)
          - "coding"  → takes the coding pipeline (software-engineering variant, see _scene_variant)
          - None      → unspecified, decided automatically by the LLM

        "working" and "coding" normalize to the same return value "coding" here: they run the same kind of
        productivity pipeline but use variant-isolated store / writer / reader. The extraction spec and whether the
        chain layer is attached are resolved separately by _scene_variant(). This way every
        `scene_forced == "coding"` check in add()/search() applies to both at once, with no per-site branch.

        Returns:
            "coding" | "chat" | None
        Raises ValueError on an invalid value.
        """
        if scene is None:
            return None
        s = str(scene).strip().lower()
        if s == "chat":
            return "chat"
        if s in ("working", "coding"):
            return "coding"
        raise ValueError(
            f"scene must be 'chat', 'working' or 'coding', got {scene!r}"
        )

    @staticmethod
    def _scene_variant(scene: Optional[str]) -> str:
        """
        Resolve the **variant** of the coding pipeline. Orthogonal to _normalize_scene: that one decides which chain to take,
        this one decides which spec to use once on it.

          - "coding"        → "coding"   software-engineering scenario
          - other / None    → "working"  general-productivity scenario (the original behavior)

        This helper returning working for None is just a compatibility fallback; the real scene=None auto
        routing is filled back in as working/coding by the three-way judge, not always swallowed to working.

        The two variants share the table schema and processing flow but use separate SQLite files and vector collections. Behavioral differences:
          1. Extraction spec -- coding uses EXTRACT_PROMPT_CODING (HARD CONSTRAINT header +
             CODE WORK section: branch names verbatim, which version won, exact coordinates, verified vs written).
          2. Version-evolution chain layer -- coding additionally attaches coding.chain_store, storing every version of each
             code's real patch with zero LLM. This addresses the very class of information the extraction spec itself cuts (it explicitly requires
             "EXACT COORDINATES, NOT PASTED CODE", yet the reverted version's code is not in the working tree,
             not in HEAD, only in some intermediate commit, and cannot be read back).
        """
        return "coding" if str(scene or "").strip().lower() == "coding" else "working"

    @staticmethod
    def _parse_input(data: AddInput):
        """
        Parse add()'s input, uniformly returning (content, chat_messages, content_type).

        str  → ("text", [], "text")
        list → ("", [ChatMessage, ...], "messages")

        In list mode it supports two tool-call styles, normalizing both into a unified ChatMessage:
        - OpenAI: assistant.tool_calls (function.arguments is a JSON string) +
                   a standalone role="tool" message (with tool_call_id / name)
        - Anthropic: assistant.content is a list[content_block] containing type=tool_use;
                     tool_result is nested in the content list of a later role="user" message,
                     and must be split into standalone normalized role="tool" messages

        A single raw message may yield 0 or more ChatMessages (Anthropic tool_result splitting).
        """
        if isinstance(data, str):
            return data, [], "text"

        if isinstance(data, list):
            chat_messages: List[ChatMessage] = []
            for raw in data:
                chat_messages.extend(HyMemoryClient._normalize_message(raw))
            return "", chat_messages, "messages"

        raise TypeError(
            f"add() first argument should be str or list[dict], got {type(data).__name__}"
        )

    @staticmethod
    def _normalize_message(m: Dict[str, Any]) -> List[ChatMessage]:
        """
        Normalize a single raw message into a ChatMessage list.

        The return is usually a single item; when the raw message is an Anthropic-style user message whose content
        list contains a tool_result block, it splits into [user_text_msg?, tool_msg, ...].
        """
        role = m.get("role", "user")
        raw_content = m.get("content")
        raw_tool_calls = m.get("tool_calls", []) or []

        # Some clients (e.g. codebuddy) encode tool_calls as a JSON string rather than a list;
        # normalize to a list here so the for loop below does not iterate over the string.
        if isinstance(raw_tool_calls, str):
            try:
                parsed_tool_calls = json.loads(raw_tool_calls) if raw_tool_calls else []
            except Exception:
                logger.warning(
                    "tool_calls is an invalid JSON string and was ignored: %r", raw_tool_calls[:200]
                )
                parsed_tool_calls = []
            raw_tool_calls = parsed_tool_calls if isinstance(parsed_tool_calls, list) else []
        elif not isinstance(raw_tool_calls, list):
            raw_tool_calls = []

        # ── 1. OpenAI assistant.tool_calls ──
        tool_calls: List[ToolCall] = []
        for tc in raw_tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {}) or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                # OpenAI encodes it as a JSON string
                try:
                    args = json.loads(args) if args else {}
                except Exception:
                    args = {"_raw": args}
            elif args is None:
                args = {}
            elif not isinstance(args, dict):
                args = {"_raw": args}
            tool_calls.append(ToolCall(
                id=tc.get("id", "") or "",
                name=fn.get("name", "") or "",
                arguments=args,
            ))

        # ── 2. OpenAI standalone tool message ──
        # Note: a tool message's `name` is the **tool name**, not the speaker name, so it still maps to tool_name.
        if role == "tool":
            return [ChatMessage(
                role="tool",
                content=str(raw_content or ""),
                tool_call_id=m.get("tool_call_id"),
                tool_name=m.get("name"),
            )]

        # For non-tool roles, `name` is the **speaker display name** (OpenAI messages spec).
        # Multi-person dialogues use it to tell who is who: role=assistant + name → a real person, not the AI.
        speaker_name = m.get("name") or None

        # ── 3. Anthropic content blocks ──
        if isinstance(raw_content, list):
            text_parts: List[str] = []
            tool_result_msgs: List[ChatMessage] = []
            for block in raw_content:
                if not isinstance(block, dict):
                    # Non-standard shape: treat as a string
                    text_parts.append(str(block))
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", "") or "")
                elif btype == "tool_use":
                    # Anthropic: input is already a dict
                    tool_input = block.get("input", {}) or {}
                    if not isinstance(tool_input, dict):
                        tool_input = {"_raw": tool_input}
                    tool_calls.append(ToolCall(
                        id=block.get("id", "") or "",
                        name=block.get("name", "") or "",
                        arguments=tool_input,
                    ))
                elif btype == "tool_result":
                    # Split into a standalone role=tool message
                    tc_id = block.get("tool_use_id")
                    inner = block.get("content", "")
                    if isinstance(inner, list):
                        # Nested content list (usually [{type:text,text:...}])
                        inner = "\n".join(
                            b.get("text", "") if isinstance(b, dict) and b.get("type") == "text"
                            else str(b)
                            for b in inner
                        )
                    elif not isinstance(inner, str):
                        inner = str(inner)
                    tool_result_msgs.append(ChatMessage(
                        role="tool",
                        content=inner,
                        tool_call_id=tc_id,
                        tool_name=None,  # Anthropic tool_result carries no name
                    ))
                # Silently ignore other unknown block types
            text = "\n".join(p for p in text_parts if p).strip()

            outputs: List[ChatMessage] = []
            # Keep the assistant message even when text is empty, as long as tool_calls is non-empty
            if text or tool_calls:
                outputs.append(ChatMessage(
                    role=role,
                    content=text,
                    name=speaker_name,
                    tool_calls=tool_calls,
                ))
            outputs.extend(tool_result_msgs)
            return outputs

        # ── 4. Plain string content ──
        # role=="tool" already returned early in branch 2, so this is never a tool message, hence tool_name is not filled
        # (the old code stuffed `name` into tool_name, which was meaningless for non-tool roles).
        content_str = "" if raw_content is None else str(raw_content)
        return [ChatMessage(
            role=role,
            content=content_str,
            name=speaker_name,
            tool_calls=tool_calls,
            tool_call_id=m.get("tool_call_id"),
        )]

    # ================================================================
    # Coding memory lazy init
    # ================================================================

    async def _get_coding_writer(self, variant: str = "working"):
        """
        Lazily initialize the coding write pipeline. **One per variant**.

        Only actually builds store / extractor / reconciler when it hits is_coding=True.
        Users on the chat path never incur this overhead.

        legacy CodingWriter = extractor + reconciler.

        variant("working" | "coding") decides two things:
          1. Which extraction spec to use (see _scene_variant)
          2. Which store to read/write -- the two variants' sqlite and vector collections are separate

        Each variant caches its own writer/store, so they can be mixed on the same client:
        add(scene="working") and add(scene="coding") each write their own store,
        and search fetches the matching one by scene, so they never cross.

        `self._coding_store` / `self._coding_writer` also point to the **most recently** used one,
        only for compatibility with existing code paths (e.g. list_memories). To fetch by variant, use
        _store_for_variant().
        """
        cached = self._coding_writers.get(variant)
        if cached is not None:
            self._coding_writer = cached
            self._coding_store = self._coding_stores.get(variant)
            return cached
        async with self._coding_init_lock:
            cached = self._coding_writers.get(variant)
            if cached is not None:
                self._coding_writer = cached
                self._coding_store = self._coding_stores.get(variant)
                return cached

            # Local import to avoid cold-start overhead
            from .agent.llm_provider import LLMProvider
            from .coding.store import CodingMemoryStore

            # Shared LLMProvider (same LLMConfig.model as the chat pipeline)
            if self._coding_llm_provider is None:
                self._coding_llm_provider = LLMProvider(self._config)

            # Store -- separate DB per variant (db_path and both vector collections carry a suffix); see
            # the _suffix comment in CodingMemoryStore.__init__. When variant="working"
            # the suffix is empty and the path and collection names are byte-identical to before the change.
            store = CodingMemoryStore(
                self._config,
                embed_service=self._embed_service,
                db_path=self._config.coding.db_path,
                variant=variant,
            )
            await store.initialize()

            # extractor + reconciler chained
            from .coding.extractor import CodingMemoryExtractor
            from .coding.reconciler import CodingMemoryReconciler
            from .coding.writer import CodingWriter

            extractor = CodingMemoryExtractor(
                self._coding_llm_provider,
                cache=self._cache,
                scene=variant,
                verbatim_fallback_enabled=bool(
                    getattr(
                        self._config.coding,
                        "verbatim_fallback_enabled",
                        False,
                    )
                ),
            )
            reconciler = CodingMemoryReconciler(
                store=store,
                embed_service=self._embed_service,
                llm_provider=self._coding_llm_provider,
                cache=self._cache,
            )

            writer = CodingWriter(
                store=store,
                extractor=extractor,
                reconciler=reconciler,
                llm_provider=self._coding_llm_provider,
                embed_service=self._embed_service,
                cache=self._cache,
            )
            self._coding_stores[variant] = store
            self._coding_writers[variant] = writer
            # Compatibility with existing paths: point to the most recently used one
            self._coding_store = store
            self._coding_writer = writer
            logger.info(
                "[coding] writer chain initialized (lazy), variant=%s", variant
            )
            return writer

    def _store_for_variant(self, variant: str = "working"):
        """Get a variant's store. Returns None if never initialized.

        The read side must go through this rather than reading self._coding_store directly -- the latter is "the most recently
        used one", so after add(working), search(coding) would get working's store.
        """
        return self._coding_stores.get(variant) or None

    @staticmethod
    def _chain_owner(user_id=None, workspace_id=None) -> str:
        """The chain layer's isolation key.

        `<user_id>|<workspace_id>` -- returns "" when both are empty. An empty string matches only an empty string,
        not a shared pool, so old data is not mixed in by newly written rows.

        Why user_id must be in the key: the thread key used to be only (branch, file, pkey),
        and without an owner, the same branch/file/pkey across different users merged into one chain, leaking data.
        """
        u = str(user_id or "").strip()
        w = str(workspace_id or "").strip()
        return "" if not (u or w) else "%s|%s" % (u, w)

    def _get_chain_store(self, owner: str = ""):
        """
        Lazily initialize the version-evolution chain layer (unique to scene="coding").

        Zero LLM, zero embedding, pure sqlite -- construction cost is negligible, so no async lock is needed.
        Returns None on failure: this layer is an **increment**; if unavailable it falls back to the pure-vector two-layer (the working behavior)
        rather than failing a whole add/search.

        A changed owner needs a new instance (when one client serves multiple users) -- the instance is just a sqlite
        connection + owner string, cheap to rebuild, whereas reusing would break owner filtering.
        The database file is the same, so different owners share the file and isolate via WHERE.

        Store location: hy_code_chain/hy_code.db in the same directory as coding_memory.db. Not merged into one
        file because CodingMemoryStore manages its own schema migrations; cramming three tables in would make the two migrations
        step on each other; two files in the same directory means backups/cleanup miss neither.
        """
        if self._chain_store is not None and getattr(
                self._chain_store, "owner", "") == owner:
            return self._chain_store
        if self._chain_store_failed:
            return None
        try:
            import os as _os
            from .coding.chain_store import HyCodeMem
            _dir = _os.path.join(
                _os.path.dirname(_os.path.abspath(self._config.coding.db_path)),
                "hy_code_chain",
            )
            self._chain_store = HyCodeMem(
                _dir, repo_root=self._chain_repo_root, owner=owner)
            logger.info("[coding] chain store initialized at %s (owner=%r)",
                        _dir, owner)
            return self._chain_store
        except Exception as e:
            # Try only once. Failures are usually environmental (unwritable directory / sqlite unavailable), and retrying pays the cost on every
            # add without ever improving.
            self._chain_store_failed = True
            logger.warning(
                "[coding] chain store init failed, falling back to vector-only: %s", e
            )
            return None

    # ----------------------------------------------------------------
    # Two glue methods for the version-evolution chain layer (write / read)
    # ----------------------------------------------------------------

    @staticmethod
    def _chain_text(chat_messages) -> str:
        """Render one add's messages into plain text the chain layer can read.

        The shape deliberately matches the lowercase prefixes _USER_LC / _DIFF_END recognize in chain_store
        (`user:` / `assistant:` / `[tool:name]`). Uppercase `USER:` is also recognized by _USER,
        but tool messages cannot enter the same boundary detection.

        Why not reuse CodingMemoryExtractor._format_conversation: it prefixes each line with an
        `[i] ` index prefix (for the LLM to read), and the chain layer's regex anchors at line start, so any prefix breaks every match.
        """
        rows = []
        for m in (chat_messages or []):
            try:
                content = m.content if getattr(m, "content", None) else ""
                if getattr(m, "is_tool_message", None) and m.is_tool_message():
                    rows.append(f"[tool:{getattr(m, 'tool_name', None) or '?'}] {content}")
                    continue
                role = getattr(m, "role", "") or "user"
                rows.append(f"{role}: {content}")
            except Exception:
                continue
        return "\n".join(rows)

    def _observe_chain(self, chat_messages, *, user_id=None, workspace_id=None,
                       branch=None) -> int:
        """Write side: store the code changes in this conversation segment into the version-evolution chain.

        Returns the number of hunks stored (0 = no recognizable code change in this segment, a normal case, not an error).

        Zero LLM, zero embedding, pure sqlite + regex throughout, suitable for synchronous execution on the write pipeline.

        Best-effort: any exception only logs a warning. This layer is an increment; when it fails the vector two-layer is already written,
        so dragging the whole add into failure is a net loss.
        """
        # Go through the class rather than self: _chain_owner is a staticmethod, and these two methods get bound by tests onto a bare
        # SimpleNamespace (as with _scene_variant), so self has no such attribute.
        store = self._get_chain_store(
            HyMemoryClient._chain_owner(user_id, workspace_id))
        if store is None:
            return 0
        try:
            text = self._chain_text(chat_messages)
            if not text.strip():
                return 0
            # If branch is given, do not guess (the branch in meta takes precedence over regex from the text).
            # workspace_id is not passed to observe separately: it is already in owner (see _chain_owner),
            # so the thread key no longer needs a second repo dimension.
            return int(store.observe(text, meta={"branch": branch or ""}) or 0)
        except Exception as e:
            logger.warning("[coding] chain observe failed: %s", e)
            return 0

    def _decorate_with_chain(self, facts, instruction: str = "", scope=None,
                             user_id=None, workspace_id=None):
        """Read side: attach version-evolution chain blocks to the facts recalled by the vector layer.

        Returns (block, used_idx):
          block    -- the multiline structured block to place **before** the fact list ("" when no chain attaches)
          used_idx -- indices of facts already placed into block; callers should remove them from the fact list to
                      avoid the same fact appearing twice.

        facts order is not re-sorted -- top-k is the vector layer's job; this layer only decides "what follows each fact".

        Best-effort: on failure returns ("", []), degrading to the pure-vector two-layer (the working behavior).
        """
        if not facts:
            return "", []
        # Go through the class rather than self: _chain_owner is a staticmethod, and these two methods get bound by tests onto a bare
        # SimpleNamespace (as with _scene_variant), so self has no such attribute.
        store = self._get_chain_store(
            HyMemoryClient._chain_owner(user_id, workspace_id))
        if store is None:
            return "", []
        try:
            return store.decorate(facts, instruction=instruction, scope=scope)
        except Exception as e:
            logger.warning("[coding] chain decorate failed: %s", e)
            return "", []

    async def _search_coding(
        self,
        target_query: str,
        *,
        user_ids: List[str],
        workspace_id: Optional[str],
        branch: Optional[str],
        limit: int,
        working_fact_min_score: Optional[float] = None,
        working_experience_min_score: Optional[float] = None,
        request_id: str,
        t0: float,
        variant: str = "working",
        include_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Coding retrieval path (fact-led two-layer retrieval):
          1. Get the store (lazy init), ensure it is initialized
          2. embed(target_query) -- once; both layers share the same vector
          3. Fact primary index: recall top-K active facts
          4. Follow links to experiences: pull experiences along the hit facts' derived_experience_ids
          5. Experience direct-query fallback: search the experience layer with the same vector and merge in
          6. Deduplicate (many facts pointing to one experience keep one) + existence filter (dead links drop out naturally)
          7. Label via: fact_link / direct, then return

        limit is the **per-layer return cap**: facts ≤ limit, experiences ≤ limit, so one call
        yields at most 2*limit items. The two layers count separately rather than share one total budget -- they play different roles:
        facts answer "what", experiences answer "how", and crowding either out is unreasonable.

        Both fetch some extra before trimming back to limit: the fact layer takes max(limit, 5), and the experience layer's link-follow and direct-query
        each take at most limit; in between they pass thresholds, dedup, and drop dead links, so without oversampling they cannot fill up.

        Two gates, one per layer, neither exempt:
          working_fact_min_score        fact layer, default 0.30
          working_experience_min_score  experience layer, default fact threshold + 0.10

        The experience gate treats link-follow and direct-query alike. Both paths gate on the same batch of scores anyway (all from the direct query below;
        link-follow just borrows them), so splitting into two tiers on one scale has no basis. Fact endorsement shows up in the **ordering**:
        link-follow ranks ahead of direct-query and is kept first when limit truncates -- not as a discount on admission.

        workspace_id / branch are **scope filters**, applied to both layers: unset means no filter (full recall),
        set means recall only within the scope. The rules are defined by coding.store._scope_where and shared with
        list_memories: workspace_id matches exactly; branch applies only when workspace_id
        is non-empty, and tolerates NULL (a row with no branch recorded is not treated as "recording a different branch"). The filter is pushed down to
        SQLite (the authoritative source for these two columns), and the vector layer widens fetching by scope selectivity, so filtering does not cost recall.

        When the scope empties the fact layer, the link-follow path loses its source too, and this retrieval degrades into a single layer of
        experience direct-query only. That is acceptable -- facts are "what happened in this repo", and using them across workspaces to drive
        experiences is meaningless -- but it is invisible in the result, so a warning is logged when degrading.

        On failure it is fail-safe: returns success=True with both layers empty (no exception raised).
        """
        fact_min_score = (_WORKING_FACT_DEFAULT_MIN_SCORE
                          if working_fact_min_score is None else working_fact_min_score)
        exp_min_score = (fact_min_score + _WORKING_EXPERIENCE_MIN_SCORE_DELTA
                         if working_experience_min_score is None
                         else working_experience_min_score)
        # coding-path trace (shares pipeline_logs with the chat path; best-effort)
        _tracer = None
        if getattr(self, "_cache", None) is not None:
            try:
                from .pipelines._retrieval.trace import ReadTraceLogger
                _tracer = ReadTraceLogger(
                    cache=self._cache,
                    request_id=request_id,
                    user_id=(user_ids[0] if user_ids else ""),
                    agent_id="default_agent",
                    reader_version="coding",
                )
            except Exception:
                _tracer = None

        def _empty() -> Dict[str, Any]:
            _r = {
                "request_id": request_id,
                "scene": "productivity",
                "facts": [],
                "experiences": [],
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            }
            # Keep the key set identical to the success path -- otherwise callers need two code paths for the fail-safe empty result,
            # yet fail-safe is by definition "no exception, return empty".
            if variant == "coding":
                _r["chain_block"] = ""
                _r["chain_used"] = []
            return _r

        try:
            # ★ Must pass variant: the two variants' stores are separate; omitting it reads working's store
            #   (silently reads the wrong one, symptom: "it was clearly ingested but cannot be recalled").
            #   Now each variant caches its own writer/store, so passing it does not pin the other variant's
            #   extractor -- only the earlier single-instance cache had that problem.
            #
            # Catch a zero-arg call with TypeError once: tests replace this method with a zero-arg stub
            # (three _stub_get_coding_writer in tests/coding/), and they only want
            # the "store is ready" side effect, not the variant. Forcing a kwarg on the stub would turn
            # a batch of tests unrelated to this change red, which is API fastidiousness, not correctness.
            try:
                await self._get_coding_writer(variant=variant)  # side effect: initialize the store
            except TypeError as _te:
                if "variant" not in str(_te):
                    raise
                await self._get_coding_writer()
        except Exception as e:
            logger.warning(f"[search-coding] init failed: {e}")
            return _empty()

        # Get **this variant's** store. Do not use self._coding_store directly -- that is "the most recently used
        # one", so after add(working), search(coding) would get the wrong store.
        #
        # Both fallback levels exist for tests, not defensive programming: the tests/coding/ fixture binds this method
        # onto a bare SimpleNamespace (self has neither _coding_stores nor
        # _store_for_variant), and they attach _coding_store directly onto the ns.
        _store = (getattr(self, "_coding_stores", None) or {}).get(variant) \
            or getattr(self, "_coding_store", None)

        # With multiple user_ids take the first (coding memory usually binds a single user;
        # cross-user coding memory retrieval is a future extension)
        primary_user = user_ids[0]

        try:
            _t_embed = time.perf_counter()
            embedding, embed_meta = await self._embed_service.embed_with_meta(target_query)
            if _tracer is not None:
                await _tracer.log_coding_embed(
                    query=target_query, dims=len(embedding or []),
                    elapsed_ms=(time.perf_counter() - _t_embed) * 1000,
                    provider_meta=embed_meta,
                )
        except Exception as e:
            logger.warning(f"[search-coding] embed failed: {e}")
            return _empty()

        # ---- 1. Fact primary index ----
        try:
            fact_hits = await _store.search_facts(
                embedding, user_id=primary_user,
                workspace_id=workspace_id, branch=branch,
                top=max(limit, 5),
                # lexical channel + move the gate below ahead of rerank: drop candidates that fail the gate first, then rerank,
                # so the returned count always equals pure-dense; the lexical channel only changes who is chosen (see store.search_facts).
                query_text=target_query,
                min_score=fact_min_score,
                include_sources=include_sources,
                exclude_sources=exclude_sources,
            )
        except Exception as e:
            logger.warning(f"[search-coding] fact search failed: {e}")
            fact_hits = []
        # The fact layer is the primary index; noise would be amplified along derived_experience_ids into a string of unrelated experiences,
        # so this gate matters even more than the experience direct-query path
        _fact_top_score = max((s for _, s in fact_hits), default=0.0)
        _fact_raw = len(fact_hits)
        fact_hits = [(f, s) for f, s in fact_hits if s >= fact_min_score]
        if _fact_raw and not fact_hits:
            # Facts were recalled but all blocked by the threshold: the link-follow path loses its source too, and this retrieval actually degrades into
            # a single layer of experience direct-query only. This degradation is invisible in the result (experiences are still there), so it can only
            # be reported here, otherwise nobody knows when the threshold is tuned too tight.
            logger.warning(
                f"[search-coding] fact layer emptied by threshold: "
                f"{_fact_raw} hit(s), top={_fact_top_score:.3f} < "
                f"working_fact_min_score={fact_min_score}; "
                f"fact_link path disabled for this query"
            )
        elif workspace_id and not fact_hits:
            # The scope filter emptied the fact layer (or this workspace simply has no facts). The consequence is the same as above:
            # the link-follow path loses its source and this retrieval degrades into a single experience-direct-query layer. It is invisible in the
            # result, so it can only be reported here.
            logger.warning(
                f"[search-coding] fact layer empty under scope "
                f"workspace_id={workspace_id!r} branch={branch!r}; "
                f"fact_link path disabled for this query"
            )
        # The fact layer over-fetched (top has a floor of 5, and the threshold cuts some), so trim back to limit here.
        # Link-follow must run on the finally returned facts, otherwise you get "an experience labeled via=fact_link whose endorsing
        # fact is not in the results".
        fact_hits = fact_hits[:limit]

        out_facts: List[Dict[str, Any]] = []
        linked_ids: List[str] = []
        _seen_linked = set()
        for fact, score in fact_hits:
            out_facts.append({
                "fact_id": fact.fact_id,
                "statement": fact.statement,
                "files": list(fact.files),
                # Conversation subject. The coverage gate uses it; the retrieval side surfaces it so consumers can tell apart
                # "these entries are about different people/files" instead of reading them as restatements of one thing.
                "subject": fact.subject,
                # The source text summarized away by statement (plan tables, config blocks, command output). The summary is responsible for
                # being recalled; this section is responsible for the value actually being there after recall -- point-lookup questions want the value in the cell,
                # not an overview of "what was done this time". Not involved in retrieval.
                "details": fact.details,
                # The LLM-written "how this conclusion was changed step by step into what it is now", fed straight into the prompt;
                # an empty string if never overwritten. Not involved in retrieval -- recall keys on the statement's current value.
                "fact_chain_summary": fact.fact_chain_summary,
                "created_at": fact.created_at.isoformat() if fact.created_at else None,
                "score": round(float(score), 4),
                "source": fact.source,
            })
            # ---- 2. Follow links to experiences (by fact-score order; many facts to one experience keep the first) ----
            for eid in fact.derived_experience_ids:
                if eid not in _seen_linked:
                    _seen_linked.add(eid)
                    linked_ids.append(eid)
        # No truncation here. The id list may contain dead links (the experience was DELETEd but the reference on the fact remains);
        # how many are dead is only known after get_many, so any fixed margin can be exhausted. Wait until details are fetched and
        # dead links have naturally dropped out, then truncate once by limit.
        # The length is naturally bounded: facts are already ≤ limit, and each derived_experience_ids is single-digit.

        # ---- 3. Experience direct-query fallback ----
        try:
            _t_vdb = time.perf_counter()
            hits = await _store.search_by_query_embedding(
                embedding,
                user_id=primary_user,
                workspace_id=workspace_id,
                branch=branch,
                top=max(limit * 4, 8),  # oversample to leave headroom for dedup
                include_sources=include_sources,
                exclude_sources=exclude_sources,
            )
        except Exception as e:
            logger.warning(f"[search-coding] vdb search failed: {e}")
            hits = []
            _t_vdb = time.perf_counter()

        _raw_hits = len(hits)
        _top_scores = [h.get("score", 0.0) for h in hits[:10]]
        # Save a copy of the scores before filtering: the link-follow experiences have no scores of their own (they were pulled by id,
        # not searched), so gating on a threshold can only borrow this round's scores. Same query vector, no extra call.
        score_by_id = {h["experience_id"]: h.get("score", 0.0) for h in hits}
        hits = [h for h in hits if h.get("score", 0.0) >= exp_min_score]
        # As above, no truncation here. Length is bounded by the VDB top above.

        # ---- 2b. The link-follow batch must pass the gate too ----
        # This path was originally exempt (rationale: "relevance is guaranteed by the hit fact"), but that guarantee is too weak:
        # a fact may match the query on one facet yet drag in experiences from another of its facets.
        # Now it shares the direct-query gate: both sides gate on the same batch of scores anyway, so two tiers have no basis.
        # Those whose score is unavailable (absent from the direct-query top-N) are treated as failing: if they cannot even make the top N,
        # their similarity is necessarily below the Nth, so on normal corpora they do not reach the gate.
        #
        # The same threshold means link-follow admits no extra experiences (anything passing the gate is already in direct_ids),
        # so its role reduces to ordering: link-follow ranks first and is kept first when limit truncates.
        linked_ids = [e for e in linked_ids
                      if score_by_id.get(e, 0.0) >= exp_min_score]
        linked_set = set(linked_ids)
        if _tracer is not None:
            await _tracer.log_coding_vdb(
                query=target_query, raw_hits=_raw_hits, kept_hits=len(hits),
                min_score=exp_min_score, top_scores=_top_scores,
                workspace_id=workspace_id, branch=branch,
                elapsed_ms=(time.perf_counter() - _t_vdb) * 1000,
            )

        direct_ids = [h["experience_id"] for h in hits]
        hit_by_id = {h["experience_id"]: h for h in hits}

        # ---- 4. Deduplicate + existence filter ----
        # Link-follow first, direct-query second; those get_many cannot find (experience already DELETEd / the two stores disagree)
        # are dropped naturally, so dead links need no separate handling
        ordered_ids = linked_ids + [e for e in direct_ids if e not in linked_set]
        # Pure safety valve: get_many uses a SQLite IN clause with a bound-variable limit (999 by default).
        # Normal volumes never come close; this only guards against pathologically long chains. Semantic count control is done by limit below.
        ordered_ids = ordered_ids[:_CODING_GET_MANY_MAX_IDS]
        try:
            details = await _store.get_many(
                ordered_ids, user_id=primary_user,
                include_sources=include_sources,
                exclude_sources=exclude_sources,
            )
        except Exception as e:
            logger.warning(f"[search-coding] fetch details failed: {e}")
            details = []
        # The single semantic truncation. Placed after get_many: dead links have already dropped out by this point,
        # so truncating first would let them waste slots and the actual return count would fall short of limit.
        # Link-follow is first, so what gets cut is preferentially the direct-query batch -- fact-endorsed ones deserve to stay.
        details = details[:limit]

        # ---- 5. Label the source ----
        out_experiences: List[Dict[str, Any]] = []
        for d in details:
            h = hit_by_id.get(d.experience_id) or {}
            out_experiences.append({
                "experience_id": d.experience_id,
                "agent_id": getattr(d, "agent_id", "") or "",
                "task": d.task,
                "search_keys": list(d.search_keys),
                "solution": d.solution,
                "boundary_envs": d.boundary_envs,
                "workspace_id": d.workspace_id,
                "branch": d.branch,
                "files": list(d.files),
                "source_fact_ids": list(d.source_fact_ids),
                "via": "fact_link" if d.experience_id in linked_set else "direct",
                "matched_key": h.get("matched_key_text"),
                "matched_key_kind": h.get("matched_key_kind"),
                # The link-follow batch is not in hit_by_id (it needed link-follow rescue precisely because its score fell short of the direct-query threshold),
                # but score_by_id kept their pre-filter scores, so surface them here
                "score": (round(float(score_by_id[d.experience_id]), 4)
                          if d.experience_id in score_by_id else None),
                "confidence": d.confidence,
                "updated_at": d.updated_at.isoformat() if d.updated_at else None,
                "source": d.source,
                "extraction_source": d.extraction_source,
            })

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            f"[search-coding] facts={len(out_facts)} experiences={len(out_experiences)} "
            f"(fact_link={sum(1 for m in out_experiences if m['via'] == 'fact_link')}) "
            f"elapsed_ms={elapsed} workspace_id={workspace_id!r} branch={branch!r}"
        )
        if _tracer is not None:
            await _tracer.log_coding_result(
                query=target_query,
                returned=len(out_experiences),
                # memory_ids is pipeline_logs' generic column name (shared with the chat path);
                # here it holds the coding-side experience_id
                memory_ids=[m["experience_id"] for m in out_experiences],
                previews=[
                    {"experience_id": m["experience_id"], "task": (m.get("task") or "")[:120],
                     "score": m.get("score"), "via": m.get("via")}
                    for m in out_experiences[:10]
                ],
                elapsed_ms=elapsed,
            )
        resp = {
            "request_id": request_id,
            "scene": "productivity",
            "facts": out_facts,
            "experiences": out_experiences,
            "elapsed_ms": elapsed,
        }
        # Version-evolution chain block (unique to scene="coding"). Placed last: it consumes the already-finalized out_facts above,
        # and the chain layer takes no part in recall or ordering.
        #
        # Note the return shape: the block goes in its own "chain_block" field, and chain-attached facts are **not** removed from
        # out_facts. Different callers may assemble prompts themselves, and silently withholding facts is more dangerous than adding one
        # extra field; when you need "block + remaining facts", filter yourself by chain_used.
        if variant == "coding":
            _blk, _used = self._decorate_with_chain(
                out_facts, instruction=target_query,
                scope=(workspace_id or None),
                user_id=primary_user, workspace_id=workspace_id,
            )
            resp["chain_block"] = _blk
            resp["chain_used"] = _used
        return resp

    # ================================================================
    # Synchronous API
    # ================================================================

    def add(
        self,
        data: AddInput,
        *,
        user_id: str = "",
        agent_id: str = "default_agent",
        session_id: str = "default_session",
        source: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        memory_at: Optional[datetime] = None,
        scene: Optional[str] = None,
        workspace_id: Optional[str] = None,
        branch: Optional[str] = None,
        request_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        embed_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Write a memory.

        Processing depth is determined by the mode at Client init:
        - lite:  pure embedding write
        - pro:   embedding + LLM agent extraction (facts/entities/reconcile)
        - ultra: pro plus System 2 cognitive processing explicitly triggerable via evolve()

        Args:
            data: a text string, or a messages list
                  - "The user likes sci-fi movies"
                  - [{"role": "user", "content": "..."}, ...]
            user_id:      user ID (level-1 isolation key)
            agent_id:     Agent ID (level-2 isolation key)
            session_id:   Session ID (level-3 isolation key)
            source:       call source, e.g. openclaw / cursor / codex; orthogonal to agent_id.
                          Before writing, strip().lower() is applied; default "default".
            metadata:     custom metadata
            memory_at:    memory timestamp (current time if omitted), for importing historical memories
            scene:        explicitly specify the write route, skipping the LLM's auto decision.
                          - None (default): with tools, the LLM decides chat/working/coding;
                            without tools it goes straight to chat
                          - "chat": force the chat pipeline (tool messages are stripped)
                          - "working": force the coding pipeline (general-productivity scenario)
                          - "coding": force the coding pipeline (software-engineering scenario)
                          Note: when forcing working / coding the input must contain tool messages,
                          otherwise it falls back to the chat pipeline.

                          "working" and "coding" reuse the same productivity pipeline and table schema,
                          but use separate SQLite files and vector collections. Behavioral differences:
                            1. Different extraction specs. coding uses EXTRACT_PROMPT_CODING: it requires
                               branch names kept verbatim, stating which version won and which was discarded, giving exact
                               repo-relative paths + symbol names, and distinguishing "verified" from "merely written".
                            2. coding additionally maintains the version-evolution chain layer (zero-LLM sqlite), storing
                               the real patch of every version of each code change. It recovers the class of information the extraction spec itself cuts --
                               it explicitly requires "coordinates, not pasted code", yet the reverted version's code is neither in the working tree
                               nor in HEAD, and cannot be read back.
                          To get the chain block, the search side must also pass scene="coding"; see search().
            workspace_id: for the Coding memory path only. Repo identifier (a normalized git remote URL is recommended).
                          Persisted with experiences and facts, so search / list can filter by scope. The chat path ignores it.
            branch:       for the Coding memory path only. Branch name. Same as workspace_id. The chat path ignores it.
            meta:         caller-defined business meta, stored with the memories produced this call and returned verbatim on query.
                          Not embedded, not involved in recall, not a filter condition.
                          Unlike metadata: metadata is merged into extra (which contains SDK internal control keys),
                          whereas meta is stored independently and only passed through.
                          Its semantics are "the context at the moment of that write", so it is stored as a list -- when the same node is
                          written multiple times the merge appends rather than overwrites; see the meta field returned by get().

        Returns:
            chat pipeline: {"success": True, "memory_id": "...", "request_id": "...", "elapsed_ms": ...}
            coding pipeline: {"success": True, "scene": "productivity", "ops": [...],
                        "experience_ids": [...], "fact_ops": [...], "fact_ids": [...], ...}
        """
        return self._loop_thread.run(
            self.async_add(
                data, user_id=user_id, agent_id=agent_id, session_id=session_id,
                source=source,
                metadata=metadata, memory_at=memory_at,
                scene=scene,
                workspace_id=workspace_id, branch=branch,
                request_id=request_id, meta=meta,
                embed_override=embed_override,
            ),
            include_runtime_diag=True,
        )

    def finalize_session(
        self,
        *,
        user_id: str,
        agent_id: str = "default_agent",
        session_id: str,
    ) -> Dict[str, Any]:
        """Finalize one caller session into a first-class episode narrative.

        The feature is opt-in through MEMORY_SESSION_EPISODE_ENABLED. Calls are
        idempotent: the same user/agent/session tuple upserts one deterministic node.
        """
        return self._loop_thread.run(
            self.async_finalize_session(
                user_id=user_id, agent_id=agent_id, session_id=session_id,
            ),
            include_runtime_diag=True,
        )

    def search(
        self,
        query: Optional[str] = None,
        *,
        queries: Optional[List[str]] = None,
        scene: Optional[str] = None,
        search_mode: str = "default",
        user_ids: Optional[List[str]] = None,
        agent_ids: Optional[List[str]] = None,
        session_ids: Optional[List[str]] = None,
        include_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
        limit: int = 10,
        min_score: float = 0.4,
        working_fact_min_score: Optional[float] = None,
        working_experience_min_score: Optional[float] = None,
        schema_min_score: float = 0.4,
        schema_limit: int = 10,
        intention_limit: int = 0,
        created_after: Optional[float] = None,
        reader: str = "",
        workspace_id: Optional[str] = None,
        branch: Optional[str] = None,
        request_id: Optional[str] = None,
        layers: Optional[List[Any]] = None,
        embed_override: Optional[Dict[str, Any]] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Search memories.

        Three-way recall (chat path):
        - **Schema path**: L6_SCHEMA (high-level behavior/state schema)
        - **Proactive path**: L7_INTENTION (proactive intention, off by default)
        - **Normal path**: L2_FACT + L3_SUMMARY + L4_IDENTITY + L5_KNOWLEDGE, etc.

        The L0 user profile is no longer mixed into Search; call get_user_profile() to fetch it separately.

        Coding path (auto-decided):
        - When the queries context is decided to be coding, use CodingMemoryStore's two-layer recall:
          the fact layer is the primary index, hit facts bring out experiences along derived_experience_ids, then the same
          vector directly queries the experience layer as a fallback. The return contains "scene": "productivity" and the "facts" /
          "experiences" lists.
        - otherwise use the chat path, returning results identical to before.

        Args:
            query:              search query (for legacy calls; mutually exclusive with queries)
            queries:            multiple queries (the last is the target; earlier ones resolve context).
                                In default mode, when Coding is on it classifies and rewrites; when Coding is off but
                                Query Rewrite is on it only rewrites/times; with both off there is zero LLM.
            scene:              explicitly specify the retrieval route, skipping the LLM's auto decision. Same values as add():
                                - None (default): search_mode decides whether to auto-decide
                                - "chat": force chat recall
                                - "working": force coding (productivity) recall
                                - "coding": as above, additionally returning the version-evolution chain block
                                  (the "chain_block" / "chain_used" fields; see Returns below)
                                (the legacy aliases "normal"/"productivity" remain supported)
            search_mode:        search execution mode:
                                - "default" (default): query="..." is a light search, going straight to
                                  chat/normal with zero LLM; with explicit queries=[...], whether to classify/rewrite is
                                  decided jointly by the Coding and Query Rewrite toggles.
                                - "agentic": with scene=None, both single query and multiple queries use
                                  one LLM call to do three-way classification, query rewrite, and time-range resolution.
                                An explicit scene always routes directly and never calls the LLM.
            user_ids:           list of user IDs (level-1 isolation key, supports cross-user search)
            agent_ids:          list of Agent IDs (empty list = search all agents)
            session_ids:        list of Session IDs (empty list = search all sessions)
            include_sources:    call-source allowlist; empty list = unrestricted.
            exclude_sources:    call-source blocklist; when it conflicts with the allowlist the blocklist wins.
            limit:              topk count for the Normal path, default 10
            min_score:          minimum score threshold **for the chat path only**, default 0.4. The working path ignores
                                this value; each layer has its own threshold (see the next two items).
            working_fact_min_score:
                                working-path fact layer (primary index) threshold, default 0.30. Deliberately lower than
                                chat: facts are single-sentence conclusions whose scores run systematically lower at the same relevance, so reusing
                                0.4 would block clearly relevant facts and also cut off the link-follow-to-experience path.
            working_experience_min_score:
                                working-path experience layer threshold, shared by link-follow and direct-query. The experience layer embeds
                                short segments like task + search_keys, whose similarity baseline is higher than the fact layer's
                                full-sentence statement, so sharing one threshold with facts makes experiences clearly too loose;
                                so when unset it takes working_fact_min_score + 0.10.
            schema_min_score:   minimum score threshold for the Schema path, default 0.4
            schema_limit:       maximum return count for the L6 Schema path, default 10
            intention_limit:    topk count for the Proactive path, default 0 (off)
            created_after:      Unix timestamp (float); return only memories with gmt_created >= this value
            reader:             Reader type, switched dynamically. Supports only legacy / hybrid.
                                Empty string = use the HY_MEMORY_READER env var or the default legacy.
            workspace_id:       for the Coding path only (the chat path ignores it). Scope filter: unset means
                                no filter (full semantic recall); set means both the fact and experience layers recall only within that
                                workspace. Filtering happens on the SQLite side, and vector fetching is widened by scope
                                selectivity, so filtering does not cost recall.
            branch:             same as workspace_id, but applied only when workspace_id is non-empty (consistent with
                                list_memories); passing it alone has no effect. Tolerates NULL: experiences/facts with no branch recorded
                                are still recalled, otherwise old data would be filtered out wholesale.
            request_id:         optional. Pre-generated by the App layer to align with SEARCH_API_ACCEPTED; if unset the SDK generates one
            layers:             optional. Under pro/ultra, recall only the specified layers (strings or
                                MemoryLayer), applied at the recall stage. Under lite this parameter is ignored and
                                recall is forced to L1_RAW only.

        Returns:
            chat path: {"request_id": "...", "memories": [...], "elapsed_ms": ...}
            coding path: {"request_id": "...", "scene": "productivity",
                         "facts": [...], "experiences": [...], "elapsed_ms": ...}
            scene="coding" additionally carries two fields:
                "chain_block" -- the version-evolution chain block (multiline text, "" when nothing attaches).
                                 It should go before facts for the model to read.
                "chain_used"  -- list of facts indices already placed into chain_block.
                                 facts are **not** deduplicated (nothing is silently withheld); for the
                                 "block + remaining fact" layout, filter yourself using this list.
        """
        return self._search_loop_thread.run(self.async_search(
            query, queries=queries, scene=scene, search_mode=search_mode,
            user_ids=user_ids, agent_ids=agent_ids, session_ids=session_ids,
            include_sources=include_sources, exclude_sources=exclude_sources,
            limit=limit, min_score=min_score,
            working_fact_min_score=working_fact_min_score,
            working_experience_min_score=working_experience_min_score,
            schema_min_score=schema_min_score, schema_limit=schema_limit,
            intention_limit=intention_limit, created_after=created_after,
            reader=reader,
            workspace_id=workspace_id, branch=branch,
            request_id=request_id, layers=layers,
            embed_override=embed_override,
            query_embedding=query_embedding,
        ))

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single memory.

        Args:
            memory_id: memory ID

        Returns:
            {"memory_id", "content", "layer", "request_id", "elapsed_ms"} or None (when it does not exist)
        """
        return self._loop_thread.run(self.async_get(memory_id))

    def update(
        self,
        memory_id: str,
        content: str,
    ) -> Dict[str, Any]:
        """
        Update memory content.

        Args:
            memory_id: memory ID
            content:   new content

        Returns:
            {"success": True, "memory_id": "...", "request_id": "...", "elapsed_ms": ...}
        """
        return self._loop_thread.run(self.async_update(memory_id, content))

    def delete(self, memory_id: str) -> Dict[str, Any]:
        """
        Delete a single memory.

        Args:
            memory_id: memory ID

        Returns:
            {"success": True, "deleted_count": 1, "request_id": "...", "elapsed_ms": ...}
        """
        return self._loop_thread.run(self.async_delete(memory_id))

    def delete_all(
        self,
        *,
        user_id: str = "",
        agent_ids: Optional[List[str]] = None,
        session_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Delete a user's memories.

        By default it deletes all of the user's memories (when neither agent_ids nor session_ids is passed).
        The delete scope can be narrowed via agent_ids/session_ids.

        Args:
            user_id:      user ID (level-1 isolation key)
            agent_ids:    list of Agent IDs (empty list = all agents)
            session_ids:  list of Session IDs (empty list = all sessions)

        Returns:
            {"success": True, "deleted_count": N, "request_id": "...", "elapsed_ms": ...}
        """
        return self._loop_thread.run(
            self.async_delete_all(user_id=user_id, agent_ids=agent_ids, session_ids=session_ids)
        )

    def list_memories(
        self,
        *,
        user_id: str = "",
        agent_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        order: str = "desc",
        workspace_id: Optional[str] = None,
        branch: Optional[str] = None,
        user_ids: Optional[List[str]] = None,
        agent_ids: Optional[List[str]] = None,
        layers: Optional[List[Any]] = None,
        with_evolution: bool = False,
    ) -> Dict[str, Any]:
        """
        List a user's memories (synchronous).

        Reads directly from the vector database, returning only memories in ACTIVE state.
        Sorted by created_at, with pagination support.

        Args:
            user_id:  user ID
            agent_id: optional, restrict to one agent
            limit:    items per page (default 100)
            offset:   offset (default 0)
            order:    sort order, "desc" (default, newest first) or "asc"
            workspace_id: optional (empty by default). Non-empty goes straight to the coding memory list path,
                          filtering by workspace_id and returning {"scene": "productivity", ...}.
            branch:   optional (empty by default). Adds exact branch filtering only when workspace_id is non-empty.
            user_ids: optional, batch listing across users. Non-empty takes precedence over user_id and returns the union
                      (pagination is applied uniformly after the merged sort).
            agent_ids: optional, batch listing across agents. Non-empty takes precedence over agent_id.
            layers:   optional, return only the specified layers (strings or MemoryLayer). When unset, returns
                      all layers except L1_RAW, identical to before the change.
            with_evolution: optional, attach evolution_chain to current-page entries (off by default).

        Returns:
            chat path: {"vdb": {...}, "graph": {...} | omitted, "elapsed_ms": ...}
            coding path: {"scene": "productivity", "experiences": [...], ...}
        """
        return self._loop_thread.run(
            self.async_list_memories(
                user_id=user_id, agent_id=agent_id,
                user_ids=user_ids, agent_ids=agent_ids,
                limit=limit, offset=offset, order=order,
                workspace_id=workspace_id, branch=branch,
                layers=layers, with_evolution=with_evolution,
            )
        )

    def get_user_basic_info(
        self,
        *,
        user_id: str = "",
    ) -> Dict[str, Any]:
        """Return the user's L0 basic info memories (synchronous).

        When an evolution chain exists, the chain head represents it and an attached `evolution_chain` is returned, consistent with search / list.

        Returns:
            {"request_id", "memories": [...], "elapsed_ms"}
        """
        return self._loop_thread.run(
            self.async_get_user_basic_info(user_id=user_id)
        )

    def get_user_profile(
        self,
        *,
        user_id: str,
    ) -> Dict[str, Any]:
        """Return the user's single assembled L0 profile.

        This is the public replacement for the L0 row previously mixed into
        Search results. It performs no embedding or LLM call.
        In Ultra mode, the active balanced catalog and specific supplement are
        appended to the same profile content.

        Returns:
            {"request_id": str, "profile": dict | None, "elapsed_ms": float}
        """
        return self._loop_thread.run(
            self.async_get_user_profile(user_id=user_id)
        )

    def clone_user(
        self,
        src_user_id: str,
        dst_user_id: str,
        *,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Clone a user's memories.

        Deep-copies all memories under src_user_id into dst_user_id;
        thereafter the two are fully independent and do not affect each other.

        Args:
            src_user_id: source user ID
            dst_user_id: destination user ID
            agent_id:    optional, restrict the clone to memories under one agent

        Returns:
            {"success": True, "request_id": "...", "cloned_count": N,
             "by_layer": {"l0_basic_info": 1, ...}, "elapsed_ms": ...}
        """
        return self._loop_thread.run(
            self.async_clone_user(
                src_user_id, dst_user_id, agent_id=agent_id,
            )
        )

    # ================================================================
    # System 2 Evolve API
    # ================================================================

    def evolve(
        self,
        user_id: str,
        agent_id: str = "default_agent",
        max_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Manually trigger System 2 cognitive processing (ultra mode only, blocking synchronously until done).

        One evolve = run System 2 once over the user's entire VDB (Schema induction → L6_SCHEMA
        + cross-domain sweeper). System 2's input is a full VDB scan, independent of "how many adds happened in between",
        so evolve does not consume the write queue and runs one complete cognitive cycle.

        Not idempotent: each evolve sees the latest full VDB, so running evolve multiple times in a row is meaningful
        (add a batch → evolve → add another batch → evolve again).

        Note: intentions (L7_INTENTION) are extracted and persisted immediately by System 1 at add/write time (present in pro/ultra
        modes) and are not handled in evolve. evolve is responsible only for System 2's Schema layer.

        The caller controls when it triggers: System 2 is not auto-triggered inside add/write and has no in-process queue.
        Typical usage: batch add() (running only S1) → call evolve() once to do full Schema induction.

        Callable only in ultra mode. Calling it in lite/pro mode raises RuntimeError.

        Args:
            user_id:  User ID
            agent_id: Agent ID

        Returns:
            {"success": True, "request_id": "...", "tasks_processed": 1, "results": {...},
             "cross_domain_sweeper": {...}, "elapsed_ms": ...}
        """
        if self._mode != "ultra":
            raise RuntimeError(
                f"evolve() requires mode='ultra', but this client is mode='{self._mode}'. "
                f"Create a new client with HyMemoryClient(mode='ultra') to use System 2."
            )
        return self._loop_thread.run(
            self.async_evolve(
                user_id=user_id, agent_id=agent_id, max_batches=max_batches,
            )
        )

    @_ensure_internal_loop
    async def async_evolve(
        self,
        user_id: str,
        agent_id: str = "default_agent",
        max_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Manually trigger System 2 cognitive processing (async)."""
        t0 = __import__("time").perf_counter()

        # Get the System2Writer
        try:
            writer = await self._registry.get_system2_writer()
        except (ValueError, KeyError):
            return {
                "success": False,
                "error": "System2Writer not available.",
                "tasks_processed": 0,
            }

        # Check whether evolve is supported
        if not hasattr(writer, "evolve"):
            return {
                "success": False,
                "error": f"Writer '{writer.VERSION}' does not support evolve()",
                "tasks_processed": 0,
            }

        result = await writer.evolve(
            user_id=user_id, agent_id=agent_id, max_batches=max_batches,
        )
        result["elapsed_ms"] = round((__import__("time").perf_counter() - t0) * 1000, 2)
        return result

    def sweep(
        self,
        user_ids: List[str],
        agent_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Manually trigger the cross-domain sweeper (synchronous).

        Independent of the evolve flow: it runs no S2 Agent and only does:
        1. Backfill the schema's beh abstraction + beh_embedding
        2. Collision scanning + core induction

        Runs the sweeper once for each user_id × agent_id combination.

        Args:
            user_ids:  List of SDK user_id values
            agent_ids: Agent ID list (default ["default_agent"])

        Returns:
            {"success": True, "results": [{...}, ...]}
        """
        if self._mode != "ultra":
            raise RuntimeError(
                f"sweep() requires mode='ultra', but this client is mode='{self._mode}'."
            )
        return self._loop_thread.run(
            self.async_sweep(user_ids=user_ids, agent_ids=agent_ids)
        )

    @_ensure_internal_loop
    async def async_sweep(
        self,
        user_ids: List[str],
        agent_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Manually trigger the cross-domain sweeper (async)

        Runs the sweeper once for each user_id × agent_id combination.
        """
        t0 = __import__("time").perf_counter()

        try:
            writer = await self._registry.get_system2_writer()
        except (ValueError, KeyError):
            return {"success": False, "error": "System2Writer not available.", "results": []}

        effective_agents = agent_ids or ["default_agent"]
        results = []
        for uid in user_ids:
            for aid in effective_agents:
                r = await writer.run_sweeper(user_id=uid, agent_id=aid)
                results.append({"user_id": uid, "agent_id": aid, **r})

        elapsed_ms = round((__import__("time").perf_counter() - t0) * 1000, 2)
        return {"success": True, "results": results, "elapsed_ms": elapsed_ms}

    # ================================================================
    # System Metrics
    # ================================================================

    def get_metrics(self, minutes: int = 5) -> Dict[str, Any]:
        """
        Get overall system load metrics.

        Args:
            minutes: Query data for the last N minutes (default 5 minutes)

        Returns:
            {
                "uptime_seconds": float,
                "requests": {"total", "active_sys1", "active_sys2", "queued_sys2", "completed", "failed"},
                "avg_latency_ms": {"sys1_waiting", "sys1_l1_process", "sys1_workflow", ...},
                "throughput": {"sys1_completed_last_60s", "sys2_completed_last_60s"},
                "storage_ops": {"vdb_ops_total", "vdb_ops_avg_ms", "graph_ops_total", "graph_ops_avg_ms"},
            }
        """
        return self._loop_thread.run(self.async_get_metrics(minutes=minutes))

    @_ensure_internal_loop
    async def async_get_metrics(self, minutes: int = 5) -> Dict[str, Any]:
        """Get overall system load metrics (async)."""
        from .metrics import MetricsCollector
        return await MetricsCollector.get().get_snapshot(minutes=minutes)

    def close(self):
        """Close the client and release all resources."""
        if self._registry is not None:
            self._loop_thread.run(self._close_async())
        # solo mode: this client owns loop_thread, so stopping it is safe
        # shared mode: loop_thread belongs to the runtime, leave it to runtime.aclose()
        if hasattr(self, '_loop_thread') and self._owns_runtime:
            self._loop_thread.stop()

    def history(self, memory_id: str) -> List[Dict[str, Any]]:
        """
        Query the change history of a memory.

        Args:
            memory_id: Memory ID

        Returns:
            History records sorted by time, each with event/old_memory/new_memory/created_at among other fields.
            Returns an empty list if history recording is not enabled.
        """
        return self._loop_thread.run(self.async_history(memory_id))

    def get_recent_history(
        self,
        user_id: str = "",
        agent_ids: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Query recent operation history.

        Args:
            user_id:    User ID
            agent_ids:  Agent ID list (empty list = query history of all agents)
            limit:      Maximum number of records to return

        Returns:
            Recent operations in reverse chronological order.
            Returns an empty list if history recording is not enabled.
        """
        return self._loop_thread.run(
            self.async_get_recent_history(user_id=user_id, agent_ids=agent_ids, limit=limit)
        )

    async def _close_async(self):
        if self._owns_runtime:
            from .metrics import MetricsCollector

            await MetricsCollector.get().stop_background_tasks()

        # Close the coding pipeline first (it depends on no other component)
        # ⚠️ Iterate _coding_stores instead of closing only self._coding_store -- the latter is only the "most
        #    recently used" one; when both variants are mixed the other one is missed (sqlite connections and VDB handles leak).
        for _v, _st in list(getattr(self, "_coding_stores", {}).items()):
            try:
                await _st.close()
            except Exception as e:
                logger.warning(f"[close] coding store close failed (variant={_v}): {e}")
        if getattr(self, "_coding_stores", None) is not None:
            self._coding_stores.clear()
        if getattr(self, "_coding_writers", None) is not None:
            self._coding_writers.clear()
        if getattr(self, "_coding_store", None) is not None:
            self._coding_store = None
            self._coding_writer = None
            self._coding_llm_provider = None
        # Chain layer: plain sqlite, no async resources
        if getattr(self, "_chain_store", None) is not None:
            try:
                self._chain_store.close()
            except Exception as e:
                logger.warning(f"[close] chain store close failed: {e}")
            self._chain_store = None
        if self._registry:
            await self._registry.close()
            self._registry = None
        if self._graph_store:
            await self._graph_store.close()
            self._graph_store = None
        if self._history_store:
            await self._history_store.close()
            self._history_store = None
        # cache: owned by the client in solo mode, so close it; owned by the runtime in shared mode, leave it to runtime.aclose()
        if self._cache and self._owns_runtime:
            await self._cache.close()
            self._cache = None
        elif self._cache and not self._owns_runtime:
            # shared mode: drop the reference only, do not actually close the pool
            self._cache = None
        logger.info("HyMemoryClient closed")

    # ================================================================
    # Delete helpers -- evolution chain repair + Graph evidence cleanup
    # ================================================================

    async def _repair_evolution_chain_on_delete(self, node) -> None:
        """
        Repair the evolution chain before a node is deleted.

        Case 1: delete the chain head (is_latest=True + non-empty supersedes)
          → Restore the predecessors listed in supersedes to is_latest=True + ACTIVE
          → Remove this node ID from each predecessor's superseded_by

        Case 2: delete a middle node (is_latest=False + non-empty superseded_by + non-empty supersedes)
          → Point the predecessor's superseded_by at the successor
          → Point the successor's supersedes at the predecessor (skipping this node)

        Case 3: delete the chain tail (is_latest=False + empty supersedes)
          → Only remove this node ID from the successor's supersedes
        """
        from .models.memory import MemoryStatus

        node_id = node.node_id
        supersedes = node.supersedes or []        # older nodes that this node supersedes
        superseded_by = node.superseded_by or []  # newer nodes that supersede this node
        is_latest = node.is_latest

        if not supersedes and not superseded_by:
            return  # not on any chain

        try:
            # Case 1: delete the chain head -- restore the predecessors
            if is_latest and supersedes:
                for pred_id in supersedes:
                    pred = await self._vector_store.get_by_id(pred_id)
                    if pred is None:
                        continue
                    # Remove this node from the predecessor's superseded_by
                    pred_sb = [x for x in (pred.superseded_by or []) if x != node_id]
                    # If the predecessor has no other successor, restore it as the chain head
                    if not pred_sb:
                        await self._vector_store.update_payload(pred_id, {
                            "is_latest": True,
                            "status": MemoryStatus.ACTIVE.value,
                            "superseded_by": [],
                        })
                        logger.info(f"[delete] chain repair: restored {pred_id} as chain head")
                    else:
                        await self._vector_store.update_payload(pred_id, {
                            "superseded_by": pred_sb,
                        })

            # Case 2: delete a middle node -- bridge predecessor and successor
            elif not is_latest and supersedes and superseded_by:
                # successor's supersedes: replace this node with this node's predecessors
                for succ_id in superseded_by:
                    succ = await self._vector_store.get_by_id(succ_id)
                    if succ is None:
                        continue
                    succ_sup = list(succ.supersedes or [])
                    if node_id in succ_sup:
                        succ_sup.remove(node_id)
                        succ_sup.extend(supersedes)
                    await self._vector_store.update_payload(succ_id, {
                        "supersedes": succ_sup,
                    })

                # predecessor's superseded_by: replace this node with this node's successors
                for pred_id in supersedes:
                    pred = await self._vector_store.get_by_id(pred_id)
                    if pred is None:
                        continue
                    pred_sb = list(pred.superseded_by or [])
                    if node_id in pred_sb:
                        pred_sb.remove(node_id)
                        pred_sb.extend(superseded_by)
                    await self._vector_store.update_payload(pred_id, {
                        "superseded_by": pred_sb,
                    })
                logger.info(f"[delete] chain repair: bridged {supersedes} → {superseded_by}")

            # Case 3: delete the chain tail -- remove it from the successor's supersedes
            elif not is_latest and not supersedes and superseded_by:
                for succ_id in superseded_by:
                    succ = await self._vector_store.get_by_id(succ_id)
                    if succ is None:
                        continue
                    succ_sup = [x for x in (succ.supersedes or []) if x != node_id]
                    await self._vector_store.update_payload(succ_id, {
                        "supersedes": succ_sup,
                    })

        except Exception as e:
            logger.warning(f"[delete] chain repair failed for {node_id}: {e}")

    async def _cleanup_graph_evidence(self, deleted_node_id: str) -> None:
        """
        Clean up the Graph evidence references of a deleted VDB node.

        After a VDB node is deleted, a Schema/Intention may still reference it as a VdbRef shadow
        node through DERIVED_FROM; that dangling reference has to be cleaned up.

        Note: VdbRef lives in its own node table, so delete_node cannot be used (it deletes Memory).
        Use the low-level execute to delete the VdbRef plus its DERIVED_FROM edges.
        """
        if self._graph_store is None:
            return
        try:
            gs = self._graph_store
            # Kuzu backend
            if hasattr(gs, '_execute') and hasattr(gs, '_available'):
                if not gs._available:
                    return
                gs._execute(
                    "MATCH (v:VdbRef {node_id: $nid}) DETACH DELETE v;",
                    {"nid": deleted_node_id},
                )
            # Neo4j backend
            elif hasattr(gs, '_run_write'):
                await gs._run_write(
                    "MATCH (v:VdbRef {node_id: $nid}) DETACH DELETE v",
                    {"nid": deleted_node_id},
                )
            logger.debug(f"[delete] VdbRef {deleted_node_id} cleaned from Graph")
        except Exception as e:
            logger.debug(f"[delete] VdbRef cleanup for {deleted_node_id}: {e}")

    # ================================================================
    # Async API
    # ================================================================

    @_ensure_internal_loop
    async def async_add(
        self,
        data: AddInput,
        *,
        user_id: str = "",
        agent_id: str = "default_agent",
        session_id: str = "default_session",
        source: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        memory_at: Optional[datetime] = None,
        scene: Optional[str] = None,
        workspace_id: Optional[str] = None,
        branch: Optional[str] = None,
        request_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        embed_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Write memory (async)

        embed_override: Per-call embedding endpoint override for this write (model/dims unchanged, only the endpoint switches).
            dict that may contain base_url / api_key / extra_headers / timeout; None = use the client config.
            Lets a single request switch to any OpenAI-compatible embedding endpoint.
        """
        if embed_override:
            with embed_endpoint_override(
                base_url=embed_override.get("base_url"),
                api_key=embed_override.get("api_key"),
                extra_headers=embed_override.get("extra_headers"),
                timeout=embed_override.get("timeout"),
            ):
                return await self._async_add_impl(
                    data, user_id=user_id, agent_id=agent_id, session_id=session_id,
                    source=source,
                    metadata=metadata, memory_at=memory_at,
                    scene=scene, workspace_id=workspace_id, branch=branch,
                    request_id=request_id, meta=meta,
                )
        return await self._async_add_impl(
            data, user_id=user_id, agent_id=agent_id, session_id=session_id,
            source=source,
            metadata=metadata, memory_at=memory_at,
            scene=scene, workspace_id=workspace_id, branch=branch,
            request_id=request_id, meta=meta,
        )

    @_ensure_internal_loop
    async def async_finalize_session(
        self,
        *,
        user_id: str,
        agent_id: str = "default_agent",
        session_id: str,
    ) -> Dict[str, Any]:
        """Async counterpart of :meth:`finalize_session`."""
        series = bool(getattr(self._config.recall, "series_episode_enabled", False))
        dynamic = bool(
            getattr(self._config.recall, "dynamic_episode_enabled", False) or series
        )
        if not (
            getattr(self._config.recall, "session_episode_enabled", False) or dynamic
        ):
            return {"success": False, "status": "disabled",
                    "error_message": "episode finalization is disabled"}
        if not user_id or not session_id:
            raise ValueError("user_id and session_id are required")
        from .agent.episode_summary import (
            finalize_dynamic_episode_capsules,
            finalize_series_episode_containers,
            finalize_session_episode,
        )
        writer = await self._registry.get_writer()
        vector_store = await writer._get_vector_store()
        finalize = (
            finalize_dynamic_episode_capsules if dynamic
            else finalize_session_episode
        )
        finalize_kwargs = dict(
            vector_store=vector_store,
            embed_service=writer.embed_service,
            llm_provider=writer.llm_provider,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        if finalize is finalize_dynamic_episode_capsules:
            finalize_kwargs["summary_concurrency"] = getattr(
                self._config.recall, "dynamic_episode_summary_concurrency", 3,
            )
        from .core.workload import episode_slot, workload_scope

        async with episode_slot(), workload_scope("derived"):
            result = await finalize(**finalize_kwargs)
        if series and result.get("status") != "failed":
            try:
                result["series_episode"] = await finalize_series_episode_containers(
                    vector_store=vector_store,
                    graph_store=self._graph_store,
                    user_id=user_id,
                    agent_id=agent_id,
                )
            except Exception as exc:
                logger.warning("[series-episode] finalize failed: %s", exc)
                result["series_episode"] = {
                    "status": "failed", "error": str(exc),
                }
        return {"success": result.get("status") != "failed", **result}

    async def _async_add_impl(
        self,
        data: AddInput,
        *,
        user_id: str = "",
        agent_id: str = "default_agent",
        session_id: str = "default_session",
        source: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        memory_at: Optional[datetime] = None,
        scene: Optional[str] = None,
        workspace_id: Optional[str] = None,
        branch: Optional[str] = None,
        request_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Write memory (async)

        Args:
            scene: Explicitly force the write route ("chat" / "working" / "coding");
                   None = auto-detect. See the add() docstring.
            workspace_id / branch: Coding memory path only; see the add() docstring.
            meta: Caller-defined business meta, passed through to storage; see the add() docstring.
        """
        from .utils.source_filter import normalize_source
        source = normalize_source(source)
        # Normalize the explicit scene (None = auto-detect)
        scene_forced = self._normalize_scene(scene)
        # Variant of the coding pipeline ("working" | "coding"). An explicit scene decides it
        # directly; with scene=None the 3-way judge fills it in, so it must not silently default to working.
        scene_variant = self._scene_variant(scene)
        # Generate the request_id and inject it into the log context (queue consumers can pass a fixed id to keep idempotency and traceability)
        request_id = request_id or str(uuid.uuid4())
        set_request_id(request_id)
        t0 = time.perf_counter()

        content, chat_messages, content_type = self._parse_input(data)
        uid = user_id

        # Pure input token count (exact tiktoken estimate; excludes prompt overhead such as system/few-shot/history)
        from .utils.token_count import count_input_tokens
        input_tokens = count_input_tokens(content, chat_messages)

        # ────────────────────────────────────────────────────────────
        # Coding pipeline routing
        # An explicit scene skips the LLM decision:
        #   scene="chat"   → force chat (chat pipeline after stripping tool messages)
        #   scene="coding" → force coding (tool messages required, otherwise falls back to chat)
        # scene=None (default) → automatic 3-way classification:
        #   no tool messages: O(1) short circuit → chat pipeline
        #   tool messages present but judged chat: chat pipeline after stripping tool messages
        #   tool messages present and judged working/coding: productivity pipeline of that variant
        # Key point: the existing chat path is left untouched
        # ────────────────────────────────────────────────────────────
        coding_available = (
            getattr(self, "_mode", "pro") != "lite"
            and content_type == "messages"
            and getattr(self._config, "coding", None) is not None
            and self._config.coding.enable
        )

        if scene_forced == "chat":
            # Forced chat: skip every coding decision, strip tool messages and take the chat pipeline
            if content_type == "messages" and has_any_tool_message(chat_messages):
                logger.info("[add] scene=chat forced; stripping tool messages")
                chat_messages = strip_tool_messages(chat_messages)
        elif coding_available and (
            scene_forced == "coding" or has_any_tool_message(chat_messages)
        ):
            has_tool = has_any_tool_message(chat_messages)

            if scene_forced == "coding" and not has_tool:
                # Forced coding but no tool messages: no coding memory can be extracted, fall back to chat
                logger.warning(
                    "[add] scene=coding forced but no tool messages found; "
                    "falling back to chat path"
                )
                is_coding = False
            elif scene_forced == "coding":
                # Explicitly set to coding: skip the LLM judge
                logger.info("[add] scene=coding forced; skip LLM judge")
                is_coding = True
            else:
                # Auto-detect (scene=None, tool messages present)
                # Lazily initialize the coding LLM provider (used by the judge)
                if self._coding_llm_provider is None:
                    from .agent.llm_provider import LLMProvider
                    self._coding_llm_provider = LLMProvider(self._config)
                try:
                    judged_scene = await classify_messages_scene(
                        chat_messages, self._coding_llm_provider,
                        cache=self._cache,
                        request_id=request_id,
                        user_id=uid,
                        agent_id=agent_id or "default_agent",
                        session_id=session_id or "",
                    )
                    is_coding = judged_scene != "chat"
                    if is_coding:
                        scene_variant = judged_scene
                except Exception as e:
                    logger.warning(f"[add] coding judge failed, fallback to chat: {e}")
                    is_coding = False

            if is_coding:
                logger.info(
                    f"[add] routing to PRODUCTIVITY path variant={scene_variant}: "
                    f"user={uid} agent_id={agent_id} "
                    f"workspace_id={workspace_id!r} branch={branch!r}"
                )
                writer = await self._get_coding_writer(variant=scene_variant)
                resp = await writer.write(
                    chat_messages,
                    user_id=uid,
                    agent_id=agent_id,
                    workspace_id=workspace_id,
                    branch=branch,
                    session_id=session_id,
                    request_id=request_id,
                    source=source,
                )
                # Version evolution chain layer (specific to scene="coding"). Placed after writer.write
                # and before the return, because it only depends on the source conversation, not on the vector-layer write result.
                # Best-effort: failing to attach this layer does not affect the vector-layer output of this add.
                if scene_variant == "coding":
                    resp["chain_rounds"] = self._observe_chain(
                        chat_messages, user_id=uid,
                        workspace_id=workspace_id, branch=branch,
                    )
                # Refresh elapsed in the same path
                resp["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                resp["input_tokens"] = input_tokens
                # Aligned with the chat pipeline: chat always returns the singular memory_id, while the
                # coding two-tier writer returns the plural experience_ids (one add usually yields 0 or 1 ADD).
                # An extra singular memory_id holds the first experience_id, so a caller can simply read
                # resp["memory_id"].
                _coding_eids = resp.get("experience_ids") or []
                resp["memory_id"] = _coding_eids[0] if _coding_eids else None
                return resp

            # Classified as chat: strip tool messages and continue
            if has_tool:
                logger.info(
                    "[add] classified as CHAT; stripping tools"
                )
                chat_messages = strip_tool_messages(chat_messages)

        if content_type == "messages":
            preview = f"messages({len(chat_messages)} turns)"
            logger.info(
                f"[add] writing {preview} user={uid} "
                f"agent_id={agent_id} session_id={session_id} mode={self._mode}"
            )
            logger.info(f"[add] messages detail: {[m.to_dict() for m in chat_messages]}")
        else:
            logger.info(
                f"[add] writing text({len(content)} chars) user={uid} "
                f"agent_id={agent_id} session_id={session_id} mode={self._mode} "
                f"content={content}"
            )

        # mode → agent_mode mapping
        # lite: no LLM agent call
        # pro/ultra: call the LLM agent for fact extraction
        agent_mode = "disabled" if self._mode == "lite" else "full"
        extra = dict(metadata) if metadata else {}
        extra["agent_mode"] = agent_mode
        extra["mode"] = self._mode

        write_req = WriteRequest(
            content=content,
            messages=chat_messages,
            user_id=uid,
            agent_id=agent_id,
            source=source,
            session_id=session_id,
            content_type=content_type,
            extra=extra,
            memory_at=memory_at,
            # Explicitly pass request_id through (a local variable, not polluted by the contextvar) as the single
            # source of truth for memory_ops / WRITE_SUMMARY / pipeline_logs along the whole write pipeline.
            request_id=request_id,
            client_meta=meta,
        )

        # mode → writer:
        #   lite/pro → MemoryWriter (they differ only in agent_mode)
        #   ultra    → System2Writer (write runs System 1 only; System 2 is triggered by evolve())
        if self._mode == "ultra":
            writer = await self._registry.get_system2_writer()
        else:
            writer = await self._registry.get_writer()
        logger.debug(f"[add] using writer: {writer.VERSION} (mode={self._mode})")
        result: WriteResponse = await writer.write(write_req)

        resp = {
            "success": result.success,
            "scene": "chat",
            "memory_id": result.memory_id,
            "request_id": request_id,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            "error_code": result.error_code if result.error_code else None,
            "error_message": result.error_message or None,
        }

        # timing breakdown
        _timing = result.extra.get("timing")
        if _timing:
            resp["timing"] = _timing

        # reconcile op counts + stored memory ids (analytics passthrough)
        if result.extra.get("reconcile_ops"):
            resp["ops_summary"] = result.extra["reconcile_ops"]
        if result.extra.get("agent_stored_ids"):
            resp["memory_ids"] = result.extra["agent_stored_ids"]
        _attach_write_diagnostics(resp, result.extra)

        # Token usage: pure input (tiktoken) plus the prompt/completion consumed by the
        # extract and reconcile LLM pipelines (lite mode runs no LLM, so extract/reconcile are 0).
        resp["input_tokens"] = input_tokens
        resp["extract_tokens"] = result.extra.get(
            "extract_tokens", {"prompt": 0, "completion": 0, "total": 0}
        )
        resp["reconcile_tokens"] = result.extra.get(
            "reconcile_tokens", {"prompt": 0, "completion": 0, "total": 0}
        )

        if result.success:
            logger.info(f"[add] done: memory_id={result.memory_id} elapsed={result.elapsed_ms:.0f}ms")

            # Record history (non-blocking; a failure only logs a warning)
            if self._history_store is not None:
                try:
                    isolation_key = self._build_isolation_key(uid, agent_id)
                    # Derive the text to record from content or messages
                    history_content = content or "; ".join(
                        f"{m.role}: {m.content}" for m in chat_messages
                    )
                    await self._history_store.record_add(
                        memory_id=result.memory_id,
                        content=history_content,
                        layer=result.layer or "",
                        isolation_key=isolation_key,
                    )
                except Exception as e:
                    logger.warning(f"[add] history record failed: {e}")
        else:
            logger.warning(f"[add] failed: {result.error_message}")

        return resp

    def add_extracted(
        self,
        contents: Union[str, List[str]],
        *,
        user_id: str = "",
        agent_id: str = "default_agent",
        session_id: str = "default_session",
        source: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        memory_at: Optional[datetime] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Write memories already extracted by the caller: skip extract and go straight to reconcile.

        Use it when the caller (for example the openclaw memory_add tool or an upstream agent) has
        already distilled the conversation into conclusive memory content, so the SDK does not need to run LLM extract again.
        The SDK still runs the full reconcile (deduplicate against existing memories / merge / contradiction evolution) and persists to L2_FACT.

        Differences from add():
        - The LLM extractor is not called, which saves one LLM call and is faster and cheaper in tokens.
        - No L1_RAW source node is written (the content is itself the fact).
        - Each entry in contents is one independent fact; several entries enter reconcile together and are deduplicated against each other.

        Args:
            contents: A single fact string, or a list of fact strings.
            All remaining arguments behave as in add().

        Returns:
            {"success", "memory_id", "memory_ids", "request_id", "elapsed_ms",
             "ops_summary", ...}
        """
        return self._loop_thread.run(
            self.async_add_extracted(
                contents, user_id=user_id, agent_id=agent_id,
                session_id=session_id, source=source, metadata=metadata,
                memory_at=memory_at, request_id=request_id,
            ),
            include_runtime_diag=True,
        )

    @_ensure_internal_loop
    async def async_add_extracted(
        self,
        contents: Union[str, List[str]],
        *,
        user_id: str = "",
        agent_id: str = "default_agent",
        session_id: str = "default_session",
        source: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        memory_at: Optional[datetime] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Write pre-extracted memories (async); skip extract and go straight to reconcile. See add_extracted()."""
        request_id = request_id or str(uuid.uuid4())
        set_request_id(request_id)
        t0 = time.perf_counter()
        from .utils.source_filter import normalize_source
        source = normalize_source(source)

        if isinstance(contents, str):
            content_list = [contents]
        else:
            content_list = list(contents or [])
        content_list = [c.strip() for c in content_list if isinstance(c, str) and c.strip()]

        if not content_list:
            return {
                "success": False,
                "request_id": request_id,
                "error_code": 400,
                "error_message": "contents is required (str or non-empty List[str])",
            }

        from .utils.token_count import count_input_tokens
        input_tokens = sum(count_input_tokens(c, None) for c in content_list)

        extra = dict(metadata) if metadata else {}
        extra["extracted_contents"] = content_list
        extra["agent_mode"] = "extracted"
        extra["mode"] = self._mode

        write_req = WriteRequest(
            content=content_list[0],
            user_id=user_id,
            agent_id=agent_id,
            source=source,
            session_id=session_id,
            content_type="text",
            extra=extra,
            memory_at=memory_at,
            request_id=request_id,
        )

        # lite/pro → MemoryWriter; ultra → System2Writer (which reuses MemoryWriter internally)
        if self._mode == "ultra":
            writer = await self._registry.get_system2_writer()
        else:
            writer = await self._registry.get_writer()
        result: WriteResponse = await writer.write_extracted(write_req)

        resp = {
            "success": result.success,
            "scene": "chat",
            "memory_id": result.memory_id,
            "request_id": request_id,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            "error_code": result.error_code if result.error_code else None,
            "error_message": result.error_message or None,
        }
        if result.extra.get("reconcile_ops"):
            resp["ops_summary"] = result.extra["reconcile_ops"]
        if result.extra.get("agent_stored_ids"):
            resp["memory_ids"] = result.extra["agent_stored_ids"]
        _attach_write_diagnostics(resp, result.extra)
        resp["input_tokens"] = input_tokens
        resp["reconcile_tokens"] = result.extra.get(
            "reconcile_tokens", {"prompt": 0, "completion": 0, "total": 0}
        )

        if result.success:
            logger.info(
                f"[add_extracted] done: {len(result.extra.get('agent_stored_ids', []))} "
                f"nodes from {len(content_list)} contents, elapsed={resp['elapsed_ms']:.0f}ms"
            )
        else:
            logger.warning(f"[add_extracted] failed: {result.error_message}")

        return resp

    def _spawn_search_dedup(
        self,
        all_mems: List[Dict[str, Any]],
        *,
        request_id: str,
        user_id: str,
        agent_id: str,
    ) -> None:
        """Deduplicate the L2_FACT + L4_IDENTITY items in search results (fire-and-forget).

        Only these two layers are considered (where duplication is worst); an item carrying evolution_chain is treated as a chain head and its chain node_ids
        are taken from evolution_chain so they are handled together with it. The whole job runs in the background and does not block search from returning.

        Deduplication batches by owner (group_items_by_owner): search does not pass agent_ids by default,
        so results may mix nodes of several agents (or even users) -- memories with owner=user belong
        to the user side and are merged across agents; those with owner=agent stay isolated per agent.

        Duplicate hits are set to SHADOW (a recoverable logical delete), never hard deleted.
        """
        if os.getenv("MEMORY_SEARCH_DEDUP_ENABLED", "true").strip().lower() in {
            "0", "false", "no", "off",
        }:
            return
        import asyncio
        from .pipelines._retrieval.dedup import (
            DedupItem, execute_dedup, group_items_by_owner,
        )

        _DEDUP_LAYERS = {"l2_fact", "l4_identity"}
        targets = [
            m for m in all_mems
            if (m.get("layer") or "").lower() in _DEDUP_LAYERS
            and m.get("memory_id")
            and m.get("_memory_kind") != "canonical_state"
        ]
        if len(targets) < 2:
            return

        vector_store = self._vector_store
        cache = getattr(self, "_cache", None)

        async def _run():
            try:
                import time as _time
                node_ids = [m["memory_id"] for m in targets]
                _t_emb = _time.perf_counter()
                embs = await vector_store.get_embeddings(node_ids)
                _get_emb_ms = round((_time.perf_counter() - _t_emb) * 1000, 2)
                items: List[DedupItem] = []
                for m in targets:
                    mid = m["memory_id"]
                    emb = embs.get(mid)
                    if not emb:
                        continue
                    chain = m.get("evolution_chain") or []
                    chain_ids = [c.get("node_id") for c in chain if c.get("node_id")] if chain else [mid]
                    items.append(DedupItem(
                        node_id=mid,
                        embedding=emb,
                        content=m.get("content", ""),
                        is_latest=True,
                        is_chain_head=bool(chain),
                        gmt_created=(float(m["gmt_created"]) if m.get("gmt_created") else None),
                        chain_node_ids=chain_ids or [mid],
                        # Batching dimension: search does not pass agent_ids by default, so _all_mems may mix
                        # nodes of several agents (or even users). owner=user items are merged across agents,
                        # owner=agent items stay isolated per agent.
                        owner=m.get("owner"),
                        user_id=m.get("user_id", "") or user_id,
                        agent_id=m.get("agent_id", "") or agent_id,
                        source=m.get("source", "default") or "default",
                    ))
                if len(items) >= 2:
                    for _batch in group_items_by_owner(
                        items,
                        default_user_id=user_id,
                        default_agent_id=agent_id,
                    ):
                        await execute_dedup(
                            _batch["items"],
                            vector_store=vector_store,
                            cache=cache,
                            trigger="search",
                            request_id=request_id,
                            # This used to pass effective_*_ids[0] unconditionally, so in cross-user or cross-agent
                            # retrieval the DEDUP log attribution was wrong (the first id was recorded while the
                            # deleted data could belong to another one). After batching each batch has a definite owner.
                            user_id=_batch["user_id"],
                            agent_id=_batch["agent_id"],
                            delete_from_store=True,
                            extra_parsed={
                                "embedding_source": "vdb_get_embeddings",
                                "get_embeddings_ms": _get_emb_ms,
                                "embed_count": len(node_ids),
                                "embed_ok_count": len(items),
                                "owner_bucket": _batch["owner"],
                                "source_bucket": _batch["source"],
                                "batch_size": len(_batch["items"]),
                            },
                        )
            except Exception as e:
                logger.debug(f"[search] background dedup failed: {e}")

        try:
            # dedup is a background task invisible to the user and touches only the VDB (thread pool, no loop affinity).
            # Dispatch it explicitly to the write loop, so it does not take scheduling away from the search loop and make the next search wait.
            asyncio.run_coroutine_threadsafe(_run(), self._loop_thread._loop)
        except RuntimeError:
            # No event loop (very rare): skip synchronously, the main flow is unaffected
            logger.debug("[search] no running loop for dedup; skipped")

    @_ensure_search_loop
    async def async_search(
        self,
        query: Optional[str] = None,
        *,
        queries: Optional[List[str]] = None,
        scene: Optional[str] = None,
        search_mode: str = "default",
        user_ids: Optional[List[str]] = None,
        agent_ids: Optional[List[str]] = None,
        session_ids: Optional[List[str]] = None,
        include_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
        limit: int = 10,
        min_score: float = 0.4,
        working_fact_min_score: Optional[float] = None,
        working_experience_min_score: Optional[float] = None,
        schema_min_score: float = 0.4,
        schema_limit: int = 10,
        intention_limit: int = 0,
        created_after: Optional[float] = None,
        reader: str = "",
        workspace_id: Optional[str] = None,
        branch: Optional[str] = None,
        request_id: Optional[str] = None,
        layers: Optional[List[Any]] = None,
        embed_override: Optional[Dict[str, Any]] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Search memories (async). See the search() docstring.

        embed_override: Per-call endpoint override for the query embedding of this search (model/dims unchanged, only the endpoint switches).
            dict that may contain base_url / api_key / extra_headers / timeout; None = use the client config.
            The contextvar must be set **after** @_ensure_search_loop delegates into the search loop --
            a contextvar does not cross the run_coroutine_threadsafe boundary, so it is wrapped here (already on the search loop).
        """
        if embed_override:
            with embed_endpoint_override(
                base_url=embed_override.get("base_url"),
                api_key=embed_override.get("api_key"),
                extra_headers=embed_override.get("extra_headers"),
                timeout=embed_override.get("timeout"),
            ):
                return await self._async_search_impl(
                    query, queries=queries, scene=scene, search_mode=search_mode,
                    user_ids=user_ids,
                    agent_ids=agent_ids, session_ids=session_ids, limit=limit,
                    include_sources=include_sources, exclude_sources=exclude_sources,
                    min_score=min_score,
                    working_fact_min_score=working_fact_min_score,
                    working_experience_min_score=working_experience_min_score,
                    schema_min_score=schema_min_score, schema_limit=schema_limit,
                    intention_limit=intention_limit, created_after=created_after,
                    reader=reader, workspace_id=workspace_id, branch=branch,
                    request_id=request_id, layers=layers,
                    query_embedding=query_embedding,
                )
        return await self._async_search_impl(
            query, queries=queries, scene=scene, search_mode=search_mode,
            user_ids=user_ids,
            agent_ids=agent_ids, session_ids=session_ids, limit=limit,
            include_sources=include_sources, exclude_sources=exclude_sources,
            min_score=min_score,
            working_fact_min_score=working_fact_min_score,
            working_experience_min_score=working_experience_min_score,
            schema_min_score=schema_min_score, schema_limit=schema_limit,
            intention_limit=intention_limit, created_after=created_after,
            reader=reader, workspace_id=workspace_id, branch=branch,
            request_id=request_id, layers=layers,
            query_embedding=query_embedding,
        )

    async def _async_search_impl(
        self,
        query: Optional[str] = None,
        *,
        queries: Optional[List[str]] = None,
        scene: Optional[str] = None,
        search_mode: str = "default",
        user_ids: Optional[List[str]] = None,
        agent_ids: Optional[List[str]] = None,
        session_ids: Optional[List[str]] = None,
        include_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
        limit: int = 10,
        min_score: float = 0.4,
        working_fact_min_score: Optional[float] = None,
        working_experience_min_score: Optional[float] = None,
        schema_min_score: float = 0.4,
        schema_limit: int = 10,
        intention_limit: int = 0,
        created_after: Optional[float] = None,
        reader: str = "",
        workspace_id: Optional[str] = None,
        branch: Optional[str] = None,
        request_id: Optional[str] = None,
        layers: Optional[List[Any]] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Search memories (async). See the search() docstring.

        layers (default empty): recall only the given layers, accepting strings or MemoryLayer values.
        The filter takes effect during recall rather than after recall, so unrelated layers cannot consume the limit budget.
        """
        # ── Input normalization ──
        if queries is not None and query is not None:
            raise ValueError("query and queries are mutually exclusive; choose one")
        if queries is None and query is None:
            raise ValueError("either query or queries is required")
        queries_supplied = queries is not None
        if queries is None:
            queries_list: List[str] = [query]
        else:
            if not isinstance(queries, list) or not queries:
                raise ValueError("queries must be a non-empty list[str]")
            queries_list = [str(q) for q in queries]
        target_query = queries_list[-1]

        search_mode_norm = str(search_mode or "default").strip().lower()
        if search_mode_norm not in ("default", "agentic"):
            raise ValueError(
                f"search_mode must be 'default' or 'agentic', got {search_mode!r}"
            )

        # scene validation + normalization
        # The public API standardizes on "chat" / "working" / "coding" (consistent with add()); the internal retrieval pipeline still uses
        # "normal" / "productivity", so this input boundary performs the mapping once.
        #   "working"      → "productivity" (coding recall, general productivity variant)
        #   "coding"       → "productivity" (coding recall, software engineering variant: additionally attaches
        #                     the version evolution chain block; see _scene_variant / _decorate_with_chain)
        #   "chat"         → "normal" (chat recall)
        #   "productivity" / "normal" are kept as legacy aliases, for backward compatibility.
        scene_norm: Optional[str] = None
        if scene is not None:
            sn = str(scene).strip().lower()
            _scene_map = {
                "working": "productivity",
                "coding": "productivity",
                "chat": "normal",
                "productivity": "productivity",  # backward compatible
                "normal": "normal",              # backward compatible
            }
            if sn not in _scene_map:
                raise ValueError(
                    f"scene must be 'chat', 'working' or 'coding', got {scene!r}"
                )
            scene_norm = _scene_map[sn]

        # Generate or reuse the request_id and inject it into the log context (the app may pass one to line up with SEARCH_API_ACCEPTED)
        request_id = (request_id or "").strip() or str(uuid.uuid4())
        set_request_id(request_id)
        t0 = time.perf_counter()

        # Stall attribution baseline: sampled at the same point as t0, the end value is taken before log_scene_judge and the delta lands in
        # READ_SCENE_JUDGE.parsed. It explains why "pure synchronous code costs hundreds of ms", separating
        # cgroup CPU throttling (throttle_ms≈elapsed_ms) / long GC pauses (gc_delta>=1) /
        # GIL contention (both ≈0 yet still slow). Both are evaluated synchronously and are as trustworthy as elapsed_ms.
        _stall_gc0 = _gc_collection_count()
        _stall_thr0 = _read_throttled_usec()

        effective_user_ids = user_ids if user_ids is not None else []
        effective_agent_ids = agent_ids if agent_ids is not None else []
        effective_session_ids = session_ids if session_ids is not None else []
        lite_mode = getattr(self, "_mode", "pro") == "lite"
        from .utils.source_filter import normalize_source_filters
        effective_include_sources, effective_exclude_sources = normalize_source_filters(
            include_sources, exclude_sources,
        )

        # Validation: user_ids must be provided
        if not effective_user_ids:
            raise ValueError("user_ids is required and cannot be empty")

        # Validation: multiple agent_ids together with multiple session_ids is invalid
        if len(effective_agent_ids) > 1 and len(effective_session_ids) > 1:
            raise ValueError(
                "Cannot specify multiple agent_ids and multiple session_ids simultaneously. "
                "Use single agent_id with multiple session_ids, or multiple agent_ids without session_ids."
            )

        # ────────────────────────────────────────────────────────────
        # Search mode + scene routing
        #
        #   scene explicitly set                     → direct routing, zero LLM
        #   default + query="..."                    → pinned to normal, zero LLM
        #   default + queries=[...]                  → keep the original multi-query judge/rewrite logic
        #   agentic + query="..." / queries=[...]    → 3-way classification + rewrite + time
        #   default multi-query input too long       → skip rewrite; classification only when needed
        #
        # queries= is itself the caller's explicit choice of context judge/rewrite; agentic goes further and
        # applies the same capability to a single query. query_rewrite_max_chars guards the length in both cases.
        # ────────────────────────────────────────────────────────────
        coding_enabled = (
            getattr(self._config, "coding", None) is not None
            and self._config.coding.enable
        )
        agentic_enabled = search_mode_norm == "agentic"
        rewrite_enabled = bool(
            getattr(self._config.llm, "enable_query_rewrite", False)
        )
        _max_chars = int(getattr(self._config.llm, "query_rewrite_max_chars", 1000) or 1000)
        _combined_len = sum(len(q) for q in queries_list)
        _too_long = _combined_len > _max_chars

        effective_target_query = target_query  # defaults to queries[-1] or query
        rewrite_created_after: Optional[float] = None  # lower time bound extracted by rewrite
        decided_scene: str
        decided_variant: str = HyMemoryClient._scene_variant(scene)
        rewrite_used: bool = False
        scene_mode: str = "default"          # explicit / llm / default
        scene_is_coding: bool = False
        _t_scene = t0
        # Net LLM time actually spent by the scene judge. Only the LLM branch (scene_mode="llm") has a value;
        # the pure synchronous default/explicit branches stay None, so parsed.llm_ms is not persisted.
        # Strictly separate from the overall duration of this step (elapsed_ms → parsed.step_ms).
        _scene_llm_ms: Optional[float] = None

        # Current system time (lets rewrite resolve relative times)
        from datetime import datetime as _dt_now
        _current_time_iso = _dt_now.now().isoformat(timespec="seconds")

        def _ensure_llm():
            if self._coding_llm_provider is None:
                from .agent.llm_provider import LLMProvider
                self._coding_llm_provider = LLMProvider(self._config)
            return self._coding_llm_provider

        if lite_mode:
            # lite is a strict raw-only vector retrieval mode. scene / search_mode must not route it
            # to working/coding, and no LLM judge/rewrite should be triggered either.
            decided_scene = "normal"
            scene_mode = "lite"
            scene_is_coding = False
            decided_variant = "working"
            logger.info(
                "[search] mode=lite forces raw-only normal search "
                "(ignore scene/search_mode/layers; no LLM)"
            )
        elif scene_norm is not None:
            # 1) Explicitly provided → no LLM call
            decided_scene = scene_norm
            scene_mode = "explicit"
            scene_is_coding = (decided_scene == "productivity")
            decided_variant = HyMemoryClient._scene_variant(scene)
            logger.info(
                f"[search] scene explicit={decided_scene} (no LLM judge); "
                f"target={effective_target_query!r}"
            )
        elif not agentic_enabled and not queries_supplied:
            # The default light search covers only query="...": pinned to normal, the LLM is never called.
            decided_scene = "normal"
            logger.info(
                f"[search] search_mode=default single-query scene=normal (no LLM); "
                f"target={target_query!r}"
            )
        elif agentic_enabled:
            # agentic: single or multiple queries force one LLM call that does 3-way classification + rewrite + time parsing.
            # The judge already builds a bounded query view, so arbitrarily long raw queries never degrade it to classify-only.
            want_classify = True
            _t_llm = time.perf_counter()
            try:
                from .coding.judge import analyze_queries
                ar = await analyze_queries(
                    queries_list, _ensure_llm(),
                    want_classify=want_classify, want_rewrite=True,
                    current_time=_current_time_iso,
                )
            except Exception as e:
                logger.warning(f"[search] analyze_queries failed, fallback normal: {e}")
                ar = {"is_coding": False, "rewrite_query": target_query,
                      "created_after": None, "ok": False}
            # Time failures too: the failure usually is a timeout, so the duration itself is key information
            _scene_llm_ms = (time.perf_counter() - _t_llm) * 1000

            scene_mode = "llm"
            if ar.get("ok"):
                judged_variant = str(ar.get("scene") or "chat") if want_classify else "chat"
                scene_is_coding = judged_variant in ("working", "coding")
                if scene_is_coding:
                    decided_variant = judged_variant
                decided_scene = "productivity" if scene_is_coding else "normal"
                new_q = ar.get("rewrite_query") or target_query
                if new_q and new_q != target_query:
                    effective_target_query = new_q
                    rewrite_used = True
                rewrite_created_after = ar.get("created_after")
                logger.info(
                    f"[search] agentic scene={decided_scene} via rewrite-LLM "
                    f"(classify={want_classify}); orig={target_query!r} "
                    f"effective={effective_target_query!r} rewrite_used={rewrite_used} "
                    f"created_after={rewrite_created_after}"
                )
            else:
                decided_scene = "normal"
                logger.info(
                    f"[search] agentic rewrite-LLM fallback=normal; target={target_query!r}"
                )
        elif rewrite_enabled and not _too_long:
            # Original default + queries= logic: one LLM call does rewrite/time when enabled in config,
            # and also 3-way classification when working memory is enabled.
            want_classify = coding_enabled
            _t_llm = time.perf_counter()
            try:
                from .coding.judge import analyze_queries
                ar = await analyze_queries(
                    queries_list, _ensure_llm(),
                    want_classify=want_classify, want_rewrite=True,
                    current_time=_current_time_iso,
                )
            except Exception as e:
                logger.warning(f"[search] multi-query analyze failed, fallback normal: {e}")
                ar = {"scene": "chat", "is_coding": False,
                      "rewrite_query": target_query, "created_after": None, "ok": False}
            _scene_llm_ms = (time.perf_counter() - _t_llm) * 1000
            scene_mode = "llm"
            if ar.get("ok"):
                judged_variant = str(ar.get("scene") or "chat") if want_classify else "chat"
                scene_is_coding = judged_variant in ("working", "coding")
                if scene_is_coding:
                    decided_variant = judged_variant
                decided_scene = "productivity" if scene_is_coding else "normal"
                new_q = ar.get("rewrite_query") or target_query
                if new_q and new_q != target_query:
                    effective_target_query = new_q
                    rewrite_used = True
                rewrite_created_after = ar.get("created_after")
            else:
                decided_scene = "normal"
        elif coding_enabled and not _too_long:
            # Original default + queries= logic: classification + coreference rewrite when working memory is enabled.
            _t_llm = time.perf_counter()
            try:
                from .coding.judge import classify_and_rewrite_queries
                jr = await classify_and_rewrite_queries(queries_list, _ensure_llm())
            except Exception as e:
                logger.warning(f"[search] multi-query classify/rewrite failed: {e}")
                jr = {"scene": "chat", "is_coding": False,
                      "rewrite_query": target_query, "ok": False}
            _scene_llm_ms = (time.perf_counter() - _t_llm) * 1000
            scene_mode = "llm"
            if jr.get("ok"):
                judged_variant = str(jr.get("scene") or "chat")
                scene_is_coding = judged_variant in ("working", "coding")
                if scene_is_coding:
                    decided_variant = judged_variant
                decided_scene = "productivity" if scene_is_coding else "normal"
                new_q = jr.get("rewrite_query") or target_query
                if new_q and new_q != target_query:
                    effective_target_query = new_q
                    rewrite_used = True
            else:
                decided_scene = "normal"
        elif coding_enabled:
            # default with over-long queries: legacy behaviour kept, classification only without rewriting.
            _t_llm = time.perf_counter()
            try:
                from .coding.judge import classify_queries_scene
                judged_variant = await classify_queries_scene(
                    queries_list, _ensure_llm()
                )
                scene_is_coding = judged_variant in ("working", "coding")
                if scene_is_coding:
                    decided_variant = judged_variant
            except Exception as e:
                logger.warning(f"[search] multi-query classify failed: {e}")
                scene_is_coding = False
            _scene_llm_ms = (time.perf_counter() - _t_llm) * 1000
            scene_mode = "llm"
            decided_scene = "productivity" if scene_is_coding else "normal"
        else:
            # default + queries=, but both related capabilities are disabled.
            decided_scene = "normal"
            logger.info(
                f"[search] default multi-query features disabled; no LLM; "
                f"target={target_query!r}"
            )

        # ────────────────────────────────────────────────────────────
        # Persist the scene decision to the pipeline log (the first search step, before the reader)
        # ────────────────────────────────────────────────────────────
        if getattr(self, "_cache", None) is not None:
            # Stall attribution end values: must be sampled before constructing ReadTraceLogger / calling log_scene_judge,
            # covering the same code span as the perf_counter window of elapsed_ms (t0 → here).
            _stall_gc_delta = _gc_collection_count() - _stall_gc0
            _stall_throttle_ms = (_read_throttled_usec() - _stall_thr0) / 1000.0
            try:
                from .pipelines._retrieval.trace import ReadTraceLogger
                _scene_trace = ReadTraceLogger(
                    cache=self._cache,
                    request_id=request_id,
                    user_id=(effective_user_ids[0] if effective_user_ids else ""),
                    agent_id=(effective_agent_ids[0] if effective_agent_ids else "default_agent"),
                    reader_version="client",
                    session_id=(effective_session_ids[0] if len(effective_session_ids) == 1 else ""),
                )
                await _scene_trace.log_scene_judge(
                    query=target_query or "",
                    scene=decided_scene,
                    mode=scene_mode,
                    is_coding=scene_is_coding,
                    rewrite_query=(effective_target_query if rewrite_used else None),
                    rewrite_used=rewrite_used,
                    created_after=(
                        datetime.fromtimestamp(rewrite_created_after).strftime("%Y-%m-%d %H:%M:%S")
                        if rewrite_created_after is not None else None
                    ),
                    elapsed_ms=(time.perf_counter() - _t_scene) * 1000 if _t_scene else 0.0,
                    llm_ms=_scene_llm_ms,
                    gc_delta=_stall_gc_delta,
                    throttle_ms=_stall_throttle_ms,
                )
            except Exception as _e:
                logger.debug(f"[search] scene judge trace failed: {_e}")

        # ────────────────────────────────────────────────────────────
        # Route by the scene that was decided
        # ────────────────────────────────────────────────────────────
        if decided_scene == "productivity":
            if not coding_enabled:
                logger.warning(
                    "[search] scene=productivity requested but productivity link disabled; "
                    "falling back to normal"
                )
                decided_scene = "normal"
            else:
                logger.info(
                    f"[search] routing to PRODUCTIVITY path variant={decided_variant}: "
                    f"target={effective_target_query!r} "
                    f"workspace_id={workspace_id!r} branch={branch!r}"
                )
                return await self._search_coding(
                    effective_target_query,
                    user_ids=effective_user_ids,
                    workspace_id=workspace_id,
                    branch=branch,
                    limit=limit,
                    working_fact_min_score=working_fact_min_score,
                    working_experience_min_score=working_experience_min_score,
                    request_id=request_id,
                    t0=t0,
                    # Called on the class instead of self: _scene_variant is a staticmethod and async_search
                    # is bound to a SimpleNamespace and called on its own in tests (see
                    # tests/coding/test_search_scene_routing.py), so the attribute is missing on self.
                    variant=decided_variant,
                    include_sources=effective_include_sources,
                    exclude_sources=effective_exclude_sources,
                )

        # ────────────────────────────────────────────────────────────
        # Chat retrieval pipeline (original logic)
        # ────────────────────────────────────────────────────────────
        logger.info(
            f"[search] query='{effective_target_query}' user_ids={effective_user_ids} "
            f"agent_ids={effective_agent_ids} session_ids={effective_session_ids} limit={limit}"
        )

        # Take the later (stricter) of the time lower bound extracted by rewrite and an explicit created_after
        effective_created_after = created_after
        if rewrite_created_after is not None:
            effective_created_after = (
                rewrite_created_after if effective_created_after is None
                else max(effective_created_after, rewrite_created_after)
            )

        # lite is pure vector storage: a write only yields an ACTIVE L1_RAW and never extracts L2/L4 again.
        # This is a hard mode-level constraint, not a default: even when the caller passes other layers
        # explicitly they must be ignored and only L1_RAW queried. pro / ultra keep the original layer semantics.
        from .models.memory import MemoryLayer

        effective_search_layers = [MemoryLayer.L1_RAW] if lite_mode else layers
        effective_schema_limit = 0 if lite_mode else schema_limit
        effective_intention_limit = 0 if lite_mode else intention_limit

        read_req = ReadRequest(
            query=effective_target_query,
            user_ids=effective_user_ids,
            agent_ids=effective_agent_ids,
            session_ids=effective_session_ids,
            include_sources=effective_include_sources,
            exclude_sources=effective_exclude_sources,
            limit=limit,
            min_score=min_score,
            schema_min_score=schema_min_score,
            schema_limit=effective_schema_limit,
            intention_limit=effective_intention_limit,
            request_id=request_id,
            created_after=effective_created_after,
            # ReadRequest.layers is List[str]; the reader converts it back into enums.
            # It still goes through _normalize_layers so invalid values fail at the entry and legacy numeric aliases are unified.
            layers=(
                [
                    layer.value
                    for layer in self._normalize_layers(effective_search_layers)
                ]
                if effective_search_layers else None
            ),
            query_embedding=query_embedding,
        )

        reader_impl = await self._registry.get_reader(reader_name=reader)
        logger.debug(f"[search] using reader: {reader_impl.VERSION}")
        result: ReadResponse = await reader_impl.read(read_req)

        if not result.success:
            logger.warning(f"[search] failed: {result.error_message}")
            return {
                "request_id": request_id,
                "memories": {"schema": [], "proactive": [], "normal": []},
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            }

        # Fallback layers filter: targeted recall is implemented by each reader, but support varies
        # (the profile / graph paths often recall fixed layers and ignore request.layers), so trusting the reader
        # alone leaks layers that were never requested. The contract is to return only the hit layers, hence this extra filter.
        _wanted_layers = (
            {
                layer.value
                for layer in self._normalize_layers(effective_search_layers)
            }
            if effective_search_layers else None
        )

        _all_mems = []
        for mem in result.memories:
            if _wanted_layers is not None and mem.get("layer", "") not in _wanted_layers:
                continue
            if lite_mode and mem.get("layer", "") != MemoryLayer.L1_RAW.value:
                continue
            score = mem.get("score", 0.0)
            item = {
                "memory_id": mem.get("memory_id", ""),
                "content": mem.get("content", ""),
                "score": round(score, 4),
                "layer": mem.get("layer", ""),
                # Callers need this to trace ownership in multi-user retrieval (R1)
                "user_id": mem.get("user_id", "") or "",
                "agent_id": mem.get("agent_id", "") or "",
                "source": mem.get("source", "default") or "default",
                "retrieval_source": mem.get("retrieval_source", "") or "",
                "owner": mem.get("owner"),
                "speculate": mem.get("speculate"),
                # source_raw_memory_id is not exposed: it points at an L1_RAW node, and raw nodes are never
                # visible on external read paths, so the id resolves to nothing (fetching by ID reports it missing)
                # and would only give callers the impression that they can trace the source text through it.
                # The field stays on the node; internal tracing and clone remapping are unaffected.
                "tags": mem.get("tags") or [],
                "memory_at": mem.get("memory_at"),
                "gmt_created": mem.get("gmt_created"),
                # Caller-defined meta (pass-through payload), which the reader takes from custom["meta"];
                # a reader that does not implement this field returns an empty list.
                "meta": mem.get("meta") or [],
                "_schema_channel": bool(mem.get("_schema_channel")),
            }
            if mem.get("schema_type"):
                item["schema_type"] = mem["schema_type"]
            if mem.get("workspace_id"):
                item["workspace_id"] = mem["workspace_id"]
            if mem.get("branch"):
                item["branch"] = mem["branch"]
            if mem.get("evolution_chain"):
                item["evolution_chain"] = mem["evolution_chain"]
            if mem.get("capsule_memories") is not None:
                item["capsule_memories"] = mem["capsule_memories"]
            if mem.get("episode_id"):
                item["episode_id"] = mem["episode_id"]
            if mem.get("member_ids") is not None:
                item["member_ids"] = mem["member_ids"]
            if mem.get("timeline") is not None:
                item["timeline"] = mem["timeline"]
            if mem.get("rerank_score") is not None:
                item["rerank_score"] = mem["rerank_score"]
            _all_mems.append(item)

        _dedup_mems = []
        for source, public in zip(result.memories, _all_mems):
            copy = dict(public)
            copy["_memory_kind"] = source.get("_memory_kind", "")
            _dedup_mems.append(copy)

        # Search pipeline dedup: only L2_FACT + L4_IDENTITY (where duplication is worst) are deduplicated and removed.
        # L0 and the other layers do not take part. Fire-and-forget, it does not block this return (the cleanup serves later queries).
        try:
            self._spawn_search_dedup(
                _dedup_mems,
                request_id=request_id,
                user_id=(effective_user_ids[0] if effective_user_ids else ""),
                agent_id=(effective_agent_ids[0] if effective_agent_ids else ""),
            )
        except Exception as e:
            logger.debug(f"[search] spawn dedup failed (non-fatal): {e}")

        # Memory Strength: best-effort write-back of access_count+1 / last_accessed_at=now after a search
        # (fire-and-forget, does not block the response, failures stay silent). The +1 is computed from the
        # access_count the reader already returned, avoiding an extra read. Gated by the master switch
        # strength_enabled (off by default), and the write-back alone can be turned off with MEMORY_RECALL_ACCESS_TRACKING=false.
        _recall_cfg = self._config.recall
        if (getattr(_recall_cfg, "strength_enabled", False)
                and getattr(_recall_cfg, "access_tracking_enabled", True)):
            _bump_items = [
                (mem.get("memory_id", ""), mem.get("access_count", 0))
                for mem in result.memories
                if mem.get("memory_id")
            ]
            if _bump_items:
                from .pipelines._retrieval.strength import bump_access as _bump_access

                async def _do_bump():
                    try:
                        await _bump_access(self._vector_store, _bump_items)
                    except Exception as _be:
                        logger.debug(f"[search] access bump failed: {_be}")

                try:
                    # access-bump touches only the VDB and is not on the critical path → dispatch it to the write loop,
                    # so it does not take scheduling away from the search loop.
                    asyncio.run_coroutine_threadsafe(_do_bump(), self._loop_thread._loop)
                except Exception as _be:
                    logger.debug(f"[search] access bump schedule failed: {_be}")

        # Group by channel: schema contains only L6; L0 is returned separately by get_user_profile().
        # proactive = intention; everything else (including L2/L3/L4) belongs to normal.
        _SCHEMA_LAYERS = {"l6_schema"}
        _PROACTIVE_LAYERS = {"l7_intention"}
        memories = {"schema": [], "proactive": [], "normal": []}
        total = 0
        for mem in _all_mems:
            channel = _search_memory_channel(
                mem,
                schema_layers=_SCHEMA_LAYERS,
                proactive_layers=_PROACTIVE_LAYERS,
                schema_min_score=schema_min_score,
            )
            memories[channel].append(mem)
            total += 1

        logger.info(
            f"[search] done: {total} results "
            f"(schema={len(memories['schema'])} proactive={len(memories['proactive'])} "
            f"normal={len(memories['normal'])}), elapsed={result.elapsed_ms:.0f}ms"
        )
        for ch, mems in memories.items():
            for i, mem in enumerate(mems):
                logger.debug(
                    f"[search] {ch}#{i} score={mem['score']:.4f} "
                    f"[{mem['layer']}] {mem['content']}"
                )

        # Record search history (optional, non-blocking)
        if self._history_store is not None and self._config.history.record_searches:
            try:
                hist_user = effective_user_ids[0] if effective_user_ids else "default"
                hist_agent = effective_agent_ids[0] if effective_agent_ids else "default"
                isolation_key = self._build_isolation_key(hist_user, hist_agent)
                await self._history_store.record_search(
                    query=effective_target_query,
                    isolation_key=isolation_key,
                    results_count=total,
                )
            except Exception as e:
                logger.warning(f"[search] history record failed: {e}")

        output = {
            "request_id": request_id,
            "memories": memories,
            "rewrite": {
                "used": rewrite_used,
                "original_query": target_query,
                "effective_query": effective_target_query,
                "created_after": rewrite_created_after,
                "scene": decided_scene,
            },
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        for key in (
            "episode_capsule",
            "episode_member_expansion",
            "retrieval_clauses",
            "local_rerank",
            "hierarchy_bundle",
            "multi_seed_bundle",
            "supplemental_propositions",
            "dynamic_entity_tag_embedding",
            "tag_index_nominee",
        ):
            if key in result.extra:
                output[key] = result.extra[key]
        return output

    # ================================================================
    # CRUD API (async) -- get / update / delete / delete_all
    # ================================================================

    @_ensure_internal_loop
    async def async_get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single memory (async).

        Queries the VDB and the graph in parallel and returns whichever side has a result.

        Returns:
            {"memory_id", "content", "layer", "source", "request_id", "elapsed_ms"} or None
        """
        import asyncio as _asyncio_get
        from .models.memory import MemoryLayer

        request_id = str(uuid.uuid4())
        set_request_id(request_id)
        t0 = time.perf_counter()

        logger.info(f"[get] memory_id={memory_id}")

        # Two paths in parallel
        async def _get_vdb():
            return await self._vector_store.get_by_id(memory_id)

        async def _get_graph():
            if self._graph_store is None:
                return None
            try:
                return await self._graph_store.get_node(memory_id)
            except Exception as e:
                logger.debug(f"[get] graph_store query failed: {e}")
                return None

        vdb_node, graph_node = await _asyncio_get.gather(_get_vdb(), _get_graph())

        # L1_RAW is not externally visible: fetching directly by id uses a primary key lookup and bypasses the VDB layer filter,
        # so it is blocked here uniformly. async_get is the mandatory common path of GET /v1/memories/{id} and
        # /v1/memories/batch, the two external endpoints, while the internal write pipeline never reaches it
        # (the writer post-write check calls vector_store.get_by_id directly), so blocking here leaves internals untouched.
        #
        # Placed after gather and before tracing: a raw hit still writes the GET trace as usual (keeping observability,
        # so it stays visible when a caller probes with a raw id); only the content is withheld from the caller.
        # The caller sees exactly what it would see for a nonexistent id, with no hint that the node exists but is blocked.
        if vdb_node is not None and vdb_node.layer == MemoryLayer.L1_RAW:
            vdb_node = None
        if graph_node is not None and getattr(graph_node, "layer", None) == MemoryLayer.L1_RAW:
            graph_node = None

        elapsed = round((time.perf_counter() - t0) * 1000, 2)

        # pipeline trace (best-effort, controlled by MEMORY_PIPELINE_TRACE_ENABLED)
        _hit_node = vdb_node if vdb_node is not None else graph_node
        _source = "vdb" if vdb_node is not None else ("graph" if graph_node is not None else "")
        _ident = self._backend_identity()
        _parsed = {
            "memory_id": memory_id,
            **_ident,
            "hit": _hit_node is not None,
            "source": _source,
        }
        if _hit_node is not None:
            try:
                _parsed["layer"] = _hit_node.layer.value
            except Exception:
                pass
        await self._maybe_store_op_log(
            step="GET",
            request_id=request_id,
            user_id=(_hit_node.user_id if _hit_node else "") or "",
            agent_id=(getattr(_hit_node, "agent_id", "") if _hit_node else "") or "",
            parsed=_parsed,
            memory_ids=[memory_id],
            elapsed_ms=elapsed,
        )

        # VDB takes precedence (when both sides have a result, the VDB wins)
        if vdb_node is not None:
            return {
                "memory_id": vdb_node.node_id,
                "content": vdb_node.content,
                "layer": vdb_node.layer.value,
                "status": vdb_node.status.value if vdb_node.status else "active",
                "is_latest": vdb_node.is_latest,
                "supersedes": vdb_node.supersedes or [],
                "superseded_by": vdb_node.superseded_by or [],
                "tags": vdb_node.tags or [],
                "memory_at": int(vdb_node.memory_at.timestamp()) if isinstance(vdb_node.memory_at, datetime) else None,
                "gmt_created": int(vdb_node.gmt_created.timestamp()) if isinstance(vdb_node.gmt_created, datetime) else None,
                "speculate": getattr(vdb_node, "speculate", None),
                # source_raw_memory_id is not exposed: see the note at the same place in async_search.
                "user_id": vdb_node.user_id,
                "agent_id": vdb_node.agent_id,
                "confidence": vdb_node.confidence,
                "custom": vdb_node.custom or {},
                "meta": list((vdb_node.custom or {}).get("meta") or []),
                "source": "vdb",
                "request_id": request_id,
                "elapsed_ms": elapsed,
            }

        if graph_node is not None:
            return {
                "memory_id": graph_node.node_id,
                "content": graph_node.content,
                "layer": graph_node.layer.value if graph_node.layer else "",
                "status": graph_node.status.value if graph_node.status else "active",
                "is_latest": getattr(graph_node, "is_latest", True),
                "supersedes": graph_node.supersedes or [],
                "superseded_by": graph_node.superseded_by or [],
                "tags": graph_node.tags or [],
                "memory_at": int(graph_node.memory_at.timestamp()) if isinstance(graph_node.memory_at, datetime) else None,
                "gmt_created": int(graph_node.gmt_created.timestamp()) if isinstance(graph_node.gmt_created, datetime) else None,
                "speculate": getattr(graph_node, "speculate", None),
                # source_raw_memory_id is not exposed: see the note at the same place in async_search.
                "user_id": graph_node.user_id,
                "agent_id": graph_node.agent_id,
                "confidence": graph_node.confidence,
                "custom": getattr(graph_node, "custom", {}) or {},
                "source": "graph",
                "request_id": request_id,
                "elapsed_ms": elapsed,
            }

        logger.info(f"[get] not found: memory_id={memory_id}")
        return None

    @_ensure_internal_loop
    async def async_update(self, memory_id: str, content: str) -> Dict[str, Any]:
        """
        Update memory content (async).

        Flow: get_by_id → 404 check → embed → update node fields → upsert → record history

        Returns:
            {"success", "memory_id", "request_id", "elapsed_ms"}
        """
        request_id = str(uuid.uuid4())
        set_request_id(request_id)
        t0 = time.perf_counter()

        logger.info(f"[update] memory_id={memory_id} new_content({len(content)} chars)")

        # Fetch the old node
        node = await self._vector_store.get_by_id(memory_id)
        if node is None:
            logger.warning(f"[update] not found: memory_id={memory_id}")
            return {
                "success": False,
                "memory_id": memory_id,
                "request_id": request_id,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
                "error_message": f"memory_id '{memory_id}' not found",
            }

        old_content = node.content
        old_status = node.status.value

        # Generate the new embedding
        new_embedding = await self._embed_service.embed(content)

        # Update the node
        node.content = content
        node.embedding = new_embedding

        await self._vector_store.upsert(node)
        try:
            from .pipelines._retrieval.retrieval_clauses import (
                sync_retrieval_clause_index,
            )
            await sync_retrieval_clause_index(
                vector_store=self._vector_store,
                embed_service=self._embed_service,
                nodes=[node],
                enabled=bool(getattr(
                    self._config.recall, "retrieval_clauses_enabled", False
                )),
                min_words=int(getattr(
                    self._config.recall, "retrieval_clauses_min_words", 70
                )),
                clear_empty=True,
            )
        except Exception as e:
            logger.warning("[retrieval-clause] client.update maintain failed: %s", e)
        logger.info(f"[update] done: memory_id={memory_id}")

        # Record history
        if self._history_store is not None:
            try:
                uid = node.user_id or ""
                agent_id = node.agent_id or "default"
                isolation_key = self._build_isolation_key(uid, agent_id)
                await self._history_store.record_update(
                    memory_id=memory_id,
                    old_content=old_content,
                    new_content=content,
                    old_status=old_status,
                    new_status=node.status.value,
                    change_reason="client.update",
                    isolation_key=isolation_key,
                )
            except Exception as e:
                logger.warning(f"[update] history record failed: {e}")

        return {
            "success": True,
            "memory_id": memory_id,
            "request_id": request_id,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def _backend_identity(self) -> Dict[str, str]:
        """Resolve the identity of the current VDB backend: provider / db_name / collection_name.

        Used when delete/get/list and similar operations persist a pipeline trace. Backends that have no
        notion of a database namespace return an empty db_name.
        """
        vs = self._vector_store
        provider = ""
        try:
            provider = str(getattr(self._config.vector_store, "provider", "") or "")
        except Exception:
            provider = ""
        return {
            "provider": provider,
            "db_name": str(getattr(vs, "_db_name", "") or ""),
            "collection_name": str(getattr(vs, "_collection_name", "") or ""),
        }

    async def _maybe_store_op_log(
        self,
        *,
        step: str,
        request_id: str,
        user_id: str = "",
        agent_id: str = "",
        session_id: str = "",
        parsed: Optional[Dict[str, Any]] = None,
        memory_ids: Optional[List[str]] = None,
        elapsed_ms: float = 0.0,
        event_at_ms: int = 0,
    ) -> None:
        """Persist one pipeline trace for non-LLM operations such as delete/get/list (best-effort).

        Reuses the pipeline_logs_v2 table: prompt/response stay empty, tokens are 0, and all structured
        information goes into parsed (JSON). Controlled by MEMORY_PIPELINE_TRACE_ENABLED, same as search.
        Failures degrade silently and never affect the main operation.

        event_at_ms: The real moment the event occurred (epoch milliseconds). Most callers
            call it directly as `await _maybe_store_op_log(...)`, so the first line of the observability
            wrapper's `time.time()` differs from the call site by only a few microseconds, so 0 can be passed.
            **Only callers dispatched via `create_task` / `ensure_future` must pass it explicitly** --
            dispatching delays the wrapper by several milliseconds, so the timestamp it stamps is already late,
            which inverts the pipeline log timestamp relative to the real order of events.
        """
        if self._cache is None:
            return
        try:
            from .utils.pipeline_observability import is_pipeline_trace_enabled
            if not is_pipeline_trace_enabled():
                return
            await self._cache.store_pipeline_log(
                request_id=request_id,
                user_id=user_id or "",
                agent_id=agent_id or "",
                session_id=session_id or "",
                step=step,
                prompt="",
                response="",
                parsed=json.dumps(parsed or {}, ensure_ascii=False),
                memory_ids=memory_ids or None,
                elapsed_ms=float(elapsed_ms or 0.0),
                event_at_ms=int(event_at_ms or 0),
            )
        except Exception as e:
            logger.debug(f"[{step.lower()}] store op trace failed (non-fatal): {e}")

    @_ensure_internal_loop
    async def async_delete(self, memory_id: str) -> Dict[str, Any]:
        """
        Delete a single memory (async).

        Also removes the corresponding node from the vector database and the graph database (when present).

        Evolution chain handling:
        - When the deleted node is the chain head (is_latest=True + non-empty supersedes),
          its predecessors are restored to is_latest=True + status=ACTIVE
        - When a middle node is deleted, the predecessors in its supersedes are reattached to the successor's supersedes

        Graph evidence handling:
        - Deletes the node in the Graph plus its associated VdbRef shadow node and DERIVED_FROM edges

        Returns:
            {"success", "deleted_count", "request_id", "elapsed_ms"}
        """
        request_id = str(uuid.uuid4())
        set_request_id(request_id)
        t0 = time.perf_counter()

        logger.info(f"[delete] memory_id={memory_id}")

        # Fetch the node info first (for chain repair + history + tag cleanup)
        old_content = ""
        old_tags: List[str] = []
        old_user_id = ""
        old_isolation_key = ""
        node = await self._vector_store.get_by_id(memory_id)
        if node is not None:
            old_content = node.content
            try:
                old_tags = list(node.tags or [])
            except Exception:
                old_tags = []
            old_user_id = node.user_id or ""
            try:
                old_isolation_key = node.get_isolation_key() or ""
            except Exception:
                old_isolation_key = ""

        # ── Evolution chain repair (runs before the VDB delete) ──
        chain_repaired = False
        if node is not None:
            await self._repair_evolution_chain_on_delete(node)
            chain_repaired = bool(node.supersedes or node.superseded_by)

        # Delete from the vector database
        deleted = await self._vector_store.delete(memory_id)
        try:
            await self._vector_store.delete_retrieval_clauses(memory_id)
        except Exception as e:
            logger.debug("[retrieval-clause] delete cleanup failed: %s", e)

        # Delete the graph database node + clean up evidence (when present)
        graph_deleted = False
        if self._graph_store is not None:
            try:
                await self._graph_store.delete_node(memory_id)
                graph_deleted = True
            except Exception as e:
                logger.warning(f"[delete] graph_store delete failed: {e}")
            # Clean up the case where this node is referenced as a VdbRef by other schemas
            try:
                await self._cleanup_graph_evidence(memory_id)
            except Exception as e:
                logger.debug(f"[delete] graph evidence cleanup failed (non-fatal): {e}")

        deleted_count = 1 if deleted else 0

        # Clean up tag_index: once this memory is deleted, check whether each of its tags is still referenced
        # by any other memory, and drop an unreferenced tag from tag_index. Failures degrade silently and do not change delete semantics.
        if deleted and old_tags and old_user_id:
            try:
                from .pipelines._retrieval import tag_index as _tag_index_helper
                removed = await _tag_index_helper.cleanup_tags_on_delete(
                    vector_store=self._vector_store,
                    user_id=old_user_id,
                    tags=old_tags,
                    isolation_key=old_isolation_key,
                )
                if removed:
                    logger.debug(f"[delete] tag_index cleaned {removed} orphan tags")
            except Exception as e:
                logger.debug(f"[delete] tag_index cleanup failed (non-fatal): {e}")

        # Record history
        if self._history_store is not None and deleted:
            try:
                uid = (node.user_id if node else "") or ""
                agent_id = (node.agent_id if node else "") or "default"
                isolation_key = self._build_isolation_key(uid, agent_id)
                await self._history_store.record_delete(
                    memory_id=memory_id,
                    content=old_content,
                    isolation_key=isolation_key,
                )
            except Exception as e:
                logger.warning(f"[delete] history record failed: {e}")

        logger.info(f"[delete] done: deleted_count={deleted_count}")

        # pipeline trace (best-effort, controlled by MEMORY_PIPELINE_TRACE_ENABLED)
        _ident = self._backend_identity()
        _parsed = {
            "memory_id": memory_id,
            **_ident,
            "found": node is not None,
            "vdb_deleted": bool(deleted),
            "graph_deleted": graph_deleted,
            "deleted_count": deleted_count,
            "chain_repaired": chain_repaired,
        }
        if node is not None:
            _parsed["user_id"] = node.user_id or ""
            _parsed["agent_id"] = node.agent_id or ""
        await self._maybe_store_op_log(
            step="DELETE",
            request_id=request_id,
            user_id=(node.user_id if node else "") or "",
            agent_id=(node.agent_id if node else "") or "",
            parsed=_parsed,
            memory_ids=[memory_id],
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

        return {
            "success": deleted,
            "deleted_count": deleted_count,
            "request_id": request_id,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    async def _acquire_graph_store_for_purge(self):
        """
        Acquire the GraphStore used for purge (independent of mode).

        ultra mode reuses the persistent connection; pro/lite open one temporarily when leftover graph data is detected, purge it, then close.
        Returns:
            (graph_store | None, ephemeral: bool) -- the caller must close it when ephemeral=True.
        """
        if self._graph_store is not None:
            return self._graph_store, False

        from .data.graph_store import create_graph_store, graph_data_may_exist

        if not graph_data_may_exist(self._config):
            return None, False

        gs = create_graph_store(self._config)
        await gs.initialize()
        if getattr(gs, "_available", True) is False:
            logger.warning(
                "[purge] ephemeral graph store unavailable (e.g. Kuzu file lock); "
                "skip graph purge"
            )
            await gs.close()
            return None, False

        provider = getattr(self._config.graph_store, "provider", None) or "kuzu"
        logger.info(
            f"[purge] opened ephemeral GraphStore for delete_all "
            f"(mode={self._mode}, provider={provider})"
        )
        return gs, True

    async def _purge_graph_data(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> tuple:
        """Purge graph data scoped by user (the VDB is mode-independent, and so is the graph DB).

        Returns:
            (deleted_count: int, elapsed_ms: float, ephemeral: bool)
            Lets the caller persist a DELETE_ALL_GRAPH pipeline log, telling
            "slow VDB" apart from "slow ephemeral graph connection (initialize() cost when ephemeral=True)".
        """
        _t = time.perf_counter()
        gs, ephemeral = await self._acquire_graph_store_for_purge()
        if gs is None:
            return 0, round((time.perf_counter() - _t) * 1000, 2), False
        try:
            n = await gs.delete_by_metadata(
                user_id=user_id, agent_id=agent_id, session_id=session_id,
            )
            elapsed_ms = round((time.perf_counter() - _t) * 1000, 2)
            logger.info(
                f"[delete_all] graph purged: user={user_id} agent={agent_id} "
                f"session={session_id} memory_nodes={n} ephemeral={ephemeral}"
            )
            return n, elapsed_ms, ephemeral
        except Exception as e:
            logger.warning(f"[delete_all] graph_store purge failed: {e}")
            return 0, round((time.perf_counter() - _t) * 1000, 2), ephemeral
        finally:
            if ephemeral:
                await gs.close()

    @_ensure_internal_loop
    async def async_delete_all(
        self,
        *,
        user_id: str = "",
        agent_ids: Optional[List[str]] = None,
        session_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Delete the user's memories (async).

        Deletes all of the user's memories by default. Narrow the scope with agent_ids/session_ids.

        - agent_ids=[], session_ids=[] -> delete every agent and every session under the user
        - agent_ids=["a1"] -> delete all sessions of a1 under the user
        - agent_ids=["a1"], session_ids=["s1","s2"] -> delete the specified combinations

        Returns:
            {"success", "deleted_count", "request_id", "elapsed_ms"}
        """
        request_id = str(uuid.uuid4())
        set_request_id(request_id)
        t0 = time.perf_counter()

        uid = user_id
        effective_agent_ids = agent_ids if agent_ids is not None else []
        effective_session_ids = session_ids if session_ids is not None else []

        total_deleted = 0
        total_graph_deleted = 0
        # Per-stage timing aggregates (persisted for DELETE_ALL_SUMMARY)
        vdb_ms_total = 0.0
        graph_ms_total = 0.0
        rounds = 0   # How many rounds actually ran (number of agent x session combinations)

        async def _do_round(
            aid: Optional[str] = None,
            sid: Optional[str] = None,
        ) -> None:
            """VDB + Graph deletion for one (agent, session) combination, each writing one pipeline log."""
            nonlocal total_deleted, total_graph_deleted, vdb_ms_total, graph_ms_total, rounds
            rounds += 1

            # ── VDB deletion ──────────────────────────────────────────────
            _t_vdb = time.perf_counter()
            try:
                cnt = await self._vector_store.delete_by_metadata(
                    user_id=uid,
                    agent_id=aid,
                    session_id=sid,
                )
            except Exception as e:
                logger.warning(f"[delete_all] vdb delete failed aid={aid} sid={sid}: {e}")
                cnt = -1
            try:
                await self._vector_store.delete_retrieval_clauses_by_metadata(
                    user_id=uid,
                    agent_id=aid,
                    session_id=sid,
                )
            except Exception as e:
                logger.debug(
                    "[retrieval-clause] delete_all cleanup failed "
                    "aid=%s sid=%s: %s", aid, sid, e,
                )
            _vdb_ms = round((time.perf_counter() - _t_vdb) * 1000, 2)
            if cnt and cnt > 0:
                total_deleted += cnt
            vdb_ms_total += _vdb_ms

            await self._maybe_store_op_log(
                step="DELETE_ALL_VDB",
                request_id=request_id,
                user_id=uid,
                agent_id=aid or "",
                parsed={
                    "agent_id": aid,
                    "session_id": sid,
                    "deleted_count": cnt if cnt and cnt > 0 else 0,
                    "vdb_ms": _vdb_ms,
                },
                elapsed_ms=_vdb_ms,
            )

            # ── Graph deletion ────────────────────────────────────────────
            g_cnt, _graph_ms, ephemeral = await self._purge_graph_data(
                uid, agent_id=aid, session_id=sid,
            )
            total_graph_deleted += g_cnt
            graph_ms_total += _graph_ms

            await self._maybe_store_op_log(
                step="DELETE_ALL_GRAPH",
                request_id=request_id,
                user_id=uid,
                agent_id=aid or "",
                parsed={
                    "agent_id": aid,
                    "session_id": sid,
                    "graph_deleted_count": g_cnt,
                    "graph_ms": _graph_ms,
                    # When ephemeral=True this includes the ephemeral initialize() cost, a common cause of Graph timeouts
                    "ephemeral": ephemeral,
                },
                elapsed_ms=_graph_ms,
            )

        # ── Route to the concrete deletion mode ──────────────────────────────────────
        if not effective_agent_ids and not effective_session_ids:
            logger.info(f"[delete_all] deleting all for user={uid}")
            await _do_round()

        elif effective_agent_ids and not effective_session_ids:
            for aid in effective_agent_ids:
                logger.info(f"[delete_all] deleting user={uid} agent={aid}")
                await _do_round(aid=aid)

        else:
            a_ids = effective_agent_ids or ["default_agent"]
            for aid in a_ids:
                for sid in effective_session_ids:
                    logger.info(f"[delete_all] deleting user={uid} agent={aid} session={sid}")
                    await _do_round(aid=aid, sid=sid)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            f"[delete_all] done: vdb_deleted={total_deleted} "
            f"graph_memory_nodes={total_graph_deleted} mode={self._mode}"
        )

        # ── Summary log ──────────────────────────────────────────────────
        await self._maybe_store_op_log(
            step="DELETE_ALL_SUMMARY",
            request_id=request_id,
            user_id=uid,
            parsed={
                "total_deleted": total_deleted,
                "graph_deleted": total_graph_deleted,
                "rounds": rounds,
                "vdb_ms_total": round(vdb_ms_total, 2),
                "graph_ms_total": round(graph_ms_total, 2),
                # scope: lets the inspector see at a glance which range was deleted
                "scope": (
                    "all" if not effective_agent_ids and not effective_session_ids
                    else f"agents={effective_agent_ids}" if not effective_session_ids
                    else f"agents={effective_agent_ids} sessions={effective_session_ids}"
                ),
            },
            elapsed_ms=elapsed_ms,
        )

        return {
            "success": True,
            "deleted_count": total_deleted,
            "graph_deleted_count": total_graph_deleted,
            "request_id": request_id,
            "elapsed_ms": elapsed_ms,
        }

    # ================================================================
    # List Memories API (async)
    # ================================================================

    @staticmethod
    def _memory_node_to_list_item(node) -> Dict[str, Any]:
        """Serialize a MemoryNode into a list API item."""
        _memory_at = None
        if isinstance(node.memory_at, datetime):
            _memory_at = int(node.memory_at.timestamp())
        _gmt_created = None
        if isinstance(node.gmt_created, datetime):
            _gmt_created = int(node.gmt_created.timestamp())
        if _gmt_created is None and isinstance(getattr(node, "valid_from", None), datetime):
            _gmt_created = int(node.valid_from.timestamp())

        return {
            "memory_id": node.node_id,
            "content": node.content,
            "layer": node.layer.value if node.layer else "",
            "status": node.status.value if node.status else "active",
            "memory_at": _memory_at,
            "gmt_created": _gmt_created,
            "user_id": node.user_id,
            "agent_id": node.agent_id,
            "session_id": getattr(node, "session_id", None),
            "speculate": getattr(node, "speculate", None),
            # source_raw_memory_id is not exposed: see the note at the same spot in async_search.
            "tags": node.tags or [],
            "custom": node.custom or {},
            # Caller-supplied meta (pass-through payload). Kept at the top level, as in search / get,
            # so callers do not need two extraction paths for different APIs; custom holds internal fields
            # and must not be exposed to business code directly.
            "meta": list((node.custom or {}).get("meta") or []),
        }

    @staticmethod
    def _normalize_layers(layers: Optional[List[Any]]) -> Optional[List]:
        """Normalize a mixed str / MemoryLayer layers argument into a deduplicated MemoryLayer list.

        Strings are accepted to stay aligned with search (ReadRequest.layers is List[str]),
        and to reuse the legacy numbering compatibility of from_string (e.g. l5_schema -> l6_schema).
        Invalid values raise ValueError right away, which the caller (App layer) turns into 400; they are never silently ignored --
        ignoring them would make callers believe the filter took effect while they actually get the full set.
        """
        from .models.memory import MemoryLayer

        if not layers:
            return None
        out: List[MemoryLayer] = []
        for item in layers:
            layer = item if isinstance(item, MemoryLayer) else MemoryLayer.from_string(str(item))
            if layer not in out:
                out.append(layer)
        return out

    @staticmethod
    def _sort_memory_nodes(nodes: List, *, order: str) -> List:
        reverse = (order.lower() != "asc")

        def _sort_key(n):
            val = n.memory_at if n.memory_at is not None else n.gmt_created
            if val is None:
                val = getattr(n, "valid_from", None)
            if val is None:
                return datetime.min
            if isinstance(val, datetime):
                return val
            return datetime.min

        nodes.sort(key=_sort_key, reverse=reverse)
        return nodes

    async def _list_vdb_bucket(
        self,
        *,
        user_id: str,
        agent_id: Optional[str],
        limit: int,
        offset: int,
        order: str,
        user_ids: Optional[List[str]] = None,
        agent_ids: Optional[List[str]] = None,
        layers: Optional[List] = None,
        with_evolution: bool = False,
    ) -> Dict[str, Any]:
        from .models.memory import MemoryLayer, MemoryStatus
        from .pipelines._retrieval.user_level_l0 import collapse_user_level_l0_hits

        # Filters stay exactly as before this change (status=ACTIVE only); only the multi-value enums are added.
        # only_latest is not layered on top: with healthy data the writer writes is_latest/status in pairs, so the two are equivalent;
        # adding it would only catch a few extra dirty rows while changing what list returns in production.
        # The list page only needs the payload (content/tags/custom/chain-head info) plus the sort fields;
        # no vector math is involved, so with_vectors=False is used throughout to avoid pulling back large embeddings.
        #
        # L1_RAW is excluded on the vector store side (allow_raw_layer defaults to False and the condition travels with the
        # request, so raw never leaves the store). It used to fetch raw into this process and filter it out in Python,
        # which both routed raw conversations through this process and re-fetched the largest layer for nothing; removed.
        all_nodes = await self._vector_store.list_by_user(
            user_id=user_id,
            agent_id=agent_id,
            user_ids=user_ids,
            agent_ids=agent_ids,
            status_filter=[MemoryStatus.ACTIVE],
            layers=layers,
            with_vectors=False,
        )

        # L0 is user-level and must not be filtered by agent. For the default list, or when L0 is explicitly included, also
        # query L0 by user and fold the historical agent-level chain heads into one entry per user.
        # L0 no longer has agent isolation, so it must be skipped without a user scope, to avoid returning other users' profiles.
        include_l0 = layers is None or MemoryLayer.L0_BASIC_INFO in layers
        if include_l0 and (user_id or user_ids):
            all_nodes = [n for n in all_nodes if n.layer != MemoryLayer.L0_BASIC_INFO]
            l0_nodes = await self._vector_store.list_by_user(
                user_id=user_id,
                user_ids=user_ids,
                status_filter=[MemoryStatus.ACTIVE],
                layers=[MemoryLayer.L0_BASIC_INFO],
                with_vectors=False,
            )
            l0_hits = collapse_user_level_l0_hits([
                {"node_id": n.node_id, "node": n, "score": 1.0}
                for n in l0_nodes
            ])
            all_nodes.extend(hit["node"] for hit in l0_hits if hit.get("node") is not None)

        all_nodes = self._sort_memory_nodes(all_nodes, order=order)
        total = len(all_nodes)
        page_nodes = all_nodes[offset: offset + limit]
        items = [self._memory_node_to_list_item(n) for n in page_nodes]
        if with_evolution:
            await self._attach_evolution_chains(page_nodes, items)
        return {
            "memories": items,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    async def _attach_evolution_chains(self, nodes: List, items: List[Dict[str, Any]]) -> None:
        """Attach evolution_chain to list items in place, using the same structure as search.

        Only the current page is expanded: walking back a chain needs one lookup per chain, and expanding everything would scale with the data set,
        while the list page renders one page at a time.

        Fields are only added; no item is added or removed. expand_evolution_chains deduplicates chains on its own (search
        may recall several nodes from the same chain), but list already filters on status=ACTIVE, so historical versions are
        always SUPERSEDED and several entries per chain should not appear; even if they do, list must not return fewer items,
        so results are mapped back to the original items by chain-head node_id instead of taking its output directly.
        """
        from .pipelines._retrieval.evolution import expand_evolution_chains

        linked = [
            n for n in nodes
            if getattr(n, "supersedes", None) or getattr(n, "superseded_by", None)
        ]
        if not linked:
            return

        try:
            expanded = await expand_evolution_chains(
                self._vector_store,
                [{"node_id": n.node_id, "score": 0.0, "node": n} for n in linked],
            )
        except Exception as e:
            # The evolution chain is supplementary; failing to fetch it must not break the whole list
            logger.warning(f"[list] evolution chain expand failed: {e}")
            return

        chains = {
            exp.get("node_id"): exp["evolution_chain"]
            for exp in expanded
            if isinstance(exp, dict) and exp.get("evolution_chain")
        }
        for item in items:
            chain = chains.get(item.get("memory_id"))
            if chain:
                item["evolution_chain"] = chain

    async def _list_graph_bucket(
        self,
        *,
        user_id: str,
        agent_id: Optional[str],
        limit: int,
        offset: int,
        order: str,
        user_ids: Optional[List[str]] = None,
        agent_ids: Optional[List[str]] = None,
        layers: Optional[List] = None,
    ) -> Optional[Dict[str, Any]]:
        """List the graph DB's L6/L7 nodes; returns None when there is no graph data, it is unavailable, or layers excludes both graph layers."""
        from .models.memory import MemoryLayer, MemoryNode

        graph_layers = [MemoryLayer.L6_SCHEMA, MemoryLayer.L7_INTENTION]
        if layers:
            graph_layers = [layer for layer in graph_layers if layer in layers]
            # The requested layers are all VDB-side, so no graph connection is needed (connecting has a cost)
            if not graph_layers:
                return None

        gs, ephemeral = await self._acquire_graph_store_for_purge()
        if gs is None:
            return None

        # The graph DB is organized by isolation_key and has no in-query capability, so multi-value requests are fetched per combination and merged
        effective_uids = list(user_ids) if user_ids else ([user_id] if user_id else [])
        effective_aids = list(agent_ids) if agent_ids else [agent_id or "default_agent"]
        iks = [
            MemoryNode.build_isolation_key(u, a)
            for u in effective_uids
            for a in effective_aids
        ]

        try:
            async def _fetch(ik: str, layer) -> List:
                try:
                    return await gs.get_all_nodes(ik, layer=layer, limit=10_000)
                except Exception as e:
                    logger.warning(
                        f"[list] graph list ik={ik} layer={layer.value} failed: {e}"
                    )
                    return []

            chunks = await asyncio.gather(*[
                _fetch(ik, layer)
                for ik in iks
                for layer in graph_layers
            ])
            graph_nodes = [n for chunk in chunks for n in chunk]

            graph_nodes = self._sort_memory_nodes(graph_nodes, order=order)
            total = len(graph_nodes)
            page_nodes = graph_nodes[offset: offset + limit]
            out = {
                "nodes": [self._memory_node_to_list_item(n) for n in page_nodes],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
            # Keep the original field shape for a single combination so existing callers reading isolation_key keep working
            if len(iks) == 1:
                out["isolation_key"] = iks[0]
            else:
                out["isolation_keys"] = iks
            return out
        finally:
            if ephemeral:
                await gs.close()

    async def _list_coding(
        self,
        *,
        user_id: str,
        workspace_id: str,
        branch: Optional[str],
        limit: int,
        offset: int,
        request_id: str,
        t0: float,
    ) -> Dict[str, Any]:
        """List full experience-layer details by pure filtering on workspace_id (plus optional branch).

        This is enumeration rather than retrieval, so only the experience layer is returned, without the fact layer. There is no
        query and no semantic ranking; results are ordered by updated_at descending. Fail-safe: failures return an empty list.
        """
        empty = {
            "request_id": request_id,
            "scene": "productivity",
            "experiences": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        try:
            await self._get_coding_writer()  # Side effect: lazy-init the store
        except Exception as e:
            logger.warning(f"[list-coding] init failed: {e}")
            return empty

        try:
            mems = await self._coding_store.list_by_workspace(
                user_id=user_id, workspace_id=workspace_id, branch=branch,
                limit=limit, offset=offset,
            )
        except Exception as e:
            logger.warning(f"[list-coding] list_by_workspace failed: {e}")
            return empty

        out_memories = [
            {
                "experience_id": d.experience_id,
                "agent_id": getattr(d, "agent_id", "") or "",
                "task": d.task,
                "search_keys": list(d.search_keys),
                "solution": d.solution,
                "boundary_envs": d.boundary_envs,
                "workspace_id": d.workspace_id,
                "branch": d.branch,
                "files": list(d.files),
                "source_fact_ids": list(d.source_fact_ids),
                "confidence": d.confidence,
                "updated_at": d.updated_at.isoformat() if d.updated_at else None,
                "source": d.source,
            }
            for d in mems
        ]
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            f"[list-coding] user={user_id} workspace_id={workspace_id!r} "
            f"branch={branch!r} returned={len(out_memories)} elapsed={elapsed}ms"
        )
        return {
            "request_id": request_id,
            "scene": "productivity",
            "experiences": out_memories,
            "total": len(out_memories),
            "limit": limit,
            "offset": offset,
            "elapsed_ms": elapsed,
        }

    @_ensure_internal_loop
    async def async_list_memories(
        self,
        *,
        user_id: str = "",
        agent_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        order: str = "desc",
        workspace_id: Optional[str] = None,
        branch: Optional[str] = None,
        user_ids: Optional[List[str]] = None,
        agent_ids: Optional[List[str]] = None,
        layers: Optional[List[Any]] = None,
        with_evolution: bool = False,
    ) -> Dict[str, Any]:
        """
        List the user's memories (async).

        The VDB and graph buckets are returned separately (the graph key is present only when Kuzu/Neo4j data exists).

        user_ids / agent_ids (multi-value, empty by default):
            - When provided they take precedence over the single-value user_id / agent_id, returning in one call the union of
              several users' (or agents') memories; pagination is applied after the merged result is sorted.
            - Each memory carries its own user_id / agent_id so callers can trace ownership.
            - The coding path (workspace_id non-empty) does not support multi-value; only the single user_id is used.

        layers (empty by default):
            - Return only the requested layers, accepting strings or MemoryLayer (aligned with the search API);
              invalid values raise ValueError.
            - When omitted the behaviour matches the previous version: every layer except L1_RAW is returned.
              When specified the argument is authoritative, including a request for L1_RAW only.
            - When none of the requested layers live in the graph DB (L6/L7), the graph query is skipped and the graph key is omitted.

        with_evolution (False by default):
            - Attach `evolution_chain` (latest -> oldest) to VDB items, with the same structure as search,
              and it is present only when that memory actually has an evolution chain.
            - Off by default: expansion needs one lookup per chain and would slow down the default list.
            - Only the current page is expanded; total and the item count are unaffected.
            - Note that only contradiction-supersede evolution is visible; supplement-refine evolution is an in-place update at write time,
              so historical versions are never persisted and no chain can be returned.

        workspace_id / branch (empty by default):
            - workspace_id non-empty -> go straight to the coding memory list path, listing that user's full experience layer by
              pure workspace_id (plus optional branch) filtering (updated_at descending, no semantic ranking),
              returning {"scene": "productivity", "experiences": [...]}.
            - Both empty -> the regular chat path (VDB + graph buckets).

        Returns:
            chat path:
            {
                "vdb": {"memories": [...], "total", "limit", "offset"},
                "graph": {"nodes": [...], "total", ...} | omitted,
                "elapsed_ms": ...,
            }
            coding path:
            {
                "scene": "productivity",
                "experiences": [...],
                "total": ...,
                "limit": ..., "offset": ...,
                "elapsed_ms": ...,
            }
        """
        request_id = str(uuid.uuid4())
        set_request_id(request_id)
        t0 = time.perf_counter()

        # workspace_id non-empty -> use the coding memory list path
        if workspace_id:
            return await self._list_coding(
                user_id=user_id, workspace_id=workspace_id, branch=branch,
                limit=limit, offset=offset, request_id=request_id, t0=t0,
            )

        effective_layers = self._normalize_layers(layers)

        vdb = await self._list_vdb_bucket(
            user_id=user_id, agent_id=agent_id,
            user_ids=user_ids, agent_ids=agent_ids,
            limit=limit, offset=offset, order=order,
            layers=effective_layers, with_evolution=with_evolution,
        )
        graph = await self._list_graph_bucket(
            user_id=user_id, agent_id=agent_id,
            user_ids=user_ids, agent_ids=agent_ids,
            limit=limit, offset=offset, order=order,
            layers=effective_layers,
        )

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            f"[list] user={user_ids or user_id} agent={agent_ids or agent_id} "
            f"vdb_total={vdb['total']} graph_total={graph['total'] if graph else 'n/a'} "
            f"elapsed={elapsed}ms"
        )

        # pipeline trace (best-effort, controlled by MEMORY_PIPELINE_TRACE_ENABLED)
        _page_items = vdb.get("memories", []) or []
        _page_ids = [
            (m.get("memory_id") or m.get("id") or "")
            for m in _page_items
            if isinstance(m, dict)
        ]
        _page_ids = [i for i in _page_ids if i]
        _ident = self._backend_identity()
        _parsed = {
            **_ident,
            "user_id": user_id,
            "agent_id": agent_id or "",
            "limit": limit,
            "offset": offset,
            "order": order,
            "vdb_total": vdb.get("total", 0),
            "returned_count": len(_page_items),
        }
        # The op log's user_id/agent_id fields are single-valued; the multi-value scope is also recorded in parsed so the audit trail stays complete
        if user_ids:
            _parsed["user_ids"] = list(user_ids)
        if agent_ids:
            _parsed["agent_ids"] = list(agent_ids)
        if effective_layers:
            _parsed["layers"] = [layer.value for layer in effective_layers]
        if with_evolution:
            _parsed["with_evolution"] = True
        if graph is not None:
            _parsed["graph_total"] = graph.get("total", 0)
        await self._maybe_store_op_log(
            step="LIST",
            request_id=request_id,
            user_id=user_id or (user_ids[0] if user_ids else ""),
            agent_id=agent_id or (agent_ids[0] if agent_ids else ""),
            parsed=_parsed,
            memory_ids=_page_ids or None,
            elapsed_ms=elapsed,
        )

        out: Dict[str, Any] = {"vdb": vdb, "elapsed_ms": elapsed, "request_id": request_id}
        if graph is not None:
            out["graph"] = graph
        return out

    # ================================================================
    # User Profile / Basic Info (L0) API
    # ================================================================

    @_ensure_internal_loop
    async def async_get_user_profile(
        self,
        *,
        user_id: str,
    ) -> Dict[str, Any]:
        """Return one complete user profile, including enabled Ultra artifacts."""
        from .agent.tools.basic_profile import (
            _assemble_full_kv_from_nodes,
            _latest_l0_heads_from_nodes,
            _render_content,
        )
        from .models.memory import MemoryLayer
        from .pipelines.balanced_catalog import (
            attach_balanced_catalog_to_l0,
            get_active_balanced_catalog,
        )
        from .pipelines.specific_supplement import (
            attach_specific_supplement_to_l0,
            get_active_specific_supplement,
        )

        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            raise ValueError("user_id is required and cannot be empty")
        request_id = str(uuid.uuid4())
        set_request_id(request_id)
        t0 = time.perf_counter()

        # Read all L0 versions once: the latest head supplies identity/timestamps,
        # while the complete history is required to reconstruct the full current KV
        # because each L0 version may contain only that write's changed fields.
        l0_nodes = await self._vector_store.list_by_user(
            user_id=normalized_user_id,
            layers=[MemoryLayer.L0_BASIC_INFO],
            limit=1000,
            with_vectors=False,
        )
        full_kv = _assemble_full_kv_from_nodes(list(l0_nodes))
        heads = _latest_l0_heads_from_nodes(list(l0_nodes))
        head = heads[0] if heads else None
        configured_order = list(self._config.basic_profile.effective_fields().keys())
        field_order = configured_order + sorted(
            key for key in full_kv if key not in configured_order
        )
        base_content = _render_content(
            full_kv,
            field_order=field_order,
        )

        l0_results: List[Dict[str, Any]] = []
        if head is not None:
            l0_results.append({
                "node_id": head.node_id,
                "node": head,
                "content": base_content or head.content,
                "layer": MemoryLayer.L0_BASIC_INFO.value,
                "score": 1.0,
                "source": "user_profile",
            })

        catalog = await get_active_balanced_catalog(
            self._graph_store,
            user_id=normalized_user_id,
        )
        l0_results, catalog_info = attach_balanced_catalog_to_l0(
            l0_results, catalog,
        )
        supplement = await get_active_specific_supplement(
            self._graph_store,
            user_id=normalized_user_id,
        )
        l0_results, supplement_info = attach_specific_supplement_to_l0(
            l0_results, supplement,
        )

        profile: Optional[Dict[str, Any]] = None
        if len(l0_results) == 1:
            item = l0_results[0]
            node = item.get("node")
            content = str(
                item.get("content") or getattr(node, "content", "") or ""
            ).strip()
            for suffix_key in (
                "_balanced_catalog_suffix",
                "_specific_supplement_suffix",
            ):
                suffix = str(item.get(suffix_key) or "").strip()
                if suffix:
                    content = content.rstrip() + ("\n\n" if content else "") + suffix
            memory_at = getattr(node, "memory_at", None) if node else None
            gmt_created = getattr(node, "gmt_created", None) if node else None
            profile = {
                "memory_id": str(item.get("node_id") or ""),
                "content": content,
                "layer": MemoryLayer.L0_BASIC_INFO.value,
                "user_id": normalized_user_id,
                "basic_info": full_kv,
                "tags": list(getattr(node, "tags", None) or []),
                "memory_at": (
                    int(memory_at.timestamp()) if isinstance(memory_at, datetime) else None
                ),
                "gmt_created": (
                    int(gmt_created.timestamp()) if isinstance(gmt_created, datetime) else None
                ),
                "catalog": catalog_info,
                "supplement": supplement_info,
            }

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        await self._maybe_store_op_log(
            step="GET_USER_PROFILE",
            request_id=request_id,
            user_id=normalized_user_id,
            agent_id="",
            parsed={
                **self._backend_identity(),
                "found": profile is not None,
                "basic_info_fields": len(full_kv),
                "catalog_attached": bool(catalog_info.get("attached")),
                "supplement_attached": bool(supplement_info.get("attached")),
            },
            memory_ids=[profile["memory_id"]] if profile else None,
            elapsed_ms=elapsed,
        )
        return {
            "request_id": request_id,
            "profile": profile,
            "elapsed_ms": elapsed,
        }

    @_ensure_internal_loop
    async def async_get_user_basic_info(
        self,
        *,
        user_id: str = "",
    ) -> Dict[str, Any]:
        """Return the user's L0 basic info memories (async).

        Consistent with search / list: when a memory has an evolution chain (supersede/evolution), the chain head
        (is_latest) is returned as the representative, together with `evolution_chain` (latest -> oldest).

        Returns:
            {
                "request_id": str,
                "memories": [
                    {
                        "memory_id", "content", "layer", "status",
                        "memory_at", "gmt_created", "tags",
                        "basic_info_kv": {...},          # Structured L0 KV (from custom)
                        "evolution_chain": [...],        # Only when a chain exists
                    },
                    ...
                ],
                "elapsed_ms": float,
            }
        """
        from .models.memory import MemoryLayer, MemoryStatus
        from .pipelines._retrieval.evolution import expand_evolution_chains
        from .pipelines._retrieval.user_level_l0 import collapse_user_level_l0_hits

        request_id = str(uuid.uuid4())
        set_request_id(request_id)
        t0 = time.perf_counter()

        # The L0 head is ACTIVE; earlier versions on the chain are fetched on demand during evolution expansion.
        # Assembling basic_info only reads custom.basic_info_kv / content / tags and the chain metadata,
        # and evolution expansion needs no embeddings, so with_vectors=False from the start avoids fetching vectors.
        # L0 Basic Info is the user-level global profile; agent_id is kept only for API backward compatibility.
        nodes = await self._vector_store.list_by_user(
            user_id=user_id,
            status_filter=[MemoryStatus.ACTIVE],
            layers=[MemoryLayer.L0_BASIC_INFO],
            with_vectors=False,
        )

        hits = collapse_user_level_l0_hits([
            {"node_id": n.node_id, "node": n, "score": 1.0}
            for n in nodes
        ])
        expanded = await expand_evolution_chains(self._vector_store, hits)

        memories: List[Dict[str, Any]] = []
        for exp in expanded:
            node = exp.get("node")
            if node is None:
                continue
            _memory_at = (
                int(node.memory_at.timestamp())
                if isinstance(node.memory_at, datetime) else None
            )
            _gmt_created = (
                int(node.gmt_created.timestamp())
                if isinstance(node.gmt_created, datetime) else None
            )
            item = {
                "memory_id": node.node_id,
                "content": node.content,
                "layer": node.layer.value if node.layer else "",
                "status": node.status.value if node.status else "active",
                "memory_at": _memory_at,
                "gmt_created": _gmt_created,
                "tags": node.tags or [],
                "basic_info_kv": (node.custom or {}).get("basic_info_kv", {}),
            }
            if exp.get("evolution_chain"):
                item["evolution_chain"] = exp["evolution_chain"]
            memories.append(item)

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            f"[basic_info] user={user_id} "
            f"count={len(memories)} elapsed={elapsed}ms"
        )

        _parsed = {
            **self._backend_identity(),
            "user_id": user_id,
            "returned_count": len(memories),
        }
        await self._maybe_store_op_log(
            step="GET_BASIC_INFO",
            request_id=request_id,
            user_id=user_id,
            agent_id="",
            parsed=_parsed,
            memory_ids=[m["memory_id"] for m in memories] or None,
            elapsed_ms=elapsed,
        )

        return {
            "request_id": request_id,
            "memories": memories,
            "elapsed_ms": elapsed,
        }

    # ================================================================
    # Entity Store build / migration API
    # ================================================================

    def build_memory_graph(
        self,
        user_id: str,
        *,
        agent_id: Optional[str] = None,
        limit: int = 10000,
        clear_first: bool = False,
    ) -> Dict[str, Any]:
        """Backfill the memory relation graph for all L2_FACT of the given user (sync).

        Why: graph data accumulates incrementally, so memories written before `MEMORY_ENABLE_GRAPH` was enabled are not in the graph.
        This entry point projects the historical L2_FACT into graph nodes and builds `TAGGED_WITH` + `semantic_neighbor` edges.

        Built: `TAGGED_WITH` (Topic) + `semantic_neighbor` + the `TEMPORAL_NEXT` temporal spine
        (the latter requires `MEMORY_GRAPH_TEMPORAL_SPINE=true`).

        ⚠️ Backfill **cannot build** `co_occurs`, nor the LLM-annotated `causes`/`elaborates`/`contrasts`:
        The former needs the "same write batch" information (lost for historical data), the latter needs a reconcile re-run.
        In other words, backfill cannot produce the two highest-value edge types; cross-topic causality still accumulates through new writes.

        The projection itself is idempotent (nodes/labels/edges all go through MERGE), so re-running does not stack data. It **does not
        delete old nodes**, though -- if the projection rules ever changed (e.g. the Topic isolation key narrowed from `user::agent::session`
        to `user::agent`), old session-granularity Topics remain in the store next to the new ones. In that case pass
        `clear_first=True` to clear this user's l2_fact projection and orphan Topics before rebuilding
        (**without touching** System 2's l6_schema).

        Returns:
            {"success", "memories_scanned", "nodes", "tags", "semantic", "errors",
             "elapsed_ms", "cleared"}
        """
        return self._loop_thread.run(
            self.async_build_memory_graph(
                user_id=user_id, agent_id=agent_id, limit=limit,
                clear_first=clear_first,
            )
        )

    @_ensure_internal_loop
    async def async_build_memory_graph(
        self,
        user_id: str,
        *,
        agent_id: Optional[str] = None,
        limit: int = 10000,
        clear_first: bool = False,
    ) -> Dict[str, Any]:
        """Async version of `build_memory_graph`."""
        import time as _t
        from .models.memory import MemoryLayer, MemoryStatus
        from .pipelines._graph.projector import project_memories_to_graph

        t0 = _t.perf_counter()
        if self._graph_store is None:
            return {
                "success": False,
                "error": "graph store not enabled (set MEMORY_ENABLE_GRAPH=true for pro mode)",
            }

        cleared: Dict[str, int] = {}
        if clear_first:
            # Clear only the l2_fact projection + orphan Topics; System 2's l6_schema is left alone
            cleared = await self._graph_store.clear_projection(
                user_id, agent_id, layer=MemoryLayer.L2_FACT.value,
            )
            logger.info(f"[graph] cleared before rebuild: {cleared}")

        vector_store = self._registry._get_shared_vector_store()
        nodes = await vector_store.list_by_user(
            user_id=user_id,
            agent_id=agent_id,
            layers=[MemoryLayer.L2_FACT],
            status_filter=[MemoryStatus.ACTIVE],
            limit=limit,
            with_vectors=True,
        )
        nodes = list(nodes or [])

        # semantic_neighbor backfill: pairwise similarity from the vectors already in the store, no embedding API calls
        neighbors_by_node: Dict[str, List[Tuple[str, float]]] = {}
        gcfg = getattr(self._config, "graph_store", None)
        thr = getattr(gcfg, "edge_min_score", 0.55) if gcfg else 0.55
        with_emb = [n for n in nodes if getattr(n, "embedding", None)]
        for i, a in enumerate(with_emb):
            na = sum(x * x for x in a.embedding) ** 0.5
            if na == 0:
                continue
            for b in with_emb[i + 1:]:
                nb_ = sum(x * x for x in b.embedding) ** 0.5
                if nb_ == 0:
                    continue
                sim = sum(x * y for x, y in zip(a.embedding, b.embedding)) / (na * nb_)
                if sim >= thr:
                    neighbors_by_node.setdefault(a.node_id, []).append((b.node_id, sim))

        stats = await project_memories_to_graph(
            graph_store=self._graph_store,
            embed_service=self._registry._get_shared_embed_service(),
            nodes=nodes,
            vector_store=vector_store,
            neighbors_by_node=neighbors_by_node,
            edge_min_score=thr,
            # Backfill has no notion of a "batch", so co_occurs is disabled (otherwise every memory pair would be linked)
            co_occurs_max_batch=0,
            # The temporal spine uses the bulk variant: the whole batch is all of history, so predecessors are ordered inside the batch.
            # Using the incremental variant by mistake would create no edges at all, because it excludes nodes inside the batch.
            temporal_spine_bulk=bool(
                getattr(gcfg, "temporal_spine", False) if gcfg else False
            ),
            hierarchy_topic_write=bool(
                getattr(gcfg, "hierarchy_topic_write", False) if gcfg else False
            ),
        )
        return {
            "success": True,
            "memories_scanned": len(nodes),
            **stats,
            "cleared": cleared,
            "elapsed_ms": round((_t.perf_counter() - t0) * 1000, 2),
        }

    def build_entity_store(
        self,
        user_id: str,
        *,
        agent_id: str = "default_agent",
        rebuild: bool = False,
    ) -> Dict[str, Any]:
        """Extract entities from all of the user's L2_FACT (legacy data included) into the entity store (sync).

        Used to backfill the entity store for historical collections, for the write path and internal migration tools.
        Independent of mode (also callable in pro).

        Args:
            user_id:  User ID
            agent_id: Agent ID (defaults to default_agent)
            rebuild:  When True, clear this user's existing entities before rebuilding (default False, incremental merge)

        Returns:
            {"success", "memories_scanned", "entities_indexed", "elapsed_ms"}
        """
        return self._loop_thread.run(
            self.async_build_entity_store(
                user_id=user_id, agent_id=agent_id, rebuild=rebuild,
            )
        )

    @_ensure_internal_loop
    async def async_build_entity_store(
        self,
        user_id: str,
        *,
        agent_id: str = "default_agent",
        rebuild: bool = False,
    ) -> Dict[str, Any]:
        """Async implementation of build_entity_store."""
        from .models.memory import MemoryLayer, MemoryStatus
        from .pipelines._retrieval.entity_store import index_memory_entities

        t0 = time.perf_counter()
        vs = self._vector_store

        # Scan all ACTIVE L2_FACT for this user
        # Entity rebuild only needs node_id/content/user_id/agent_id; the downstream index_memory_entities re-embeds the
        # entity names itself and does not depend on the original node embeddings, so use with_vectors=False here.
        try:
            nodes = await vs.list_by_user(
                user_id=user_id,
                agent_id=agent_id,
                limit=100000,
                status_filter=[MemoryStatus.ACTIVE],
                layers=[MemoryLayer.L2_FACT],
                with_vectors=False,
            )
        except Exception as e:
            logger.error(f"[build_entity_store] list_by_user failed: {e}")
            return {"success": False, "error": str(e), "memories_scanned": 0, "entities_indexed": 0}

        # rebuild: first drop these memories' associations in the entity store (best-effort)
        if rebuild:
            for n in nodes:
                try:
                    await vs.delete_entities_for_memory(
                        memory_id=n.node_id, user_id=user_id, agent_ids=[agent_id] if agent_id else None,
                    )
                except NotImplementedError:
                    return {
                        "success": False,
                        "error": "entity store not supported by this vector store backend",
                        "memories_scanned": 0, "entities_indexed": 0,
                    }
                except Exception as e:
                    logger.debug(f"[build_entity_store] rebuild cleanup failed for {n.node_id}: {e}")

        indexed = 0
        for n in nodes:
            cnt = await index_memory_entities(
                vector_store=vs,
                embed_service=self._embed_service,
                memory_id=n.node_id,
                content=n.content or "",
                user_id=user_id,
                agent_id=agent_id or "default_agent",
            )
            indexed += cnt

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            f"[build_entity_store] user={user_id} agent={agent_id} "
            f"scanned={len(nodes)} entities_indexed={indexed} rebuild={rebuild} elapsed={elapsed}ms"
        )
        return {
            "success": True,
            "memories_scanned": len(nodes),
            "entities_indexed": indexed,
            "elapsed_ms": elapsed,
        }

    # ================================================================
    # Clone User API (async)
    # ================================================================

    async def _load_clone_source_snapshot(
        self,
        src_user_id: str,
        agent_id: Optional[str],
    ) -> Dict[str, Any]:
        """Read immutable source state once for a concurrent clone fan-out."""
        from .models.memory import MemoryLayer

        reads = [self._vector_store.list_by_user(
            src_user_id, agent_id=agent_id, allow_raw_layer=True,
        )]
        include_user_l0 = bool(agent_id)
        if include_user_l0:
            reads.append(self._vector_store.list_by_user(
                src_user_id, layers=[MemoryLayer.L0_BASIC_INFO],
                allow_raw_layer=True,
            ))

        graph_export = getattr(self._graph_store, "export_clone_snapshot", None)
        if callable(graph_export):
            reads.append(graph_export(src_user_id))
        clause_export = getattr(
            self._vector_store, "export_retrieval_clause_snapshot", None,
        )
        if callable(clause_export):
            reads.append(clause_export(
                src_user_id=src_user_id, agent_id=agent_id,
            ))

        values = await asyncio.gather(*reads)
        src_nodes = list(values[0] or [])
        cursor = 1
        if include_user_l0:
            seen_node_ids = {node.node_id for node in src_nodes}
            src_nodes.extend(
                node for node in (values[cursor] or [])
                if not str(getattr(node, "agent_id", "") or "").strip()
                and node.node_id not in seen_node_ids
            )
            cursor += 1
        graph_snapshot = values[cursor] if callable(graph_export) else None
        if callable(graph_export):
            cursor += 1
        clause_snapshot = values[cursor] if callable(clause_export) else None
        return {
            "nodes": tuple(src_nodes),
            "graph": graph_snapshot,
            "retrieval_clauses": clause_snapshot,
        }

    async def _acquire_clone_source_snapshot(
        self,
        src_user_id: str,
        agent_id: Optional[str],
    ) -> Tuple[Dict[str, Any], Tuple[Any, Any]]:
        """Single-flight a source snapshot across concurrent destinations.

        Entries live only while at least one caller is waiting/using the
        snapshot.  They are removed immediately after the final waiter exits,
        so a later clone observes later source writes rather than stale state.
        """
        lock = getattr(self, "_clone_snapshot_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._clone_snapshot_lock = lock
        entries = getattr(self, "_clone_snapshot_entries", None)
        if entries is None:
            entries = {}
            self._clone_snapshot_entries = entries

        key = (src_user_id, agent_id or "")
        async with lock:
            entry = entries.get(key)
            if entry is None:
                entry = {
                    "task": asyncio.create_task(
                        self._load_clone_source_snapshot(src_user_id, agent_id)
                    ),
                    "waiters": 0,
                }
                entries[key] = entry
            entry["waiters"] += 1
            task = entry["task"]
        try:
            # One cancelled HTTP clone must not cancel the shared source read
            # for its sibling rollout destinations.
            return await asyncio.shield(task), (key, entry)
        except BaseException:
            await self._release_clone_source_snapshot((key, entry))
            raise

    async def _release_clone_source_snapshot(
        self, lease: Tuple[Any, Any],
    ) -> None:
        key, entry = lease
        lock = getattr(self, "_clone_snapshot_lock", None)
        entries = getattr(self, "_clone_snapshot_entries", None)
        if lock is None or entries is None:
            return
        async with lock:
            current = entries.get(key)
            if current is entry:
                current["waiters"] -= 1
                if current["waiters"] <= 0:
                    entries.pop(key, None)

    @_ensure_internal_loop
    async def async_clone_user(
        self,
        src_user_id: str,
        dst_user_id: str,
        *,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Clone user memories (async).

        Deep copies all of src's memories to dst; embeddings are copied as-is and never recomputed.
        dst must be empty, otherwise an error is returned. The operation completes synchronously and rolls back written data on failure.

        Returns:
            {"success": True, "request_id": "...", "cloned_count": N, "elapsed_ms": ...}
        """
        request_id = str(uuid.uuid4())
        set_request_id(request_id)
        t0 = time.perf_counter()

        if not src_user_id or not dst_user_id:
            return {
                "success": False,
                "request_id": request_id,
                "cloned_count": 0,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
                "error_message": "src_user_id and dst_user_id are required",
            }

        if src_user_id == dst_user_id:
            return {
                "success": False,
                "request_id": request_id,
                "cloned_count": 0,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
                "error_message": "src_user_id and dst_user_id must be different",
            }

        logger.info(
            f"[clone] src={src_user_id} → dst={dst_user_id} agent_id={agent_id}"
        )

        snapshot_lease = None
        try:
            from .models.memory import MemoryLayer, MemoryNode

            # 1. Check whether dst is empty
            # Emptiness is judged by length only, no embeddings needed; avoids pulling a large payload for an existence probe.
            # allow_raw_layer=True: the emptiness check means "is there any data at all", raw included;
            # otherwise a destination holding only raw would be misread as empty and the clone would overwrite it and mix data.
            dst_nodes = await self._vector_store.list_by_user(
                dst_user_id, agent_id=agent_id, limit=1, with_vectors=False,
                allow_raw_layer=True,
            )
            # L0_BASIC_INFO is user-scoped and is intentionally stored with an
            # empty agent_id. An agent-filtered clone must still treat an
            # existing user profile at the destination as data, otherwise it
            # can silently overwrite/mix user-level state.
            if agent_id and not dst_nodes:
                dst_user_nodes = await self._vector_store.list_by_user(
                    dst_user_id, layers=[MemoryLayer.L0_BASIC_INFO],
                    with_vectors=False, allow_raw_layer=True,
                )
                dst_nodes = [
                    node for node in dst_user_nodes
                    if not str(getattr(node, "agent_id", "") or "").strip()
                ][:1]
            if dst_nodes:
                return {
                    "success": False,
                    "request_id": request_id,
                    "cloned_count": 0,
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
                    "error_message": (
                        f"dst_user_id '{dst_user_id}' already has memories. "
                        "Clone requires an empty destination."
                    ),
                }

            # 2. List all of src's memories (embeddings included). When one winner fans out to several
            # rollout destinations at the same time, they share one in-flight source snapshot instead of scrolling repeatedly.
            # allow_raw_layer=True: clone means "copy this user's memories in full", raw conversation layer included --
            # the destination user's extraction pipeline also relies on raw to piece conversation history together,
            # and dropping raw would leave the cloned user without any historical context.
            source_snapshot, snapshot_lease = await self._acquire_clone_source_snapshot(
                src_user_id, agent_id,
            )
            src_nodes = list(source_snapshot["nodes"])
            if not src_nodes:
                return {
                    "success": True,
                    "request_id": request_id,
                    "cloned_count": 0,
                    "by_layer": {},
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
                    "error_message": None,
                }

            logger.info(f"[clone] found {len(src_nodes)} nodes to clone")

            # 3. Deep copy: first pass -- generate new node_ids and build the complete mapping table
            id_map: Dict[str, str] = {}  # old_node_id → new_node_id

            for src_node in src_nodes:
                id_map[src_node.node_id] = str(uuid.uuid4())

            # Second pass -- deep copy with the complete id_map plus pointer remapping
            from .utils.clone_references import remap_memory_references
            new_nodes: List = []
            for src_node in src_nodes:
                new_id = id_map[src_node.node_id]

                # Deep copy via to_dict + from_dict
                source_dict = src_node.to_dict()
                node_dict = remap_memory_references(source_dict, id_map)
                node_dict["node_id"] = new_id
                node_dict["user_id"] = dst_user_id
                # Reset retrieval metadata
                node_dict["access_count"] = 0
                node_dict["last_accessed_at"] = None

                # Remap evolution chain pointers (supersedes / superseded_by)
                # IDs outside the map (not part of this clone) are cleared
                if source_dict.get("supersedes"):
                    node_dict["supersedes"] = [
                        id_map[sid] for sid in source_dict["supersedes"]
                        if sid in id_map
                    ]
                if source_dict.get("superseded_by"):
                    node_dict["superseded_by"] = [
                        id_map[sid] for sid in source_dict["superseded_by"]
                        if sid in id_map
                    ]
                # Remap the IDs inside evidence_chain
                if source_dict.get("evidence_chain"):
                    node_dict["evidence_chain"] = [
                        id_map[eid] for eid in source_dict["evidence_chain"]
                        if eid in id_map
                    ]
                new_node = MemoryNode.from_dict(node_dict)
                # Keep the original embedding (content is unchanged, so no recomputation)
                new_node.embedding = src_node.embedding
                # Keep the original timestamps (do not let __post_init__ overwrite them)
                new_node.memory_at = src_node.memory_at
                new_node.gmt_created = src_node.gmt_created
                new_node.valid_from = src_node.valid_from
                new_nodes.append(new_node)

            # 5. Batch write to the vector database
            written_ids = await self._vector_store.upsert_batch(new_nodes)
            logger.info(f"[clone] upsert_batch: {len(written_ids)} nodes written")
            written_id_set = set(written_ids)
            if written_id_set != {node.node_id for node in new_nodes}:
                raise RuntimeError("clone incomplete: not all destination nodes were written")
            clone_clauses = getattr(self._vector_store, "clone_retrieval_clauses", None)
            import_clauses = getattr(
                self._vector_store, "import_retrieval_clause_snapshot", None,
            )
            clause_snapshot = source_snapshot.get("retrieval_clauses")
            if clause_snapshot is not None and callable(import_clauses):
                await import_clauses(
                    clause_snapshot,
                    dst_user_id=dst_user_id,
                    id_map=id_map,
                )
            elif callable(clone_clauses):
                await clone_clauses(
                    src_user_id=src_user_id, dst_user_id=dst_user_id,
                    id_map=id_map, agent_id=agent_id,
                )
            by_layer: Dict[str, int] = {}
            for node in new_nodes:
                if node.node_id not in written_id_set:
                    continue
                layer = node.layer.value if hasattr(node.layer, "value") else str(node.layer)
                by_layer[layer] = by_layer.get(layer, 0) + 1

            # 6. Graph database clone (when enabled)
            # Correct flow: read nodes + edges from the source Graph, remap them with id_map, then write to the destination Graph
            if self._graph_store is not None:
                graph_snapshot = source_snapshot.get("graph")
                graph_import = getattr(
                    self._graph_store, "import_clone_snapshot", None,
                )
                if graph_snapshot is not None and callable(graph_import):
                    try:
                        await graph_import(
                            graph_snapshot,
                            dst_user_id=dst_user_id,
                            id_map=id_map,
                            embed_service=self._embed_service,
                        )
                    except Exception as graph_error:
                        logger.warning(
                            "[clone] native Graph snapshot import failed; "
                            "falling back to legacy clone: %s", graph_error,
                        )
                        await self._clone_graph(
                            src_user_id=src_user_id,
                            dst_user_id=dst_user_id,
                            agent_id=agent_id,
                            id_map=id_map,
                        )
                else:
                    await self._clone_graph(
                        src_user_id=src_user_id,
                        dst_user_id=dst_user_id,
                        agent_id=agent_id,
                        id_map=id_map,
                    )

            # 7. Copy memory_operations records (src -> dst, remapped with id_map)
            if self._cache:
                try:
                    await self._clone_memory_operations(
                        src_user_id=src_user_id,
                        dst_user_id=dst_user_id,
                        clone_request_id=request_id,
                        id_map=id_map,
                    )
                except Exception as ops_err:
                    logger.warning(f"[clone] clone memory_operations failed (non-fatal): {ops_err}")

            logger.info(
                f"[clone] done: {len(written_ids)} nodes cloned "
                f"from {src_user_id} to {dst_user_id}"
            )

            return {
                "success": True,
                "request_id": request_id,
                "cloned_count": len(written_ids),
                "by_layer": by_layer,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
                "error_message": None,
            }

        except Exception as e:
            logger.error(f"[clone] failed: {e}", exc_info=True)
            # Rollback: delete whatever was already written to dst
            try:
                await self._vector_store.delete_by_metadata(user_id=dst_user_id)
                delete_clauses = getattr(self._vector_store, "delete_retrieval_clauses_by_metadata", None)
                if callable(delete_clauses):
                    await delete_clauses(user_id=dst_user_id)
                await self._purge_graph_data(dst_user_id)
                logger.info(f"[clone] rollback: cleaned dst_user_id={dst_user_id}")
            except Exception as rollback_err:
                logger.warning(f"[clone] rollback failed: {rollback_err}")

            return {
                "success": False,
                "request_id": request_id,
                "cloned_count": 0,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
                "error_message": str(e),
            }
        finally:
            if snapshot_lease is not None:
                await self._release_clone_source_snapshot(snapshot_lease)

    # ================================================================
    # Clone Graph Helper
    # ================================================================

    async def _clone_graph(
        self,
        src_user_id: str,
        dst_user_id: str,
        agent_id: Optional[str],
        id_map: Dict[str, str],
    ) -> None:
        """
        Deep copies from the source user's Graph to the destination user.

        Flow:
        1. Read all Memory nodes from the source Graph (by isolation_key)
        2. Remap node IDs + user_id + isolation_key with id_map and write to the destination Graph
        3. Copy VdbRef shadow nodes (with ID remapping) + DERIVED_FROM edges
        4. Copy Memory -> Memory relation edges (RELATED_TO)
        5. Copy TAGGED_WITH edges (Topic tags)
        """
        from .models.memory import MemoryNode
        gs = self._graph_store
        if gs is None:
            return

        # Build the source/destination isolation_key
        # The source user may have several sessions, so get_all_nodes matches by user_id prefix;
        # but get_all_nodes needs an exact isolation_key, so the source nodes in id_map are scanned for isolation_key
        try:
            # Step 1: read all Memory nodes from the source Graph
            # Query with Cypher by user_id (independent of isolation_key, more general)
            src_graph_nodes = []
            if hasattr(gs, '_execute') and hasattr(gs, '_available'):
                # Kuzu
                if not gs._available:
                    return
                result = gs._execute(
                    "MATCH (m:Memory) WHERE m.user_id = $uid RETURN m;",
                    {"uid": src_user_id},
                )
                while result is not None and result.has_next():
                    row = result.get_next()
                    src_graph_nodes.append(gs._row_to_memory_node(row[0]))
            elif hasattr(gs, '_run'):
                # Neo4j -- also fetch embedding / beh_embedding so the clone can copy them as well
                rows = await gs._run(
                    "MATCH (m:Memory) WHERE m.user_id = $uid "
                    "RETURN m, m.embedding AS _emb, m.beh_embedding AS _beh_emb",
                    {"uid": src_user_id},
                )
                for r in rows:
                    node = gs._record_to_memory_node(r["m"])
                    # Stash the embedding on the node for the later upsert
                    if r.get("_emb") is not None:
                        node._graph_embedding = list(r["_emb"])
                    if r.get("_beh_emb") is not None:
                        node._graph_beh_embedding = list(r["_beh_emb"])
                    src_graph_nodes.append(node)

            if not src_graph_nodes:
                logger.info("[clone] no Graph nodes to clone")
                return

            logger.info(f"[clone] Graph: found {len(src_graph_nodes)} source nodes")

            # Remember each source node's old_id so id_map covers it
            graph_old_ids = []  # One entry per element of src_graph_nodes
            for src_node in src_graph_nodes:
                old_nid = src_node.node_id
                graph_old_ids.append(old_nid)
                if old_nid not in id_map:
                    id_map[old_nid] = str(uuid.uuid4())

            # Step 2: write the destination Graph Memory nodes (deep copy + ID remapping)
            from .models.memory import MemoryNode as _MN
            for i, src_node in enumerate(src_graph_nodes):
                old_nid = graph_old_ids[i]
                new_id = id_map[old_nid]
                # Deep copy; the source node is left untouched
                from .utils.clone_references import remap_memory_references
                d = remap_memory_references(src_node.to_dict(), id_map)
                d["node_id"] = new_id
                d["user_id"] = dst_user_id
                dst_node = _MN.from_dict(d)
                # Carry over the source node's embedding (to_dict omits it, so pass it manually)
                src_emb = getattr(src_node, '_graph_embedding', None)
                src_beh = getattr(src_node, '_graph_beh_embedding', None)
                if src_emb:
                    dst_node._graph_embedding = src_emb
                try:
                    await gs.upsert_memory_node(dst_node)
                    # Patch beh_embedding after the upsert (upsert only writes embedding, not beh)
                    if src_beh:
                        await gs.update_embedding(new_id, beh_embedding=src_beh)
                except Exception as e:
                    logger.warning(f"[clone] Graph node upsert failed {new_id}: {e}")

            # Step 3: copy VdbRef + DERIVED_FROM edges
            for i, old_nid in enumerate(graph_old_ids):
                new_id = id_map[old_nid]
                try:
                    evidence = await gs.get_evidence_vdbrefs(old_nid)
                except Exception:
                    evidence = []

                for ev in evidence:
                    old_vdbref_id = ev["node_id"]
                    new_vdbref_id = id_map.get(old_vdbref_id, old_vdbref_id)
                    try:
                        await gs.ensure_vdbref(new_vdbref_id, ev.get("layer", ""))
                        await gs.add_derived_from(new_id, new_vdbref_id)
                    except Exception as e:
                        logger.debug(f"[clone] DERIVED_FROM edge failed: {e}")

            # Step 4: copy Memory -> Memory relation edges.
            #
            # The relation_type / weight of RELATED_TO are core semantics for read-side graph reranking; copying only
            # the Neo4j/Kuzu relation table type is not enough. The old implementation passed empty properties, which turned
            # semantic_neighbor/co_occurs/typed edges all into the default related,
            # so the clone bank and the seed bank were no longer equivalent at retrieval time.
            seen_relation_edges = set()

            for i, old_nid in enumerate(graph_old_ids):
                try:
                    if hasattr(gs, '_execute'):
                        result = gs._execute(
                            """
                            MATCH (a:Memory {node_id: $nid})-[r:RELATED_TO]->(b:Memory)
                            RETURN b.node_id, r.relation_type, r.weight;
                            """,
                            {"nid": old_nid},
                        )
                        edges = []
                        while result is not None and result.has_next():
                            row = result.get_next()
                            edges.append({
                                "target": row[0],
                                "edge_type": "RELATED_TO",
                                "properties": {
                                    "relation_type": row[1] or "related",
                                    "weight": float(row[2]) if row[2] is not None else 1.0,
                                },
                            })
                    elif hasattr(gs, '_run'):
                        rows = await gs._run(
                            """
                            MATCH (a:Memory {node_id: $nid})-[r:RELATED_TO]->(b:Memory)
                            RETURN b.node_id AS target, type(r) AS edge_type,
                                   properties(r) AS properties
                            """,
                            {"nid": old_nid},
                        )
                        edges = [{
                            "target": r["target"],
                            "edge_type": r["edge_type"],
                            "properties": {
                                "relation_type": (r.get("properties") or {}).get(
                                    "relation_type", "related"
                                ),
                                "weight": (r.get("properties") or {}).get("weight", 1.0),
                            },
                        } for r in rows]
                    else:
                        edges = []
                except Exception:
                    edges = []

                for edge in edges:
                    et = edge["edge_type"]
                    if et != "RELATED_TO":
                        continue
                    new_src = id_map.get(old_nid)
                    new_tgt = id_map.get(edge["target"])
                    if new_src and new_tgt:
                        properties = dict(edge.get("properties") or {})
                        relation_type = properties.get("relation_type") or "related"
                        edge_key = (
                            frozenset((old_nid, edge["target"])),
                            relation_type,
                        )
                        if edge_key in seen_relation_edges:
                            continue
                        seen_relation_edges.add(edge_key)
                        try:
                            await gs.add_edge(new_src, new_tgt, et, properties)
                        except Exception as e:
                            logger.debug(f"[clone] edge {et} failed: {e}")

            # Step 4b: copy the directed temporal spine. Only the source end is read here, so the same directed edge is
            # not replayed once from each endpoint; topic_id takes part in the temporal semantics and must be kept.
            for old_nid in graph_old_ids:
                try:
                    if hasattr(gs, '_execute'):
                        result = gs._execute(
                            """
                            MATCH (a:Memory {node_id: $nid})-[r:TEMPORAL_NEXT]->(b:Memory)
                            RETURN b.node_id, r.topic_id;
                            """,
                            {"nid": old_nid},
                        )
                        temporal_edges = []
                        while result is not None and result.has_next():
                            row = result.get_next()
                            temporal_edges.append({"target": row[0], "topic_id": row[1] or ""})
                    elif hasattr(gs, '_run'):
                        temporal_edges = await gs._run(
                            """
                            MATCH (a:Memory {node_id: $nid})-[r:TEMPORAL_NEXT]->(b:Memory)
                            RETURN b.node_id AS target, r.topic_id AS topic_id
                            """,
                            {"nid": old_nid},
                        )
                    else:
                        temporal_edges = []
                except Exception:
                    temporal_edges = []
                for edge in temporal_edges:
                    new_src = id_map.get(old_nid)
                    new_tgt = id_map.get(edge["target"])
                    if new_src and new_tgt:
                        try:
                            await gs.add_temporal_edge(
                                new_src, new_tgt, edge.get("topic_id") or ""
                            )
                        except Exception as e:
                            logger.debug(f"[clone] TEMPORAL_NEXT edge failed: {e}")

            # Step 4c: copy System 2's cross-abstraction directed edges.
            for old_nid in graph_old_ids:
                try:
                    if hasattr(gs, '_execute'):
                        result = gs._execute(
                            """
                            MATCH (a:Memory {node_id: $nid})-[:CROSS_ABSTRACTS_TO]->(b:Memory)
                            RETURN b.node_id;
                            """,
                            {"nid": old_nid},
                        )
                        cross_targets = []
                        while result is not None and result.has_next():
                            cross_targets.append(result.get_next()[0])
                    elif hasattr(gs, '_run'):
                        rows = await gs._run(
                            """
                            MATCH (a:Memory {node_id: $nid})-[:CROSS_ABSTRACTS_TO]->(b:Memory)
                            RETURN b.node_id AS target
                            """,
                            {"nid": old_nid},
                        )
                        cross_targets = [row["target"] for row in rows]
                    else:
                        cross_targets = []
                except Exception:
                    cross_targets = []
                for old_tgt in cross_targets:
                    new_src = id_map.get(old_nid)
                    new_tgt = id_map.get(old_tgt)
                    if new_src and new_tgt:
                        try:
                            await gs.add_cross_abstracts_to(new_src, new_tgt)
                        except Exception as e:
                            logger.debug(f"[clone] CROSS_ABSTRACTS_TO edge failed: {e}")

            # Step 5: rebuild Topic and hierarchy membership in the destination
            # scope. Hierarchy node ids include user scope + subject + full path,
            # so copying source edges/ids would leak source-user identity.
            for i, old_nid in enumerate(graph_old_ids):
                new_id = id_map[old_nid]
                src_node = src_graph_nodes[i]
                dst_ik = MemoryNode.build_isolation_key(
                    dst_user_id,
                    src_node.agent_id or "default_agent",
                    src_node.session_id or "default_session",
                )

                try:
                    if hasattr(gs, '_execute'):
                        result = gs._execute(
                            "MATCH (m:Memory {node_id: $nid})-[:TAGGED_WITH]->(t:Topic) RETURN t.name;",
                            {"nid": old_nid},
                        )
                        tag_rows = []
                        while result is not None and result.has_next():
                            tag_rows.append({"name": result.get_next()[0], "embedding": None})
                    elif hasattr(gs, '_run'):
                        tag_rows = await gs._run(
                            "MATCH (m:Memory {node_id: $nid})-[:TAGGED_WITH]->(t:Topic) "
                            "RETURN t.name AS name, t.embedding AS embedding",
                            {"nid": old_nid},
                        )
                    else:
                        tag_rows = []
                except Exception:
                    tag_rows = []

                for tr in tag_rows:
                    tag_name = tr["name"]
                    tag_emb = tr.get("embedding")
                    try:
                        await gs.add_topic_tag(new_id, tag_name, dst_ik)
                        # Copy the source Topic's embedding onto the destination Topic (when add_topic_tag matches by name
                        # exactly it leaves the embedding unset, so it has to be patched in)
                        if tag_emb is not None and hasattr(gs, '_run_write'):
                            await gs._run_write(
                                "MATCH (t:Topic {isolation_key: $ik, name: $name}) "
                                "WHERE t.embedding IS NULL "
                                "SET t.embedding = $emb",
                                {"ik": dst_ik, "name": tag_name, "emb": list(tag_emb)},
                            )
                    except Exception as e:
                        logger.debug(f"[clone] TAGGED_WITH failed: {e}")

                custom = getattr(src_node, "custom", None) or {}
                from .structure import topic_assignments_from_custom
                assignments = topic_assignments_from_custom(custom)
                if assignments and hasattr(gs, "add_topic_path"):
                    dst_topic_scope = MemoryNode.build_topic_scope_key(
                        dst_user_id, src_node.agent_id or "default_agent",
                    )
                    for assignment in assignments:
                        try:
                            try:
                                await gs.add_topic_path(
                                    new_id, assignment["path"], dst_topic_scope,
                                    assignment["subject"],
                                    node_types=assignment.get("node_types"),
                                    role=assignment.get("role") or "participant",
                                    embed_service=getattr(self, "_embed_service", None),
                                )
                            except TypeError:
                                await gs.add_topic_path(
                                    new_id, assignment["path"], dst_topic_scope,
                                    assignment["subject"],
                                    embed_service=getattr(self, "_embed_service", None),
                                )
                        except Exception as e:
                            logger.debug(f"[clone] hierarchy topic failed: {e}")

                # Body-text NER is an independent scoring channel in normal recall. MENTIONS must be rebuilt after a clone,
                # otherwise the same bank loses entity recall merely because the user_id was renamed.
                try:
                    if hasattr(gs, '_execute'):
                        result = gs._execute(
                            """
                            MATCH (m:Memory {node_id: $nid})-[:MENTIONS]->(e:Entity)
                            RETURN e.name, e.label;
                            """,
                            {"nid": old_nid},
                        )
                        entity_rows = []
                        while result is not None and result.has_next():
                            row = result.get_next()
                            entity_rows.append({"name": row[0], "label": row[1]})
                    elif hasattr(gs, '_run'):
                        entity_rows = await gs._run(
                            """
                            MATCH (m:Memory {node_id: $nid})-[:MENTIONS]->(e:Entity)
                            RETURN e.name AS name, e.label AS label
                            """,
                            {"nid": old_nid},
                        )
                    else:
                        entity_rows = []
                except Exception:
                    entity_rows = []
                for entity in entity_rows:
                    try:
                        await gs.add_entity_mention(
                            new_id, entity.get("name") or "",
                            entity.get("label") or "", dst_ik,
                        )
                    except Exception as e:
                        logger.debug(f"[clone] MENTIONS failed: {e}")

            logger.info(f"[clone] Graph: cloned {len(src_graph_nodes)} nodes with edges")

        except Exception as e:
            logger.warning(f"[clone] Graph clone failed (non-fatal): {e}")

    async def _clone_memory_operations(
        self,
        src_user_id: str,
        dst_user_id: str,
        clone_request_id: str,
        id_map: Dict[str, str],
    ) -> None:
        """
        Copy the source user's memory_operations to the destination user.

        - user_id is replaced with dst_user_id
        - memory_id / old_memory_id are remapped via id_map (values without a mapping keep the original)
        - IDs in the supersedes list are remapped as well
        - request_id keeps the source value (preserving per_request grouping semantics)
        - One extra CLONE op record is appended (marking the clone origin)
        """
        import json as _json

        # 1. Load all ops of src
        src_ops = await self._cache.get_memory_operations(user_id=src_user_id, limit=50000)
        if not src_ops:
            logger.info("[clone] no memory_operations to clone")
            return

        # 2. Write them one by one into dst (with ID remapping)
        cloned_count = 0
        for op in src_ops:
            # Remap memory_id
            src_mem_id = op.get("memory_id", "")
            dst_mem_id = id_map.get(src_mem_id, src_mem_id)

            # Remap old_memory_id
            src_old_id = op.get("old_memory_id") or ""
            dst_old_id = id_map.get(src_old_id, src_old_id) if src_old_id else None

            # Remap the supersedes list
            raw_supersedes = op.get("supersedes", [])
            if isinstance(raw_supersedes, str):
                try:
                    raw_supersedes = _json.loads(raw_supersedes)
                except Exception:
                    raw_supersedes = []
            dst_supersedes = [id_map.get(sid, sid) for sid in (raw_supersedes or [])]

            await self._cache.store_memory_operation(
                request_id=op.get("request_id", clone_request_id),
                user_id=dst_user_id,
                agent_id=op.get("agent_id", ""),
                op=op.get("op", "ADD"),
                memory_id=dst_mem_id,
                content=op.get("content", ""),
                layer=op.get("layer", ""),
                old_memory_id=dst_old_id,
                reason=op.get("reason", ""),
                supersedes=dst_supersedes,
            )
            cloned_count += 1

        # 3. Append one CLONE meta record (marks the clone origin for traceability)
        await self._cache.store_memory_operation(
            request_id=clone_request_id,
            user_id=dst_user_id,
            agent_id="",
            op="CLONE",
            memory_id="",
            content=_json.dumps({
                "src_user_id": src_user_id,
                "dst_user_id": dst_user_id,
                "ops_cloned": cloned_count,
            }, ensure_ascii=False),
            layer="",
            reason=f"cloned from {src_user_id}",
        )

        logger.info(f"[clone] memory_operations: cloned {cloned_count} ops from {src_user_id} → {dst_user_id}")

    # ================================================================
    # History API (async)
    # ================================================================

    @_ensure_internal_loop
    async def async_history(self, memory_id: str) -> List[Dict[str, Any]]:
        """Query the change history of a memory (async)"""
        if self._history_store is None:
            return []
        return await self._history_store.get_history(memory_id)

    @_ensure_internal_loop
    async def async_get_recent_history(
        self,
        user_id: str = "",
        agent_ids: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Query the recent operation history (async).

        When agent_ids is empty, the history of every agent of that user is queried.
        """
        if self._history_store is None:
            return []

        uid = user_id
        effective_agent_ids = agent_ids if agent_ids is not None else []

        if not effective_agent_ids:
            # Query every agent's history: match on the "user_id:" prefix
            isolation_key = self._build_isolation_key(uid, "default")
            return await self._history_store.get_recent(isolation_key, limit=limit)
        else:
            # Query the history of the specified agents
            all_results = []
            for aid in effective_agent_ids:
                isolation_key = self._build_isolation_key(uid, aid)
                results = await self._history_store.get_recent(isolation_key, limit=limit)
                all_results.extend(results)
            # Sort by time descending and take the top limit
            all_results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return all_results[:limit]

    # ================================================================
    # Context manager
    # ================================================================

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self._close_async()
