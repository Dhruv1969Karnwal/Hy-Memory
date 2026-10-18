# -*- coding: utf-8 -*-
"""
Pipeline observability: Log (file JSONL, always on) and Trace (SQLite pipeline_logs, can be disabled).

- Log: for operators to tail/grep, one line per stage as soon as it happens, independent of whether later steps succeed.
- Trace: for the Inspector / debugging, persisted into pipeline_logs (needs a durable backend such as MEMORY_CACHE_BACKEND=sqlite).

Environment variables:
  MEMORY_PIPELINE_TRACE_ENABLED  — persist Trace rows, default true; OpenClaw client deployments may set false
  MEMORY_MEMORY_OPERATIONS_ENABLED — persist the memory_operations audit table, default false
  MEMORY_LOG_DIR                 — Log directory (preferred)
  MEMORY_TRACE_LOG_DIR           — Log directory (legacy name, still supported)

Subdirectories ({log_dir}/{subdir}/{date}.log) are derived by PipelineLogWriter from the prefix of each record's
user_id (everything left of the first "__"); without a prefix it falls back to "default". That prefix is a
generic namespace and can be used for projects, deployments or data bucketing.

Write semantics (search/write hot paths):
  store_pipeline_log **returns immediately** to the caller once wrapped; JSONL / DB flushing happens in background tasks.
  event_at_ms / step-to-step elapsed_ms are stamped synchronously before enqueueing and are unaffected by background IO.
  When the caller (e.g. the App-side write_trace()) already captured event_at_ms synchronously at the very moment the event
  happened and passes it explicitly, this module reuses it instead of overwriting it with a local time.time() --
  so even if the call reached here through cross-thread/cross-queue forwarding and is already delayed, the persisted time still aligns with the real
  event moment; only a missing value or 0 falls back to the local current time.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import threading
import time
import weakref
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Set

from .async_executor import run_in_executor_guarded

logger = logging.getLogger(__name__)

# The synchronous JSONL open/write runs in a dedicated thread pool so it cannot stall the asyncio event loop.
_jsonl_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="pipeline-jsonl",
)

# Tests can await drain_pipeline_log_tasks() to wait for background flushing to finish.
_pending_log_tasks: Set["asyncio.Task[Any]"] = set()

_TERMINAL_STEPS = {"READ_SUMMARY", "READ_CODING_RESULT", "WRITE_TIMELINE"}
_LAST_EVENT_MAX = 4096
# step -> semantic key of that step's dominant cost. The injected value is the elapsed_ms the business code passed into store_pipeline_log,
# i.e. the overall wall-clock duration of that stage.
#
# Naming rule: use a concrete semantic key only when the step's duration **necessarily equals** that of a specific operation
# (embed_ms / llm_ms / vdb_search_ms); otherwise use the neutral step_ms.
#
# READ_SCENE_JUDGE uses step_ms rather than llm_ms: it only calls the LLM in mode="llm";
# mode="default" (single query / feature off) and "explicit" (scene specified explicitly) are both
# purely synchronous branches that never call the LLM at all. Mapping everything to llm_ms used to invite reading "stage latency" as
# "LLM latency" (production once showed mode=default with llm_ms=445). Real LLM latency is
# now injected by business code as parsed.llm_ms in the three LLM branches (see client._async_search_impl).
#
# S1_EMBED uses embed_ms: the writer brackets this single call precisely with `datetime.now()`,
# `embed_queued_with_meta()`, covering enqueue -> lock wait -> batch wait -> HTTP,
# which is exactly the whole embed stage duration as the user experiences it. **Alerts should watch this field** -- looking only at
# provider_meta.embed_request_ms (pure HTTP) hides the entire queuing/lock-contention problem
# (production incident: embed_request_ms was only 1.7s while queuing pushed the stage duration to 48s).
# S1_EMBED used to be missing from this table, so the duration computed by business code was discarded by _inject_op_ms,
# parsed.embed_ms stayed None forever, and alerts could only fall back to the top-level elapsed_ms (step interval).
_STEP_TIMING_KEY = {
    "READ_EMBED_QUERY": "embed_ms", "READ_CODING_EMBED": "embed_ms",
    "READ_RECALL_VEC": "vdb_search_ms", "READ_HIDDEN_L0_CONTEXT": "vdb_search_ms",
    "READ_RECALL_TAG": "vdb_search_ms", "READ_BM25": "vdb_search_ms",
    "READ_TAG_MATCH": "vdb_search_ms", "READ_CODING_VDB": "vdb_search_ms",
    "READ_KEYWORD_EMBED": "embed_ms",
    "READ_SCENE_JUDGE": "step_ms",
    "S1_EMBED": "embed_ms",
    "EXTRACT": "llm_ms", "RECONCILE": "llm_ms",
    "SEARCH_QUERY": "llm_ms", "SUMMARY": "llm_ms",
}


def is_pipeline_trace_enabled() -> bool:
    """Whether pipeline steps are written to the cache (SQLite pipeline_logs) for the Inspector."""
    return os.getenv("MEMORY_PIPELINE_TRACE_ENABLED", "true").lower() in (
        "true",
        "1",
        "yes",
    )


def is_memory_operations_enabled() -> bool:
    """Whether memory base changes are written to the memory_operations audit table. Off by default (false)."""
    return os.getenv("MEMORY_MEMORY_OPERATIONS_ENABLED", "false").lower() in (
        "true",
        "1",
        "yes",
    )


def is_llm_content_logging_enabled() -> bool:
    """Whether pipeline logs/traces may persist full LLM input and output."""
    return os.getenv("MEMORY_LOG_LLM_CONTENT", "false").strip().lower() in (
        "true", "1", "yes", "on",
    )


def redact_content(value: str) -> str:
    """Keep diagnostics without persisting the original sensitive text."""
    import hashlib

    text = str(value or "")
    if not text:
        return ""
    if text.startswith("<redacted len=") and text.endswith(">"):
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"<redacted len={len(text)} sha256={digest}>"


def resolve_pipeline_log_dir() -> str:
    """
    Root directory of the JSONL pipeline logs (always used, has a default).

    Precedence: MEMORY_LOG_DIR > MEMORY_TRACE_LOG_DIR > {MEMORY_DATA_DIR}/logs/pipeline
    """
    explicit = os.getenv("MEMORY_LOG_DIR") or os.getenv("MEMORY_TRACE_LOG_DIR")
    if explicit:
        return explicit
    data_dir = os.getenv("MEMORY_DATA_DIR", os.path.join(Path.home(), ".hy_memory"))
    return os.path.join(data_dir, "logs", "pipeline")


def split_namespace_user(user_id: str) -> tuple:
    """Split an SDK user_id on the first "__" into (namespace, user_id).

    - "project__u1"        -> ("project", "u1")
    - "project__u1__extra" -> ("project", "u1__extra")
    - "u1" (no "__")       -> ("", "u1")
    - ""               -> ("", "")
    """
    if not user_id:
        return "", ""
    if "__" in user_id:
        namespace, rest = user_id.split("__", 1)
        return namespace, rest
    return "", user_id




def _inject_op_ms(parsed_str: str, step: str, op_ms: float) -> str:
    """Inject this stage's operation duration into the parsed JSON under its semantic key (embed_ms/llm_ms/vdb_search_ms)."""
    key = _STEP_TIMING_KEY.get(step)
    if not key or not op_ms:
        return parsed_str
    try:
        obj = json.loads(parsed_str) if parsed_str else {}
        if isinstance(obj, dict):
            obj[key] = round(float(op_ms), 2)
            return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        pass
    return parsed_str


def _track_task(task: "asyncio.Task[Any]") -> None:
    _pending_log_tasks.add(task)

    def _done(t: "asyncio.Task[Any]") -> None:
        _pending_log_tasks.discard(t)
        try:
            exc = t.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            logger.debug(f"[pipeline-log] background task failed: {exc}")

    task.add_done_callback(_done)


async def drain_pipeline_log_tasks(timeout: float = 5.0) -> None:
    """Wait for all background pipeline log tasks to finish (tests / graceful shutdown only)."""
    if not _pending_log_tasks:
        return
    current_loop = asyncio.get_running_loop()
    pending = []
    stale = []
    for task in list(_pending_log_tasks):
        if task.done():
            stale.append(task)
            continue
        try:
            task_loop = task.get_loop()
        except Exception:
            stale.append(task)
            continue
        if task_loop is current_loop:
            pending.append(task)
        elif task_loop.is_closed():
            stale.append(task)
    for task in stale:
        _pending_log_tasks.discard(task)
    if not pending:
        return
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
        pending = [task for task in pending if not task.done()]
    if pending:
        logger.warning(
            f"[pipeline-log] drain timeout after {timeout}s, "
            f"{len(_pending_log_tasks)} tasks still pending"
        )


def wrap_store_pipeline_log(
    original_store: Callable[..., Any],
    *,
    log_writer: Any,
    trace_enabled: bool,
    owner_loop: Any = None,
) -> Callable[..., Any]:
    """Wrap cache.store_pipeline_log: timing stays synchronous while JSONL/DB persistence is asynchronous, so the main request is never blocked.

    - step-to-step elapsed_ms / event_at_ms are stamped at the moment of the call (aligned with the business wall clock)
    - event_at_ms: a non-zero value passed by the caller is reused directly (the App already stamped the very moment
      it was invoked, which is closer to the real event moment than here); only 0 or an omitted value uses local time.time()
    - JSONL goes through the thread pool; DB Trace goes through create_task
    - the caller gets True back immediately after awaiting, so a slow disk/DB never lengthens search RTT
    - owner_loop: the loop the cache is bound to (e.g. the aiomysql pool). Once search runs on its own loop,
      DB persistence triggered from the search loop must be dispatched back to owner_loop, otherwise using
      aiomysql across loops would poison the connection pool, so it has to return to the loop the cache owns.
      When it is None the code falls back to "the current loop" (the old solo/single-loop behaviour).
    """
    _last_event: OrderedDict[str, float] = OrderedDict()
    try:
        _trace_max_pending = max(
            1, int(os.getenv("MEMORY_PIPELINE_TRACE_MAX_PENDING", "64")),
        )
    except (TypeError, ValueError):
        _trace_max_pending = 64
    try:
        _trace_max_concurrency = max(
            1, int(os.getenv("MEMORY_PIPELINE_TRACE_MAX_CONCURRENCY", "2")),
        )
    except (TypeError, ValueError):
        _trace_max_concurrency = 2
    try:
        _trace_timeout = max(
            0.05, float(os.getenv("MEMORY_PIPELINE_TRACE_TIMEOUT_SECONDS", "1.0")),
        )
    except (TypeError, ValueError):
        _trace_timeout = 1.0
    _trace_lock = threading.Lock()
    _trace_pending = 0
    _trace_dropped = 0
    _trace_semaphores = weakref.WeakKeyDictionary()

    def _reserve_trace_slot() -> bool:
        nonlocal _trace_pending, _trace_dropped
        with _trace_lock:
            if _trace_pending >= _trace_max_pending:
                _trace_dropped += 1
                if _trace_dropped == 1 or not (_trace_dropped & (_trace_dropped - 1)):
                    logger.warning(
                        "[pipeline-log] dropping DB trace: pending=%d max=%d dropped=%d",
                        _trace_pending, _trace_max_pending, _trace_dropped,
                    )
                return False
            _trace_pending += 1
            return True

    def _trace_stats() -> dict:
        with _trace_lock:
            return {
                "pending": _trace_pending,
                "dropped": _trace_dropped,
                "max_pending": _trace_max_pending,
                "max_concurrency": _trace_max_concurrency,
                "timeout_seconds": _trace_timeout,
            }

    def _release_once(state: dict) -> None:
        nonlocal _trace_pending
        with _trace_lock:
            if state.get("released"):
                return
            state["released"] = True
            _trace_pending = max(0, _trace_pending - 1)

    async def _store_pipeline_observability(
        *,
        request_id: str = "",
        user_id: str = "",
        agent_id: str = "",
        step: str = "",
        prompt: str = "",
        response: str = "",
        parsed: str = "",
        memory_ids=None,
        elapsed_ms: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        session_id: str = "",
        event_at_ms: int = 0,
        **kwargs,
    ):
        # If the caller (e.g. the App layer) already stamped the moment it was invoked and passes
        # event_at_ms explicitly, that value wins -- it is the real wall-clock moment of the event. Only the default 0 falls back to
        # the time.time() of when this hook was scheduled and actually reached, which is only the moment this wrapper function itself
        # was called; the caller may already be late by then because of create_task/cross-thread scheduling.
        # The parameters are named explicitly (instead of letting them land in **kwargs) so that the original_store(
        # event_at_ms=event_at_ms, **extra_kwargs) call below cannot fail with a duplicate keyword argument error.
        _now = time.time()
        event_at_ms = int(event_at_ms) if event_at_ms else int(_now * 1000)

        parsed = _inject_op_ms(parsed, step, elapsed_ms)
        if not is_llm_content_logging_enabled():
            prompt = redact_content(prompt)
            response = redact_content(response)

        _prev = _last_event.get(request_id)
        step_elapsed_ms = round((_now - _prev) * 1000, 2) if _prev is not None else 0.0
        if step in _TERMINAL_STEPS:
            _last_event.pop(request_id, None)
        else:
            _last_event[request_id] = _now
            _last_event.move_to_end(request_id)
            while len(_last_event) > _LAST_EVENT_MAX:
                _last_event.popitem(last=False)

        # Snapshot the mutable arguments so background tasks never read a list that was later changed
        mem_ids = list(memory_ids) if memory_ids is not None else None
        extra_kwargs = dict(kwargs)

        async def _persist_jsonl() -> None:
            try:
                await run_in_executor_guarded(
                    _jsonl_executor,
                    log_writer.write_step,
                    request_id=request_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    step=step,
                    prompt=prompt,
                    response=response,
                    parsed=parsed,
                    memory_ids=mem_ids,
                    elapsed_ms=step_elapsed_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    event_at_ms=event_at_ms,
                )
            except Exception as e:
                logger.debug(f"[pipeline-log] JSONL write failed: {e}")

        release_state = {"released": False}

        async def _persist_trace() -> None:
            if not trace_enabled:
                return
            try:
                execution_loop = asyncio.get_running_loop()
                with _trace_lock:
                    semaphore = _trace_semaphores.get(execution_loop)
                    if semaphore is None:
                        semaphore = asyncio.Semaphore(_trace_max_concurrency)
                        _trace_semaphores[execution_loop] = semaphore

                async def _limited_store() -> None:
                    async with semaphore:
                        await original_store(
                            request_id=request_id,
                            user_id=user_id,
                            agent_id=agent_id,
                            session_id=session_id,
                            step=step,
                            prompt=prompt,
                            response=response,
                            parsed=parsed,
                            memory_ids=mem_ids,
                            elapsed_ms=step_elapsed_ms,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            event_at_ms=event_at_ms,
                            **extra_kwargs,
                        )

                await asyncio.wait_for(_limited_store(), timeout=_trace_timeout)
            except asyncio.TimeoutError:
                logger.warning(
                    "[pipeline-log] trace store timed out after %.2fs step=%s",
                    _trace_timeout, step,
                )
            except Exception as e:
                logger.debug(f"[pipeline-log] trace store failed: {e}")
            finally:
                _release_once(release_state)

        try:
            loop = asyncio.get_running_loop()
            # JSONL uses the thread pool and has no affinity to a loop, so it stays on the current loop.
            _track_task(loop.create_task(_persist_jsonl()))
            # DB trace uses the cache pool (possibly aiomysql, bound to owner_loop). If the current loop is not
            # owner_loop (e.g. search on its own loop), the work is dispatched back to owner_loop to avoid crossing loops.
            if trace_enabled:
                if _reserve_trace_slot():
                    try:
                        if owner_loop is not None and owner_loop is not loop:
                            trace_coro = _persist_trace()
                            try:
                                future = asyncio.run_coroutine_threadsafe(
                                    trace_coro, owner_loop,
                                )
                            except Exception:
                                trace_coro.close()
                                raise

                            def _consume(f):
                                try:
                                    f.result()
                                except Exception as e:
                                    logger.debug(
                                        "[pipeline-log] cross-loop trace failed: %s", e,
                                    )
                                finally:
                                    _release_once(release_state)

                            future.add_done_callback(_consume)
                        else:
                            _track_task(loop.create_task(_persist_trace()))
                    except Exception as e:
                        _release_once(release_state)
                        logger.debug(
                            "[pipeline-log] trace scheduling failed: %s", e,
                        )
        except RuntimeError:
            # No running loop (extremely rare): degrade to synchronous writes so no log is lost
            try:
                log_writer.write_step(
                    request_id=request_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    step=step,
                    prompt=prompt,
                    response=response,
                    parsed=parsed,
                    memory_ids=mem_ids,
                    elapsed_ms=step_elapsed_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    event_at_ms=event_at_ms,
                )
            except Exception:
                pass
            if trace_enabled:
                try:
                    result = original_store(
                        request_id=request_id,
                        user_id=user_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        step=step,
                        prompt=prompt,
                        response=response,
                        parsed=parsed,
                        memory_ids=mem_ids,
                        elapsed_ms=step_elapsed_ms,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        event_at_ms=event_at_ms,
                        **extra_kwargs,
                    )
                    if asyncio.iscoroutine(result):
                        # Cannot be awaited: discard it so nothing gets stuck
                        result.close()
                except Exception:
                    pass

        return True

    _store_pipeline_observability.trace_stats = _trace_stats
    return _store_pipeline_observability


def install_pipeline_observability(
    cache: Any,
    *,
    log_label: str = "init",
) -> None:
    """Install the JSONL + Trace wrapper on a cache (fire-and-forget).

    Call this on the loop the cache belongs to (write loop / _LoopThread); the function captures that loop
    and passes it to the wrapper as owner_loop: once search runs on its own loop, DB writes triggered from the search
    loop are dispatched back to this owner_loop, avoiding aiomysql cross-loop connection pool poisoning.
    """
    from .pipeline_log_writer import PipelineLogWriter

    try:
        owner_loop = asyncio.get_running_loop()
    except RuntimeError:
        owner_loop = None

    log_dir = resolve_pipeline_log_dir()
    log_writer = PipelineLogWriter(log_dir)
    trace_enabled = is_pipeline_trace_enabled()
    logger.info(
        f"[{log_label}] Pipeline log (JSONL) enabled: {log_dir}/<subdir>/<date>.log "
        f"(subdir from user_id prefix before '__', else default) ; "
        f"pipeline trace (DB)={'on' if trace_enabled else 'off'} ; "
        f"persist=async(fire-and-forget)"
    )
    cache.store_pipeline_log = wrap_store_pipeline_log(
        cache.store_pipeline_log,
        log_writer=log_writer,
        trace_enabled=trace_enabled,
        owner_loop=owner_loop,
    )
