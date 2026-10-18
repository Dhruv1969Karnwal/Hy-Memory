"""
HY Memory - MemoryInspector debug/audit utility class

Provides read access to the internal state of the memory system, including:
- Call-chain logs (prompt and response of EXTRACT / SEARCH_QUERY / RECONCILE)
- Knowledge-base change records (ADD / UPDATE operation history)

Both sync and async interfaces are provided to fit different call sites:
- Sync (get_xxx): runs on the background loop through _loop_thread, for plain scripts / CLI
- Async (async_get_xxx): safely delegated to the _LoopThread event loop, for any async framework

Usage:
    from hy_memory import HyMemoryClient, MemoryInspector

    client = HyMemoryClient()
    inspector = MemoryInspector(client)

    # Sync
    logs = inspector.get_pipeline_logs(request_id="abc123")
    trace = inspector.get_full_trace("abc123")

    # Async (inside an async context)
    logs = await inspector.async_get_pipeline_logs(request_id="abc123")
    trace = await inspector.async_get_full_trace("abc123")
"""

import asyncio
from typing import Any, Dict, List, Optional

from .utils.async_executor import await_concurrent_future_guarded


class MemoryInspector:
    """
    Memory system debug/audit utility.

    Takes the internal cache reference from the HyMemoryClient instance;
    sync methods use _loop_thread to run on the right event loop,
    async methods await cache operations directly.
    """

    def __init__(self, client):
        """
        Args:
            client: HyMemoryClient instance
        """
        self._client = client
        self._cache = client._cache
        self._loop_thread = client._loop_thread

    async def _run_on_internal_loop(self, coro):
        """
        Ensure the coroutine runs on the _LoopThread event loop.
        Awaits directly when already on the right loop; otherwise delegates.
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        internal_loop = self._loop_thread._loop

        if current_loop is internal_loop:
            return await coro
        else:
            future = asyncio.run_coroutine_threadsafe(coro, internal_loop)
            return await await_concurrent_future_guarded(future, internal_loop)

    # ================================================================
    # Pipeline Logs (intermediate results of the LLM call chain)
    # ================================================================

    def get_pipeline_logs(
        self,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        step: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query pipeline LLM call-chain intermediate results (sync)"""
        return self._loop_thread.run(
            self.async_get_pipeline_logs(
                request_id=request_id, user_id=user_id, step=step, limit=limit,
            )
        )

    async def async_get_pipeline_logs(
        self,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        step: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query pipeline LLM call-chain intermediate results (async).

        For each add() call, the prompt, response and parsed result of the
        extract / search_query / reconcile steps of the lite pipeline are persisted.

        Args:
            request_id: filter by request_id (full call chain of one write)
            user_id:    filter by user_id
            step:       filter by step ("EXTRACT" / "SEARCH_QUERY" / "RECONCILE")
            limit:      max number of rows to return

        Returns:
            [{"step", "prompt", "response", "parsed", "memory_ids",
              "elapsed_ms", "created_at", "request_id", "user_id", "agent_id"}]
        """
        return await self._run_on_internal_loop(
            self._cache.get_pipeline_logs(
                request_id=request_id, user_id=user_id, step=step, limit=limit,
            )
        )

    # Convenience methods (sync)

    def get_extract_log(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the EXTRACT step log of a request"""
        logs = self.get_pipeline_logs(request_id=request_id, step="EXTRACT", limit=1)
        return logs[0] if logs else None

    def get_search_query_log(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the SEARCH_QUERY step log of a request"""
        logs = self.get_pipeline_logs(request_id=request_id, step="SEARCH_QUERY", limit=1)
        return logs[0] if logs else None

    def get_reconcile_log(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the RECONCILE step log of a request"""
        logs = self.get_pipeline_logs(request_id=request_id, step="RECONCILE", limit=1)
        return logs[0] if logs else None

    # Convenience methods (async)

    async def async_get_extract_log(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the EXTRACT step log of a request (async)"""
        logs = await self.async_get_pipeline_logs(request_id=request_id, step="EXTRACT", limit=1)
        return logs[0] if logs else None

    async def async_get_search_query_log(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the SEARCH_QUERY step log of a request (async)"""
        logs = await self.async_get_pipeline_logs(request_id=request_id, step="SEARCH_QUERY", limit=1)
        return logs[0] if logs else None

    async def async_get_reconcile_log(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the RECONCILE step log of a request (async)"""
        logs = await self.async_get_pipeline_logs(request_id=request_id, step="RECONCILE", limit=1)
        return logs[0] if logs else None

    # ================================================================
    # Memory Operations (knowledge-base change records)
    # ================================================================

    def get_memory_operations(
        self,
        request_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query knowledge-base change records (sync)"""
        return self._loop_thread.run(
            self.async_get_memory_operations(
                request_id=request_id, memory_id=memory_id, user_id=user_id, limit=limit,
            )
        )

    async def async_get_memory_operations(
        self,
        request_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query knowledge-base change records (async).

        Every ADD / UPDATE produced by reconcile is written to persistent storage.

        Args:
            request_id: which knowledge changes a given write triggered
            memory_id:  how a given memory came to be (ADD or UPDATE)
            user_id:    full change history of a user
            limit:      max number of rows to return

        Returns:
            [{"op", "memory_id", "old_memory_id", "content", "layer",
              "reason", "created_at", "request_id", "user_id", "agent_id"}]
        """
        return await self._run_on_internal_loop(
            self._cache.get_memory_operations(
                request_id=request_id, memory_id=memory_id, user_id=user_id, limit=limit,
            )
        )

    # ================================================================
    # Combined queries (full pipeline in one call)
    # ================================================================

    def get_full_trace(self, request_id: str) -> Dict[str, Any]:
        """Get the full trace of a write (sync)"""
        return self._loop_thread.run(
            self.async_get_full_trace(request_id)
        )

    async def async_get_full_trace(self, request_id: str) -> Dict[str, Any]:
        """
        Get the full trace of a write (async).

        One call returns the call chain plus the knowledge changes.

        Args:
            request_id: request_id returned by add()

        Returns:
            {
                "request_id": "...",
                "pipeline_logs": [EXTRACT, SEARCH_QUERY, RECONCILE],
                "memory_operations": [ADD, UPDATE, ...],
            }
        """
        import asyncio
        logs, ops = await asyncio.gather(
            self.async_get_pipeline_logs(request_id=request_id),
            self.async_get_memory_operations(request_id=request_id),
        )
        return {
            "request_id": request_id,
            "pipeline_logs": logs,
            "memory_operations": ops,
        }

    # ================================================================
    # Search / Read specific tracing (post15+)
    # ================================================================

    # All read-side step strings start with this prefix by convention; the inspector filters on it
    _READ_STEP_PREFIX = "READ_"

    # Typical step sequence each reader writes, so the UI can render in order
    READ_STEP_ORDER = [
        "READ_REQUEST",
        "READ_EMBED_QUERY",
        "READ_RECALL_VEC",
        "READ_HIDDEN_L0_CONTEXT",
        "READ_BM25",
        "READ_GRAPH_RERANK",
        "READ_MERGE_SCHEMA",
        "READ_EVOLUTION",
        "READ_SUMMARY",
    ]
    def get_read_trace(self, request_id: str) -> Dict[str, Any]:
        """Query the full read pipeline trace of a search (sync)."""
        return self._loop_thread.run(self.async_get_read_trace(request_id))

    async def async_get_read_trace(self, request_id: str) -> Dict[str, Any]:
        """
        Query the full read pipeline trace of a search (async).

        Pulls every `READ_*` step of this request_id from the `pipeline_logs` table and groups
        the rows by step name. Also returns a recommended render order for the UI.

        Args:
            request_id: request_id returned by `client.search()` / `client.async_search()`

        Returns:
            {
                "request_id": "...",
                "reader": "hybrid" / "legacy" / None,
                "steps": [  # ascending by created_at
                    {"step", "prompt", "response", "parsed" (already parsed into a dict),
                     "memory_ids", "elapsed_ms", "created_at"}
                ],
                "by_step": { "READ_RECALL_VEC": [...], ... },
                "recommended_order": [...],   # typical step sequence for the reader version
                "summary": { ... } | None,    # shortcut to the parsed field of the READ_SUMMARY row
            }
        """
        all_logs = await self.async_get_pipeline_logs(request_id=request_id, limit=500)
        read_logs = [log for log in all_logs if (log.get("step") or "").startswith(self._READ_STEP_PREFIX)]

        # Ascending by event_at_ms (when the step ran, most accurate); falls back to created_at for stability
        read_logs.sort(key=lambda log: (log.get("event_at_ms") or 0, log.get("created_at") or ""))

        import json as _json
        parsed_logs: List[Dict[str, Any]] = []
        by_step: Dict[str, List[Dict[str, Any]]] = {}
        reader_version = None
        summary_parsed: Optional[Dict[str, Any]] = None

        for log in read_logs:
            item = dict(log)
            # Deserialize the parsed field so the UI can read it directly
            raw_parsed = log.get("parsed")
            if isinstance(raw_parsed, str) and raw_parsed:
                try:
                    item["parsed"] = _json.loads(raw_parsed)
                except Exception:
                    item["parsed"] = raw_parsed
            # Pick up the reader tag from any log
            if reader_version is None and isinstance(item.get("parsed"), dict):
                reader_version = item["parsed"].get("_reader") or reader_version
            parsed_logs.append(item)
            by_step.setdefault(item.get("step", ""), []).append(item)
            if item.get("step") == "READ_SUMMARY" and isinstance(item.get("parsed"), dict):
                summary_parsed = item["parsed"]

        # Recommended order
        order = self.READ_STEP_ORDER

        return {
            "request_id": request_id,
            "reader": reader_version,
            "steps": parsed_logs,
            "by_step": by_step,
            "recommended_order": order,
            "summary": summary_parsed,
        }

    # Convenience methods: single-step reads
    def get_read_summary(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the READ_SUMMARY overview log of a search (sync)."""
        return self._loop_thread.run(self.async_get_read_summary(request_id))

    async def async_get_read_summary(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the READ_SUMMARY overview log of a search (async)."""
        logs = await self.async_get_pipeline_logs(request_id=request_id, step="READ_SUMMARY", limit=1)
        return logs[0] if logs else None

    # ================================================================
    # Statistics (for the paper / analysis)
    # ================================================================

    @staticmethod
    def _classify_op(op_record: Dict[str, Any]) -> str:
        """
        Normalize the op type into a single classification.

        Backwards compatible: the old format stored supersede as op="ADD" with a non-empty supersedes=[target_id],
        which has to be reclassified here as "SUPERSEDE".
        """
        op_type = op_record.get("op", "")
        # Legacy compatibility: op=ADD with non-empty supersedes is really SUPERSEDE
        if op_type == "ADD":
            supersedes = op_record.get("supersedes", [])
            if supersedes and len(supersedes) > 0:
                return "SUPERSEDE"
        return op_type

    def get_evolve_stats(self, request_id: str) -> Dict[str, Any]:
        """
        Get operation stats for a single evolve (or add) call.

        Returns:
            {
                "request_id": "...",
                "op_counts": {"ADD": N, "SUPERSEDE": N, "UPDATE": N, ...},
                "net_change": N,       # add + supersede - delete (net memory growth)
            }
        """
        ops = self.get_memory_operations(request_id=request_id)
        from collections import Counter
        op_counts = Counter(self._classify_op(o) for o in ops)

        # net_change: ops that create a new node minus ops that delete one
        _ADD_OPS = {"ADD", "SUPERSEDE", "GRAPH_ADD", "SWEEPER_CORE_CREATE"}
        _DEL_OPS = {"DELETE", "GRAPH_DELETE"}
        net = sum(op_counts.get(k, 0) for k in _ADD_OPS) - sum(op_counts.get(k, 0) for k in _DEL_OPS)

        return {
            "request_id": request_id,
            "op_counts": dict(op_counts),
            "net_change": net,
        }

    def get_user_stats(
        self,
        user_id: str,
        limit: int = 10000,
    ) -> Dict[str, Any]:
        """
        Get cumulative per-user operation stats (broken down by op type).

        Backwards compatible: the old format stored supersede as op="ADD" with non-empty supersedes,
        and it is counted as SUPERSEDE here.

        Returns:
            {
                "user_id": "...",
                "total_ops": N,
                "op_counts": {"ADD": N, "SUPERSEDE": N, "UPDATE": N, "GRAPH_ADD": N, ...},
                "total_net_change": N,      # net growth (ADD+SUPERSEDE+GRAPH_ADD+SWEEPER_CORE_CREATE) - (DELETE+GRAPH_DELETE)
                "total_requests": N,         # number of distinct request_id values
                "per_request": [             # per write/evolve stats, ordered by time
                    {"request_id": "...", "op_counts": {...}, "net": N, "created_at": "..."},
                    ...
                ]
            }
        """
        ops = self.get_memory_operations(user_id=user_id, limit=limit)

        from collections import OrderedDict, Counter

        # Group by request_id
        request_map: OrderedDict = OrderedDict()
        for o in reversed(ops):  # ops come back DESC, reversing makes them ASC
            rid = o.get("request_id", "")
            if rid not in request_map:
                request_map[rid] = {"op_counts": Counter(), "created_at": o.get("created_at", "")}
            op_type = self._classify_op(o)
            request_map[rid]["op_counts"][op_type] += 1

        # Aggregate
        total_counter = Counter()
        for d in request_map.values():
            total_counter.update(d["op_counts"])

        _ADD_OPS = {"ADD", "SUPERSEDE", "GRAPH_ADD", "SWEEPER_CORE_CREATE"}
        _DEL_OPS = {"DELETE", "GRAPH_DELETE"}

        per_request = []
        for rid, d in request_map.items():
            oc = d["op_counts"]
            net = sum(oc.get(k, 0) for k in _ADD_OPS) - sum(oc.get(k, 0) for k in _DEL_OPS)
            per_request.append({
                "request_id": rid,
                "op_counts": dict(oc),
                "net": net,
                "created_at": d["created_at"],
            })

        total_net = sum(total_counter.get(k, 0) for k in _ADD_OPS) - sum(total_counter.get(k, 0) for k in _DEL_OPS)

        return {
            "user_id": user_id,
            "total_ops": sum(total_counter.values()),
            "op_counts": dict(total_counter),
            "total_net_change": total_net,
            "total_requests": len(request_map),
            "per_request": per_request,
        }


    # ================================================================
    # System Metrics
    # ================================================================

    def get_system_metrics(self, minutes: int = 5) -> Dict[str, Any]:
        """
        Get system-level load metrics (last 5 minutes by default).

        Returns:
            {
                "uptime_seconds", "requests", "avg_latency_ms",
                "throughput", "storage_ops", "sys2_requests", "since"
            }
        """
        from .metrics import MetricsCollector
        return self._loop_thread.run(MetricsCollector.get().get_snapshot(minutes=minutes))

    async def async_get_system_metrics(self, minutes: int = 5) -> Dict[str, Any]:
        """Get system-level load metrics (async, last 5 minutes by default)"""
        from .metrics import MetricsCollector
        return await self._run_on_internal_loop(
            MetricsCollector.get().get_snapshot(minutes=minutes)
        )
