"""
HY Memory - Pipeline Tracer (per-user debug logging)

A straightforward request-level trace tool that records:
1. Per-stage latency (ms)
2. Key intermediate results per stage (truncated to a sane length)
3. The full timeline of the whole request

Usage:
    tracer = PipelineTracer(
        operation="ProcessMemory",
        pipeline_version="system1",
        uid="user_123",
        agent_id="chat_scene",
    )
    
    with tracer.span("embed") as s:
        embedding = await embed_service.embed(text)
        s.set_output({"dims": len(embedding), "cache_hit": False})
    
    with tracer.span("qdrant_upsert") as s:
        memory_id = await vector_store.upsert(node)
        s.set_output({"memory_id": memory_id})
    
    tracer.finish()
    # -> automatically writes the log file + Python logger

Log file location (one directory per uid, rotated by uid+agent_id+date):
    {MEMORY_TRACE_LOG_DIR}/{uid}/{uid}_{agent_id}_{date}.log

Example:
    /memory/logs/traces/user_123/user_123_chat_scene_2026-03-17.log

Each line is one complete JSON trace:
{
    "trace_id": "abc123",
    "operation": "ProcessMemory",
    "pipeline": "system1",
    "uid": "user_123",
    "agent_id": "chat_scene",
    "start_time": "2026-03-11T14:30:00.123",
    "total_ms": 1720.5,
    "spans": [
        {"name": "route", "start_ms": 0.1, "end_ms": 0.3, "duration_ms": 0.2, "output": {"layer": "profile"}},
        {"name": "embed", "start_ms": 0.3, "end_ms": 1500.0, "duration_ms": 1499.7, "output": {"dims": 1024}},
        {"name": "qdrant_upsert", "start_ms": 1500.0, "end_ms": 1700.0, "duration_ms": 200.0, "output": {"memory_id": "xxx"}},
    ],
    "input_summary": "The user likes sci-fi movies...",
    "output_summary": {"success": true, "layer": "profile"},
    "error": null
}
"""

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Log root directory
# Cloud deploy: set MEMORY_TRACE_LOG_DIR or MEMORY_DATA_DIR to a mounted path
# Local use: writes to ~/.hy_memory/logs/traces/ by default
def _default_trace_log_dir() -> str:
    explicit = os.getenv("MEMORY_TRACE_LOG_DIR")
    if explicit:
        return explicit
    data_dir = os.getenv("MEMORY_DATA_DIR", os.path.join(Path.home(), ".hy_memory"))
    return os.path.join(data_dir, "logs", "traces")

_TRACE_LOG_DIR = _default_trace_log_dir()

# Max characters per output field (keeps logs from exploding)
# Raised to 10000 so full text payloads print intact (vectors and other big data still need external filtering)
_MAX_OUTPUT_LEN = 10000

# Whether tracing is enabled (an env var can turn it off)
_TRACE_ENABLED = os.getenv("MEMORY_TRACE_ENABLED", "true").lower() in ("true", "1", "yes")


def _truncate(obj: Any, max_len: int = _MAX_OUTPUT_LEN) -> Any:
    """Truncate overlong output."""
    if obj is None:
        return None
    if isinstance(obj, str):
        if len(obj) > max_len:
            return obj[:max_len] + f"...(truncated, total {len(obj)})"
        return obj
    if isinstance(obj, (list, tuple)):
        # If the items are floats (likely a vector), truncate immediately
        if len(obj) > 10 and all(isinstance(x, (int, float)) for x in obj[:5]):
            return {"_type": "vector", "_len": len(obj), "_sample": list(obj[:3])}
        if len(obj) > 100:
            return {"_type": type(obj).__name__, "_len": len(obj), "_sample": list(obj[:5])}
        return [_truncate(item, max_len) for item in obj]
    if isinstance(obj, dict):
        return {k: _truncate(v, max_len) for k, v in obj.items()}
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, (int, bool)):
        return obj
    # fallback: string form
    s = str(obj)
    if len(s) > max_len:
        return s[:max_len] + "...(truncated)"
    return s


@dataclass
class Span:
    """One stage within a trace."""
    name: str
    start_time: float = 0.0  # time.perf_counter()
    end_time: float = 0.0
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    # Offsets from the trace start (ms)
    start_ms: float = 0.0
    end_ms: float = 0.0
    
    def set_output(self, data: Dict[str, Any]) -> None:
        """Set the key output of this stage."""
        self.output.update(data)
    
    def set_error(self, err: str) -> None:
        """Set the error message."""
        self.error = err
    
    @property
    def duration_ms(self) -> float:
        if self.end_time <= 0:
            return 0.0
        return (self.end_time - self.start_time) * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "start_ms": round(self.start_ms, 2),
            "end_ms": round(self.end_ms, 2),
            "duration_ms": round(self.duration_ms, 2),
        }
        if self.output:
            d["output"] = _truncate(self.output)
        if self.error:
            d["error"] = self.error
        return d


class PipelineTracer:
    """
    Request-level pipeline tracer
    
    Records the full timeline and key intermediate results of one request.
    Supports nested spans and context-manager usage.
    """
    
    def __init__(
        self,
        operation: str,
        pipeline_version: str = "",
        uid: str = "",
        agent_id: str = "",
        request_id: str = "",
        content_preview: str = "",
    ):
        self.trace_id = uuid.uuid4().hex[:12]
        self.operation = operation
        self.pipeline_version = pipeline_version
        self.uid = uid
        self.agent_id = agent_id
        self.request_id = request_id
        self.content_preview = content_preview or ""
        
        self._start_time = time.perf_counter()
        self._start_datetime = datetime.now()
        self._spans: List[Span] = []
        self._current_span: Optional[Span] = None
        self._output_summary: Dict[str, Any] = {}
        self._error: Optional[str] = None
        self._finished = False
        
        if _TRACE_ENABLED:
            logger.debug(
                f"[trace] START {self.operation} "
                f"pipeline={self.pipeline_version} uid={self.uid} "
                f"content={self.content_preview}"
            )
    
    @contextmanager
    def span(self, name: str):
        """
        Record a stage via context manager.
        
        Usage:
            with tracer.span("embed") as s:
                result = await embed(text)
                s.set_output({"dims": len(result)})
        """
        s = Span(name=name)
        s.start_time = time.perf_counter()
        s.start_ms = (s.start_time - self._start_time) * 1000
        
        prev_span = self._current_span
        self._current_span = s
        
        try:
            yield s
        except Exception as e:
            s.set_error(str(e))
            raise
        finally:
            s.end_time = time.perf_counter()
            s.end_ms = (s.end_time - self._start_time) * 1000
            self._spans.append(s)
            self._current_span = prev_span
            
            if _TRACE_ENABLED:
                err_tag = f" ERROR={s.error}" if s.error else ""
                out_tag = ""
                if s.output:
                    # Condensed output
                    short_out = {k: v for k, v in s.output.items() if not isinstance(v, (list, dict)) or len(str(v)) < 100}
                    if short_out:
                        out_tag = f" → {short_out}"
                logger.debug(
                    f"[trace] {name}: {s.duration_ms:.1f}ms{out_tag}{err_tag}"
                )
    
    def start_span(self, name: str) -> Span:
        """
        Manual start/end style (used without with)
        
        Usage:
            s = tracer.start_span("embed")
            result = await embed(text)
            s.set_output({"dims": len(result)})
            tracer.end_span(s)
        """
        s = Span(name=name)
        s.start_time = time.perf_counter()
        s.start_ms = (s.start_time - self._start_time) * 1000
        self._current_span = s
        return s
    
    def end_span(self, s: Span) -> None:
        """End a manually started span."""
        s.end_time = time.perf_counter()
        s.end_ms = (s.end_time - self._start_time) * 1000
        self._spans.append(s)
        self._current_span = None
        
        if _TRACE_ENABLED:
            err_tag = f" ERROR={s.error}" if s.error else ""
            logger.debug(
                f"[trace] {s.name}: {s.duration_ms:.1f}ms{err_tag}"
            )
    
    def set_output(self, data: Dict[str, Any]) -> None:
        """Set the request-level output summary."""
        self._output_summary.update(data)
    
    def set_error(self, error: str) -> None:
        """Set the request-level error."""
        self._error = error
    
    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._start_time) * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Build the complete trace JSON."""
        return {
            "trace_id": self.trace_id,
            "operation": self.operation,
            "pipeline": self.pipeline_version,
            "uid": self.uid,
            "agent_id": self.agent_id,
            "request_id": self.request_id,
            "start_time": self._start_datetime.isoformat(),
            "total_ms": round(self.total_ms, 2),
            "spans": [s.to_dict() for s in self._spans],
            "input_summary": self.content_preview,
            "output_summary": _truncate(self._output_summary),
            "error": self._error,
        }
    
    def to_summary_line(self) -> str:
        """Build a one-line human-readable trace summary."""
        spans_info = " → ".join(
            f"{s.name}({s.duration_ms:.0f}ms)" for s in self._spans
        )
        status = "✅" if not self._error else f"❌ {self._error}"
        return (
            f"[{self.trace_id}] {self.operation} {self.pipeline_version} "
            f"total={self.total_ms:.0f}ms | {spans_info} | {status}"
        )
    
    def finish(self, *, write_file: bool = False) -> Dict[str, Any]:
        """
        Finish the trace.

        The request-level timeline goes to SQLite via the writer (WRITE_TIMELINE, used by the Inspector).
        By default the old per-uid trace file is not written; write_file=True keeps the legacy behavior.

        Returns:
            the complete trace dict
        """
        if self._finished:
            return self.to_dict()
        self._finished = True

        trace_data = self.to_dict()

        if _TRACE_ENABLED:
            logger.debug(f"[trace] FINISH {self.to_summary_line()}")
            if write_file:
                self._write_to_file(trace_data)

        return trace_data
    
    def _write_to_file(self, trace_data: Dict[str, Any]) -> None:
        """
        Write the JSONL log file.
        
        File path format:
            {TRACE_LOG_DIR}/{uid}/{uid}_{agent_id}_{date}.log

        Example:
            /memory/logs/traces/user_123/user_123_chat_scene_2026-03-17.log

        One JSON trace per line, rotated daily.
        """
        try:
            uid = self.uid or "_unknown"
            agent_id = self.agent_id or "_default"
            date_str = self._start_datetime.strftime("%Y-%m-%d")

            # Directory: {trace_log_dir}/{uid}/
            log_dir = Path(_TRACE_LOG_DIR) / uid
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # File: {uid}_{agent_id}_{date}.log
            log_file = log_dir / f"{uid}_{agent_id}_{date_str}.log"
            
            # Append one JSON line
            line = json.dumps(trace_data, ensure_ascii=False, default=str)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                
        except Exception as e:
            logger.warning(f"[TRACE:{self.trace_id}] Failed to write trace file: {e}")


def create_tracer(
    operation: str,
    pipeline_version: str = "",
    uid: str = "",
    agent_id: str = "",
    request_id: str = "",
    content_preview: str = "",
) -> PipelineTracer:
    """
    Convenience factory for creating a PipelineTracer.

    With MEMORY_TRACE_ENABLED=false the returned tracer still works;
    it simply emits no logs and writes no files.
    """
    return PipelineTracer(
        operation=operation,
        pipeline_version=pipeline_version,
        uid=uid,
        agent_id=agent_id,
        request_id=request_id,
        content_preview=content_preview,
    )
