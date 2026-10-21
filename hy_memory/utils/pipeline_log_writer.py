# -*- coding: utf-8 -*-
"""
Pipeline Log Writer -- per-step JSONL logs (always on, separate from the SQLite trace).

Directory: {log_dir}/{subdir}/{date}.log
Per line: request_id, namespace, user_id, agent_id, step, prompt, content, parsed,
tokens, elapsed_ms, timestamp. prompt/content are empty by default; the full text is recorded only when
MEMORY_LOG_LLM_CONTENT=true is explicitly set.

Where the subdirectory (subdir) comes from (general namespace convention):
  1. Take the part left of the first "__" in the record's user_id as the subdirectory;
     callers may format user_id as "{namespace}__{user_id}" to bucket different data domains into different directories;
  2. user_id without a "__" prefix (e.g. a single person using the SDK directly) -> falls back to the
     default_subdir passed at construction time (default "default").
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Compatibility with legacy imports
TraceFileWriter = None  # set after class definition


class PipelineLogWriter:
    """Appends JSONL as {subdir}/{date}.log; best-effort, failures silent.

    subdir resolution (general namespace prefix):
      1. Take the part left of the first "__" in the record's user_id as the subdir
         (callers can use the "{namespace}__{user_id}" convention for bucketing);
      2. user_id without a "__" prefix -> fall back to the default_subdir given at construction;
      3. Fall back to "default" after that.
    """

    def __init__(self, log_dir: str, default_subdir: str = "default"):
        self._log_dir = log_dir
        self._default_subdir = default_subdir or "default"
        try:
            (Path(self._log_dir) / self._default_subdir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def _resolve_subdir(self, user_id: str) -> str:
        """Take the prefix from the user_id's "{prefix}__{user_id}" form as the subdirectory; fall back when there is no prefix."""
        if user_id and "__" in user_id:
            prefix = user_id.split("__", 1)[0].strip()
            if prefix:
                return prefix
        return self._default_subdir

    def write_step(
        self,
        *,
        request_id: str = "",
        user_id: str = "",
        agent_id: str = "",
        session_id: str = "",
        step: str = "",
        prompt: str = "",
        response: str = "",
        parsed: str = "",
        memory_ids: Optional[List[str]] = None,
        elapsed_ms: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        event_at_ms: int = 0,
        **kwargs: Any,
    ) -> None:
        try:
            from .pipeline_observability import (
                is_llm_content_logging_enabled,
                redact_content,
                split_namespace_user,
            )

            tokens = (prompt_tokens or 0) + (completion_tokens or 0) + (total_tokens or 0)
            namespace, uid = split_namespace_user(user_id or "")
            if not is_llm_content_logging_enabled():
                prompt = redact_content(prompt)
                response = redact_content(response)
            # timestamp = when the step happened (readable ISO ms). Prefer the event_at_ms captured uniformly
            # by the upper layer (same source as the DB's event_at_ms, guaranteeing JSONL/DB reconciliation); otherwise fall back to the current time.
            if event_at_ms:
                _ts = datetime.fromtimestamp(event_at_ms / 1000.0).isoformat(timespec="milliseconds")
            else:
                _ts = datetime.now().isoformat(timespec="milliseconds")
            record = {
                "request_id": request_id or "",
                "namespace": namespace,
                "user_id": uid,
                "agent_id": agent_id or "",
                "session_id": session_id or "",
                "step": step or "",
                "prompt": prompt or "",
                "content": response or "",
                "parsed": parsed or "",
                "tokens": tokens,
                "elapsed_ms": round(float(elapsed_ms or 0), 2),
                "timestamp": _ts,
            }
            if memory_ids:
                record["memory_ids"] = list(memory_ids)

            line = json.dumps(record, ensure_ascii=False, default=str)
            date_str = datetime.now().strftime("%Y-%m-%d")
            subdir = self._resolve_subdir(user_id)
            log_dir = Path(self._log_dir) / subdir
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{date_str}.log"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            logger.debug(f"[pipeline-log] write failed: {e}")


# Backward compatibility
TraceFileWriter = PipelineLogWriter
