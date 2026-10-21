# -*- coding: utf-8 -*-
"""
HY Memory - logging configuration

Standard Python logging + TimedRotatingFileHandler,
one file per day, kept for 15 days. Supports request_id injection,
so the complete request pipeline can be filtered by grepping the logs for a request_id.

Usage:
    from hy_memory.utils.log_setup import setup_logging, set_request_id

    setup_logging()  # call once at application startup
    set_request_id("abc123def456")  # set at each request entry point

Log file location:
    {MEMORY_DATA_DIR}/logs/hy_memory_{date}.log
    Default: ~/.hy_memory/logs/hy_memory_{date}.log
"""

import contextvars
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

# ================================================================
# request_id context variable (coroutine safe)
# ================================================================

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


def set_request_id(request_id: str) -> None:
    """Set the request_id for the current coroutine/thread"""
    _request_id_var.set(request_id)


def get_request_id() -> str:
    """Return the request_id of the current coroutine/thread"""
    return _request_id_var.get()


class request_id_scope:
    """
    Temporarily set request_id to the given value (the "-" sentinel by default), restoring the old value on exit.

    For process-level / initialization logging: even when an outer scope (e.g. an App request context) has already set a business
    request_id, initialization logs must not inherit it, otherwise filtering by request_id would pick up
    one-off init noise.

    Usage:
        with request_id_scope():   # "-" by default
            ... initialization, emitting init logs ...
    """

    def __init__(self, request_id: str = "-"):
        self._rid = request_id
        self._token = None

    def __enter__(self):
        self._token = _request_id_var.set(self._rid)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._token is not None:
            _request_id_var.reset(self._token)
        return False


# ================================================================
# Custom filter: inject request_id into the LogRecord
# ================================================================

class RequestIdFilter(logging.Filter):
    """
    Reads request_id from contextvars and injects it into the LogRecord.

    This lets every handler's formatter use %(request_id)s.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get()  # type: ignore[attr-defined]
        return True


class _StdoutLevelFilter(logging.Filter):
    """
    For the stdout handler only: lets DEBUG/INFO through and rejects WARNING and above.

    Paired with a separate stderr handler (setLevel=WARNING) this realizes the Unix-conventional
    two-stream split, so INFO logs are not mislabeled as WARN by monitoring systems (e.g. OpenClaw) that treat stderr as WARN.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING


# ================================================================
# Logging configuration
# ================================================================

_LOG_FORMAT = (
    "%(asctime)s [%(levelname)s] [%(request_id)s] %(name)s: %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Global flag to avoid configuring twice
_logging_configured = False


class SingleLineFormatter(logging.Formatter):
    """
    Plain-text formatter that guarantees every log record is emitted on a single line.

    Problem: message / exception may contain real newlines (a user query may carry \\n),
    and once the single-line template is cut by a newline, downstream line-based collectors get half a record and report parse errors.
    Fix: after rendering, control newlines such as \\r\\n are escaped into literal \\n / \\r.
    """

    def format(self, record: logging.LogRecord) -> str:
        s = super().format(record)
        if "\n" in s or "\r" in s:
            s = s.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n")
        return s


class JsonLineFormatter(logging.Formatter):
    """
    Structured JSON Lines formatter (each log record becomes one valid JSON line).

    For log collection platforms that parse every line as JSON: newlines inside message are escaped naturally by json,
    so lines are never cut and no "[json] error field" style parse errors appear.

    Fields: ts / level / request_id / logger / msg (plus exc for an exception stack, when present).
    """

    def format(self, record: logging.LogRecord) -> str:
        import json

        payload = {
            "ts": self.formatTime(record, _DATE_FORMAT),
            "level": record.levelname,
            "request_id": getattr(record, "request_id", "-"),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _make_formatter() -> logging.Formatter:
    """
    Select the formatter from MEMORY_LOG_FORMAT:
    - "json"          → JsonLineFormatter (structured, for JSON collection platforms)
    - other/unset (default) → SingleLineFormatter (plain text, but a single line is never cut by newlines)
    """
    fmt = os.getenv("MEMORY_LOG_FORMAT", "text").strip().lower()
    if fmt == "json":
        return JsonLineFormatter()
    return SingleLineFormatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)


def _default_log_dir() -> str:
    """
    Determine the log directory:
    1. MEMORY_LOG_DIR environment variable
    2. {MEMORY_DATA_DIR}/logs/
    3. ~/.hy_memory/logs/
    """
    explicit = os.getenv("MEMORY_LOG_DIR")
    if explicit:
        return explicit
    data_dir = os.getenv(
        "MEMORY_DATA_DIR",
        os.path.join(str(Path.home()), ".hy_memory"),
    )
    return os.path.join(data_dir, "logs")


def setup_logging(
    log_dir: Optional[str] = None,
    retention_days: int = 15,
    level: Optional[int] = None,
) -> None:
    """
    Configure logging for the hy_memory module.

    - File output: TimedRotatingFileHandler, rolls at midnight each day, retained for retention_days
    - Console output: two-stream split -- DEBUG/INFO go to stdout, WARNING/ERROR/CRITICAL go to stderr
      (following the Unix convention; monitoring platforms such as OpenClaw label all of stderr as WARN,
       so INFO must use stdout to avoid being misclassified as an alert)
    - every log line carries a request_id field

    Args:
        log_dir:        log directory, {MEMORY_DATA_DIR}/logs/ by default
        retention_days: days of log retention, 15 by default
        level:          log level, read from the MEMORY_LOG_LEVEL env var and defaulting to INFO

    Safe to call repeatedly; only the first call takes effect.
    """
    global _logging_configured
    if _logging_configured:
        return
    _logging_configured = True

    if level is None:
        level = getattr(logging, os.getenv("MEMORY_LOG_LEVEL", "INFO").upper(), logging.INFO)

    # Determine the log directory and create it
    resolved_dir = log_dir or _default_log_dir()
    os.makedirs(resolved_dir, exist_ok=True)

    # hy_memory root logger
    root_logger = logging.getLogger("hy_memory")
    root_logger.setLevel(level)

    # request_id filter -- it must be attached to handlers, not to the logger.
    # Reason: in Python logging propagation, when a child logger (e.g. hy_memory.client)
    # propagates a record to its parent, only the parent's handlers are invoked,
    # never the parent's filters. If the filter sat only on the logger,
    # child records would reach the handler without a request_id, and the
    # formatter would crash with a KeyError.
    req_filter = RequestIdFilter()

    formatter = _make_formatter()

    # ---- File handler ----
    log_file = os.path.join(resolved_dir, "hy_memory.log")
    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=retention_days,
        encoding="utf-8",
    )
    # Rotated file name: hy_memory.log.2026-03-21
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    file_handler.addFilter(req_filter)
    root_logger.addHandler(file_handler)

    # ---- Console handler: two-stream split ----
    # stdout: DEBUG / INFO (a filter excludes WARNING and above)
    # stderr: WARNING / ERROR / CRITICAL
    # Issue fix: previously every level went through a single StreamHandler defaulting to stderr,
    # and monitoring platforms such as OpenClaw labeled the whole stderr WARN, so INFO lines
    # were mislabeled as alerts and flooded the view.
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(level)
    stdout_handler.addFilter(req_filter)
    stdout_handler.addFilter(_StdoutLevelFilter())
    root_logger.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    # stderr always takes only WARNING+, independent of the global level (even with level=DEBUG
    # it never receives INFO/DEBUG lines)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.addFilter(req_filter)
    root_logger.addHandler(stderr_handler)

    # Avoid propagating records up to the root logger, which would duplicate output
    root_logger.propagate = False

    # Suppress noisy third-party debug logs
    for noisy in ("openai", "httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root_logger.debug(
        f"Logging configured: dir={resolved_dir} "
        f"retention={retention_days}d level={logging.getLevelName(level)}"
    )
