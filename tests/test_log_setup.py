# -*- coding: utf-8 -*-
"""
Logging stream split tests -- verify INFO/DEBUG go to stdout and WARNING/ERROR/CRITICAL go to stderr.

Fixes an issue: every level used to flow through a single StreamHandler defaulting to stderr, and monitoring
platforms such as OpenClaw flagged the whole stderr as WARN, mislabeling INFO logs as alerts.
"""

import io
import logging
import sys

import pytest

from hy_memory.utils import log_setup
from hy_memory.utils.log_setup import _StdoutLevelFilter

@pytest.fixture(autouse=True)
def _reset_logging_state(tmp_path, monkeypatch):
    """Reset the hy_memory logger and _logging_configured before each case"""
    # reset the module-level guard
    monkeypatch.setattr(log_setup, "_logging_configured", False)
    # clear existing handlers
    hy_logger = logging.getLogger("hy_memory")
    old_handlers = hy_logger.handlers[:]
    old_propagate = hy_logger.propagate
    hy_logger.handlers.clear()

    # redirect the log directory to tmp (avoid polluting ~/.hy_memory)
    monkeypatch.setenv("MEMORY_LOG_DIR", str(tmp_path / "logs"))

    yield

    hy_logger.handlers.clear()
    for h in old_handlers:
        hy_logger.addHandler(h)
    hy_logger.propagate = old_propagate
    monkeypatch.setattr(log_setup, "_logging_configured", False)


class TestStdoutLevelFilter:
    """Verify the filter class behavior directly"""

    def test_passes_debug(self):
        f = _StdoutLevelFilter()
        rec = logging.LogRecord("x", logging.DEBUG, "", 0, "msg", None, None)
        assert f.filter(rec) is True

    def test_passes_info(self):
        f = _StdoutLevelFilter()
        rec = logging.LogRecord("x", logging.INFO, "", 0, "msg", None, None)
        assert f.filter(rec) is True

    def test_blocks_warning(self):
        f = _StdoutLevelFilter()
        rec = logging.LogRecord("x", logging.WARNING, "", 0, "msg", None, None)
        assert f.filter(rec) is False

    def test_blocks_error(self):
        f = _StdoutLevelFilter()
        rec = logging.LogRecord("x", logging.ERROR, "", 0, "msg", None, None)
        assert f.filter(rec) is False

    def test_blocks_critical(self):
        f = _StdoutLevelFilter()
        rec = logging.LogRecord("x", logging.CRITICAL, "", 0, "msg", None, None)
        assert f.filter(rec) is False


class TestSetupLoggingHandlerRouting:
    """Verify setup_logging configures a stdout/stderr split handler set"""

    def test_two_stream_handlers_present(self):
        log_setup.setup_logging(level=logging.DEBUG)
        hy_logger = logging.getLogger("hy_memory")
        stream_handlers = [
            h for h in hy_logger.handlers if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            and getattr(h, "stream", None) in {sys.stdout, sys.stderr}
        ]
        # stdout + stderr = 2 stream handlers
        assert len(stream_handlers) == 2, f"expected 2 stream handlers, got {len(stream_handlers)}"

        targets = {h.stream for h in stream_handlers}
        assert sys.stdout in targets, "expected one handler targeting sys.stdout"
        assert sys.stderr in targets, "expected one handler targeting sys.stderr"

    def test_stdout_handler_has_level_filter(self):
        log_setup.setup_logging(level=logging.INFO)
        hy_logger = logging.getLogger("hy_memory")
        stdout_handler = next(
            h for h in hy_logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            and h.stream is sys.stdout
        )
        # must carry _StdoutLevelFilter
        has_filter = any(isinstance(f, _StdoutLevelFilter) for f in stdout_handler.filters)
        assert has_filter, "stdout handler must have _StdoutLevelFilter"

    def test_stderr_handler_level_warning(self):
        log_setup.setup_logging(level=logging.DEBUG)  # even with a global DEBUG level
        hy_logger = logging.getLogger("hy_memory")
        stderr_handler = next(
            h for h in hy_logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            and h.stream is sys.stderr
        )
        # the stderr handler is always WARNING+
        assert stderr_handler.level == logging.WARNING


class TestEndToEndStreamRouting:
    """End to end: redirect stdout/stderr to buffers and verify records really land in the right place"""

    def _setup_with_redirected_streams(self, monkeypatch):
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out_buf)
        monkeypatch.setattr(sys, "stderr", err_buf)
        log_setup.setup_logging(level=logging.DEBUG)
        return out_buf, err_buf

    def test_info_to_stdout(self, monkeypatch):
        out_buf, err_buf = self._setup_with_redirected_streams(monkeypatch)
        log = logging.getLogger("hy_memory.test_info")
        log.info("INFO_LINE_TOKEN")
        for h in logging.getLogger("hy_memory").handlers:
            h.flush()
        assert "INFO_LINE_TOKEN" in out_buf.getvalue()
        assert "INFO_LINE_TOKEN" not in err_buf.getvalue()

    def test_debug_to_stdout(self, monkeypatch):
        out_buf, err_buf = self._setup_with_redirected_streams(monkeypatch)
        log = logging.getLogger("hy_memory.test_debug")
        log.debug("DEBUG_LINE_TOKEN")
        for h in logging.getLogger("hy_memory").handlers:
            h.flush()
        assert "DEBUG_LINE_TOKEN" in out_buf.getvalue()
        assert "DEBUG_LINE_TOKEN" not in err_buf.getvalue()

    def test_warning_to_stderr(self, monkeypatch):
        out_buf, err_buf = self._setup_with_redirected_streams(monkeypatch)
        log = logging.getLogger("hy_memory.test_warn")
        log.warning("WARN_LINE_TOKEN")
        for h in logging.getLogger("hy_memory").handlers:
            h.flush()
        assert "WARN_LINE_TOKEN" not in out_buf.getvalue()
        assert "WARN_LINE_TOKEN" in err_buf.getvalue()

    def test_error_to_stderr(self, monkeypatch):
        out_buf, err_buf = self._setup_with_redirected_streams(monkeypatch)
        log = logging.getLogger("hy_memory.test_err")
        log.error("ERROR_LINE_TOKEN")
        for h in logging.getLogger("hy_memory").handlers:
            h.flush()
        assert "ERROR_LINE_TOKEN" not in out_buf.getvalue()
        assert "ERROR_LINE_TOKEN" in err_buf.getvalue()

    def test_critical_to_stderr(self, monkeypatch):
        out_buf, err_buf = self._setup_with_redirected_streams(monkeypatch)
        log = logging.getLogger("hy_memory.test_crit")
        log.critical("CRITICAL_LINE_TOKEN")
        for h in logging.getLogger("hy_memory").handlers:
            h.flush()
        assert "CRITICAL_LINE_TOKEN" not in out_buf.getvalue()
        assert "CRITICAL_LINE_TOKEN" in err_buf.getvalue()

    def test_no_duplicate_emission_per_line(self, monkeypatch):
        """One log line must not appear on both stdout and stderr"""
        out_buf, err_buf = self._setup_with_redirected_streams(monkeypatch)
        log = logging.getLogger("hy_memory.test_dup")
        log.info("UNIQUE_INFO_TOKEN")
        log.warning("UNIQUE_WARN_TOKEN")
        for h in logging.getLogger("hy_memory").handlers:
            h.flush()

        out = out_buf.getvalue()
        err = err_buf.getvalue()
        # INFO on stdout, not on stderr
        assert out.count("UNIQUE_INFO_TOKEN") == 1
        assert err.count("UNIQUE_INFO_TOKEN") == 0
        # WARN on stderr, not on stdout
        assert out.count("UNIQUE_WARN_TOKEN") == 0
        assert err.count("UNIQUE_WARN_TOKEN") == 1


class TestSingleLineAndJsonFormat:
    """Newlines must not truncate log lines; JSON mode emits valid JSON."""

    def _record(self, msg):
        rec = logging.LogRecord(
            "hy_memory.test", logging.INFO, "f.py", 1, msg, None, None
        )
        rec.request_id = "req-xyz"
        return rec

    def test_text_formatter_escapes_newlines(self, monkeypatch):
        monkeypatch.setenv("MEMORY_LOG_FORMAT", "text")
        fmt = log_setup._make_formatter()
        out = fmt.format(self._record("query='a\nb\rc'"))
        # single line: the rendered output contains no raw newline/carriage return
        assert "\n" not in out and "\r" not in out
        assert "\\n" in out and "\\r" in out
        assert "req-xyz" in out

    def test_json_formatter_is_valid_json_and_single_line(self, monkeypatch):
        import json as _json
        monkeypatch.setenv("MEMORY_LOG_FORMAT", "json")
        fmt = log_setup._make_formatter()
        out = fmt.format(self._record("query='a\nb\rc'"))
        assert "\n" not in out and "\r" not in out
        obj = _json.loads(out)  # must be valid JSON
        assert obj["request_id"] == "req-xyz"
        assert obj["level"] == "INFO"
        assert obj["logger"] == "hy_memory.test"
        # raw newlines stay inside the field value (stored JSON-escaped, without truncating the line)
        assert obj["msg"] == "query='a\nb\rc'"

    def test_default_is_text(self, monkeypatch):
        monkeypatch.delenv("MEMORY_LOG_FORMAT", raising=False)
        fmt = log_setup._make_formatter()
        assert isinstance(fmt, log_setup.SingleLineFormatter)
