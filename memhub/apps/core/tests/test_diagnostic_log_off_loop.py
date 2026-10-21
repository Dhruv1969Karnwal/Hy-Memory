"""The diagnostic log must not do disk I/O on the calling thread

Real-machine evidence: of the six `loop_watchdog event loop blocked` lines in \
`~/.one-memory/logs/core.log`, three had MainThread blocked at
the top of the stack in `diagnostic_log.write` -- twice on that global lock,
once on `os.open`. The events are emitted from the event loop, and writing one line costs a \
directory scan
plus a full flush; once the disk gets busy, nothing in the process answers any more.

These guards are written as 'first prove it should fail': each one was made red on the old \
implementation first, then turned into an assertion.
The first three are red by behavior (the old implementation really did the I/O and really did \
block the caller),
not because some new API is missing.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import threading
import time
from pathlib import Path

import pytest

import one_memory.diagnostic_log as module
from one_memory.diagnostic_log import DiagnosticLog

# Injected 'slow disk' cost of a single write. Larger than any sane event loop budget,
# so while the calling thread still writes, the measurement is bound to see it.
SLOW_WRITE_SECONDS = 0.5
# Maximum tolerated event loop lag. The real watchdog threshold is 2.0s; this tightens it by an
# order of magnitude,
# and the slack left for CI jitter is still far larger than one slow write.
MAX_LOOP_LAG_SECONDS = 0.3
LINE_PREFIX = b'{"schema_version"'


class FilesystemSpy:
    """Records which thread issued every filesystem action that lands in the log directory.

    Only paths under the log directory count, so pytest's own stats are not included -- \
otherwise the calling thread
    always 'has I/O' and the assertion becomes a fake guard that is red forever.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.calls: list[tuple[str, int]] = []
        self._descriptors: set[int] = set()
        self._lock = threading.Lock()

    def _mine(self, path: object) -> bool:
        try:
            resolved = Path(os.fspath(path))  # type: ignore[arg-type]
        except TypeError:
            return False
        try:
            return self.root in resolved.parents or resolved == self.root
        except (OSError, ValueError):
            return False

    def _record(self, name: str) -> None:
        with self._lock:
            self.calls.append((name, threading.get_ident()))

    def names_for_thread(self, ident: int) -> list[str]:
        with self._lock:
            return sorted({name for name, caller in self.calls if caller == ident})

    def count(self, *names: str) -> int:
        wanted = set(names) if names else None
        with self._lock:
            return sum(1 for name, _ in self.calls if wanted is None or name in wanted)

    def install(self, monkeypatch: pytest.MonkeyPatch, *, slow_write: bool = False) -> None:
        real_open = os.open
        real_write = os.write
        real_fsync = os.fsync
        real_chmod = os.chmod
        real_close = os.close
        real_glob = pathlib.Path.glob
        real_stat = pathlib.Path.stat

        def spy_open(path, flags, mode=0o777, **kwargs):  # type: ignore[no-untyped-def]
            descriptor = real_open(path, flags, mode, **kwargs)
            if self._mine(path):
                self._record("os.open")
                with self._lock:
                    self._descriptors.add(descriptor)
            return descriptor

        def spy_write(descriptor, data):  # type: ignore[no-untyped-def]
            with self._lock:
                mine = descriptor in self._descriptors
            if mine:
                self._record("os.write")
                if slow_write and data.startswith(LINE_PREFIX):
                    time.sleep(SLOW_WRITE_SECONDS)
            return real_write(descriptor, data)

        def spy_fsync(descriptor):  # type: ignore[no-untyped-def]
            with self._lock:
                mine = descriptor in self._descriptors
            if mine:
                self._record("os.fsync")
            return real_fsync(descriptor)

        def spy_close(descriptor):  # type: ignore[no-untyped-def]
            # Descriptor numbers get reused. Forgetting to untrack them here
            # would let an unrelated write be attributed to the diagnostic file.
            with self._lock:
                self._descriptors.discard(descriptor)
            return real_close(descriptor)

        def spy_chmod(path, mode, **kwargs):  # type: ignore[no-untyped-def]
            if self._mine(path):
                self._record("os.chmod")
            return real_chmod(path, mode, **kwargs)

        def spy_glob(this, pattern, **kwargs):  # type: ignore[no-untyped-def]
            if self._mine(this):
                self._record("Path.glob")
            return real_glob(this, pattern, **kwargs)

        def spy_stat(this, **kwargs):  # type: ignore[no-untyped-def]
            if self._mine(this):
                self._record("Path.stat")
            return real_stat(this, **kwargs)

        monkeypatch.setattr(os, "open", spy_open)
        monkeypatch.setattr(os, "write", spy_write)
        monkeypatch.setattr(os, "fsync", spy_fsync)
        monkeypatch.setattr(os, "chmod", spy_chmod)
        monkeypatch.setattr(os, "close", spy_close)
        monkeypatch.setattr(pathlib.Path, "glob", spy_glob)
        monkeypatch.setattr(pathlib.Path, "stat", spy_stat)


def _emit(log: DiagnosticLog, index: int) -> None:
    log.write(
        level="info",
        component="memory.scheduler",
        event="memory_scheduler.idle",
        outcome="succeeded",
        details={"count": index},
    )


def _read_events(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line]


def _wait_for_events(path: Path, expected: int, timeout: float = 10.0) -> list[dict[str, object]]:
    """Wait until expected events show up on disk.

    This does not guess at time; it only keeps the guard from being written as 'never write and \
stay green':
    nothing on disk is a failure, and so is slow output.
    """
    deadline = time.monotonic() + timeout
    events: list[dict[str, object]] = []
    while time.monotonic() < deadline:
        events = _read_events(path)
        if len(events) >= expected:
            return events
        time.sleep(0.01)
    return events


def test_emitting_an_event_does_no_disk_io_on_the_caller_thread(tmp_path, monkeypatch):
    """Red on the old implementation: write() globbed, stated, opened and fsynced on the calling \
thread."""
    log = DiagnosticLog(tmp_path)
    spy = FilesystemSpy(tmp_path / "logs")
    spy.install(monkeypatch)
    caller = threading.get_ident()

    for index in range(8):
        _emit(log, index)
    # Sample before this test reads the file itself, otherwise its own reads are charged to the
    # calling thread.
    on_caller = spy.names_for_thread(caller)

    events = _wait_for_events(log.path, 8)
    # Prove the events really were written first; otherwise 'no I/O on the calling thread' would
    # only mean nothing happened.
    assert len(events) == 8, (
        f"only {len(events)} events reached disk, the guard would be green forever"
    )
    assert spy.count("os.write") >= 1

    assert not on_caller, f"the calling thread is still doing log disk I/O: {on_caller}"
    log.close()


def test_no_fsync_per_event_on_any_thread(tmp_path, monkeypatch):
    """One full flush per event is the most expensive step on this path, and a diagnostic log \
needs no such durability."""
    log = DiagnosticLog(tmp_path)
    spy = FilesystemSpy(tmp_path / "logs")
    spy.install(monkeypatch)

    for index in range(20):
        _emit(log, index)

    events = _wait_for_events(log.path, 20)
    assert len(events) == 20
    assert spy.count("os.fsync") < 20, "still one fsync per event"
    log.close()


def test_directory_scan_is_not_repeated_for_every_event(tmp_path, monkeypatch):
    """_prune() globbed once per event, so a large directory meant a full directory scan per \
event."""
    log = DiagnosticLog(tmp_path)
    spy = FilesystemSpy(tmp_path / "logs")
    spy.install(monkeypatch)

    for index in range(20):
        _emit(log, index)

    assert len(_wait_for_events(log.path, 20)) == 20
    assert spy.count("Path.glob") < 20, "still one directory scan per event"
    log.close()


def test_an_event_emitted_on_the_event_loop_does_not_stall_it(tmp_path, monkeypatch):
    """Red on the old implementation: with the slow disk injected, every event emitted on the \
loop stopped it for 0.5s."""
    log = DiagnosticLog(tmp_path)
    spy = FilesystemSpy(tmp_path / "logs")
    spy.install(monkeypatch, slow_write=True)

    async def scenario() -> float:
        worst = 0.0
        running = True

        async def heartbeat() -> None:
            nonlocal worst
            last = time.monotonic()
            while running:
                await asyncio.sleep(0.01)
                now = time.monotonic()
                worst = max(worst, now - last - 0.01)
                last = now

        beat = asyncio.create_task(heartbeat())
        await asyncio.sleep(0.05)
        for index in range(5):
            _emit(log, index)
            await asyncio.sleep(0)
        await asyncio.sleep(0.05)
        running = False
        await beat
        return worst

    worst = asyncio.run(scenario())
    assert len(_wait_for_events(log.path, 5, timeout=20.0)) == 5
    assert worst < MAX_LOOP_LAG_SECONDS, (
        f"the event loop was stalled by log writes for {worst:.2f}s"
    )
    log.close()


def test_caller_never_waits_on_a_slow_disk_even_when_the_queue_overflows(tmp_path, monkeypatch):
    """Red on the old implementation: the caller waited for a full slow write on every event.

    The new implementation drops when the queue is full and leaves a trace, but it must never \
hold the caller back.
    """
    monkeypatch.setattr(module, "MAX_QUEUED_EVENTS", 4, raising=False)
    log = DiagnosticLog(tmp_path)
    spy = FilesystemSpy(tmp_path / "logs")
    spy.install(monkeypatch, slow_write=True)

    slowest = 0.0
    for index in range(40):
        started = time.monotonic()
        _emit(log, index)
        slowest = max(slowest, time.monotonic() - started)
    assert slowest < 0.05, (
        f"the slowest write() took {slowest:.3f}s; the caller was held by the disk"
    )

    log.close()
    events = _read_events(log.path)
    assert len(events) < 40, "the queue limit did not take effect"
    dropped = [item for item in events if item["event"] == "diagnostic.events_dropped"]
    assert dropped, "events were dropped without leaving any record"
    assert int(dropped[-1]["details"]["count"]) > 0  # type: ignore[index]


def test_dropped_record_says_how_many_and_over_what_window(tmp_path, monkeypatch):
    """A count alone is not enough: whoever reads the log must be able to tell 'this window is \
empty' from 'this window was dropped'.

    Otherwise the next investigation again concludes from absence -- a missing event cannot be \
told from a dropped one.
    The window is timestamp (window end) minus duration_ms (window length);
    both are fields the schema already has, so the field set on disk does not change.
    """
    monkeypatch.setattr(module, "MAX_QUEUED_EVENTS", 4, raising=False)
    log = DiagnosticLog(tmp_path)
    spy = FilesystemSpy(tmp_path / "logs")
    spy.install(monkeypatch, slow_write=True)

    first_drop_at = None
    for index in range(40):
        _emit(log, index)
        if index == 8 and first_drop_at is None:
            first_drop_at = time.time()
    time.sleep(0.3)  # give the dropped time window a measurable width
    for index in range(40, 60):
        _emit(log, index)
    log.close()

    dropped = [
        item for item in _read_events(log.path) if item["event"] == "diagnostic.events_dropped"
    ]
    assert dropped, "events were dropped without leaving any record"
    record = dropped[-1]
    assert int(record["details"]["count"]) > 0  # type: ignore[index]
    assert "duration_ms" in record, "it says how many were dropped, not which span of time"
    window = int(record["duration_ms"])  # type: ignore[arg-type]
    assert window >= 300, f"the {window} ms window does not cover the actual drop interval"
    # The window ends at this record's own timestamp; only the two together locate the interval.
    assert str(record["timestamp"]).endswith("Z")


def test_close_drains_everything_that_was_queued(tmp_path):
    """A clean exit must not drop events."""
    log = DiagnosticLog(tmp_path)
    for index in range(200):
        _emit(log, index)
    log.close()

    events = _read_events(log.path)
    assert len(events) == 200


def test_order_and_line_format_are_unchanged(tmp_path):
    log = DiagnosticLog(tmp_path, session_id="11111111-1111-4111-8111-111111111111")
    for index in range(50):
        _emit(log, index)
    log.close()

    text = log.path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    events = [json.loads(line) for line in text.splitlines()]
    assert [int(item["details"]["count"]) for item in events] == list(range(50))  # type: ignore[index]
    # Still the original compact ASCII JSON: no space separators, no non-ASCII literals.
    for line in text.splitlines():
        assert ", " not in line and '": ' not in line
        assert line == line.encode("ascii", "strict").decode("ascii")
    assert events[0]["schema_version"] == 1
    assert events[0]["service"] == "core"
    assert events[0]["session_id"] == "11111111-1111-4111-8111-111111111111"


def test_events_from_several_threads_keep_each_thread_internal_order(tmp_path):
    log = DiagnosticLog(tmp_path)
    started = threading.Barrier(4)

    def emit_many(base: int) -> None:
        started.wait()
        for index in range(50):
            _emit(log, base + index)

    threads = [threading.Thread(target=emit_many, args=(base,)) for base in (0, 100, 200, 300)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()
    log.close()

    events = _read_events(log.path)
    assert len(events) == 200
    for base in (0, 100, 200, 300):
        mine = [
            int(item["details"]["count"])  # type: ignore[index]
            for item in events
            if base <= int(item["details"]["count"]) < base + 50  # type: ignore[index]
        ]
        assert mine == list(range(base, base + 50))


def test_symlinked_log_file_is_still_refused(tmp_path):
    log = DiagnosticLog(tmp_path)
    log.path.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_text("private")
    log.path.symlink_to(outside)

    _emit(log, 1)

    assert not log.close(timeout=5), "refused writes must not be reported as flushed"
    assert outside.read_text() == "private"


def test_symlinked_log_directory_is_still_refused(tmp_path):
    log = DiagnosticLog(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "logs").symlink_to(elsewhere, target_is_directory=True)

    _emit(log, 1)

    assert not log.close(timeout=5), "refused writes must not be reported as flushed"
    assert list(elsewhere.iterdir()) == []


def test_directory_and_file_permissions_are_still_owner_only(tmp_path):
    """Loosen the permissions first; otherwise the open/mkdir mode already suffices and the \
assertion guards nothing.

    Mutation testing caught it exactly this way in the first round: with both chmod calls \
deleted the test stayed green,
    because a new file is already 0600 and a new directory is already 0700.
    """
    log = DiagnosticLog(tmp_path)
    log.path.parent.mkdir(parents=True)
    log.path.write_text("")
    os.chmod(log.path.parent, 0o755)
    os.chmod(log.path, 0o644)

    _emit(log, 1)
    log.close()

    assert log.path.stat().st_mode & 0o777 == 0o600
    assert log.path.parent.stat().st_mode & 0o777 == 0o700


def test_rotation_still_happens_when_the_file_grows_past_the_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "MAX_FILE_BYTES", 4096)
    log = DiagnosticLog(tmp_path)
    for index in range(250):
        _emit(log, index)
    log.close()

    rotated = log.path.with_suffix(log.path.suffix + ".1")
    assert rotated.exists(), "no rotation after the limit was exceeded"
    assert log.path.stat().st_size <= 4096 + 1024


def test_old_logs_are_still_pruned_on_the_first_write(tmp_path):
    log = DiagnosticLog(tmp_path)
    log.path.parent.mkdir(parents=True)
    stale = log.path.parent / "diagnostic-runtime.jsonl"
    stale.write_text("{}\n")
    old = time.time() - 20 * 24 * 3600
    os.utime(stale, (old, old))

    _emit(log, 1)
    log.close()

    assert not stale.exists(), "expired logs were not pruned"


def test_invalid_input_is_still_rejected_on_the_caller_side(tmp_path):
    """Validation must stay on the calling thread, or the error is swallowed by the background \
thread and the caller never sees it."""
    log = DiagnosticLog(tmp_path)
    with pytest.raises(ValueError, match="diagnostic_detail_key_invalid"):
        log.write(
            level="error",
            component="hy.bridge",
            event="hy.failed",
            outcome="failed",
            details={"prompt": "private"},
        )
    log.close()


def test_a_failing_disk_never_reaches_the_caller(tmp_path, monkeypatch):
    """The caller must not see an exception, but 'nothing happened' is not a pass.

    Checking only that `_emit` does not raise would also let an implementation that throws every \
event away pass.
    So this also requires that the writer thread really tried to persist, and that it did not \
claim a clean shutdown.
    """
    real_open = os.open
    attempts = []

    def refuse(path, flags, mode=0o777, **kwargs):  # type: ignore[no-untyped-def]
        if str(path).endswith("diagnostic-core.jsonl"):
            attempts.append(path)
            raise OSError("disk is full")
        return real_open(path, flags, mode, **kwargs)

    monkeypatch.setattr(os, "open", refuse)
    log = DiagnosticLog(tmp_path)
    for index in range(10):
        _emit(log, index)

    assert not log.close(timeout=5), "not one byte was written, yet close() reported success"
    assert attempts, "the writer thread never even tried to open the file; this guard is empty"
