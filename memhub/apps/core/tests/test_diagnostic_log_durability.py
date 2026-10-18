"""The write thread's report of "persisted" has to be true.

After the first version moved writes to a background thread, a new class of lie appeared: the
batch was already taken off the queue, the write raised and was swallowed, `_drained` was set
anyway -- so `flush()` returned True while the disk held nothing. Faults are injected here to \
pin these as regression guards.

The shared shape of these assertions is: **never pass off "I did my best" as "the data is on \
disk".**
"""

from __future__ import annotations

import json
import os
import threading
import time

import pytest

import one_memory.diagnostic_log as module
from one_memory.diagnostic_log import DiagnosticLog


def _emit(log: DiagnosticLog, index: int) -> None:
    log.write(
        level="info",
        component="memory.scheduler",
        event="memory_scheduler.idle",
        outcome="succeeded",
        details={"count": index},
    )


def _lines(log: DiagnosticLog) -> list[str]:
    if not log.path.exists():
        return []
    return [line for line in log.path.read_text(encoding="utf-8").splitlines() if line]


def _events(log: DiagnosticLog) -> list[dict[str, object]]:
    return [json.loads(line) for line in _lines(log)]


class _BrokenDisk:
    """Make the diagnostic file unwritable, and fixable again at will."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.failing = True
        self.opened = 0
        real_open = os.open

        def guarded(path, flags, mode=0o777, **kwargs):  # type: ignore[no-untyped-def]
            if str(path).endswith("diagnostic-core.jsonl"):
                self.opened += 1
                if self.failing:
                    raise OSError("disk is full")
            return real_open(path, flags, mode, **kwargs)

        monkeypatch.setattr(os, "open", guarded)


def test_flush_does_not_claim_success_when_the_write_failed(tmp_path, monkeypatch):
    """The old implementation fails: batch taken, write raised and swallowed, still marked \
drained."""
    disk = _BrokenDisk(monkeypatch)
    log = DiagnosticLog(tmp_path)
    for index in range(5):
        _emit(log, index)

    persisted = log.flush(timeout=3)

    assert disk.opened > 0, "the write thread never tried to write, which empties this guard"
    assert _lines(log) == [], "the disk should hold nothing"
    assert not persisted, "the write failed, yet flush() reported it persisted"
    log.close(timeout=3)


def test_a_failed_batch_is_not_thrown_away_before_the_disk_recovers(tmp_path, monkeypatch):
    """Events taken while the disk was broken are not treated as written; once it heals they are \
rewritten or accounted for."""
    disk = _BrokenDisk(monkeypatch)
    log = DiagnosticLog(tmp_path)
    for index in range(5):
        _emit(log, index)
    log.flush(timeout=3)

    disk.failing = False
    for index in range(5, 10):
        _emit(log, index)
    assert log.close(timeout=10)

    events = _events(log)
    kept = [
        int(item["details"]["count"])  # type: ignore[index]
        for item in events
        if item["event"] == "memory_scheduler.idle"
    ]
    lost = sum(
        int(item["details"]["count"])  # type: ignore[index]
        for item in events
        if item["event"] == "diagnostic.events_dropped"
    )
    assert kept, "after the disk recovered, not even new events were written"
    # Either all 10 are present, or the missing ones are explicitly accounted for -- no silent loss.
    assert len(kept) + lost == 10, (
        f"of 10 only {len(kept)} kept + {lost} accounted for were explained"
    )


def test_a_short_write_never_leaves_a_half_line_on_disk(tmp_path, monkeypatch):
    """os.write can write only part of a buffer. Ignore the return value and half-written \
JSON appears."""
    real_write = os.write
    state = {"short": True}

    def short(descriptor, data):  # type: ignore[no-untyped-def]
        if state["short"] and data.startswith(b'{"schema_version"') and len(data) > 24:
            state["short"] = False
            return real_write(descriptor, data[:20])
        return real_write(descriptor, data)

    monkeypatch.setattr(os, "write", short)
    log = DiagnosticLog(tmp_path)
    for index in range(6):
        _emit(log, index)
        log.flush(timeout=3)
    assert log.close(timeout=5)

    for line in _lines(log):
        json.loads(line)  # truncated JSON blows up here
    assert len(_lines(log)) == 6, "the line count is wrong after a short write"


def test_a_clean_shutdown_actually_reaches_the_disk(tmp_path, monkeypatch):
    """A promised fsync at shutdown has to actually happen.

    The old implementation only fsynced if the write thread happened to take one more batch
    after observing closed. A normal shutdown drains the last entry in flush() first, so by the
    time close() arrives there is no batch left, and that fsync never happened.
    """
    synced = []
    real_fsync = os.fsync

    def counting(descriptor):  # type: ignore[no-untyped-def]
        synced.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", counting)
    log = DiagnosticLog(tmp_path)
    _emit(log, 1)
    assert log.flush(timeout=3)  # the order of a normal shutdown: drain first
    assert not synced, "the drain phase must not fsync, that was the per-event fsync habit"

    assert log.close(timeout=5)
    assert synced, "a clean exit never fsynced, so the durability promise in STATUS is empty"


def test_close_does_not_claim_success_while_the_writer_is_still_stuck(tmp_path, monkeypatch):
    """After the first close times out, the second must not succeed because closing was flagged."""
    release = threading.Event()
    real_write = os.write

    def blocking(descriptor, data):  # type: ignore[no-untyped-def]
        if data.startswith(b'{"schema_version"'):
            release.wait(30)
        return real_write(descriptor, data)

    monkeypatch.setattr(os, "write", blocking)
    log = DiagnosticLog(tmp_path)
    _emit(log, 1)
    time.sleep(0.1)

    try:
        assert not log.close(timeout=0.2), (
            "the write thread is still stuck, the first close must not report success"
        )
        assert not log.close(timeout=0.2), (
            "the write thread is still stuck, the second close even less so"
        )
    finally:
        release.set()
    assert log.close(timeout=10), "once released it should finish normally"


def test_every_emitted_event_is_either_on_disk_or_counted_as_dropped(tmp_path, monkeypatch):
    """Keeps the books: kept + dropped == emitted. Asserting only "dropped > 0" lets rows go \
missing."""
    monkeypatch.setattr(module, "MAX_QUEUED_EVENTS", 4, raising=False)
    real_write = os.write

    def slow(descriptor, data):  # type: ignore[no-untyped-def]
        if data.startswith(b'{"schema_version"'):
            time.sleep(0.05)
        return real_write(descriptor, data)

    monkeypatch.setattr(os, "write", slow)
    log = DiagnosticLog(tmp_path)
    emitted = 120
    for index in range(emitted):
        _emit(log, index)
    assert log.close(timeout=30)

    events = _events(log)
    kept = sum(1 for item in events if item["event"] == "memory_scheduler.idle")
    dropped = sum(
        int(item["details"]["count"])  # type: ignore[index]
        for item in events
        if item["event"] == "diagnostic.events_dropped"
    )
    assert dropped > 0, "the queue cap never triggered, so this round tested nothing"
    assert kept + dropped == emitted, (
        f"emitted {emitted}, only {kept} kept + {dropped} dropped were accounted for"
    )


def test_the_directory_is_scanned_once_per_throttle_window_not_once_per_batch(
    tmp_path, monkeypatch
):
    """Uses flush to force one batch per event, so batching cannot carry a weak assertion."""
    scans = []
    real_glob = type(tmp_path).glob
    root = (tmp_path / "logs").resolve()

    def counting(this, pattern, **kwargs):  # type: ignore[no-untyped-def]
        if this.resolve() == root:
            scans.append(pattern)
        return real_glob(this, pattern, **kwargs)

    monkeypatch.setattr(type(tmp_path), "glob", counting)
    log = DiagnosticLog(tmp_path)
    for index in range(20):
        _emit(log, index)
        assert log.flush(timeout=3)

    assert len(_lines(log)) == 20, "not every event landed on its own, so this guard means nothing"
    assert len(scans) == 1, (
        f"20 separate batches scanned the directory {len(scans)} times, throttling is off"
    )
    log.close(timeout=5)


def test_no_fsync_happens_before_shutdown_even_across_many_batches(tmp_path, monkeypatch):
    """Also split into 20 separate batches with flush: not one fsync before shutdown."""
    synced = []
    real_fsync = os.fsync

    def counting(descriptor):  # type: ignore[no-untyped-def]
        synced.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", counting)
    log = DiagnosticLog(tmp_path)
    for index in range(20):
        _emit(log, index)
        assert log.flush(timeout=3)

    assert len(_lines(log)) == 20
    assert synced == [], f"{len(synced)} fsyncs happened before shutdown"
    assert log.close(timeout=5)
    assert len(synced) == 1, f"shutdown should fsync exactly once, actually {len(synced)}"
