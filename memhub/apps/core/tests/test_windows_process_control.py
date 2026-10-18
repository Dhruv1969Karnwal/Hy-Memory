"""R-29: on Windows `os.kill` is not a signal, it is an unconditional process kill.

CPython's official docs (os.kill):

    Windows: The `signal.CTRL_C_EVENT` and `signal.CTRL_BREAK_EVENT` signals
    are special signals which can only be sent to console processes which
    share a common console window... Any other value for sig will cause the
    process to be unconditionally killed by the TerminateProcess API, and the
    exit code will be set to sig.

So `parent_watchdog`, the module whose whole job is to avoid a leftover Core, becomes on Windows
the very failure it exists to prevent:

- `process_is_alive(pid)` probes with `os.kill(pid, 0)`, and that one call TerminateProcess'es
  the desktop app itself with the exit code set to sig (0), which on the app side looks like the \
user quit normally.
  The watchdog checks every 2 seconds, so the app is gone 2 seconds after Core starts.
- `_terminate_self()` uses `os.kill(os.getpid(), SIGTERM)`, which also goes through
  TerminateProcess, so uvicorn's SIGTERM handler never runs, and per the comment at the top of the \
module
  that handler is what releases the data directory lock and deletes the runtime receipt. A \
leftover lock makes the next start
  fail straight away with "another MemHub Core owns this directory".

This machine is macOS, so both cases are asserted through injectable platforms: what is verified \
is the branch choice, not
Windows' syscalls themselves. The syscall side can only wait for real hardware and is listed as a
to-be-tested item in `docs/mvp/2026-09-08-r29-windows-survey.md`.
"""

import os
import signal
import threading
import time

from one_memory.parent_watchdog import ParentWatchdog, _terminate_self
from one_memory.runtime import (
    _ERROR_INVALID_PARAMETER,
    _STILL_ACTIVE,
    _windows_process_is_alive,
    process_is_alive,
)


def test_probing_a_pid_on_windows_does_not_go_through_os_kill() -> None:
    """Probing must not fall through to `os.kill` on Windows; that one line kills the process \
being probed."""
    calls: list[tuple[int, int]] = []

    def recording_kill(pid: int, sig: int) -> None:
        calls.append((pid, sig))

    probed: list[int] = []

    def fake_windows_probe(pid: int) -> bool:
        probed.append(pid)
        return True

    alive = process_is_alive(
        4321,
        platform="win32",
        kill=recording_kill,
        windows_probe=fake_windows_probe,
    )

    assert alive is True
    assert probed == [4321]
    assert calls == [], (
        "once probing falls through to os.kill on Windows, the probe hits the host app itself"
    )


def test_probing_a_pid_still_uses_os_kill_everywhere_else() -> None:
    """Green control: on POSIX `os.kill(pid, 0)` is the correct and cheapest probe; do not change \
it in passing."""
    calls: list[tuple[int, int]] = []

    def recording_kill(pid: int, sig: int) -> None:
        calls.append((pid, sig))

    def unreachable_probe(pid: int) -> bool:  # pragma: no cover - must never be called
        raise AssertionError("the Windows branch must not be taken on POSIX")

    alive = process_is_alive(
        4321,
        platform="darwin",
        kill=recording_kill,
        windows_probe=unreachable_probe,
    )

    assert alive is True
    assert calls == [(4321, 0)]


def test_a_real_dead_pid_is_reported_dead_on_this_machine() -> None:
    """Green control: the real semantics must not be lost in a refactor."""
    # Our own process is always alive; a pid that can never exist is always dead.
    assert process_is_alive(os.getpid()) is True
    assert process_is_alive(2**31 - 1) is False


def test_asking_for_shutdown_on_windows_keeps_the_signal_handler_in_play() -> None:
    """Suicide must not go through `os.kill`, or the uvicorn handler never runs and the lock and \
receipt stay behind.

    `signal.raise_signal` triggers the handler inside the process without TerminateProcess,
    and it is the same path on both platforms, so what is asserted here is that nobody os.kills \
itself any more.
    """
    kills: list[tuple[int, int]] = []
    raised: list[int] = []

    _terminate_self(
        kill=lambda pid, sig: kills.append((pid, sig)),
        raise_signal=lambda sig: raised.append(sig),
    )

    assert raised == [signal.SIGTERM]
    assert kills == [], "os.kill on yourself means giving up a graceful exit on Windows"


def test_the_watchdog_still_fires_once_and_only_once() -> None:
    """Green control: both places above sit on the watchdog's path, and its own behavior must not \
be dragged along."""
    shutdowns: list[str] = []
    watchdog = ParentWatchdog(
        parent_pid=4321,
        is_alive=lambda pid: False,
        request_shutdown=lambda: shutdowns.append("asked"),
    )

    assert watchdog.check_once() is True
    assert watchdog.check_once() is True
    assert shutdowns == ["asked"]


def test_the_handler_runs_even_though_the_watchdog_is_on_another_thread() -> None:
    """The watchdog runs on its own thread, and a signal only counts if it really raises the \
handler.

    `raise_signal`'s C-level handler runs on the calling thread, while the Python callback runs on \
the main thread.
    What this measures is exactly that asking for shutdown from the watchdog thread reaches the \
main thread's handler; otherwise
    switching to raise_signal only trades one place where it does not work for another.
    """
    seen: list[int] = []
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda number, frame: seen.append(number))
    try:
        thread = threading.Thread(target=_terminate_self)
        thread.start()
        thread.join(timeout=5)
        # Python only runs signal callbacks on the main thread, so the time slice has to be yielded
        # to wait for it.
        deadline = time.monotonic() + 5
        while not seen and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        signal.signal(signal.SIGTERM, previous)

    assert seen == [signal.SIGTERM]


class FakeWin32:
    """Answering four questions is enough: can it open, what is the error code, what is the exit \
code, was it closed."""

    def __init__(
        self,
        *,
        handle: int,
        last_error: int = 0,
        exit_code: int | None = _STILL_ACTIVE,
    ) -> None:
        self._handle = handle
        self._last_error = last_error
        self._exit_code = exit_code
        self.closed: list[int] = []

    def open_process(self, pid: int) -> int:
        return self._handle

    def last_error(self) -> int:
        return self._last_error

    def exit_code(self, handle: int) -> int | None:
        return self._exit_code

    def close(self, handle: int) -> None:
        self.closed.append(handle)


def test_a_windows_process_that_already_exited_is_not_read_as_alive() -> None:
    """The handle of an exited process still opens, so looking only at 'it opened' reports the \
dead as alive.

    This one came out of a mutation run: with the exit code check removed the previous version was \
all green. Erroring here means
    the watchdog never fires, so on Windows not one leftover Core case would be caught.
    """
    api = FakeWin32(handle=99, exit_code=0)

    assert _windows_process_is_alive(4321, api) is False
    assert api.closed == [99], "the handle must be closed, otherwise probing leaks handles itself"


def test_a_windows_process_still_running_is_read_as_alive() -> None:
    api = FakeWin32(handle=99, exit_code=_STILL_ACTIVE)

    assert _windows_process_is_alive(4321, api) is True
    assert api.closed == [99]


def test_a_pid_windows_says_is_not_a_pid_is_read_as_gone() -> None:
    api = FakeWin32(handle=0, last_error=_ERROR_INVALID_PARAMETER)

    assert _windows_process_is_alive(4321, api) is False
    assert api.closed == [], "a handle that was never opened must not be closed"


def test_a_process_we_are_not_allowed_to_open_is_read_as_there() -> None:
    """The same reading as the POSIX branch, which takes PermissionError to mean 'it is there'."""
    _ERROR_ACCESS_DENIED = 5
    api = FakeWin32(handle=0, last_error=_ERROR_ACCESS_DENIED)

    assert _windows_process_is_alive(4321, api) is True


def test_an_exit_code_we_cannot_read_is_not_taken_as_death() -> None:
    """A handle that opened means the process object exists; being unable to read an exit code is \
no evidence that it died."""
    api = FakeWin32(handle=99, exit_code=None)

    assert _windows_process_is_alive(4321, api) is True
    assert api.closed == [99]
