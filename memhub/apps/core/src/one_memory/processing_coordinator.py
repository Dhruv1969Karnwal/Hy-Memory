"""Continuous recovery monitor for model processing.

The monitor is intentionally retained.  It observes the global config final
state plus ``memory_tasks`` and asks the supervisor to recover work; it owns no
second request/receipt state machine.
"""

import asyncio
import time
from collections.abc import Hashable

from one_memory.memory_processing import MemoryProcessingSupervisor
from one_memory.memory_server.config_state import ConfigStore


class ProcessingCoordinator:
    RESCAN_SECONDS = 60.0
    POLL_SECONDS = 2.0

    def __init__(self, config: ConfigStore, supervisor: MemoryProcessingSupervisor):
        self.config = config
        self.supervisor = supervisor
        self._stopping = False
        self._monitor: asyncio.Task[None] | None = None
        self._error: str | None = None
        self._observed: tuple[Hashable, ...] | None = None
        self._last_scan = 0.0
        self._clock = time.monotonic

    async def recover(self) -> None:
        """Start the one process-owned monitor, whether permission is on or off."""

        if self._stopping:
            return
        if self._monitor is None or self._monitor.done():
            self._monitor = asyncio.create_task(
                self._monitor_loop(),
                name="one-memory-processing-monitor",
            )

    def _identity(self) -> tuple[Hashable, ...]:
        counts = self.supervisor.tasks.counts()
        return (
            self.config.model_content_allowed(),
            tuple(sorted((str(status), count) for status, count in counts.items())),
        )

    async def _tick(self) -> None:
        identity = await asyncio.to_thread(self._identity)
        now = self._clock()
        if identity[0] and (
            identity != self._observed or now - self._last_scan >= self.RESCAN_SECONDS
        ):
            await self.supervisor.resume_allowed()
            self._last_scan = now
            identity = await asyncio.to_thread(self._identity)
        self._observed = identity
        self._error = None

    async def _monitor_loop(self) -> None:
        while not self._stopping:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._error = "model_content_recovery_blocked"
            await asyncio.sleep(self.POLL_SECONDS)

    def status(self) -> dict[str, object]:
        return {
            "monitoring": self._monitor is not None and not self._monitor.done(),
            "allowed": self._observed[0] if self._observed else None,
            "error": self._error,
        }

    async def shutdown(self) -> None:
        self._stopping = True
        if self._monitor is not None:
            self._monitor.cancel()
            await asyncio.gather(self._monitor, return_exceptions=True)
