import asyncio
from types import SimpleNamespace

from one_memory.memory_server.config_state import ConfigStore
from one_memory.processing_coordinator import ProcessingCoordinator


class Tasks:
    def __init__(self) -> None:
        self.value = {}

    def counts(self):
        return dict(self.value)


def test_monitor_is_retained_and_observes_config_plus_tasks(tmp_path) -> None:
    async def scenario() -> None:
        config = ConfigStore(tmp_path)
        tasks = Tasks()
        resumed = 0

        async def resume_allowed() -> None:
            nonlocal resumed
            resumed += 1

        supervisor = SimpleNamespace(tasks=tasks, resume_allowed=resume_allowed)
        coordinator = ProcessingCoordinator(config, supervisor)
        coordinator.POLL_SECONDS = 0.01
        coordinator.RESCAN_SECONDS = 60
        try:
            await coordinator.recover()
            await asyncio.sleep(0.03)
            assert coordinator.status()["monitoring"] is True
            assert resumed == 0

            config.set_model_content_allowed(True)
            await asyncio.sleep(0.03)
            assert resumed == 1

            tasks.value = {"pending": 1}
            await asyncio.sleep(0.03)
            assert resumed == 2

            config.set_model_content_allowed(False)
            tasks.value = {"pending": 2}
            await asyncio.sleep(0.03)
            assert resumed == 2
            assert coordinator.status()["allowed"] is False
        finally:
            await coordinator.shutdown()

        assert coordinator.status()["monitoring"] is False

    asyncio.run(scenario())
