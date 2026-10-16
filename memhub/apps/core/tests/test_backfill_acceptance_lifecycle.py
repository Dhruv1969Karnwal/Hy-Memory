"""A processing acknowledgement belongs to one consent revision and live worker."""

import asyncio

import pytest

from one_memory.hy_memory_bridge import HyMemoryBridgeError
from tests.test_model_processing_permissions import Harness


def test_changed_consent_does_not_acknowledge_an_older_running_request(tmp_path, monkeypatch):
    async def scenario():
        h = Harness(tmp_path)
        first = h.grant()
        supervisor = h.supervisor()
        started = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def backfill(*, expected_permission_revision=None, on_first_scheduled=None):
            calls.append(expected_permission_revision)
            if on_first_scheduled is not None:
                on_first_scheduled()
            started.set()
            await release.wait()

        monkeypatch.setattr(supervisor, "backfill_off_thread", backfill)
        try:
            await asyncio.wait_for(
                supervisor.accept_backfill_off_thread(expected_permission_revision=first.revision),
                timeout=5,
            )
            await asyncio.wait_for(started.wait(), timeout=5)
            second = h.grant()
            await asyncio.wait_for(
                supervisor.accept_backfill_off_thread(expected_permission_revision=second.revision),
                timeout=5,
            )
            started.clear()
            await asyncio.wait_for(started.wait(), timeout=5)
            assert calls == [first.revision, second.revision]
        finally:
            release.set()
            await supervisor.shutdown()

    asyncio.run(scenario())


def test_late_failure_is_observed_in_processing_status(tmp_path, monkeypatch):
    async def scenario():
        h = Harness(tmp_path)
        receipt = h.grant()
        supervisor = h.supervisor()
        release = asyncio.Event()

        async def backfill(*, expected_permission_revision=None, on_first_scheduled=None):
            if on_first_scheduled is not None:
                on_first_scheduled()
            await release.wait()
            raise HyMemoryBridgeError("model_content_source_permission_changed")

        monkeypatch.setattr(supervisor, "backfill_off_thread", backfill)
        try:
            await supervisor.accept_backfill_off_thread(
                expected_permission_revision=receipt.revision
            )
            assert supervisor.accepted_backfill_status() == ("running", None)
            release.set()
            await supervisor._live_backfill
            assert supervisor.accepted_backfill_status() == (
                "failed",
                "model_content_source_permission_changed",
            )
        finally:
            release.set()
            await supervisor.shutdown()

    asyncio.run(scenario())


def test_same_consent_concurrent_requests_share_one_worker(tmp_path, monkeypatch):
    async def scenario():
        h = Harness(tmp_path)
        receipt = h.grant()
        supervisor = h.supervisor()
        entered, acknowledge, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = []

        async def backfill(*, expected_permission_revision=None, on_first_scheduled=None):
            calls.append(expected_permission_revision)
            entered.set()
            await acknowledge.wait()
            if on_first_scheduled is not None:
                on_first_scheduled()
            await finish.wait()

        monkeypatch.setattr(supervisor, "backfill_off_thread", backfill)
        first = asyncio.create_task(
            supervisor.accept_backfill_off_thread(expected_permission_revision=receipt.revision)
        )
        try:
            await entered.wait()
            second = asyncio.create_task(
                supervisor.accept_backfill_off_thread(expected_permission_revision=receipt.revision)
            )
            await asyncio.sleep(0)
            acknowledge.set()
            await asyncio.wait_for(asyncio.gather(first, second), 2)
            assert calls == [receipt.revision]
        finally:
            acknowledge.set()
            finish.set()
            await supervisor.shutdown()

    asyncio.run(scenario())


def test_background_failure_is_reported_after_acceptance_and_can_retry(tmp_path, monkeypatch):
    async def scenario():
        h = Harness(tmp_path)
        receipt = h.grant()
        supervisor = h.supervisor()

        async def broken(**_kwargs):
            raise RuntimeError("secret or provider body must not be returned")

        monkeypatch.setattr(supervisor, "backfill_off_thread", broken)
        try:
            await supervisor.accept_backfill_off_thread(
                expected_permission_revision=receipt.revision
            )
            await supervisor._live_backfill
            assert supervisor.accepted_backfill_status() == (
                "failed",
                "model_content_processing_unavailable",
            )

            async def complete(**kwargs):
                callback = kwargs.get("on_first_scheduled")
                if callback is not None:
                    callback()

            monkeypatch.setattr(supervisor, "backfill_off_thread", complete)
            await supervisor.accept_backfill_off_thread(
                expected_permission_revision=receipt.revision
            )
            await supervisor._live_backfill
            assert supervisor.accepted_backfill_status() == ("done", None)
        finally:
            await supervisor.shutdown()

    asyncio.run(scenario())


def test_shutdown_rejects_new_processing_acceptance(tmp_path):
    async def scenario():
        h = Harness(tmp_path)
        receipt = h.grant()
        supervisor = h.supervisor()
        await supervisor.shutdown()
        with pytest.raises(HyMemoryBridgeError, match="model_content_processing_unavailable"):
            await supervisor.accept_backfill_off_thread(
                expected_permission_revision=receipt.revision
            )
        assert h.tasks.list() == []
        assert h.factory_calls == []

    asyncio.run(scenario())


def test_health_reports_late_explicit_processing_failure(tmp_path, monkeypatch):
    from httpx import ASGITransport, AsyncClient

    from one_memory.api import create_app
    from one_memory.config import Settings

    monkeypatch.setattr("one_memory.model_presets._preset_data", lambda: None)

    async def scenario():
        app = create_app(Settings(_env_file=None, data_dir=tmp_path, imports_enabled=False))
        release = asyncio.Event()
        async with app.router.lifespan_context(app):
            await app.state.memory_backfill
            supervisor = app.state.memory_processing_supervisor
            supervisor._permission_guard = None

            async def failed_later(**kwargs):
                callback = kwargs.get("on_first_scheduled")
                if callback is not None:
                    callback()
                await release.wait()
                raise HyMemoryBridgeError("model_content_source_permission_changed")

            monkeypatch.setattr(supervisor, "backfill_off_thread", failed_later)
            try:
                await supervisor.accept_backfill_off_thread()
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    running = (await client.get("/api/v1/health")).json()
                    assert running["memory_backfill"] == "running"
                    release.set()
                    await supervisor._live_backfill
                    failed = (await client.get("/api/v1/health")).json()
                    assert failed["memory_backfill"] == "failed"
                    assert (
                        failed["memory_backfill_error"] == "model_content_source_permission_changed"
                    )
            finally:
                release.set()

    asyncio.run(scenario())


def test_completed_caller_does_not_cancel_accepted_background_work(tmp_path, monkeypatch):
    async def scenario():
        h = Harness(tmp_path)
        receipt = h.grant()
        supervisor = h.supervisor()
        entered, release = asyncio.Event(), asyncio.Event()

        async def backfill(**kwargs):
            entered.set()
            await release.wait()
            callback = kwargs.get("on_first_scheduled")
            if callback is not None:
                callback()

        monkeypatch.setattr(supervisor, "backfill_off_thread", backfill)
        caller = asyncio.create_task(
            supervisor.accept_backfill_off_thread(expected_permission_revision=receipt.revision)
        )
        try:
            await asyncio.wait_for(caller, timeout=2)
            assert not supervisor._live_backfill.done()
            await asyncio.wait_for(entered.wait(), timeout=5)
            release.set()
            await supervisor._live_backfill
            assert supervisor.accepted_backfill_status() == ("done", None)
        finally:
            release.set()
            await supervisor.shutdown()

    asyncio.run(scenario())
