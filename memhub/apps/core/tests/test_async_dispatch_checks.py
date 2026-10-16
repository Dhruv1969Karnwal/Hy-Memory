import asyncio
import threading

import pytest

from one_memory.model_profiles import ModelProfilePatch
from tests.test_model_processing_permissions import Harness


def test_quiet_arrival_guard_is_off_loop_and_5000_events_coalesce(tmp_path, monkeypatch):
    h = Harness(tmp_path)
    supervisor = h.supervisor()
    entered, release = threading.Event(), threading.Event()
    checked = []
    original = supervisor._guard_records

    async def scenario():
        loop_thread = threading.get_ident()

        def guard(records, *args, **kwargs):
            checked.append(threading.get_ident())
            assert threading.get_ident() != loop_thread, "arrival parsed permission on the loop"
            entered.set()
            assert release.wait(5)
            return original(records, *args, **kwargs)

        def credentials(*_args, **_kwargs):
            pytest.fail("denied preparation reached credential/profile resolution")

        monkeypatch.setattr(supervisor, "_guard_records", guard)
        monkeypatch.setattr(h.profiles, "active", credentials)
        monkeypatch.setattr(h.profiles, "resolve_snapshot_secret", credentials)
        try:
            for _ in range(5000):
                supervisor.enqueue_evidence(h.record)
            assert checked == []
            assert len(supervisor._debounces) == 1
            assert await asyncio.to_thread(entered.wait, 3)
            # This callback must run while the actual permission reader is blocked.
            responsive = asyncio.Event()
            asyncio.get_running_loop().call_soon(responsive.set)
            await asyncio.wait_for(responsive.wait(), 0.5)
            assert checked == [checked[0]]
            release.set()
            await asyncio.gather(*tuple(supervisor._debounces.values()))
            assert supervisor._preparations == {}
            assert h.tasks.list() == []
            assert len(supervisor._queue) == 0 and supervisor._workers == {}
            assert h.factory_calls == [] and h.bridge.calls == []
        finally:
            release.set()
            await supervisor.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("stop_mode", ["cancel", "shutdown"])
def test_live_preparation_cancel_joins_thread_without_late_dispatch(
    tmp_path, monkeypatch, stop_mode
):
    h = Harness(tmp_path)
    h.grant()
    supervisor = h.supervisor()
    entered, release = threading.Event(), threading.Event()
    original = supervisor._prepare_conversation_work
    completed = []

    def blocked(seed, **kwargs):
        entered.set()
        assert release.wait(5)
        result = original(seed, **kwargs)
        completed.append(True)
        return result

    monkeypatch.setattr(supervisor, "_prepare_conversation_work", blocked)

    async def scenario():
        stopping = None
        preparation = None
        try:
            supervisor.enqueue_evidence(h.record)
            assert await asyncio.to_thread(entered.wait, 3)
            preparation = next(iter(supervisor._preparations.values()))
            if stop_mode == "cancel":
                preparation.cancel()
                await asyncio.sleep(0.02)
                preparation.cancel()
            else:
                stopping = asyncio.create_task(supervisor.shutdown())
            await asyncio.sleep(0.02)
            assert not preparation.done(), "cancel detached the actual preparation thread"
            if stopping is not None:
                assert not stopping.done()
        finally:
            release.set()
            if preparation is not None:
                await asyncio.gather(preparation, return_exceptions=True)
            if stopping is not None:
                await stopping
            await supervisor.shutdown()
        assert completed == [True]
        assert supervisor._preparations == {} and supervisor._debounces == {}
        assert len(supervisor._queue) == 0 and supervisor._workers == {}
        assert h.factory_calls == [] and h.bridge.calls == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "change", ["model-revoked", "source-revoked", "profile-changed", "regranted-other-input"]
)
def test_quiet_preparation_rechecks_permission_before_credentials_or_durable_work(
    tmp_path, monkeypatch, change
):
    h = Harness(tmp_path)
    receipt = h.grant()
    supervisor = h.supervisor()
    entered, release = threading.Event(), threading.Event()
    original = supervisor._prepare_conversation_work
    guarded = []
    original_guard = supervisor._guard_records

    def blocked(seed, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(seed, **kwargs)

    def guard(*args, **kwargs):
        guarded.append(True)
        return original_guard(*args, **kwargs)

    def credentials(*_args, **_kwargs):
        pytest.fail("denied preparation reached credential/profile resolution")

    monkeypatch.setattr(supervisor, "_prepare_conversation_work", blocked)
    monkeypatch.setattr(supervisor, "_guard_records", guard)

    async def scenario():
        try:
            supervisor.enqueue_evidence(h.record)
            assert await asyncio.to_thread(entered.wait, 3)
            if change in {"model-revoked", "regranted-other-input"}:
                h.permissions.revoke(
                    user_id="local", expected_revision=receipt.revision, user_confirmed=True
                )
                if change == "regranted-other-input":
                    h.grant((h.put("other", conversation="other"),))
            elif change == "source-revoked":
                h.sources.revoke(user_id="local", source_id="cursor")
            else:
                h.profiles.update(
                    h.profile.profile_id,
                    ModelProfilePatch(expected_revision=h.profile.revision, model="changed"),
                    h.secrets,
                )
            monkeypatch.setattr(h.profiles, "active", credentials)
            monkeypatch.setattr(h.profiles, "resolve_snapshot_secret", credentials)
            release.set()
            await asyncio.gather(*tuple(supervisor._debounces.values()))
            assert guarded == [True]
            assert h.tasks.list() == []
            assert h.factory_calls == [] and h.bridge.calls == []
            assert len(supervisor._queue) == 0 and supervisor._workers == {}
        finally:
            release.set()
            await supervisor.shutdown()

    asyncio.run(scenario())


def test_cancelling_quiet_timer_does_not_start_any_permission_reader(tmp_path, monkeypatch):
    h = Harness(tmp_path)
    h.grant()
    supervisor = h.supervisor()
    checks = []
    original = supervisor._guard_records

    def guard(*args, **kwargs):
        checks.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(supervisor, "_guard_records", guard)

    async def scenario():
        try:
            supervisor.enqueue_evidence(h.record)
            timer = next(iter(supervisor._debounces.values()))
            await asyncio.sleep(0)
            timer.cancel()
            await asyncio.gather(timer, return_exceptions=True)
            assert checks == []
            assert supervisor._debounces == {} and supervisor._preparations == {}
            assert h.tasks.list() == [] and h.factory_calls == [] and h.bridge.calls == []
        finally:
            await supervisor.shutdown()

    asyncio.run(scenario())


def test_dispatch_permission_check_denies_before_queue_after_classification(tmp_path, monkeypatch):
    h = Harness(tmp_path)
    h.grant()
    supervisor = h.supervisor()
    work = supervisor._classify(supervisor.prepare_conversation(h.record))
    checked = []

    def reject(task):
        checked.append(task.task_id)
        return False

    async def scenario():
        monkeypatch.setattr(supervisor, "_guard_task", reject)
        try:
            supervisor._schedule_tasks(work)
            assert len(checked) == 1
            assert len(supervisor._queue) == 0 and supervisor._workers == {}
            assert h.factory_calls == []
        finally:
            await supervisor.shutdown()

    asyncio.run(scenario())


def test_dispatch_check_does_not_queue_work_after_shutdown(tmp_path, monkeypatch):
    h = Harness(tmp_path)
    h.grant()
    supervisor = h.supervisor()
    work = supervisor._classify(supervisor.prepare_conversation(h.record))
    checked = []

    def allowed(task):
        checked.append(task.task_id)
        return True

    monkeypatch.setattr(supervisor, "_guard_task", allowed)

    async def scenario():
        stopping = asyncio.create_task(supervisor.shutdown())
        try:
            await stopping
            supervisor._schedule_tasks(work)
        finally:
            await supervisor.shutdown()
        assert len(supervisor._queue) == 0 and supervisor._workers == {}
        assert checked == []
        assert h.factory_calls == []

    asyncio.run(scenario())
