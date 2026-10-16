import asyncio

import pytest

from one_memory.hy_memory_bridge import HyMemoryBridgeError
from one_memory.processing_coordinator import ProcessingCoordinator
from one_memory.processing_requests import ProcessingRequestStore
from tests.test_continuous_processing_bridge import enable, put_after_policy, transition
from tests.test_model_processing_permissions import h as h  # noqa: F401


def coordinator(h, monkeypatch):
    supervisor = h.supervisor()
    scheduled = []

    async def schedule(work):
        scheduled.extend(work)

    monkeypatch.setattr(supervisor, "_schedule_tasks_off_thread", schedule)
    return ProcessingCoordinator(ProcessingRequestStore(h.database), supervisor), scheduled


def test_restart_replays_lost_callback_into_same_durable_task_queue(h, monkeypatch):
    policy = enable(h)
    fresh = put_after_policy(h, "committed-before-crash")
    owner, scheduled = coordinator(h, monkeypatch)

    async def run():
        try:
            await owner._continuous_tick()
            assert scheduled
            task_ids = {task.task_id for task in h.tasks.list()}
            assert len(task_ids) == 1
            assert {
                ref["evidence_id"]
                for task in h.tasks.list()
                for ref in h.tasks.inputs(task.task_id)
            } == {fresh.evidence_id}
            assert owner.continuous_status()["observed_seq"] > policy.after_seq
            await owner._continuous_tick()
            assert {task.task_id for task in h.tasks.list()} == task_ids
            restarted, _ = coordinator(h, monkeypatch)
            try:
                await restarted._continuous_tick()
                assert {task.task_id for task in h.tasks.list()} == task_ids
            finally:
                await restarted.shutdown()
                await restarted.supervisor.shutdown()
        finally:
            await owner.shutdown()
            await owner.supervisor.shutdown()

    asyncio.run(run())


def test_new_journal_event_and_source_regrant_wake_replay_without_legacy_request(h, monkeypatch):
    from one_memory.imports.sources.registry import RootCandidateKind

    enable(h)
    owner, scheduled = coordinator(h, monkeypatch)

    async def run():
        try:
            await owner._continuous_tick()
            assert not scheduled
            fresh = put_after_policy(h, "new")
            h.sources.revoke(user_id="local", source_id="cursor")
            await owner._continuous_tick()
            assert not scheduled
            h.sources.grant(
                user_id="local",
                source_id="cursor",
                root=h.root / "synthetic-source",
                candidate_kind=RootCandidateKind.DEFAULT,
            )
            await owner._continuous_tick()
            assert scheduled
            assert h.tasks.inputs(scheduled[-1].task_id)[0]["evidence_id"] == fresh.evidence_id
            with h.evidence._connect() as db:
                assert db.execute("SELECT count(*) FROM processing_requests").fetchone()[0] == 0
        finally:
            await owner.shutdown()
            await owner.supervisor.shutdown()

    asyncio.run(run())


def test_activation_hook_and_startup_recovery_start_single_monitor_then_stop(h, monkeypatch):
    policy = enable(h)
    put_after_policy(h, "new")
    owner, scheduled = coordinator(h, monkeypatch)

    async def run():
        try:
            await owner.recover()
            worker = owner._continuous_worker
            assert worker is not None
            await owner.resume_continuous(expected_policy_revision=policy.revision)
            assert owner._continuous_worker is worker
            for _ in range(100):
                if scheduled:
                    break
                await asyncio.sleep(0.01)
            assert scheduled
        finally:
            await owner.shutdown()
            await owner.supervisor.shutdown()
        assert worker.done()
        with pytest.raises(HyMemoryBridgeError):
            await owner.resume_continuous(expected_policy_revision=policy.revision)

    asyncio.run(run())


def test_pause_refuses_recovery_without_fallback_and_resume_replays(h, monkeypatch):
    h.grant()
    enable(h)
    fresh = put_after_policy(h, "new")
    owner, scheduled = coordinator(h, monkeypatch)
    transition(h, "pause")

    async def run():
        try:
            await owner.recover()
            assert not scheduled and owner._continuous_worker is None
            with pytest.raises(HyMemoryBridgeError):
                await owner.resume_continuous()
            transition(h, "resume")
            await owner._continuous_tick()
            assert scheduled
            assert h.tasks.inputs(scheduled[-1].task_id)[0]["evidence_id"] == fresh.evidence_id
        finally:
            await owner.shutdown()
            await owner.supervisor.shutdown()

    asyncio.run(run())


def test_failed_preparation_does_not_advance_observed_watermark(h, monkeypatch):
    enable(h)
    put_after_policy(h, "new")
    owner, _ = coordinator(h, monkeypatch)
    original = owner.supervisor.resume_continuous

    async def fail(**kwargs):
        raise RuntimeError("synthetic failed preparation")

    monkeypatch.setattr(owner.supervisor, "resume_continuous", fail)

    async def run():
        try:
            with pytest.raises(RuntimeError):
                await owner._continuous_tick()
            assert owner.continuous_status()["observed_seq"] is None
            monkeypatch.setattr(owner.supervisor, "resume_continuous", original)
            await owner._continuous_tick()
            assert owner.continuous_status()["observed_seq"] is not None
        finally:
            await owner.shutdown()
            await owner.supervisor.shutdown()

    asyncio.run(run())


def test_unchanged_continuous_identity_is_rescanned_every_sixty_seconds(h, monkeypatch):
    enable(h)
    owner, _scheduled = coordinator(h, monkeypatch)
    calls = []
    now = 100.0

    async def resume(*, expected_policy_revision=None):
        calls.append(expected_policy_revision)

    monkeypatch.setattr(owner.supervisor, "resume_continuous", resume)
    owner._clock = lambda: now

    async def run():
        nonlocal now
        try:
            await owner._continuous_tick()
            now = 159.9
            await owner._continuous_tick()
            now = 160.0
            await owner._continuous_tick()
            assert calls == [1, 1]
        finally:
            await owner.shutdown()
            await owner.supervisor.shutdown()

    asyncio.run(run())
