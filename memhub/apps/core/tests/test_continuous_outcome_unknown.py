import asyncio

import pytest

from one_memory.hy_memory_bridge import HyMemoryBridgeError
from one_memory.memory_tasks import MemoryTaskOwnershipError, MemoryTaskStatus
from tests.test_continuous_processing_bridge import enable, put_after_policy
from tests.test_model_processing_permissions import h as h  # noqa: F401


def test_crashed_continuous_send_is_not_replayed_but_new_event_progresses(h):
    enable(h)
    fresh = put_after_policy(h, "sent-before-crash")
    supervisor = h.supervisor()
    task = supervisor.prepare_conversation(fresh)[0]
    claimed, _ = h.tasks.claim(task.task_id)
    runner = supervisor._create_runner()
    lease, request, _ = runner._prepare_permission_lease(claimed)
    h.bridge.fail = False
    asyncio.run(h.bridge.ingest(request))  # Controlled send returned, process lost its receipt.
    assert len(h.bridge.calls) == 1
    assert h.tasks.recover_running() == 1
    recovered = h.tasks.require(task.task_id)
    assert recovered.status == MemoryTaskStatus.UNKNOWN
    assert recovered.last_error_code == "model_content_outcome_unknown"
    with pytest.raises(MemoryTaskOwnershipError):
        h.tasks.claim(task.task_id)
    assert supervisor.prepare_resume() == []
    assert supervisor.prepare_backfill() == []
    # The continuous-policy filter now excludes an already possibly-sent
    # Evidence before task preparation. Either explicit unknown or outside
    # policy proves it cannot be blindly replayed.
    with pytest.raises(HyMemoryBridgeError, match="outcome_unknown|evidence_outside_policy"):
        asyncio.run(supervisor.process_records((fresh,)))
    # A new message in the same conversation must not resend the unknown original as context.
    newest = put_after_policy(h, "new-after-restart")
    tasks = supervisor.prepare_conversation(newest)
    assert len(tasks) == 1 and tasks[0].task_id != task.task_id

    async def run():
        try:
            result = await supervisor._create_runner().run(tasks[0].task_id)
            assert result.status == MemoryTaskStatus.COMPLETED
            sent = [
                request
                for kind, request in h.bridge.calls
                if kind in {"ingest", "ingest_extracted"}
            ]
            assert len(sent) == 2
            assert sent[-1].evidence_ids == (newest.evidence_id,)
            assert all(r.evidence_id != fresh.evidence_id for r in h.leases[-1].records)
            assert supervisor.prepare_backfill() == []
        finally:
            await supervisor.shutdown()

    asyncio.run(run())
    assert h.tasks.require(task.task_id).last_error_code == "model_content_outcome_unknown"


def test_legacy_running_recovery_remains_pending_and_no_continuous_table(h):
    legacy = h.legacy_task()
    h.tasks.claim(legacy.task_id)
    assert h.tasks.recover_running() == 1
    restored = h.tasks.require(legacy.task_id)
    assert restored.status == MemoryTaskStatus.PENDING
    assert restored.last_error_code == "worker_restarted"


def test_a_timed_out_non_continuous_task_also_becomes_unknown(h):
    """The timeout reason alone is enough to make a task possibly-sent.

    The continuous-permission record is not the only way HY can have been
    reached: `MemoryTaskRunner.run` wraps the HY call in `asyncio.wait_for`, so
    a wall-clock timeout can fire after the request was in flight. A
    non-continuous task must therefore also become sticky `unknown`, or the
    scheduler re-selects it and sends a second copy (HY has no idempotency key).
    """
    # A legacy (non-continuous) task: no continuous permission record exists.
    task = h.legacy_task()
    _claimed, runner_token = h.tasks.claim(task.task_id)
    h.tasks.record_hy_invocation(task.task_id, runner_token=runner_token)
    assert h.tasks.recover_one_running(task.task_id, reason="hy_memory_timeout") is True
    recovered = h.tasks.require(task.task_id)
    assert recovered.status == MemoryTaskStatus.UNKNOWN
    assert recovered.last_error_code == "model_content_outcome_unknown"
    with pytest.raises(MemoryTaskOwnershipError):
        h.tasks.claim(task.task_id)


def test_claim_without_proven_send_is_still_conservatively_unknown(h):
    enable(h)
    fresh = put_after_policy(h, "claim-only")
    task = h.supervisor().prepare_conversation(fresh)[0]
    h.tasks.claim(task.task_id)
    h.tasks.recover_running()
    recovered = h.tasks.require(task.task_id)
    assert recovered.status == MemoryTaskStatus.UNKNOWN
    assert recovered.last_error_code == "model_content_outcome_unknown"
    assert h.bridge.calls == []


def test_a_timed_out_continuous_task_becomes_unknown_not_pending(h):
    """The one-shot recovery must be continuous-aware, like process-start recovery.

    `recover_one_running` is the path a timeout takes. A continuous task may
    already have reached HY, so returning it to `pending` would let it be
    re-sent and duplicate the memory; it must become sticky `unknown` exactly
    as `recover_running` marks it at process start.
    """
    enable(h)
    fresh = put_after_policy(h, "timed-out-continuous-send")
    task = h.supervisor().prepare_conversation(fresh)[0]
    _claimed, runner_token = h.tasks.claim(task.task_id)
    h.tasks.record_hy_invocation(task.task_id, runner_token=runner_token)

    assert h.tasks.recover_one_running(task.task_id, reason="hy_memory_timeout") is True

    recovered = h.tasks.require(task.task_id)
    assert recovered.status == MemoryTaskStatus.UNKNOWN
    assert recovered.last_error_code == "model_content_outcome_unknown"
    with pytest.raises(MemoryTaskOwnershipError):
        h.tasks.claim(task.task_id)


def test_restart_preserves_the_original_unknown_ordering_time(h):
    """Recovery must not collapse old in-flight work into the restart instant.

    `unknown_jobs()` pages in `(updated_at, job_id)` order. If a process
    restart rewrote every recovered task's `updated_at`, a crash with many
    in-flight tasks would erase their original request order and make the
    inquiry drain start from an arbitrary same-timestamp batch. The recovery
    transition changes status/error only; the timestamp remains the durable
    moment the row last changed before the process died.
    """
    enable(h)
    fresh = put_after_policy(h, "restart-order")
    task = h.supervisor().prepare_conversation(fresh)[0]
    h.tasks.claim(task.task_id)
    original = h.tasks.require(task.task_id).updated_at

    assert h.tasks.recover_running() == 1

    recovered = h.tasks.require(task.task_id)
    assert recovered.status is MemoryTaskStatus.UNKNOWN
    assert recovered.updated_at == original


def test_an_unsettled_continuous_send_becomes_unknown_not_failed(h):
    """`failed` is retryable, so an abandoned continuous send must be `unknown`.

    `fail_unsettled` runs when a runner raised without settling. That exception
    may have arrived after HY was already called; `failed` would be re-claimed
    and re-sent, duplicating the memory. Only a task that never reached
    `running` (still pending) is provably unsent and may be `failed`.
    """
    enable(h)
    fresh = put_after_policy(h, "unsettled-continuous-send")
    task = h.supervisor().prepare_conversation(fresh)[0]
    h.tasks.claim(task.task_id)

    assert h.tasks.fail_unsettled(task.task_id, error_code="memory_task_dispatch_failed") is True

    settled = h.tasks.require(task.task_id)
    assert settled.status == MemoryTaskStatus.UNKNOWN
    with pytest.raises(MemoryTaskOwnershipError):
        h.tasks.claim(task.task_id)


def test_an_unsettled_non_continuous_task_is_still_failed(h):
    """The continuous rule must not change the ordinary path."""
    legacy = h.legacy_task()
    h.tasks.claim(legacy.task_id)

    assert h.tasks.fail_unsettled(legacy.task_id, error_code="memory_task_dispatch_failed") is True

    assert h.tasks.require(legacy.task_id).status == MemoryTaskStatus.FAILED
