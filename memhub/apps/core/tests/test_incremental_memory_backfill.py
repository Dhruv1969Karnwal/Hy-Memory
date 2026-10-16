import asyncio
import threading
from contextlib import suppress

import pytest

from one_memory.imports.models import EvidenceContentKind, WrittenBy
from one_memory.memory_processing import MemoryProcessingSupervisor, _merge_historical_chunks
from one_memory.memory_tasks import MemoryTaskStatus
from one_memory.model_profiles import ModelProfilePatch
from tests.test_memory_processing import _put, _setup
from tests.test_model_processing_permissions import Harness


def setup(tmp_path, count=3):
    evidence, profiles, tasks, governance, secrets, _profile = _setup(tmp_path)
    records = [
        _put(
            evidence,
            message_id=f"message-{index}",
            conversation_id=f"conversation-{index}",
            role="user",
            written_by=WrittenBy.PERSON,
            content=f"Synthetic conversation {index}",
            content_kind=EvidenceContentKind.MEMORY_CONTENT,
        )
        for index in range(count)
    ]
    supervisor = MemoryProcessingSupervisor(
        tasks=tasks,
        profiles=profiles,
        secrets=secrets,
        evidence=evidence,
        governance=governance,
        data_dir=tmp_path,
    )
    return supervisor, records


def test_accept_backfill_returns_after_first_batch_while_later_history_is_still_read(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("one_memory.memory_processing._BACKFILL_START_GRACE_SECONDS", 0)

    async def scenario():
        supervisor, records = setup(tmp_path)
        later_entered, release = threading.Event(), threading.Event()
        scheduled = []
        reads = []
        original = supervisor.evidence.list_conversation

        def blocked(evidence_id):
            reads.append(evidence_id)
            if len(reads) == 2:
                later_entered.set()
                assert release.wait(5)
            return original(evidence_id)

        monkeypatch.setattr(supervisor.evidence, "list_conversation", blocked)
        monkeypatch.setattr(
            supervisor, "_queue_validated_work", lambda work: scheduled.extend(work)
        )
        try:
            accepted = asyncio.create_task(supervisor.accept_backfill_off_thread())
            assert await asyncio.to_thread(later_entered.wait, 3)
            await asyncio.wait_for(accepted, 1)
            assert len(scheduled) == 1
            assert accepted.done()
            assert supervisor._live_backfill is not None
            assert not supervisor._live_backfill.done()
        finally:
            release.set()
            if supervisor._live_backfill is not None:
                await supervisor._live_backfill
            await supervisor.shutdown()
        assert len(scheduled) == len(records)

    asyncio.run(scenario())


def test_first_task_is_dispatched_before_later_conversation_is_prepared(tmp_path, monkeypatch):
    async def scenario():
        supervisor, records = setup(tmp_path)
        later_entered, release = threading.Event(), threading.Event()
        scheduled = []
        reads = []
        original = supervisor.evidence.list_conversation

        def blocked(evidence_id):
            reads.append(evidence_id)
            if len(reads) == 2:
                later_entered.set()
                assert release.wait(5)
            return original(evidence_id)

        monkeypatch.setattr(supervisor.evidence, "list_conversation", blocked)
        monkeypatch.setattr(
            supervisor, "_queue_validated_work", lambda work: scheduled.extend(work)
        )
        task = asyncio.create_task(supervisor.backfill_off_thread())
        try:
            assert await asyncio.to_thread(later_entered.wait, 3)
            assert len(scheduled) == 1, "first task still waits behind later conversation reads"
            assert supervisor.tasks.require(scheduled[0].task_id).input_ref in {
                record.evidence_id for record in records
            }
        finally:
            release.set()
            await task
            await supervisor.shutdown()
        assert len(scheduled) == len(records)
        assert {supervisor.tasks.require(item.task_id).input_ref for item in scheduled} == {
            record.evidence_id for record in records
        }

    asyncio.run(scenario())


def test_cancellation_joins_the_preparation_thread_before_return(tmp_path, monkeypatch):
    async def scenario():
        supervisor, _records = setup(tmp_path)
        entered, release, exited = threading.Event(), threading.Event(), threading.Event()
        original = supervisor.evidence.list_conversation
        scheduled = []

        def blocked(evidence_id):
            entered.set()
            assert release.wait(5)
            result = original(evidence_id)
            exited.set()
            return result

        monkeypatch.setattr(supervisor.evidence, "list_conversation", blocked)
        monkeypatch.setattr(
            supervisor, "_queue_validated_work", lambda work: scheduled.extend(work)
        )
        task = asyncio.create_task(supervisor.backfill_off_thread())
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            task.cancel()
            await asyncio.sleep(0.03)
            task.cancel()
            await asyncio.sleep(0.03)
            assert not task.done(), "backfill abandoned a thread that can still write tasks"
        finally:
            release.set()
            with suppress(asyncio.CancelledError):
                await task
            await supervisor.shutdown()
        assert exited.is_set()
        assert scheduled == []

    asyncio.run(scenario())


def test_later_failure_keeps_the_first_committed_batch_visible(tmp_path, monkeypatch):
    async def scenario():
        supervisor, _records = setup(tmp_path)
        scheduled, reads = [], []
        original = supervisor.evidence.list_conversation

        def broken(evidence_id):
            reads.append(evidence_id)
            if len(reads) == 2:
                raise RuntimeError("synthetic later read failure")
            return original(evidence_id)

        monkeypatch.setattr(supervisor.evidence, "list_conversation", broken)
        monkeypatch.setattr(
            supervisor, "_queue_validated_work", lambda work: scheduled.extend(work)
        )
        try:
            with pytest.raises(RuntimeError, match="synthetic later"):
                await supervisor.backfill_off_thread()
            assert len(scheduled) == 1
            assert len(supervisor.tasks.list()) == 1
        finally:
            await supervisor.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["revoke", "regrant"])
def test_permission_change_between_batches_stops_without_new_tasks(tmp_path, monkeypatch, change):
    async def scenario():
        h = Harness(tmp_path)
        records = [
            h.put("first", conversation="first", memory_content=True),
            h.put("second", conversation="second", memory_content=True),
        ]
        receipt = h.grant(records)
        supervisor = h.supervisor()
        scheduled = []

        def dispatch(work):
            scheduled.extend(work)
            if len(scheduled) == 1:
                if change == "revoke":
                    h.permissions.revoke(
                        user_id="local", user_confirmed=True, expected_revision=receipt.revision
                    )
                else:
                    h.grant(records)

        monkeypatch.setattr(supervisor, "_queue_validated_work", dispatch)
        try:
            with pytest.raises(RuntimeError, match="model_content_permission_"):
                await supervisor.backfill_off_thread(expected_permission_revision=receipt.revision)
            assert len(h.tasks.list()) == len(scheduled) == 1
            assert h.factory_calls == [] and h.bridge.calls == []
        finally:
            await supervisor.shutdown()

    asyncio.run(scenario())


def test_profile_change_after_first_batch_preserves_old_task_identity(tmp_path, monkeypatch):
    async def scenario():
        h = Harness(tmp_path)
        h.grant(
            [
                h.put("first", conversation="first", memory_content=True),
                h.put("second", conversation="second", memory_content=True),
            ]
        )
        supervisor = h.supervisor()
        scheduled = []

        def dispatch(work):
            scheduled.extend(work)
            h.profiles.update(
                h.profile.profile_id,
                ModelProfilePatch(model="changed-model", expected_revision=h.profile.revision),
                h.secrets,
            )

        monkeypatch.setattr(supervisor, "_queue_validated_work", dispatch)
        try:
            with pytest.raises(RuntimeError, match="model_content_profile_changed"):
                await supervisor.backfill_off_thread()
            assert len(h.tasks.list()) == len(scheduled) == 1
            assert h.tasks.list()[0].profile_revision == h.profile.revision
            assert h.factory_calls == []
        finally:
            await supervisor.shutdown()

    asyncio.run(scenario())


def test_batched_and_sync_preparation_keep_all_task_inputs_and_idempotency(tmp_path, monkeypatch):
    async def scenario():
        supervisor, records = setup(tmp_path, count=21)
        scheduled = []
        monkeypatch.setattr(
            supervisor, "_queue_validated_work", lambda batch: scheduled.append(batch)
        )
        try:
            expected = supervisor.prepare_backfill()
            expected_ids = {work.task_id for work in expected}
            assert len(expected_ids) == 21
            before = {
                record.evidence_id: supervisor.evidence.get(record.evidence_id)
                for record in records
            }
            await supervisor.backfill_off_thread()
            assert len(scheduled[0]) == 1
            assert all(len(batch) <= 8 for batch in scheduled)
            assert {work.task_id for batch in scheduled for work in batch} == expected_ids
            assert len(supervisor.tasks.list()) == 21
            assert {
                record.evidence_id: supervisor.evidence.get(record.evidence_id)
                for record in records
            } == before
        finally:
            await supervisor.shutdown()

    asyncio.run(scenario())


def test_history_backfill_merges_adjacent_unsent_turns_in_one_conversation(tmp_path):
    supervisor, records = setup(tmp_path, count=0)
    profile = supervisor.profiles.active(supervisor.secrets)
    assert profile is not None
    conversation = []
    for index in range(2):
        conversation.extend(
            (
                _put(
                    supervisor.evidence,
                    message_id=f"user-{index}",
                    role="user",
                    written_by=WrittenBy.PERSON,
                    content=f"问题 {index}",
                    conversation_id="shared-history",
                ),
                _put(
                    supervisor.evidence,
                    message_id=f"assistant-{index}",
                    role="assistant",
                    written_by=WrittenBy.AGENT,
                    content=f"回答 {index}",
                    conversation_id="shared-history",
                ),
            )
        )

    chunks = supervisor._prepare_conversation_chunks(conversation, profile, None)
    merged = _merge_historical_chunks(chunks)

    assert len(chunks) == 2
    assert len(merged) == 1
    assert {record.evidence_id for record in merged[0]} == {
        record.evidence_id for record in conversation
    }


def test_backfill_holds_an_unanswered_raw_turn_until_its_reply_arrives(tmp_path):
    supervisor, _records = setup(tmp_path, count=0)
    question = _put(
        supervisor.evidence,
        message_id="open-question",
        role="user",
        written_by=WrittenBy.PERSON,
        content="发布包上传了吗？",
        conversation_id="open-turn",
    )

    assert supervisor.prepare_conversation(question) == []
    assert supervisor.prepare_backfill() == []

    answer = _put(
        supervisor.evidence,
        message_id="closing-answer",
        role="assistant",
        written_by=WrittenBy.AGENT,
        content="已经上传完成。",
        conversation_id="open-turn",
    )

    prepared = supervisor.prepare_conversation(answer)

    assert len(prepared) == 1
    assert prepared[0].input_ref == question.evidence_id


def test_history_merge_supersedes_an_already_queued_shorter_session_task(tmp_path):
    supervisor, _records = setup(tmp_path, count=0)
    first = _put(
        supervisor.evidence,
        message_id="merge-user-1",
        role="user",
        written_by=WrittenBy.PERSON,
        content="第一问",
        conversation_id="growing-history",
    )
    first_reply = _put(
        supervisor.evidence,
        message_id="merge-assistant-1",
        role="assistant",
        written_by=WrittenBy.AGENT,
        content="第一答",
        conversation_id="growing-history",
    )
    profile = supervisor.profiles.active(supervisor.secrets)
    assert profile is not None
    old = supervisor._enqueue_chunk((first, first_reply), profile, freeze_inputs=True)
    second = _put(
        supervisor.evidence,
        message_id="merge-user-2",
        role="user",
        written_by=WrittenBy.PERSON,
        content="第二问",
        conversation_id="growing-history",
    )
    second_reply = _put(
        supervisor.evidence,
        message_id="merge-assistant-2",
        role="assistant",
        written_by=WrittenBy.AGENT,
        content="第二答",
        conversation_id="growing-history",
    )

    chunks = supervisor._prepare_conversation_chunks(
        supervisor.evidence.list_conversation(second.evidence_id), profile, None
    )
    merged_chunk = _merge_historical_chunks(chunks)[0]
    merged_task = supervisor._enqueue_chunk(merged_chunk, profile, freeze_inputs=True)

    pending = supervisor.tasks.list(statuses=(MemoryTaskStatus.PENDING,))
    merged = next(task for task in supervisor.tasks.list() if task.task_id != old.task_id)
    assert supervisor.tasks.require(old.task_id).status.value == "cancelled"
    assert [task.task_id for task in pending] == [merged.task_id]
    assert {ref["evidence_id"] for ref in supervisor.tasks.inputs(merged.task_id)} == {
        first.evidence_id,
        first_reply.evidence_id,
        second.evidence_id,
        second_reply.evidence_id,
    }
    assert merged_task.task_id == merged.task_id


def test_backfill_budget_yields_between_rounds_and_resumes_same_cursor(tmp_path, monkeypatch):
    async def scenario():
        supervisor, records = setup(tmp_path, count=3)
        scheduled = []
        clock = iter((0.0, 0.0, 331.0, 331.0, 331.0, 662.0, 662.0, 662.0))
        supervisor._backfill_clock = lambda: next(clock)
        supervisor._backfill_round_budget_seconds = 330.0
        releases = []

        async def schedule(work):
            assert supervisor._backfill_lock.locked()
            scheduled.extend(work)

        original_sleep = asyncio.sleep

        async def observed_sleep(delay):
            if delay == 0:
                releases.append(not supervisor._backfill_lock.locked())
            await original_sleep(0)

        monkeypatch.setattr(supervisor, "_schedule_tasks_off_thread", schedule)
        monkeypatch.setattr("one_memory.memory_processing.asyncio.sleep", observed_sleep)
        try:
            await supervisor.backfill_off_thread()
        finally:
            await supervisor.shutdown()

        assert releases == [True, True]
        assert len(scheduled) == len(records)
        assert len({work.task_id for work in scheduled}) == len(records)

    asyncio.run(scenario())


def test_shutdown_waits_for_backfill_preparation_and_never_dispatches_tail(tmp_path, monkeypatch):
    async def scenario():
        supervisor, _records = setup(tmp_path)
        entered, release = threading.Event(), threading.Event()
        scheduled = []
        original = supervisor.evidence.list_conversation

        def blocked(evidence_id):
            entered.set()
            assert release.wait(5)
            return original(evidence_id)

        monkeypatch.setattr(supervisor.evidence, "list_conversation", blocked)
        monkeypatch.setattr(
            supervisor, "_queue_validated_work", lambda work: scheduled.extend(work)
        )
        task = asyncio.create_task(supervisor.backfill_off_thread())
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            stopping = asyncio.create_task(supervisor.shutdown())
            await asyncio.sleep(0.03)
            assert not stopping.done()
        finally:
            release.set()
            await task
            await stopping
        assert scheduled == []
        assert supervisor._backfill_preparations == set()
        assert supervisor.tasks.list() == []

    asyncio.run(scenario())


@pytest.mark.parametrize("count", [0, 1, 8, 9, 18])
def test_empty_and_batch_boundaries_preserve_all_inputs(tmp_path, monkeypatch, count):
    async def scenario():
        supervisor, records = setup(tmp_path, count=count)
        scheduled = []
        monkeypatch.setattr(
            supervisor, "_queue_validated_work", lambda work: scheduled.extend(work)
        )
        active_calls = []
        active = supervisor.profiles.active

        def counted(secrets):
            active_calls.append(True)
            return active(secrets)

        monkeypatch.setattr(supervisor.profiles, "active", counted)
        try:
            await supervisor.backfill_off_thread()
            assert len(active_calls) == 1, "batch identity checks should not reaccess credentials"
            assert {supervisor.tasks.require(work.task_id).input_ref for work in scheduled} == {
                record.evidence_id for record in records
            }
            assert len(supervisor.tasks.list()) == count
        finally:
            await supervisor.shutdown()

    asyncio.run(scenario())


def test_empty_denied_batches_do_not_end_the_remaining_traversal(tmp_path, monkeypatch):
    async def scenario():
        supervisor, records = setup(tmp_path, count=19)
        scheduled = []
        original = supervisor.evidence.list_conversation
        earliest = records[0].evidence_id
        monkeypatch.setattr(
            supervisor.evidence,
            "list_conversation",
            lambda identity: original(identity) if identity == earliest else [],
        )
        monkeypatch.setattr(
            supervisor, "_queue_validated_work", lambda work: scheduled.append(work)
        )
        try:
            await supervisor.backfill_off_thread()
            assert scheduled[:2] == [[], []]
            useful = [work for batch in scheduled for work in batch]
            assert len(useful) == 1
            assert supervisor.tasks.require(useful[0].task_id).input_ref == earliest
        finally:
            await supervisor.shutdown()

    asyncio.run(scenario())


def test_replacement_backfill_waits_for_cancelled_writer_and_remains_idempotent(
    tmp_path, monkeypatch
):
    async def scenario():
        supervisor, records = setup(tmp_path)
        entered, release = threading.Event(), threading.Event()
        reads, scheduled = [], []
        original = supervisor.evidence.list_conversation

        def blocked(evidence_id):
            reads.append(evidence_id)
            if len(reads) == 1:
                entered.set()
                assert release.wait(5)
            return original(evidence_id)

        monkeypatch.setattr(supervisor.evidence, "list_conversation", blocked)
        monkeypatch.setattr(
            supervisor, "_queue_validated_work", lambda work: scheduled.extend(work)
        )
        old = asyncio.create_task(supervisor.backfill_off_thread())
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            old.cancel()
            replacement = asyncio.create_task(supervisor.backfill_off_thread())
            await asyncio.sleep(0.03)
            assert not old.done() and not replacement.done()
            assert len(reads) == 1
        finally:
            release.set()
            with suppress(asyncio.CancelledError):
                await old
            await replacement
            await supervisor.shutdown()
        assert len(supervisor.tasks.list()) == len(records)
        assert len({work.task_id for work in scheduled}) == len(scheduled) == len(records)

    asyncio.run(scenario())


@pytest.mark.parametrize("unclaimable", ["completed", "deferred"])
def test_unclaimable_first_seed_does_not_delay_first_useful_batch(
    tmp_path, monkeypatch, unclaimable
):
    async def scenario():
        supervisor, records = setup(tmp_path)
        first = supervisor.prepare_conversation(records[-1])[0]
        if unclaimable == "completed":
            _, token = supervisor.tasks.claim(first.task_id)
            supervisor.tasks.complete(first.task_id, runner_token=token)
        else:
            supervisor.tasks.defer_outside_window(first.task_id)
        scheduled, reads = [], []
        entered, release = threading.Event(), threading.Event()
        original = supervisor.evidence.list_conversation

        def blocked(identity):
            reads.append(identity)
            if len(reads) == 3:
                entered.set()
                assert release.wait(5)
            return original(identity)

        monkeypatch.setattr(supervisor.evidence, "list_conversation", blocked)
        monkeypatch.setattr(
            supervisor, "_queue_validated_work", lambda work: scheduled.extend(work)
        )
        task = asyncio.create_task(supervisor.backfill_off_thread())
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            assert len(scheduled) == 1
            assert scheduled[0].task_id != first.task_id
        finally:
            release.set()
            await task
            await supervisor.shutdown()

    asyncio.run(scenario())


def test_cancelled_shutdown_waits_for_inflight_generator_and_preserves_cleanup(
    tmp_path, monkeypatch
):
    async def scenario():
        supervisor, _records = setup(tmp_path)
        entered, release = threading.Event(), threading.Event()
        original = supervisor.evidence.list_conversation

        def blocked(identity):
            entered.set()
            assert release.wait(5)
            return original(identity)

        monkeypatch.setattr(supervisor.evidence, "list_conversation", blocked)
        backfill = asyncio.create_task(supervisor.backfill_off_thread())
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            stopping = asyncio.create_task(supervisor.shutdown())
            await asyncio.sleep(0.02)
            stopping.cancel()
            await asyncio.sleep(0.02)
            stopping.cancel()
            await asyncio.sleep(0.02)
            assert not stopping.done() and not backfill.done()
            assert len(supervisor._backfill_preparations) == 1
        finally:
            release.set()
            await asyncio.gather(stopping, backfill, return_exceptions=True)
        assert not backfill.cancelled() and backfill.exception() is None
        assert supervisor._backfill_preparations == set()
        assert supervisor.tasks.list() == []

    asyncio.run(scenario())


def test_exact_backfill_keeps_granted_prefix_when_same_conversation_grows(tmp_path):
    async def scenario():
        h = Harness(tmp_path)
        first = h.put("first", memory_content=True)
        second = h.put("second", memory_content=True)
        receipt = h.grant([first, second])
        later = h.put("later", memory_content=True)
        other = h.put("other", conversation="other", memory_content=True)
        supervisor = h.supervisor()
        try:
            await supervisor.backfill_off_thread(expected_permission_revision=receipt.revision)
            tasks = supervisor.tasks.list()
            # One task per turn; both granted turns, nothing after them.
            assert len(tasks) == 1
            assert {task.input_ref for task in tasks} == {second.evidence_id}
            refs = {
                item["evidence_id"]
                for task in tasks
                for item in supervisor.tasks.inputs(task.task_id)
            }
            assert {first.evidence_id, second.evidence_id} <= refs
            assert later.evidence_id not in refs
            assert other.evidence_id not in refs
            # The first turn may already be running; whatever reached the model was granted.
            assert all(
                {record.evidence_id for record in lease.records}
                <= {first.evidence_id, second.evidence_id}
                for lease, _data_dir in h.factory_calls
            )
        finally:
            await supervisor.shutdown()

    asyncio.run(scenario())
