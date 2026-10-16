"""Independent review of completed-input reuse; real stores and controlled HY only."""

import asyncio
import sqlite3
from contextlib import closing
from uuid import uuid4

import pytest

from one_memory import continuous_policy
from one_memory.continuous_permission_bridge import ContinuousPermissionReceipt
from one_memory.hy_memory_bridge import HyMemoryBridgeError
from one_memory.imports.models import EvidenceInput
from one_memory.imports.sources.registry import RootCandidateKind
from one_memory.memory_processing import MEMORY_INGEST_POLICY_VERSION
from one_memory.memory_tasks import MemoryTaskStatus
from tests.test_continuous_processing_integration import IntegrationCore


def transition(core, action):
    with closing(sqlite3.connect(core.database)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        policy = continuous_policy.current(db)
        return continuous_policy.transition(
            db,
            action=action,
            expected_revision=policy.revision,
            operation_id=str(uuid4()),
            user_confirmed=True,
        )


def bindings(core):
    with closing(sqlite3.connect(core.database)) as db:
        return db.execute(
            "SELECT task_id,policy_revision,policy_epoch,receipt_json "
            "FROM memory_task_continuous_permissions ORDER BY task_id"
        ).fetchall()


async def completed(core):
    core.grant_source()
    await core.save_and_probe()
    core.kernel_enable("none")
    record = core.put_complete("review-completed", conversation="review")
    tasks = await core.run_prepared(record)
    assert len(tasks) == 1 and tasks[0].status == MemoryTaskStatus.COMPLETED
    assert len(core.bridge.requests) == 1
    return record, tasks[0]


def frozen(core, records):
    guard = core.app.state.memory_processing_supervisor._permission_guard
    receipt = guard.freeze(guard.current("local"), records)
    assert isinstance(receipt, ContinuousPermissionReceipt)
    return guard, receipt


def lookup(guard, receipt, records, seed):
    return guard.completed_input_task(
        receipt,
        records,
        input_ref=seed.evidence_id,
        request_prefix=f"conversation:{MEMORY_INGEST_POLICY_VERSION}:{seed.source_id}:",
    )


def test_resume_reuses_completed_without_rebinding_historical_receipt(tmp_path, monkeypatch):
    core = IntegrationCore(tmp_path, monkeypatch)

    async def scenario():
        async with core.running():
            record, task = await completed(core)
            before = bindings(core)
            transition(core, "pause")
            resumed = transition(core, "resume")
            supervisor = core.app.state.memory_processing_supervisor
            profile = core.app.state.model_profile_store.active(core.secrets)

            def cannot_rebind(*args, **kwargs):
                pytest.fail("completed history was rebound to a new execution receipt")

            with monkeypatch.context() as patch:
                patch.setattr(supervisor._permission_guard, "bind", cannot_rebind)
                reused = supervisor._enqueue_chunk(core.turn_records(record), profile)
            assert reused.task_id == task.task_id
            assert reused.status == MemoryTaskStatus.COMPLETED
            assert resumed.revision == 3 and before[0][1] == 1
            assert bindings(core) == before
            assert supervisor._classify([reused]) == []
            assert len(core.bridge.requests) == 1
            assert len(core.app.state.memory_task_store.list()) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["chunk", "event-revision", "profile"])
def test_changed_input_identity_is_not_reused_as_completed(tmp_path, monkeypatch, change):
    core = IntegrationCore(tmp_path, monkeypatch)

    async def scenario():
        async with core.running():
            record, task = await completed(core)
            before = bindings(core)
            records = core.turn_records(record)
            if change == "chunk":
                other = core.put("review-added", conversation="review")
                # Same input_ref deliberately: input_ref alone cannot prove chunk identity.
                records = (other, *records)
            elif change == "event-revision":
                changed = EvidenceInput.model_validate(
                    {
                        **record.model_dump(include=set(EvidenceInput.model_fields)),
                        "verbatim": record.verbatim + " changed",
                    }
                )
                record = core.app.state.evidence_store.put(changed).record
                assert record.revision > 1
                records = (record, *records[1:])
            elif change == "profile":
                transition(core, "revoke")
                # Persist a new profile through the real API; probe remains controlled.
                saved = await core.client.post(
                    "/api/v1/model-profiles",
                    json={
                        "name": "review-second-profile",
                        "provider_kind": "openai_compatible",
                        "base_url": "https://integration.invalid/v1",
                        "model": "integration-model",
                        "api_key": "synthetic-key-not-a-credential",
                        "embedding_model": "integration-vector",
                        "embedding_provider_kind": "openai_compatible",
                        "embedding_base_url": "https://integration.invalid/v1",
                    },
                )
                assert saved.status_code == 200
                result = await core.client.post(
                    f"/api/v1/model-profiles/{saved.json()['profile_id']}/probe"
                )
                assert result.status_code == 200
                core.profile = next(
                    p
                    for p in (await core.client.get("/api/v1/model-profiles")).json()
                    if p["profile_id"] == saved.json()["profile_id"]
                )
                current = core.policy_current()
                core.app.state.model_policy_activation.enable(
                    operation_id=str(uuid4()),
                    profile_id=core.profile["profile_id"],
                    profile_revision=core.profile["revision"],
                    expected_policy_revision=current.revision,
                    expected_legacy_revision=current.legacy_revision,
                    backlog_mode="include",
                    user_confirmed=True,
                )
            guard, receipt = frozen(core, records)
            assert lookup(guard, receipt, records, record) is None
            assert bindings(core) == before
            assert (
                core.app.state.memory_task_store.require(task.task_id).status
                == MemoryTaskStatus.COMPLETED
            )
            assert len(core.bridge.requests) == 1

    asyncio.run(scenario())


def test_changed_source_root_is_not_reused_as_completed(tmp_path, monkeypatch):
    core = IntegrationCore(tmp_path, monkeypatch)

    async def scenario():
        async with core.running():
            record, task = await completed(core)
            before = bindings(core)
            grants = core.app.state.source_grant_store
            grants.revoke(user_id="local", source_id="cursor")
            replacement = core.home / "cursor-replacement"
            replacement.mkdir()
            grants.grant(
                user_id="local",
                source_id="cursor",
                root=replacement,
                candidate_kind=RootCandidateKind.DEFAULT,
            )
            records = core.turn_records(record)
            guard, receipt = frozen(core, records)

            assert lookup(guard, receipt, records, record) is None
            assert bindings(core) == before
            assert (
                core.app.state.memory_task_store.require(task.task_id).status
                == MemoryTaskStatus.COMPLETED
            )
            assert len(core.bridge.requests) == 1

    asyncio.run(scenario())


def test_reenabled_policy_reuses_exact_completed_input_without_duplicate_hy_call(
    tmp_path, monkeypatch
):
    core = IntegrationCore(tmp_path, monkeypatch)

    async def scenario():
        async with core.running():
            record, task = await completed(core)
            before = bindings(core)
            transition(core, "revoke")
            prior = core.policy_current()
            core.app.state.model_policy_activation.enable(
                operation_id=str(uuid4()),
                profile_id=core.profile["profile_id"],
                profile_revision=core.profile["revision"],
                expected_policy_revision=prior.revision,
                expected_legacy_revision=prior.legacy_revision,
                backlog_mode="include",
                user_confirmed=True,
            )
            assert core.policy_current().epoch != before[0][2]

            supervisor = core.app.state.memory_processing_supervisor
            profile = core.app.state.model_profile_store.active(core.secrets)
            reused = supervisor._enqueue_chunk(core.turn_records(record), profile)

            assert reused.task_id == task.task_id
            assert reused.status == MemoryTaskStatus.COMPLETED
            assert bindings(core) == before
            assert len(core.bridge.requests) == 1
            assert len(core.app.state.memory_task_store.list()) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("when", ["before", "after-first-check"])
@pytest.mark.parametrize("authority", ["source", "policy"])
def test_current_source_permission_is_rechecked_around_completed_lookup(
    tmp_path, monkeypatch, when, authority
):
    core = IntegrationCore(tmp_path, monkeypatch)

    async def scenario():
        async with core.running():
            record, _ = await completed(core)
            before = bindings(core)
            records = core.turn_records(record)
            guard, receipt = frozen(core, records)

            def revoke():
                if authority == "source":
                    core.app.state.source_grant_store.revoke(user_id="local", source_id="cursor")
                else:
                    transition(core, "pause")

            calls = []
            original = guard.check

            def checked(*args, **kwargs):
                value = original(*args, **kwargs)
                calls.append(True)
                if len(calls) == 1:
                    revoke()
                return value

            if when == "before":
                revoke()
            else:
                monkeypatch.setattr(guard, "check", checked)
            with pytest.raises(HyMemoryBridgeError):
                lookup(guard, receipt, records, record)
            assert bindings(core) == before
            assert len(core.bridge.requests) == 1
            if when == "after-first-check":
                assert calls == [True]

    asyncio.run(scenario())


def test_corrupt_historical_binding_is_not_accepted_as_completion(tmp_path, monkeypatch):
    core = IntegrationCore(tmp_path, monkeypatch)

    async def scenario():
        async with core.running():
            record, task = await completed(core)
            with sqlite3.connect(core.database) as db:
                db.execute(
                    "UPDATE memory_task_continuous_permissions "
                    "SET policy_revision=99 WHERE task_id=?",
                    (task.task_id,),
                )
            records = core.turn_records(record)
            guard, receipt = frozen(core, records)
            with pytest.raises(HyMemoryBridgeError, match="continuous_binding_invalid"):
                lookup(guard, receipt, records, record)
            assert len(core.bridge.requests) == 1

    asyncio.run(scenario())


def test_sdk_admitted_unknown_is_not_reused_or_resent_on_resume(tmp_path, monkeypatch):
    core = IntegrationCore(tmp_path, monkeypatch)

    async def scenario():
        async with core.running():
            core.grant_source()
            await core.save_and_probe()
            core.kernel_enable()
            record = core.put_complete("review-unknown")
            supervisor = core.app.state.memory_processing_supervisor
            tasks = supervisor.prepare_conversation(record)
            entered = asyncio.Event()

            async def interrupted(request):
                core.bridge.requests.append(request)
                entered.set()
                await asyncio.Event().wait()

            monkeypatch.setattr(core.bridge, "ingest", interrupted)
            running = asyncio.create_task(supervisor._create_runner().run(tasks[0].task_id))
            try:
                await asyncio.wait_for(entered.wait(), 3)
            finally:
                running.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await running
            # Actual restart recovery marks a previously admitted RUNNING task unknown.
            core.app.state.memory_task_store.recover_running()
            before = bindings(core)
            transition(core, "pause")
            transition(core, "resume")
            with pytest.raises(HyMemoryBridgeError, match="outcome_unknown"):
                supervisor._enqueue_chunk(
                    core.turn_records(record),
                    core.app.state.model_profile_store.active(core.secrets),
                )
            assert bindings(core) == before
            assert len(core.bridge.requests) == 1
            task = core.app.state.memory_task_store.require(tasks[0].task_id)
            assert task.last_error_code == "model_content_outcome_unknown" and task.attempts == 1
            assert len(core.app.state.memory_task_store.list()) == 1

    asyncio.run(scenario())
