import asyncio
import json
from uuid import uuid4

import pytest

from one_memory import continuous_policy as cp
from one_memory import ingest_journal
from one_memory.hy_memory_bridge import HyMemoryBridgeError
from one_memory.imports.sources.registry import RootCandidateKind
from one_memory.memory_tasks import MemoryTaskStatus
from one_memory.model_processing_permission_guard import ModelProcessingPermissionGuard
from tests.test_model_processing_permissions import h as h  # noqa: F401


def enable(h, backlog="none"):
    with h.evidence._connect(writing=True) as db:
        ingest_journal.initialize(db)
        cp.initialize(db)
        legacy = h.permissions.receipt(user_id="local")
        return cp.enable(
            db,
            profile=h.profiles.snapshot(h.profile.profile_id, h.profile.revision),
            expected_revision=0,
            expected_legacy_revision=legacy.revision if legacy else 0,
            operation_id=str(uuid4()),
            backlog_mode=backlog,
            user_confirmed=True,
        )


def transition(h, action):
    with h.evidence._connect(writing=True) as db:
        policy = cp.current(db)
        return cp.transition(
            db,
            expected_revision=policy.revision,
            action=action,
            operation_id=str(uuid4()),
            user_confirmed=True,
        )


def put_after_policy(h, key, *, source="cursor"):
    """Create one complete continuous input and record its ingest identity.

    These policy tests used a raw user-only record, which the complete-turn
    contract correctly holds back. MEMORY_CONTENT is a complete input by
    itself; recording it after policy enable proves the `after_seq` boundary
    without inventing an assistant reply that the assertions do not inspect.
    """
    record = h.put(key, source=source, memory_content=True)
    with h.evidence._connect(writing=True) as db:
        ingest_journal.record(db, record.evidence_id, cause="import")
    return h.evidence.get(record.evidence_id)


def test_continuous_prepare_lease_ingest_has_no_legacy_receipt_or_old_context(h):
    policy = enable(h)
    fresh = put_after_policy(h, "new")
    supervisor = h.supervisor()

    async def scenario():
        try:
            tasks = supervisor.prepare_conversation(fresh)
            assert len(tasks) == 1
            result = await supervisor._create_runner().run(tasks[0].task_id)
            assert result.status == MemoryTaskStatus.WAITING_FOR_MODEL  # Controlled HY failure.
            assert result.last_error_code == "hy_memory_unavailable"
            lease = h.leases[0]
            assert lease.receipt.schema_version == 2
            assert lease.receipt.authority_kind == "continuous"
            assert lease.receipt.policy == policy
            assert {r.evidence_id for r in lease.records} == {fresh.evidence_id}
            assert h.record.evidence_id not in {r.evidence_id for r in lease.records}
            assert h.permissions.receipt(user_id="local") is None
        finally:
            await supervisor.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("action", ["pause", "revoke"])
def test_continuous_inactive_never_falls_back_to_legacy(h, action):
    h.grant()
    enable(h, "include")
    transition(h, action)
    supervisor = h.supervisor()
    with pytest.raises(HyMemoryBridgeError, match="model_content_"):
        supervisor.prepare_conversation(h.record)
    assert not h.bridge.calls


@pytest.mark.parametrize("phase", ["before_run", "ingest", "profile"])
def test_revocation_rechecked_at_runner_and_publication_boundaries(h, phase):
    enable(h)
    fresh = put_after_policy(h, "new")
    supervisor = h.supervisor()
    task = supervisor.prepare_conversation(fresh)[0]
    h.bridge.fail = False
    if phase == "before_run":
        transition(h, "revoke")
    elif phase == "ingest":
        h.bridge.on_ingest = lambda: transition(h, "revoke")
    else:
        h.bridge.on_profile = lambda: transition(h, "revoke")

    async def scenario():
        try:
            result = await supervisor._create_runner().run(task.task_id)
            assert result.status == MemoryTaskStatus.CANCELLED
            assert result.last_error_code.startswith("model_content_")
            assert h.governance.list(user_id="local") == []
        finally:
            await supervisor.shutdown()

    asyncio.run(scenario())


def test_durable_binding_restarts_and_cannot_adopt_new_policy_or_new_revision(h):
    enable(h)
    fresh = put_after_policy(h, "new")
    supervisor = h.supervisor()
    task = supervisor.prepare_conversation(fresh)[0]
    guard = ModelProcessingPermissionGuard(h.permissions, h.governance)
    saved = guard.bound(task)
    assert saved.authority_kind == "continuous" and saved.inputs
    guard.lease(task, (fresh,), None).recheck()
    with h.evidence._connect(writing=True) as db:
        db.execute(
            "UPDATE evidence SET revision=revision+1,verbatim='changed' WHERE evidence_id=?",
            (fresh.evidence_id,),
        )
        ingest_journal.record(db, fresh.evidence_id, cause="import")
    with pytest.raises(HyMemoryBridgeError):
        guard.lease(task, (fresh,), None).recheck()


def test_legacy_task_not_promoted_by_continuous_policy(h):
    legacy = h.grant()
    task = h.legacy_task()
    guard = ModelProcessingPermissionGuard(h.permissions, h.governance)
    guard.bind(task, legacy)
    enable(h, "include")
    with pytest.raises(HyMemoryBridgeError):
        guard.bound(task)


def test_future_new_source_old_date_uses_continuous_ingest_boundary(h, tmp_path):
    enable(h)
    folder = tmp_path / "second"
    folder.mkdir()
    h.sources.grant(
        user_id="local",
        source_id="new-source",
        root=folder,
        candidate_kind=RootCandidateKind.DEFAULT,
    )
    fresh = put_after_policy(h, "new", source="new-source")
    with h.evidence._connect(writing=True) as db:
        db.execute(
            "UPDATE evidence SET created_at='1990-01-01T00:00:00+00:00' WHERE evidence_id=?",
            (fresh.evidence_id,),
        )
        # Actual import material revision, not authorizing an attribution repair.
        db.execute(
            "UPDATE evidence SET verbatim='Newly imported old-dated content' WHERE evidence_id=?",
            (fresh.evidence_id,),
        )
        ingest_journal.record(db, fresh.evidence_id, cause="import")
    fresh = h.evidence.get(fresh.evidence_id)
    supervisor = h.supervisor()
    tasks = supervisor.prepare_conversation(fresh)
    assert len(tasks) == 1
    assert [work.task_id for work in supervisor._classify(tasks)] == [tasks[0].task_id]
    assert supervisor.tasks.require(tasks[0].task_id).status is MemoryTaskStatus.PENDING


def test_schema2_roundtrip_rejects_unknown_and_preserves_bound_events(h):
    from one_memory.continuous_permission_bridge import decode_authority

    enable(h)
    fresh = put_after_policy(h, "new")
    supervisor = h.supervisor()
    task = supervisor.prepare_conversation(fresh)[0]
    receipt = supervisor._permission_guard.bound(task)
    assert decode_authority(receipt.model_dump_json()) == receipt
    payload = receipt.model_dump(mode="json")
    payload["schema_version"] = 99
    with pytest.raises(ValueError):
        decode_authority(json.dumps(payload))


def test_transport_rechecks_continuous_lease_without_forged_snapshot(h):
    from one_memory.hy_permission_transport import HyPermissionTransport, HyPermissionTransportError

    enable(h)
    fresh = put_after_policy(h, "new")
    supervisor = h.supervisor()
    task = supervisor.prepare_conversation(fresh)[0]
    lease = supervisor._create_runner()._prepare_permission_lease(task)[0]
    transport = HyPermissionTransport(lease)
    transport._active = True
    transport.assert_allowed()
    transition(h, "pause")
    with pytest.raises(HyPermissionTransportError, match="lease_denied"):
        transport.assert_allowed()


def test_transport_denial_names_why_the_lease_failed(h):
    from one_memory.hy_permission_transport import HyPermissionTransport, HyPermissionTransportError

    enable(h)
    task = h.supervisor().prepare_conversation(put_after_policy(h, "new"))[0]
    lease = h.supervisor()._create_runner()._prepare_permission_lease(task)[0]
    transport = HyPermissionTransport(lease)
    transport._active = True
    transition(h, "pause")
    with pytest.raises(Exception) as direct:
        lease.recheck()
    reason = str(direct.value).removeprefix("model_content_")
    assert reason and reason != "lease_denied"

    with pytest.raises(HyPermissionTransportError) as denied:
        transport.assert_allowed()

    assert denied.value.code == f"model_content_transport_lease_denied_{reason}"


def test_successful_continuous_runner_completes_without_old_backlog(h):
    enable(h)
    fresh = put_after_policy(h, "new")
    supervisor = h.supervisor()
    h.bridge.fail = False

    async def run():
        try:
            task = supervisor.prepare_conversation(fresh)[0]
            result = await supervisor._create_runner().run(task.task_id)
            assert result.status == MemoryTaskStatus.COMPLETED
            assert [kind for kind, _ in h.bridge.calls] == [
                "ingest_extracted",
                "profile",
            ]
            request = h.bridge.calls[0][1]
            assert request.evidence_ids == (fresh.evidence_id,)
            assert request.source_context is None
        finally:
            await supervisor.shutdown()

    asyncio.run(run())


def test_prepare_backfill_filters_old_inputs_and_paused_policy_refuses(h):
    policy = enable(h)
    fresh = put_after_policy(h, "new")
    supervisor = h.supervisor()
    tasks = supervisor.prepare_backfill(expected_permission_revision=policy.revision)
    assert tasks
    for work in tasks:
        task = h.tasks.require(work.task_id)
        assert {ref["evidence_id"] for ref in h.tasks.inputs(task.task_id)} == {fresh.evidence_id}
    transition(h, "pause")
    with pytest.raises(HyMemoryBridgeError):
        supervisor.prepare_backfill()


def test_bound_source_regrant_refuses_old_task(h):
    enable(h)
    fresh = put_after_policy(h, "new")
    supervisor = h.supervisor()
    task = supervisor.prepare_conversation(fresh)[0]
    h.sources.revoke(user_id="local", source_id="cursor")
    h.sources.grant(
        user_id="local",
        source_id="cursor",
        root=h.root / "synthetic-source",
        candidate_kind=RootCandidateKind.DEFAULT,
    )
    with pytest.raises(HyMemoryBridgeError):
        supervisor._create_runner()._prepare_permission_lease(task)


def test_schema2_native_sdk_child_send_denies_after_policy_revoke(h):
    import httpx

    from one_memory.hy_permission_transport import HyPermissionTransport, HyPermissionTransportError
    from tests.test_hy_permission_transport import native, response

    enable(h)
    fresh = put_after_policy(h, "new")
    supervisor = h.supervisor()
    task = supervisor.prepare_conversation(fresh)[0]
    lease = supervisor._create_runner()._prepare_permission_lease(task)[0]
    provider, _ = native(lease)
    requests = []

    def transport(request):
        requests.append(request.url.path)
        return response(request)

    async def run():
        with pytest.raises(HyPermissionTransportError):
            async with HyPermissionTransport(
                lease, http_transport=httpx.MockTransport(transport)
            ) as boundary:
                boundary.bind_provider(provider)
                await provider.complete("Synthetic permitted input")
                assert len(requests) == 1
                transition(h, "revoke")
                await provider.complete("This request must not leave")
        assert len(requests) == 1

    asyncio.run(run())


def test_recreated_event_identity_produces_new_task_even_same_revision(h):
    enable(h)
    fresh = put_after_policy(h, "new")
    supervisor = h.supervisor()
    first = supervisor.prepare_conversation(fresh)[0]
    with h.evidence._connect(writing=True) as db:
        db.execute(
            "UPDATE evidence SET verbatim='changed without revision' WHERE evidence_id=?",
            (fresh.evidence_id,),
        )
        ingest_journal.record(db, fresh.evidence_id, cause="import")
    new = h.evidence.get(fresh.evidence_id)
    second = supervisor.prepare_conversation(new)[0]
    assert second.task_id != first.task_id


def test_default_guard_does_not_create_continuous_schema(h):
    ModelProcessingPermissionGuard(h.permissions, h.governance)
    with h.evidence._connect() as db:
        assert not db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='memory_task_continuous_permissions'"
        ).fetchone()
        assert not db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='continuous_policy_history'"
        ).fetchone()


def test_publish_guard_rechecks_continuous_after_last_bridge_read(h, monkeypatch):
    enable(h)
    fresh = put_after_policy(h, "new")
    supervisor = h.supervisor()
    h.bridge.fail = False
    task = supervisor.prepare_conversation(fresh)[0]
    runner = supervisor._create_runner()
    original = runner._record_outputs

    def revoke_at_commit(**kwargs):
        transition(h, "revoke")
        return original(**kwargs)

    monkeypatch.setattr(runner, "_record_outputs", revoke_at_commit)
    result = asyncio.run(runner.run(task.task_id))
    assert result.status == MemoryTaskStatus.CANCELLED
    assert result.last_error_code.startswith("model_content_")
    assert h.tasks.extraction(task.task_id) is None
    assert h.governance.list(user_id="local") == []


def test_native_hy_continuous_source_to_governance_with_mock_network(tmp_path):
    import httpx

    from one_memory.hy_permission_context import permission_bridge_factory
    from one_memory.imports.models import (
        EvidenceContentKind,
        EvidenceInput,
        EvidenceOrigin,
        OriginMetadata,
        WrittenBy,
    )
    from tests.test_hy_permission_context import whole_client

    async def run():
        with whole_client(tmp_path) as (supervisor, old, permissions, legacy):
            with supervisor.evidence._connect(writing=True) as db:
                ingest_journal.initialize(db)
                cp.initialize(db)
                cp.enable(
                    db,
                    profile=legacy.snapshot.profile,
                    expected_revision=0,
                    expected_legacy_revision=legacy.revision,
                    operation_id=str(uuid4()),
                    backlog_mode="none",
                    user_confirmed=True,
                )
            fresh = supervisor.evidence.put(
                EvidenceInput(
                    source_id=old.source_id,
                    conversation_id=old.conversation_id,
                    message_id="new",
                    role="user",
                    written_by=WrittenBy.PERSON,
                    content_kind=EvidenceContentKind.MEMORY_CONTENT,
                    verbatim="Synthetic fresh continuous memory, not old backlog.",
                    origin_meta={EvidenceOrigin.FILE: OriginMetadata()},
                )
            ).record
            sends = []

            def handler(request):
                body = json.loads(request.content)
                sends.append(body)
                assert old.verbatim not in request.content.decode()
                if request.url.path.endswith("embeddings"):
                    return httpx.Response(
                        200,
                        json={
                            "object": "list",
                            "model": "test-embedding",
                            "data": [
                                {
                                    "index": i,
                                    "object": "embedding",
                                    "embedding": [1.0] + [0.0] * 383,
                                }
                                for i, _ in enumerate(body["input"])
                            ],
                            "usage": {"prompt_tokens": 1, "total_tokens": 1},
                        },
                    )
                return httpx.Response(
                    200,
                    json={
                        "id": "fixture",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "test-generation",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        [
                                            {
                                                "op": "ADD",
                                                "content": fresh.verbatim,
                                                "layer": "L2_FACT",
                                                "owner": "user",
                                            }
                                        ]
                                    ),
                                },
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    },
                )

            supervisor.permission_bridge_factory = permission_bridge_factory(
                supervisor._bridge, http_transport=httpx.MockTransport(handler)
            )
            try:
                task = supervisor.prepare_conversation(fresh)[0]
                result = await supervisor._create_runner().run(task.task_id)
                assert result.status == "completed", result.last_error_code
                memories = supervisor.governance.list(user_id="local")
                assert sends and len(memories) == 1
                assert memories[0].content == fresh.verbatim
                assert memories[0].evidence_ids == (fresh.evidence_id,)
                assert permissions.receipt(user_id="local") == legacy
            finally:
                await supervisor.shutdown()

    asyncio.run(run())
