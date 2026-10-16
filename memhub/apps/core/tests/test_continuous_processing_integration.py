"""Real temporary Core/stores with controlled probe and HY boundaries, never real data.

Kernel-helper tests isolate the scheduling handoff while the public activation
route lands. They do not claim atomic activation; API tests require the real route.
"""

import asyncio
import sqlite3
from contextlib import asynccontextmanager, closing
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx
import pytest

from one_memory import api, continuous_policy, ingest_journal
from one_memory.config import Settings
from one_memory.hy_memory_bridge import HyMemoryBridgeError, HyMemoryIngestResult, HyMemoryOutput
from one_memory.imports.models import EvidenceInput, EvidenceOrigin, OriginMetadata, WrittenBy
from one_memory.imports.sources.registry import RootCandidateKind
from one_memory.memory_processing import MemoryTaskRunner
from one_memory.memory_tasks import MemoryTaskStatus
from one_memory.model_probe import ModelProber
from one_memory.secrets import InMemorySecretStore

API_ROOT = "/api/v1/continuous-processing-policy"
TEXT = "用户偏好本地优先的记忆处理，并保留可追溯的原始依据。"
OLD = datetime(2019, 1, 1, tzinfo=UTC)


class FakeSecretStore(InMemorySecretStore):
    """Only the existing in-memory SecretStore protocol; no keychain calls."""


class ControlledHyBridge:
    def __init__(self):
        self.requests = []
        self.leases = []
        self.before_return = lambda: None
        self.closed = 0

    async def ingest(self, request):
        if request.source_id == "one-memory-manual":
            pytest.fail("A note the person asked to keep must be imported, not re-extracted")
        return await self.answer(request)

    async def ingest_extracted(self, request, contents):
        if request.source_id != "one-memory-manual":
            pytest.fail("Raw conversation must use HY ingest, not direct content import")
        assert list(contents) == [turn.content for turn in request.turns]
        return await self.answer(request)

    async def answer(self, request):
        self.requests.append(request)
        self.before_return()
        return HyMemoryIngestResult(
            success=True,
            memories=(
                HyMemoryOutput(
                    memory_id="integration-hy-" + request.evidence_ids[0],
                    content=TEXT,
                    layer="L2_FACT",
                    owner="user",
                    evidence_ids=request.evidence_ids,
                ),
            ),
        )

    async def profile_memories(self, **kwargs):
        return ()

    async def close(self):
        self.closed += 1


class IntegrationCore:
    def __init__(self, root, monkeypatch):
        self.root = root
        self.data = root / "data"
        self.data.mkdir()
        self.home = root / "synthetic-home"
        self.home.mkdir()
        self.database = self.data / "one-memory.sqlite3"
        self.secrets = FakeSecretStore()
        self.bridge = ControlledHyBridge()
        self.probe_requests = []
        self.profile = None
        self.app = None
        self.client = None

        async def no_network(*args, **kwargs):
            pytest.fail("Real network transport forbidden in continuous integration")

        def no_sync_network(*args, **kwargs):
            pytest.fail("Real synchronous network forbidden")

        def no_unguarded(*args, **kwargs):
            pytest.fail("Production runner bypassed its permission-isolated HY boundary")

        @asynccontextmanager
        async def controlled_context(app, lease, secret, directory):
            assert directory.resolve() == self.data.resolve()
            assert app.state.evidence_store.database_path.resolve() == self.database.resolve()
            lease.recheck()
            self.bridge.leases.append(lease)
            try:
                yield self.bridge
            finally:
                await self.bridge.close()

        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_network)
        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_sync_network)
        monkeypatch.setattr(MemoryTaskRunner, "_create_bridge", no_unguarded)
        monkeypatch.setattr(api, "_permission_context", controlled_context)

    def probe_transport(self, request):
        body = request.content.decode() if request.content else ""
        assert TEXT not in body and "old-neighbor" not in body
        self.probe_requests.append((request.url.path, body))
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "integration-model"}]})
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": '{"memory_probe":true}'}}],
                },
            )
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2, 0.3]}]})
        pytest.fail("Unexpected synthetic probe path: " + request.url.path)

    @asynccontextmanager
    async def running(self, *, enable_continuous=True, automatic_memory_processing=True):
        capability = {"continuous_processing_enabled": True} if enable_continuous else {}
        self.app = api.create_app(
            Settings(
                data_dir=self.data,
                source_home=self.home,
                api_token="integration-only-token",
                instance_id="continuous-integration",
                imports_enabled=False,
                live_sync_enabled=False,
                automatic_memory_processing=automatic_memory_processing,
            ),
            source_catalog={},
            secret_store=self.secrets,
            **capability,
        )
        async with self.app.router.lifespan_context(self.app):

            async def skip_read_bridge_refresh(_snapshot):
                # These tests replace only the permission-isolated write
                # boundary. A post-write local read-client refresh is covered
                # separately and must not invoke the forbidden write factory.
                return None

            self.app.state.memory_processing_supervisor._refresh_read_bridge = (
                skip_read_bridge_refresh
            )
            self.app.state.model_prober = ModelProber(
                store=self.app.state.model_profile_store,
                secrets=self.secrets,
                transport=httpx.MockTransport(self.probe_transport),
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.app),
                base_url="http://test",
                headers={"Authorization": "Bearer integration-only-token"},
            ) as client:
                self.client = client
                assert (await client.get("/api/v1/health")).status_code == 200
                for name in (
                    "evidence_store",
                    "memory_governance_store",
                    "memory_task_store",
                    "model_profile_store",
                    "model_content_permissions",
                ):
                    assert (
                        getattr(self.app.state, name).database_path.resolve()
                        == self.database.resolve()
                    )
                yield self
        self.client = None

    def grant_source(self, source="cursor", *, enabled=True):
        root = self.home / source
        root.mkdir(exist_ok=True)
        receipt = self.app.state.source_grant_store.grant(
            user_id="local",
            source_id=source,
            root=root,
            candidate_kind=RootCandidateKind.DEFAULT,
        )
        if enabled:
            self.app.state.config_store.set_sync(source, True)
        return receipt

    def put(
        self,
        key,
        *,
        source="cursor",
        conversation="same-conversation",
        created_at=OLD,
        role="user",
        written_by=WrittenBy.PERSON,
        verbatim=None,
    ):
        return self.app.state.evidence_store.put(
            EvidenceInput(
                user_id="local",
                source_id=source,
                conversation_id=conversation,
                message_id=key,
                role=role,
                written_by=written_by,
                created_at=created_at,
                verbatim=verbatim if verbatim is not None else TEXT + " " + key,
                origin_meta={EvidenceOrigin.FILE: OriginMetadata()},
            )
        ).record

    def put_complete(
        self,
        key,
        *,
        source="cursor",
        conversation="same-conversation",
        created_at=OLD,
        verbatim=None,
    ):
        """Store a complete user/assistant turn and return its user record."""
        user = self.put(
            key,
            source=source,
            conversation=conversation,
            created_at=created_at,
            verbatim=verbatim,
        )
        self.put(
            key + "-reply",
            source=source,
            conversation=conversation,
            created_at=created_at + timedelta(microseconds=1),
            role="assistant",
            written_by=WrittenBy.AGENT,
        )
        return user

    def turn_records(self, seed):
        """Return the complete turn beginning at ``seed`` from durable Evidence."""
        conversation = self.app.state.evidence_store.list_conversation(seed.evidence_id)
        start = next(index for index, record in enumerate(conversation) if record == seed)
        turn = []
        for record in conversation[start:]:
            if turn and record.role == "user":
                break
            turn.append(record)
        return tuple(turn)

    async def save_and_probe(self):
        response = await self.client.post(
            "/api/v1/model-profiles",
            json={
                "name": "Controlled integration",
                "provider_kind": "openai_compatible",
                "base_url": "https://integration.invalid/v1",
                "model": "integration-model",
                "api_key": "synthetic-key-not-a-credential",
                "disable_thinking": True,
                "embedding_provider_kind": "openai_compatible",
                "embedding_model": "integration-vector",
                "embedding_base_url": "https://integration.invalid/v1",
                "max_concurrency": 1,
            },
        )
        assert response.status_code == 200, response.text
        self.profile = response.json()
        probe = await self.client.post(f"/api/v1/model-profiles/{self.profile['profile_id']}/probe")
        assert probe.status_code == 200, probe.text
        caps = probe.json()["capabilities"]
        assert (
            caps["endpoint_reachable"] and caps["model_available"] and caps["embedding_available"]
        )
        profiles = await self.client.get("/api/v1/model-profiles")
        assert profiles.status_code == 200
        self.profile = next(
            p for p in profiles.json() if p["profile_id"] == self.profile["profile_id"]
        )
        assert self.profile["revision"] == probe.json()["profile_revision"]
        assert self.probe_requests and not self.bridge.requests
        assert self.app.state.model_profile_store.active_identity() is None
        assert self.app.state.memory_task_store.list() == []

    def policy_current(self):
        with closing(sqlite3.connect(self.database)) as db:
            exists = db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='continuous_policy_history'"
            ).fetchone()
            return continuous_policy.current(db) if exists else None

    def kernel_enable(self, mode="none"):
        """Known helper path only; explicitly not the atomic public enable transaction."""
        store = self.app.state.model_profile_store
        store.activate(self.profile["profile_id"], self.secrets)
        profile = store.snapshot(self.profile["profile_id"], self.profile["revision"])
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            ingest_journal.initialize(db)
            continuous_policy.initialize(db)
            return continuous_policy.enable(
                db,
                profile=profile,
                expected_revision=0,
                expected_legacy_revision=0,
                operation_id=str(uuid4()),
                backlog_mode=mode,
                user_confirmed=True,
            )

    async def enable_api(self, mode="none"):
        body = dict(
            operation_id=str(uuid4()),
            profile_id=self.profile["profile_id"],
            profile_revision=self.profile["revision"],
            expected_policy_revision=0,
            expected_legacy_revision=0,
            backlog_mode=mode,
            user_confirmed=True,
        )
        response = await self.client.post(API_ROOT + "/enable", json=body)
        assert response.status_code in (200, 202), (
            f"Real enable route not integrated: {response.status_code} {response.text}"
        )
        assert (
            "secret_ref" not in response.text
            and "synthetic-key-not-a-credential" not in response.text
        )
        receipt = self.policy_current()
        assert receipt is not None and receipt.state == "enabled"
        assert self.app.state.model_profile_store.active_identity() == (
            self.profile["profile_id"],
            self.profile["revision"],
        )
        return receipt, body

    async def run_prepared(self, record):
        supervisor = self.app.state.memory_processing_supervisor
        tasks = await asyncio.to_thread(supervisor.prepare_conversation, record)
        assert tasks, "Policy accepted input but production prepare created no tasks"
        return [await supervisor._create_runner().run(task.task_id) for task in tasks]

    async def wait_completed(self, timeout=3, *, minimum_tasks=1):
        end = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < end:
            tasks = self.app.state.memory_task_store.list()
            if len(tasks) >= minimum_tasks and all(
                t.status
                in {
                    MemoryTaskStatus.COMPLETED,
                    MemoryTaskStatus.FAILED,
                    MemoryTaskStatus.WAITING_FOR_MODEL,
                }
                for t in tasks
            ):
                assert all(t.status == MemoryTaskStatus.COMPLETED for t in tasks), [
                    (t.status, t.last_error_code) for t in tasks
                ]
                return tasks
            await asyncio.sleep(0.025)
        tasks = self.app.state.memory_task_store.list()
        pytest.fail(
            "Continuous ingest/restart did not reach completed task within bounded wait: "
            + repr([(task.status, task.last_error_code, task.input_ref) for task in tasks])
        )

    async def assert_governed(self, record):
        assert self.bridge.requests, (
            "No controlled HY invocation; governance alone is not completion"
        )
        response = await self.client.get("/api/v1/memory-library", params={"status": "confirmed"})
        assert response.status_code == 200, response.text
        items = response.json()["items"]
        assert any(record.evidence_id in item["evidence_ids"] for item in items), items


@pytest.fixture
def core(tmp_path, monkeypatch):
    return IntegrationCore(tmp_path, monkeypatch)


def test_save_probe_and_kernel_boundaries_are_real_without_auto_legacy_upgrade(core):
    async def scenario():
        async with core.running():
            core.grant_source()
            old = core.put_complete("old-neighbor")
            await core.save_and_probe()
            receipt = core.kernel_enable()
            assert receipt.backlog_events == ()
            core.grant_source("workbuddy")
            new = core.put("new-old-date", source="workbuddy")
            with closing(sqlite3.connect(core.database)) as db, db:
                db.execute("BEGIN")
                assert (
                    continuous_policy.check(db, receipt, evidence_ids=(new.evidence_id,)) == receipt
                )
                with pytest.raises(continuous_policy.ContinuousPolicyError, match="outside_policy"):
                    continuous_policy.check(db, receipt, evidence_ids=(old.evidence_id,))
            assert core.app.state.model_content_permissions.receipt(user_id="local") is None
            assert core.app.state.memory_task_store.list() == [] and not core.bridge.requests

    asyncio.run(scenario())


@pytest.mark.parametrize("backlog", ["none", "include"])
def test_kernel_policy_reaches_real_guard_runner_hy_and_governance(core, backlog):
    async def scenario():
        async with core.running():
            core.grant_source()
            old = core.put_complete("old-neighbor")
            await core.save_and_probe()
            core.kernel_enable(backlog)
            record = old if backlog == "include" else core.put_complete("new-permitted")
            results = await core.run_prepared(record)
            assert all(t.status == MemoryTaskStatus.COMPLETED for t in results), results
            await core.assert_governed(record)
            if backlog == "none":
                for request in core.bridge.requests:
                    assert old.evidence_id not in request.evidence_ids
                    context = request.source_context
                    if context:
                        assert old.evidence_id not in {
                            e.evidence_id
                            for e in (*context.current, *context.history, *context.supporting)
                        }

    asyncio.run(scenario())


def test_live_batch_drops_pre_policy_neighbor_without_cancelling_new_turn(core):
    async def scenario():
        async with core.running():
            core.grant_source()
            old = core.put("old-neighbor")
            await core.save_and_probe()
            core.kernel_enable("none")
            user = core.put(
                "new-user",
                conversation=old.conversation_id,
                created_at=datetime.now(UTC),
            )
            assistant = core.put(
                "new-assistant",
                conversation=old.conversation_id,
                created_at=datetime.now(UTC),
                role="assistant",
                written_by=WrittenBy.AGENT,
            )

            completed = await core.app.state.memory_processing_supervisor.process_records(
                (old, user, assistant)
            )

            assert completed and all(task.status == "completed" for task in completed)
            sent = {
                evidence_id
                for request in core.bridge.requests
                for evidence_id in request.evidence_ids
            }
            assert old.evidence_id not in sent
            assert {user.evidence_id, assistant.evidence_id} <= sent

    asyncio.run(scenario())


def test_real_enable_api_atomic_active_policy_and_idempotence(core):
    async def scenario():
        async with core.running():
            await core.save_and_probe()
            policy, body = await core.enable_api()
            repeated = await core.client.post(API_ROOT + "/enable", json=body)
            assert repeated.status_code in (200, 202), repeated.text
            assert core.policy_current() == policy
            current = await core.client.get(API_ROOT)
            assert current.status_code == 200 and "secret_ref" not in current.text
            assert len(core.app.state.model_profile_store.list(core.secrets)) == 1

    asyncio.run(scenario())


def test_enable_future_new_source_old_dated_ingest_normal_notification(core):
    async def scenario():
        async with core.running():
            await core.save_and_probe()
            await core.enable_api()
            core.grant_source("workbuddy")
            record = core.put_complete("later-source-old-date", source="workbuddy")
            core.app.state.memory_processing_supervisor.enqueue_evidence(record)
            await core.wait_completed()
            await core.assert_governed(record)

    asyncio.run(scenario())


def test_explicit_backlog_api_positive_control(core):
    async def scenario():
        async with core.running():
            core.grant_source()
            old = core.put_complete("old-explicit-backlog")
            await core.save_and_probe()
            policy, _ = await core.enable_api("include")
            assert policy.backlog_events == ()
            assert policy.backlog_through_seq is not None
            assert policy.backlog_source_ids == ("cursor",)
            await core.wait_completed()
            await core.assert_governed(old)

    asyncio.run(scenario())


def test_import_before_model_enable_and_later_append_both_reach_memory_within_ten_seconds(core):
    async def scenario():
        async with core.running():
            core.grant_source()
            historical = core.put_complete("before-model")
            assert core.app.state.memory_task_store.list() == []
            assert core.app.state.memory_task_store.processing_metrics()["failed"] == 0

            await core.save_and_probe()
            policy, _body = await core.enable_api("include")
            assert policy.backlog_mode == "include"
            historical_tasks = await core.wait_completed(timeout=10)
            assert any(
                historical.evidence_id
                in {
                    ref["evidence_id"]
                    for ref in core.app.state.memory_task_store.inputs(task.task_id)
                }
                for task in historical_tasks
            )
            await core.assert_governed(historical)

            later = core.put("after-model", created_at=datetime.now(UTC))
            reply = core.put(
                "after-model-reply",
                created_at=datetime.now(UTC),
                role="assistant",
                written_by=WrittenBy.AGENT,
            )
            core.app.state.memory_processing_supervisor.enqueue_evidence(reply)
            deadline = asyncio.get_running_loop().time() + 10
            later_tasks = []
            while asyncio.get_running_loop().time() < deadline:
                later_tasks = core.app.state.memory_task_store.list()
                if any(
                    later.evidence_id
                    in {
                        ref["evidence_id"]
                        for ref in core.app.state.memory_task_store.inputs(task.task_id)
                    }
                    and task.status is MemoryTaskStatus.COMPLETED
                    for task in later_tasks
                ):
                    break
                await asyncio.sleep(0.025)
            assert any(
                later.evidence_id
                in {
                    ref["evidence_id"]
                    for ref in core.app.state.memory_task_store.inputs(task.task_id)
                }
                for task in later_tasks
            )
            await core.assert_governed(later)

    asyncio.run(scenario())


@pytest.mark.parametrize("revocation", ["source", "policy"])
def test_revocation_after_hy_blocks_governance_publication(core, revocation):
    async def scenario():
        async with core.running():
            core.grant_source()
            await core.save_and_probe()
            core.kernel_enable()
            record = core.put_complete("revoked-in-flight")

            def revoke():
                if revocation == "source":
                    core.app.state.source_grant_store.revoke(user_id="local", source_id="cursor")
                else:
                    with closing(sqlite3.connect(core.database)) as db, db:
                        db.execute("BEGIN IMMEDIATE")
                        current = continuous_policy.current(db)
                        continuous_policy.transition(
                            db,
                            expected_revision=current.revision,
                            action="revoke",
                            operation_id=str(uuid4()),
                            user_confirmed=True,
                        )

            core.bridge.before_return = revoke
            results = await core.run_prepared(record)
            assert core.bridge.requests, "Revocation test requires positive send admission"
            assert all(t.status != MemoryTaskStatus.COMPLETED for t in results)
            assert core.app.state.memory_governance_store.list(user_id="local") == []

    asyncio.run(scenario())


def test_kernel_future_source_normal_enqueue_uses_existing_production_debounce(core):
    async def scenario():
        async with core.running():
            await core.save_and_probe()
            core.kernel_enable()
            core.grant_source("workbuddy")
            record = core.put("future-source", source="workbuddy")
            core.put(
                "future-source-reply",
                source="workbuddy",
                role="assistant",
                written_by=WrittenBy.AGENT,
            )
            core.app.state.memory_processing_supervisor.enqueue_evidence(record)
            await core.wait_completed()
            await core.assert_governed(record)

    asyncio.run(scenario())


def test_kernel_restart_recovers_unnotified_ingest_without_legacy_grant(core):
    async def scenario():
        async with core.running():
            core.grant_source()
            await core.save_and_probe()
            core.kernel_enable()
            record = core.put_complete("committed-before-restart", conversation="restart")
            assert not core.bridge.requests
        async with core.running():
            await core.wait_completed()
            await core.assert_governed(record)
            assert core.app.state.model_content_permissions.receipt(user_id="local") is None

    asyncio.run(scenario())


def test_no_backlog_old_conversation_is_rejected_before_bridge_with_live_positive_control(core):
    async def scenario():
        async with core.running():
            core.grant_source()
            old = core.put_complete("old-only", conversation="old-only")
            await core.save_and_probe()
            core.kernel_enable()
            supervisor = core.app.state.memory_processing_supervisor
            with pytest.raises(HyMemoryBridgeError, match="model_content_evidence_outside_policy"):
                await asyncio.to_thread(supervisor.prepare_conversation, old)
            assert not core.bridge.requests
            new = core.put_complete("new-positive", conversation="new-positive")
            results = await core.run_prepared(new)
            assert all(t.status == MemoryTaskStatus.COMPLETED for t in results)
            await core.assert_governed(new)
            assert all(
                old.evidence_id not in request.evidence_ids for request in core.bridge.requests
            )

    asyncio.run(scenario())


def test_real_current_and_disabled_capability_preserve_no_side_effects(core):
    async def scenario():
        async with core.running(enable_continuous=False):
            await core.save_and_probe()
            current = await core.client.get(API_ROOT)
            assert current.status_code == 200, current.text
            value = current.json()
            assert value["available"] is True and value["policy"] is None
            assert "secret_ref" not in current.text
            assert value["capability"]["enable"] is False
            body = dict(
                operation_id=str(uuid4()),
                profile_id=core.profile["profile_id"],
                profile_revision=core.profile["revision"],
                expected_policy_revision=0,
                expected_legacy_revision=0,
                backlog_mode="none",
                user_confirmed=True,
            )
            denied = await core.client.post(API_ROOT + "/enable", json=body)
            assert denied.status_code == 409
            assert denied.json()["detail"] == "continuous_processing_unavailable"
            resume = await core.client.post(
                API_ROOT + "/resume",
                json={
                    "operation_id": str(uuid4()),
                    "expected_policy_revision": 1,
                    "user_confirmed": True,
                },
            )
            assert resume.status_code == 409
            assert resume.json()["detail"] == "continuous_processing_unavailable"
            assert core.policy_current() is None
            assert core.app.state.model_profile_store.active_identity() is None
            assert core.app.state.memory_task_store.list() == [] and not core.bridge.requests

    asyncio.run(scenario())


def test_actual_activation_service_rejects_preprobe_revision_then_commits_exact_profile(core):
    async def scenario():
        async with core.running():
            core.grant_source()
            await core.save_and_probe()
            service = core.app.state.model_policy_activation
            assert core.profile["revision"] > 1  # Probe measured the embedding width.
            body = dict(
                operation_id=str(uuid4()),
                profile_id=core.profile["profile_id"],
                profile_revision=1,
                expected_policy_revision=0,
                expected_legacy_revision=0,
                backlog_mode="none",
                user_confirmed=True,
            )
            with pytest.raises(
                continuous_policy.ContinuousPolicyError, match="model_profile_changed"
            ):
                await asyncio.to_thread(service.enable, **body)
            assert core.policy_current() is None
            assert core.app.state.model_profile_store.active_identity() is None
            body["profile_revision"] = core.profile["revision"]
            result = await asyncio.to_thread(service.enable, **body)
            assert result.created and result.policy.profile.revision == core.profile["revision"]
            assert core.policy_current() == result.policy
            record = core.put_complete("after-atomic-service")
            tasks = await core.run_prepared(record)
            assert all(t.status == MemoryTaskStatus.COMPLETED for t in tasks)
            await core.assert_governed(record)

    asyncio.run(scenario())


def test_actual_activation_conflict_does_not_partially_activate_or_create_policy(core):
    async def scenario():
        async with core.running():
            await core.save_and_probe()
            service = core.app.state.model_policy_activation
            with pytest.raises(
                continuous_policy.ContinuousPolicyError, match="policy_revision_changed"
            ):
                await asyncio.to_thread(
                    service.enable,
                    operation_id=str(uuid4()),
                    profile_id=core.profile["profile_id"],
                    profile_revision=core.profile["revision"],
                    expected_policy_revision=9,
                    expected_legacy_revision=0,
                    backlog_mode="none",
                    user_confirmed=True,
                )
            assert core.app.state.model_profile_store.active_identity() is None
            assert core.policy_current() is None
            assert core.app.state.memory_task_store.list() == [] and not core.bridge.requests

    asyncio.run(scenario())


def test_actual_current_and_enable_require_authentication(core):
    async def scenario():
        async with core.running():
            await core.save_and_probe()
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=core.app),
                base_url="http://test",
            ) as anonymous:
                assert (await anonymous.get(API_ROOT)).status_code == 401
                response = await anonymous.post(
                    API_ROOT + "/enable",
                    json={
                        "operation_id": str(uuid4()),
                        "profile_id": core.profile["profile_id"],
                        "profile_revision": core.profile["revision"],
                        "expected_policy_revision": 0,
                        "expected_legacy_revision": 0,
                        "backlog_mode": "none",
                        "user_confirmed": True,
                    },
                )
                assert response.status_code == 401
            assert core.policy_current() is None and not core.bridge.requests

    asyncio.run(scenario())


def test_legacy_revoked_policy_not_upgraded_on_restart(core):
    async def scenario():
        async with core.running():
            core.grant_source()
            record = core.put("legacy")
            await core.save_and_probe()
            store = core.app.state.model_profile_store
            store.activate(core.profile["profile_id"], core.secrets)
            permissions = core.app.state.model_content_permissions
            receipt = permissions.grant(
                permissions.snapshot(
                    user_id="local",
                    source_ids=("cursor",),
                    profile=store.snapshot(core.profile["profile_id"], core.profile["revision"]),
                    evidence=(record,),
                    include_future_additions=False,
                ),
                user_confirmed=True,
                expected_revision=0,
            )
            revoked = permissions.revoke(
                user_id="local", expected_revision=receipt.revision, user_confirmed=True
            )
        async with core.running():
            await core.app.state.memory_backfill
            assert core.app.state.model_content_permissions.receipt(user_id="local") == revoked
            assert core.policy_current() is None
            assert core.app.state.memory_task_store.list() == [] and not core.bridge.requests

    asyncio.run(scenario())


def test_restart_recovers_future_ingest_without_lost_notification(core):
    async def scenario():
        async with core.running():
            core.grant_source()
            await core.save_and_probe()
            await core.enable_api()
        # Offline arrival uses the same normal store; no synthetic queue/task is injected.
        from one_memory.imports.storage import EvidenceStore

        record = (
            EvidenceStore(core.database)
            .put(
                EvidenceInput(
                    source_id="cursor",
                    conversation_id="offline",
                    message_id="offline-new",
                    role="user",
                    written_by=WrittenBy.PERSON,
                    created_at=OLD,
                    verbatim=TEXT,
                    origin_meta={EvidenceOrigin.FILE: OriginMetadata()},
                )
            )
            .record
        )
        EvidenceStore(core.database).put(
            EvidenceInput(
                source_id="cursor",
                conversation_id="offline",
                message_id="offline-new-reply",
                role="assistant",
                written_by=WrittenBy.AGENT,
                created_at=OLD + timedelta(microseconds=1),
                verbatim=TEXT + " reply",
                origin_meta={EvidenceOrigin.FILE: OriginMetadata()},
            )
        )
        async with core.running():
            await core.wait_completed()
            await core.assert_governed(record)

    asyncio.run(scenario())


def test_api_pause_resume_revoke_keeps_original_boundary_and_refuses_revoked_resume(core):
    async def scenario():
        async with core.running():
            await core.save_and_probe()
            original, _ = await core.enable_api()
            for action, state in (
                ("pause", "paused"),
                ("resume", "enabled"),
                ("revoke", "revoked"),
            ):
                current = core.policy_current()
                response = await core.client.post(
                    API_ROOT + "/" + action,
                    json={
                        "operation_id": str(uuid4()),
                        "expected_policy_revision": current.revision,
                        "user_confirmed": True,
                    },
                )
                assert response.status_code in (200, 202), response.text
                changed = core.policy_current()
                assert changed.state == state and changed.after_seq == original.after_seq
            denied = await core.client.post(
                API_ROOT + "/resume",
                json={
                    "operation_id": str(uuid4()),
                    "expected_policy_revision": changed.revision,
                    "user_confirmed": True,
                },
            )
            assert denied.status_code == 409

    asyncio.run(scenario())


def test_reauthorize_with_backlog_includes_conversations_received_while_revoked(core):
    async def scenario():
        async with core.running():
            core.grant_source()
            await core.save_and_probe()
            original, _ = await core.enable_api("none")
            revoked_response = await core.client.post(
                API_ROOT + "/revoke",
                json={
                    "operation_id": str(uuid4()),
                    "expected_policy_revision": original.revision,
                    "user_confirmed": True,
                },
            )
            assert revoked_response.status_code == 200, revoked_response.text
            revoked = core.policy_current()
            record = core.put_complete("received-while-revoked", conversation="revoked-backlog")
            core.app.state.memory_processing_supervisor.enqueue_evidence(record)
            await asyncio.sleep(1)
            assert core.bridge.requests == []

            response = await core.client.post(
                API_ROOT + "/enable",
                json={
                    "operation_id": str(uuid4()),
                    "profile_id": core.profile["profile_id"],
                    "profile_revision": core.profile["revision"],
                    "expected_policy_revision": revoked.revision,
                    "expected_legacy_revision": revoked.legacy_revision,
                    "backlog_mode": "include",
                    "user_confirmed": True,
                },
            )
            assert response.status_code in (200, 202), response.text
            await core.wait_completed(timeout=4)
            await core.assert_governed(record)
            assert len(core.bridge.requests) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("authority", "new_conversation"),
    [
        ("continuous", "new-event-after-unknown"),
        ("continuous", "unknown-outcome"),
        ("legacy", "new-event-after-unknown"),
    ],
)
def test_sdk_admitted_running_restart_preserves_unknown_without_blocking_new_event(
    core,
    monkeypatch,
    authority,
    new_conversation,
):
    """Interrupt AFTER controlled SDK admission, not an unsent pending task.

    CancelledError emulates loss of the caller before the result is known; the
    real runner has already claimed and bound the task. No status is forged.
    Legacy is a positive control for the existing RUNNING -> PENDING behavior.
    """

    async def scenario():
        admitted = asyncio.Event()
        never_return = asyncio.Event()
        normal_ingest = core.bridge.ingest

        async def accepted_then_interrupted(request):
            core.bridge.requests.append(request)
            admitted.set()
            await never_return.wait()
            pytest.fail("Interrupted SDK invocation must not produce a result")

        async with core.running():
            core.grant_source()
            await core.save_and_probe()
            if authority == "continuous":
                core.kernel_enable()
            original = core.put_complete("sdk-admitted", conversation="unknown-outcome")
            if authority == "legacy":
                profiles = core.app.state.model_profile_store
                profiles.activate(core.profile["profile_id"], core.secrets)
                permissions = core.app.state.model_content_permissions
                permissions.grant(
                    permissions.snapshot(
                        user_id="local",
                        source_ids=("cursor",),
                        profile=profiles.snapshot(
                            core.profile["profile_id"], core.profile["revision"]
                        ),
                        evidence=core.turn_records(original),
                        include_future_additions=True,
                    ),
                    user_confirmed=True,
                    expected_revision=0,
                )
            supervisor = core.app.state.memory_processing_supervisor
            prepared = await asyncio.to_thread(supervisor.prepare_conversation, original)
            assert len(prepared) == 1
            task_id = prepared[0].task_id
            monkeypatch.setattr(core.bridge, "ingest", accepted_then_interrupted)
            running = asyncio.create_task(supervisor._create_runner().run(task_id))
            try:
                await asyncio.wait_for(admitted.wait(), timeout=3)
                task = core.app.state.memory_task_store.require(task_id)
                assert task.status == MemoryTaskStatus.RUNNING and task.attempts == 1
                assert len(core.bridge.requests) == 1
                assert original.evidence_id in core.bridge.requests[0].evidence_ids
                assert core.app.state.memory_governance_store.list(user_id="local") == []
            finally:
                running.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await running
            stranded = core.app.state.memory_task_store.require(task_id)
            assert stranded.status == MemoryTaskStatus.RUNNING and stranded.attempts == 1
            monkeypatch.setattr(core.bridge, "ingest", normal_ingest)

        with closing(sqlite3.connect(core.database)) as db:
            row = db.execute(
                "SELECT status,attempts FROM memory_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            assert row == ("running", 1), "Fixture must preserve an actually admitted RUNNING task"

        async with core.running():
            await asyncio.wait_for(core.app.state.memory_backfill, timeout=4)
            if authority == "legacy":
                await core.wait_completed(timeout=4)
                recovered = core.app.state.memory_task_store.require(task_id)
                assert recovered.status == MemoryTaskStatus.COMPLETED
                assert recovered.attempts == 2 and len(core.bridge.requests) == 2
                await core.assert_governed(original)
                return

            recovered = core.app.state.memory_task_store.require(task_id)
            assert recovered.last_error_code == "model_content_outcome_unknown", recovered
            assert recovered.status not in {
                MemoryTaskStatus.PENDING,
                MemoryTaskStatus.RUNNING,
                MemoryTaskStatus.COMPLETED,
            }, recovered
            assert recovered.attempts == 1
            assert len(core.bridge.requests) == 1, "Same admitted event was sent again on startup"

            # Repeated import observation and notification must not mint a new
            # request identity for the already admitted event.
            repeated = core.put("sdk-admitted", conversation="unknown-outcome")
            assert repeated.evidence_id == original.evidence_id
            core.app.state.memory_processing_supervisor.enqueue_evidence(repeated)
            fresh = core.put_complete("new-event-positive", conversation=new_conversation)
            core.app.state.memory_processing_supervisor.enqueue_evidence(fresh)
            end = asyncio.get_running_loop().time() + 4
            completed = []
            while asyncio.get_running_loop().time() < end:
                completed = [
                    task
                    for task in core.app.state.memory_task_store.list()
                    if task.input_ref == fresh.evidence_id
                    and task.status == MemoryTaskStatus.COMPLETED
                ]
                if completed:
                    break
                await asyncio.sleep(0.025)
            assert completed, "Unknown task must not starve a newly permitted event"
            # Both debounce callbacks have had a chance to run; new work is the
            # positive control that the restarted queue is actually live.
            await asyncio.sleep(1)
            old_calls = [r for r in core.bridge.requests if original.evidence_id in r.evidence_ids]
            new_calls = [r for r in core.bridge.requests if fresh.evidence_id in r.evidence_ids]
            assert len(old_calls) == len(new_calls) == 1
            new_context = new_calls[0].source_context
            if new_context:
                assert original.evidence_id not in {
                    item.evidence_id
                    for item in (
                        *new_context.current,
                        *new_context.history,
                        *new_context.supporting,
                    )
                }, "Unknown old event must not be resent as context for the new event"
            unchanged = core.app.state.memory_task_store.require(task_id)
            assert unchanged.attempts == 1
            assert unchanged.last_error_code == "model_content_outcome_unknown"
            await core.assert_governed(fresh)
            assert all(
                original.evidence_id not in memory.evidence_ids
                for memory in core.app.state.memory_governance_store.list(user_id="local")
            )

    asyncio.run(scenario())


def test_public_none_policy_excludes_old_history_and_does_not_replay_legacy_startup(core):
    async def scenario():
        async with core.running():
            core.grant_source()
            old = core.put_complete("old-neighbor", conversation="mixed-public")
            await core.save_and_probe()
            await core.enable_api("none")
            fresh = core.put_complete("public-future", conversation="mixed-public")
            core.app.state.memory_processing_supervisor.enqueue_evidence(fresh)
            await core.wait_completed(timeout=4)
            await core.assert_governed(fresh)
            assert all(old.evidence_id not in r.evidence_ids for r in core.bridge.requests)
            for request in core.bridge.requests:
                if request.source_context:
                    context = request.source_context
                    assert old.evidence_id not in {
                        e.evidence_id
                        for e in (*context.current, *context.history, *context.supporting)
                    }
            calls = len(core.bridge.requests)
        async with core.running():
            await core.app.state.memory_backfill
            await asyncio.sleep(2.2)  # Covers the coordinator's normal monitor interval.
            assert len(core.bridge.requests) == calls
            assert core.app.state.model_content_permissions.receipt(user_id="local") is None
            assert all(
                t.status == MemoryTaskStatus.COMPLETED
                for t in core.app.state.memory_task_store.list()
            )

    asyncio.run(scenario())


def test_public_revoke_after_sdk_admission_blocks_publish_and_future_work(core, monkeypatch):
    async def scenario():
        admitted, release = asyncio.Event(), asyncio.Event()
        original_ingest = core.bridge.ingest

        async def hold_response(request):
            admitted.set()
            await release.wait()
            return await original_ingest(request)

        async with core.running():
            core.grant_source()
            await core.save_and_probe()
            policy, _ = await core.enable_api()
            monkeypatch.setattr(core.bridge, "ingest", hold_response)
            record = core.put_complete("public-in-flight", conversation="revoke-public")
            core.app.state.memory_processing_supervisor.enqueue_evidence(record)
            try:
                await asyncio.wait_for(admitted.wait(), 4)
                current = core.app.state.memory_task_store.list()
                assert any(t.status == MemoryTaskStatus.RUNNING for t in current)
                revoked = await core.client.post(
                    API_ROOT + "/revoke",
                    json={
                        "operation_id": str(uuid4()),
                        "expected_policy_revision": policy.revision,
                        "user_confirmed": True,
                    },
                )
                assert revoked.status_code == 200, revoked.text
                assert revoked.json()["state"] == "revoked"
            finally:
                release.set()
            deadline = asyncio.get_running_loop().time() + 4
            while asyncio.get_running_loop().time() < deadline:
                tasks = core.app.state.memory_task_store.list()
                if tasks and all(t.status != MemoryTaskStatus.RUNNING for t in tasks):
                    break
                await asyncio.sleep(0.025)
            assert tasks and all(t.status == MemoryTaskStatus.CANCELLED for t in tasks)
            assert len(core.bridge.requests) == 1
            assert core.app.state.memory_governance_store.list(user_id="local") == []
            next_record = core.put("after-revoke", conversation="after-revoke")
            core.app.state.memory_processing_supervisor.enqueue_evidence(next_record)
            await asyncio.sleep(2.2)
            assert len(core.bridge.requests) == 1
            assert core.app.state.memory_governance_store.list(user_id="local") == []

    asyncio.run(scenario())


def test_public_pause_accumulation_resumes_through_actual_callback(core):
    async def scenario():
        async with core.running():
            core.grant_source()
            await core.save_and_probe()
            policy, _ = await core.enable_api()
            paused = await core.client.post(
                API_ROOT + "/pause",
                json={
                    "operation_id": str(uuid4()),
                    "expected_policy_revision": policy.revision,
                    "user_confirmed": True,
                },
            )
            assert paused.status_code == 200, paused.text
            record = core.put_complete("while-paused", conversation="paused")
            core.app.state.memory_processing_supervisor.enqueue_evidence(record)
            await asyncio.sleep(1)
            assert core.bridge.requests == []
            resumed = await core.client.post(
                API_ROOT + "/resume",
                json={
                    "operation_id": str(uuid4()),
                    "expected_policy_revision": paused.json()["revision"],
                    "user_confirmed": True,
                },
            )
            assert resumed.status_code == 200, resumed.text
            assert resumed.json()["after_seq"] == policy.after_seq
            await core.wait_completed(timeout=4)
            await core.assert_governed(record)
            assert len(core.bridge.requests) == 1

    asyncio.run(scenario())


def test_public_memory_note_auto_processes_under_continuous_policy_without_legacy_grant(core):
    async def scenario():
        async with core.running():
            await core.save_and_probe()
            policy, _ = await core.enable_api("none")
            with closing(sqlite3.connect(core.database)) as db:
                assert (
                    db.execute(
                        "SELECT 1 FROM source_grants WHERE user_id='local' "
                        "AND source_id='one-memory-manual'"
                    ).fetchone()
                    is None
                )
            note = {"request_id": str(uuid4()), "text": TEXT, "scope": "all-apps"}
            response = await core.client.post("/api/v1/memory-notes", json=note)
            assert response.status_code == 201, response.text
            saved = response.json()
            assert saved["state"] in {
                "pending",
                "running",
                "waiting_for_model",
                "ready",
            }, saved
            assert saved["error"] is None
            record = core.app.state.evidence_store.get(saved["evidence_id"])
            assert record is not None and record.source_id == "one-memory-manual"
            assert record.verbatim == TEXT and record.written_by == WrittenBy.PERSON
            with closing(sqlite3.connect(core.database)) as db:
                db.execute("BEGIN")
                grant = db.execute(
                    "SELECT root_path,revoked_at FROM source_grants WHERE user_id='local' "
                    "AND source_id='one-memory-manual'"
                ).fetchone()
                assert grant == (str(core.data), None)
                event = ingest_journal.current(db, "local", record.evidence_id)
                assert event is not None and event.seq > policy.after_seq
                assert event.generation_eligible and ingest_journal.matches_current(db, event)
                continuous_policy.check(db, policy, evidence_ids=(record.evidence_id,))
            assert core.app.state.model_content_permissions.receipt(user_id="local") is None
            # No /process request or second content consent: the normal POST owns scheduling.
            await core.wait_completed(timeout=4)
            await core.assert_governed(record)
            final = await core.client.get(f"/api/v1/memory-notes/{record.evidence_id}")
            assert final.status_code == 200, final.text
            assert final.json()["state"] == "ready" and final.json()["memory_ids"]
            assert final.json()["error"] is None
            listing = await core.client.get("/api/v1/memory-notes")
            assert listing.status_code == 200
            assert any(
                item["evidence_id"] == record.evidence_id and item["state"] == "ready"
                for item in listing.json()
            )
            calls = [r for r in core.bridge.requests if record.evidence_id in r.evidence_ids]
            assert len(calls) == 1 and calls[0].source_id == "one-memory-manual"
            repeated = await core.client.post("/api/v1/memory-notes", json=note)
            assert repeated.status_code == 201, repeated.text
            assert repeated.json()["evidence_id"] == record.evidence_id
            assert repeated.json()["state"] == "ready"
            assert core.app.state.model_content_permissions.receipt(user_id="local") is None
            assert core.policy_current() == policy
            assert len(core.bridge.requests) == 1

    asyncio.run(scenario())


def test_the_summary_tells_running_rows_from_work_actually_being_done(core):
    """`running` rows are not the current concurrency.

    A task is marked running before anything slow happens, so a worker that
    gives up — a timeout, a crash between steps — leaves a row that still says
    running. Reporting that as "what is running" is how a stalled queue looks
    busy, so the count stays as `running` and rows nobody owns are broken out
    as `stranded_running`.
    """

    async def scenario():
        async with core.running():
            core.grant_source()
            await core.save_and_probe()
            core.kernel_enable()
            # Make the difference real: a row the store marks running that no
            # worker owns. Without this the two numbers are both zero and a
            # hardcoded zero would pass just as well.
            stranded = core.app.state.memory_task_store
            record = core.put("stranded-task")
            task = stranded.create(
                request_id=f"native:{record.evidence_id}:{record.revision}:strand",
                user_id=record.user_id,
                kind="extract",
                input_ref=record.evidence_id,
                profile=core.app.state.model_profile_store.list(core.secrets)[0],
            )
            stranded.claim(task.task_id)
            assert stranded.require(task.task_id).status == "running"

            response = await core.client.get("/api/v1/memory-processing")
            assert response.status_code == 200, response.text
            summary = response.json()
            for field in ("running", "in_flight_count", "stranded_running"):
                assert field in summary, (field, summary)
            assert summary["stranded_running"] >= 0
            assert summary["stranded_running"] == max(
                0, summary["running"] - summary["in_flight_count"]
            )
            # The strand is real here, so it must be visible: this is the
            # assertion a hardcoded zero fails.
            assert summary["stranded_running"] >= 1, summary

    asyncio.run(scenario())
