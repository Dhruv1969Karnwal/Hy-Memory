import asyncio
import sqlite3
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from one_memory.api import create_app
from one_memory.config import Settings
from one_memory.hy_memory_bridge import HyMemoryBridgeError, HyMemoryIngestResult, HyMemoryOutput
from one_memory.imports.models import WrittenBy
from one_memory.model_profiles import ModelCapabilities, ModelProfileInput, ProviderKind
from one_memory.secrets import InMemorySecretStore


def application(tmp_path):
    return create_app(
        Settings(
            data_dir=tmp_path / "data",
            source_home=tmp_path,
            api_token=SecretStr("note-test"),
            imports_enabled=False,
            automatic_memory_processing=False,
            live_sync_enabled=False,
        ),
        secret_store=InMemorySecretStore(),
    )


def client_for(app):
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer note-test"},
    )


async def wait_for_note_job(app, evidence_id, *, timeout=10.0):
    """Wait for the durable note job to leave pending/running.

    The note now reaches HY through the shared durable queue
    (`entry:<evidence_id>`), so the drain - not the supervisor's in-memory
    workers - is what completes it. Waiting on `processor._workers` directly
    races that drain and can observe zero requests.
    """
    import time as _time

    deadline = _time.monotonic() + timeout
    key = f"entry:{evidence_id}"
    while _time.monotonic() < deadline:
        with sqlite3.connect(app.state.evidence_store.database_path) as db:
            row = db.execute(
                "SELECT state FROM memory_write_jobs WHERE idempotency_key=?",
                (key,),
            ).fetchone()
        if row is not None and row[0] not in {"pending", "running"}:
            return row[0]
        await asyncio.sleep(0.05)
    raise AssertionError(f"note job for {evidence_id} never settled")


def test_empty_library_can_save_a_note_without_a_matter_or_configured_model(tmp_path):
    async def scenario():
        app = application(tmp_path)
        async with app.router.lifespan_context(app), client_for(app) as client:
            assert app.state.evidence_store.count() == 0
            note = {
                "request_id": str(uuid4()),
                "text": "请记住：项目周会不排在周五。",
                "scope": "all-apps",
            }
            response = await client.post("/api/v1/memory-notes", json=note)
            assert response.status_code == 201
            saved = response.json()
            assert saved["state"] == "waiting_for_model"
            assert saved["memory_ids"] == []
            record = app.state.evidence_store.get(saved["evidence_id"])
            assert record.source_id == "one-memory-manual"
            assert record.role == "user" and record.written_by == WrittenBy.PERSON
            assert record.verbatim == note["text"] and record.created_at is not None
            assert app.state.matter_store.list() == []
            again = await client.post("/api/v1/memory-notes", json=note)
            assert again.json()["evidence_id"] == saved["evidence_id"]
            operation = await client.get(f"/api/v1/memory-notes/operations/{note['request_id']}")
            assert operation.status_code == 200
            assert operation.json()["evidence_id"] == saved["evidence_id"]
            assert (
                await client.get(f"/api/v1/memory-notes/operations/{uuid4()}")
            ).status_code == 404
            assert app.state.evidence_store.count() == 1
            conflict = await client.post(
                "/api/v1/memory-notes", json={**note, "text": "Changed content"}
            )
            assert conflict.status_code == 409
            assert app.state.evidence_store.get(saved["evidence_id"]).verbatim == note["text"]
        restarted = application(tmp_path)
        async with restarted.router.lifespan_context(restarted), client_for(restarted) as client:
            notes = (await client.get("/api/v1/memory-notes")).json()
            assert len(notes) == 1 and notes[0]["evidence_id"] == saved["evidence_id"]

    asyncio.run(scenario())


def test_note_requires_authentication_explicit_scope_and_nonempty_content(tmp_path):
    async def scenario():
        app = application(tmp_path)
        async with app.router.lifespan_context(app), client_for(app) as client:
            base = {"request_id": str(uuid4()), "text": "Remember this.", "scope": "all-apps"}
            for change in (
                {"text": "  \n"},
                {"text": "x" * 12_001},
                {"scope": "*"},
                {"user_id": "other"},
            ):
                assert (
                    await client.post("/api/v1/memory-notes", json={**base, **change})
                ).status_code == 422
            assert (
                await client.post(
                    "/api/v1/memory-notes",
                    json={"request_id": base["request_id"], "text": "No scope"},
                )
            ).status_code == 422
            client.headers.pop("Authorization")
            assert (await client.post("/api/v1/memory-notes", json=base)).status_code == 401
            assert app.state.evidence_store.count() == 0

    asyncio.run(scenario())


def test_saved_note_evidence_returns_raw_and_consent_reference_from_same_record(tmp_path):
    async def scenario():
        app = application(tmp_path)
        async with app.router.lifespan_context(app), client_for(app) as client:
            text = "Original  spacing\n预算<5，不能改为>5。"
            saved = (
                await client.post(
                    "/api/v1/memory-notes",
                    json={
                        "request_id": str(uuid4()),
                        "text": text,
                        "scope": "all-apps",
                    },
                )
            ).json()
            endpoint = f"/api/v1/memory-notes/{saved['evidence_id']}/evidence"
            response = await client.get(endpoint)
            assert response.status_code == 200, response.text
            value = response.json()
            assert value["verbatim"] == text
            assert value["evidence_id"] == saved["evidence_id"] and value["revision"] == 1
            assert value["role"] == "user" and value["written_by"] == "person"
            assert value["available"] is True and value["source_id"] == "one-memory-manual"
            from one_memory.model_content_permissions import model_content_evidence_snapshot

            record = app.state.evidence_store.get(saved["evidence_id"])
            reference = model_content_evidence_snapshot(record).model_dump(mode="json")
            assert all(value[key] == expected for key, expected in reference.items())
            assert app.state.model_content_permissions.receipt(user_id="local") is None
            assert app.state.memory_task_store.list() == []
            assert (await client.get("/api/v1/memory-notes/missing/evidence")).status_code == 404
            app.state.source_grant_store.revoke(user_id="local", source_id="one-memory-manual")
            denied = await client.get(endpoint)
            assert denied.status_code == 409 and text not in denied.text
            client.headers.pop("Authorization")
            assert (await client.get(endpoint)).status_code == 401

    asyncio.run(scenario())


def test_note_and_source_grant_are_atomic(tmp_path):
    from one_memory.memory_notes import MemoryNoteInput

    async def scenario():
        app = application(tmp_path)
        async with app.router.lifespan_context(app):
            with sqlite3.connect(app.state.evidence_store.database_path) as db:
                db.execute(
                    "CREATE TRIGGER fail_manual_grant BEFORE INSERT ON source_grants "
                    "BEGIN SELECT RAISE(ABORT, 'grant failed'); END"
                )
            with pytest.raises(sqlite3.IntegrityError, match="grant failed"):
                app.state.memory_note_store.save(
                    MemoryNoteInput(
                        request_id=uuid4(), text="Do not leave half a save.", scope="all-apps"
                    )
                )
            assert app.state.evidence_store.count() == 0

    asyncio.run(scenario())


def test_revoked_note_source_is_not_silently_reauthorized(tmp_path):
    async def scenario():
        app = application(tmp_path)
        async with app.router.lifespan_context(app), client_for(app) as client:
            note = {"request_id": str(uuid4()), "text": "Remember this.", "scope": "all-apps"}
            saved = (await client.post("/api/v1/memory-notes", json=note)).json()
            with sqlite3.connect(app.state.evidence_store.database_path) as db:
                db.execute(
                    "UPDATE source_grants SET revoked_at='2026-09-12' "
                    "WHERE source_id='one-memory-manual'"
                )
            assert (await client.post("/api/v1/memory-notes", json=note)).status_code == 409
            assert (
                await client.post(f"/api/v1/memory-notes/{saved['evidence_id']}/process", json={})
            ).status_code == 409
            assert (await client.get("/api/v1/memory-notes")).json() == []

    asyncio.run(scenario())


@pytest.mark.parametrize("fail", [False, True])
def test_note_reuses_hy_queue_and_never_reports_saved_raw_text_as_extracted(tmp_path, fail):
    async def scenario():
        app = application(tmp_path)
        requests = []

        class Bridge:
            async def ingest(self, request):
                pytest.fail("A note the person asked to keep must be imported, not re-extracted")

            async def ingest_extracted(self, request, contents):
                assert list(contents) == [turn.content for turn in request.turns]
                requests.append(request)
                if fail:
                    raise HyMemoryBridgeError("hy_memory_unavailable")
                return HyMemoryIngestResult(
                    success=True,
                    memories=(
                        HyMemoryOutput(
                            memory_id="note-memory",
                            content="用户不在周五安排项目周会。",
                            owner="user",
                            layer="L2_FACT",
                        ),
                    ),
                )

            async def profile_memories(self, **kwargs):
                return ()

            async def close(self):
                pass

        async with app.router.lifespan_context(app), client_for(app) as client:
            profiles, secrets = app.state.model_profile_store, app.state.secret_store
            profile = profiles.create(
                ModelProfileInput(name="Test", provider_kind=ProviderKind.OLLAMA, model="fixture"),
                secrets,
            )
            profiles.record_capabilities(
                profile.profile_id,
                expected_revision=profile.revision,
                capabilities=ModelCapabilities(
                    endpoint_reachable=True, model_available=True, embedding_available=True
                ),
                secrets=secrets,
            )
            profiles.activate(profile.profile_id, secrets)
            processor = app.state.memory_processing_supervisor

            @asynccontextmanager
            async def isolated_bridge(lease, _secret, _data):
                lease.recheck()
                yield Bridge()

            processor.permission_bridge_factory = isolated_bridge
            response = await client.post(
                "/api/v1/memory-notes",
                json={
                    "request_id": str(uuid4()),
                    "text": "请记住：项目周会不排在周五。",
                    "scope": "all-apps",
                },
            )
            assert response.status_code == 201
            saved = response.json()
            assert saved["state"] == "waiting_for_permission"
            assert requests == []
            # Reproduce E5: the source is revoked while this unsent durable
            # note is parked, so the queue records a terminal cancellation.
            # Explicit processing after a fresh grant must safely revive it.
            with sqlite3.connect(app.state.evidence_store.database_path) as db:
                db.execute(
                    "UPDATE memory_write_jobs SET state='cancelled',"
                    "last_error='model_content_permission_revoked',hy_invoked=0 "
                    "WHERE idempotency_key=?",
                    (f"entry:{saved['evidence_id']}",),
                )
            preview = await client.post(
                "/api/v1/model-content-permission/preview",
                json={
                    "profile_id": profile.profile_id,
                    "profile_revision": profile.revision,
                    "source_ids": ["one-memory-manual"],
                    "evidence_ids": [saved["evidence_id"]],
                },
            )
            assert preview.status_code == 200
            consent = await client.post(
                "/api/v1/model-content-permission",
                json={
                    "snapshot": preview.json(),
                    "expected_revision": 0,
                    "user_confirmed": True,
                },
            )
            assert consent.status_code == 200
            started = await client.post(f"/api/v1/memory-notes/{saved['evidence_id']}/process")
            assert started.status_code == 200
            # The note is completed by the shared durable queue now; wait on its
            # job rather than the supervisor's in-memory workers.
            await wait_for_note_job(app, saved["evidence_id"])
            receipt = (await client.get(f"/api/v1/memory-notes/{saved['evidence_id']}")).json()
            assert (
                len(requests) == 1
                and requests[0].turns[0].content == "请记住：项目周会不排在周五。"
            )
            assert app.state.evidence_store.count() == 1
            if fail:
                assert receipt["state"] == "waiting_for_model" and not receipt["memory_ids"]
                assert receipt["error"] == "hy_memory_unavailable"
            else:
                assert receipt["state"] == "ready" and receipt["memory_ids"] == ["note-memory"]
                memory = app.state.memory_governance_store.get("note-memory", user_id="local")
                assert memory.scopes == ("all-apps",)
                recall = await client.post("/api/v1/recall", json={"query": "周五 项目周会"})
                assert recall.status_code == 200
                assert any(hit["memory_id"] == "note-memory" for hit in recall.json()["hits"])
                await client.post(f"/api/v1/memory-notes/{saved['evidence_id']}/process", json={})
                await wait_for_note_job(app, saved["evidence_id"])
                assert len(requests) == 1

    asyncio.run(scenario())
