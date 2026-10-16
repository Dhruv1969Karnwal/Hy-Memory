"""T04: what the person explicitly asks to remember is written through HY's import path.

Whole native HY client behind the production permission context; only the model
endpoints are scripted through an httpx MockTransport, never a network.
"""

import asyncio
import hashlib
import json
import sqlite3
from contextlib import asynccontextmanager, closing
from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from one_memory import continuous_policy, ingest_journal
from one_memory.api import create_app
from one_memory.config import Settings
from one_memory.hy_memory_bridge import HyMemoryIngestResult, HyMemoryOutput
from one_memory.hy_permission_context import permission_bridge_factory
from one_memory.imports.models import EvidenceInput, EvidenceOrigin, OriginMetadata, WrittenBy
from one_memory.matters.selection import select_evidence
from one_memory.memory_notes import SOURCE_ID
from one_memory.model_profiles import ModelCapabilities, ModelProfileInput, ProviderKind
from one_memory.secrets import InMemorySecretStore
from tests.source_consent import confirm_model_content
from tests.test_memory_processing import FakeBridge, _runner, _setup, _task

TEXT = "请记住：项目周会固定不排在周五。"


def _vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    values = [float(byte) + 1.0 for byte in digest * 12]
    norm = sum(value * value for value in values) ** 0.5
    return [value / norm for value in values]


class ScriptedModel:
    """The model endpoints HY reaches. Reconcile says "nothing to change" unless told."""

    def __init__(self) -> None:
        self.reconcile_prompts: list[str] = []
        self.other_prompts: list[str] = []
        self.embeddings_down = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path.endswith("embeddings"):
            if self.embeddings_down:
                return httpx.Response(503, json={"error": "down"})
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "test-embedding",
                    "data": [
                        {"index": i, "object": "embedding", "embedding": _vector(text)}
                        for i, text in enumerate(body["input"])
                    ],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
        prompt = "\n".join(str(message.get("content")) for message in body["messages"])
        reconcile = "Existing memories" in prompt
        (self.reconcile_prompts if reconcile else self.other_prompts).append(prompt)
        # Extraction or reconcile: the model finds nothing worth writing.
        content = "[]" if reconcile else '{"memory": [], "identity": [], "intentions": []}'
        return httpx.Response(
            200,
            json={
                "id": "scripted",
                "object": "chat.completion",
                "created": 1,
                "model": "test-generation",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )


@pytest.fixture
def hy_calls(monkeypatch):
    from hy_memory.client import HyMemoryClient

    calls: list[tuple[str, object]] = []
    original_add = HyMemoryClient.add
    original_extracted = HyMemoryClient.add_extracted

    def add(self, data, **kwargs):
        calls.append(("add", [turn["content"] for turn in data]))
        return original_add(self, data, **kwargs)

    def add_extracted(self, contents, **kwargs):
        raw = original_extracted(self, contents, **kwargs)
        calls.append(("add_extracted", {"contents": list(contents), "raw": raw}))
        return raw

    monkeypatch.setattr(HyMemoryClient, "add", add)
    monkeypatch.setattr(HyMemoryClient, "add_extracted", add_extracted)
    return calls


def _application(tmp_path):
    return create_app(
        Settings(
            data_dir=tmp_path / "data",
            source_home=tmp_path,
            api_token=SecretStr("t04"),
            imports_enabled=False,
            automatic_memory_processing=False,
            live_sync_enabled=False,
        ),
        secret_store=InMemorySecretStore(),
    )


async def _configure(app, model: ScriptedModel) -> None:
    profiles, secrets = app.state.model_profile_store, app.state.secret_store
    profile = profiles.create(
        ModelProfileInput(
            name="Scripted",
            provider_kind=ProviderKind.OLLAMA,
            base_url="http://127.0.0.1:1/v1",
            model="test-generation",
            embedding_provider_kind=ProviderKind.OLLAMA,
            embedding_base_url="http://127.0.0.1:2/v1",
            embedding_model="test-embedding",
        ),
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
    processor.permission_bridge_factory = permission_bridge_factory(
        processor._bridge, http_transport=httpx.MockTransport(model)
    )


async def _remember(app, client, text: str) -> dict:
    saved = (
        await client.post(
            "/api/v1/memory-notes",
            json={"request_id": str(uuid4()), "text": text, "scope": "all-apps"},
        )
    ).json()
    await confirm_model_content(
        client,
        source_ids=("one-memory-manual",),
        evidence_ids=(saved["evidence_id"],),
        future_additions=False,
    )
    started = await client.post(f"/api/v1/memory-notes/{saved['evidence_id']}/process")
    assert started.status_code == 200, started.text
    await _wait_for_note_job(app, saved["evidence_id"])
    return (await client.get(f"/api/v1/memory-notes/{saved['evidence_id']}")).json()


async def _wait_for_note_job(app, evidence_id: str, *, timeout: float = 10) -> str:
    """Wait for the production durable note job, not the legacy worker map."""
    deadline = asyncio.get_running_loop().time() + timeout
    key = f"entry:{evidence_id}"
    while asyncio.get_running_loop().time() < deadline:
        with sqlite3.connect(app.state.evidence_store.database_path) as db:
            row = db.execute(
                "SELECT state FROM memory_write_jobs WHERE idempotency_key=?",
                (key,),
            ).fetchone()
        if row is not None and row[0] not in {"pending", "running"}:
            return str(row[0])
        await asyncio.sleep(0.05)
    raise AssertionError(f"note job for {evidence_id} never settled")


def _client(app):
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer t04"},
    )


def test_note_is_written_through_hy_import_even_when_the_model_would_extract_nothing(
    tmp_path, hy_calls
):
    async def scenario():
        app, model = _application(tmp_path), ScriptedModel()
        async with app.router.lifespan_context(app), _client(app) as client:
            await _configure(app, model)
            receipt = await _remember(app, client, TEXT)

            assert receipt["state"] == "ready", (receipt, [name for name, _ in hy_calls])
            assert [name for name, _ in hy_calls] == ["add_extracted"]
            assert hy_calls[0][1]["contents"] == [TEXT]
            assert len(receipt["memory_ids"]) == 1
            memory = app.state.memory_governance_store.get(
                receipt["memory_ids"][0], user_id="local"
            )
            assert memory is not None and memory.content == TEXT
            assert memory.status == "confirmed" and memory.scopes == ("all-apps",)
            assert receipt["evidence_id"] in memory.evidence_ids
            assert memory.memory_id in hy_calls[0][1]["raw"]["memory_ids"]
            assert app.state.evidence_store.get(receipt["evidence_id"]).verbatim == TEXT
            # Still the person's own words: matter discovery keeps reading the note.
            selected = select_evidence(app.state.evidence_store.database_path)
            assert receipt["evidence_id"] in {record.evidence_id for record in selected.records}

    asyncio.run(scenario())


def test_enabled_continuous_policy_processes_a_saved_note_without_separate_consent(
    tmp_path, hy_calls
):
    async def scenario():
        app, model = _application(tmp_path), ScriptedModel()
        async with app.router.lifespan_context(app), _client(app) as client:
            await _configure(app, model)
            app.state.memory_note_store.intake = None
            profile = app.state.model_profile_store.active(app.state.secret_store)
            assert profile is not None
            with app.state.evidence_store._connect(writing=True) as db:
                ingest_journal.initialize(db)
                continuous_policy.initialize(db)
                continuous_policy.enable(
                    db,
                    profile=app.state.model_profile_store.snapshot(
                        profile.profile_id, profile.revision
                    ),
                    expected_revision=0,
                    expected_legacy_revision=0,
                    operation_id=str(uuid4()),
                    backlog_mode="none",
                    user_confirmed=True,
                )

            saved = await client.post(
                "/api/v1/memory-notes",
                json={"request_id": str(uuid4()), "text": TEXT, "scope": "all-apps"},
            )
            assert saved.status_code == 201, saved.text
            processor = app.state.memory_processing_supervisor
            await asyncio.gather(*tuple(processor._workers.values()))
            receipt = (
                await client.get(f"/api/v1/memory-notes/{saved.json()['evidence_id']}")
            ).json()

            assert receipt["state"] == "ready"
            assert [name for name, _ in hy_calls] == ["add_extracted"]
            with app.state.evidence_store._connect() as db:
                assert (
                    db.execute("SELECT COUNT(*) FROM model_content_permissions").fetchone()[0] == 0
                )

    asyncio.run(scenario())


@pytest.mark.xfail(
    strict=True,
    reason=(
        "T04 wait-user: a note's permission lease covers only that note's evidence, so HY "
        "cannot see the memory an earlier note produced and adds a second one. Letting it "
        "see earlier notes means sending them to the model with this one: a consent-scope "
        "decision. The bridge's no-ids fallback also cannot find it (decorated search "
        "content; the leased vector store has no keyword_search)."
    ),
)
def test_identical_second_note_points_at_the_memory_hy_already_holds(tmp_path, hy_calls):
    async def scenario():
        app, model = _application(tmp_path), ScriptedModel()
        async with app.router.lifespan_context(app), _client(app) as client:
            await _configure(app, model)
            first = await _remember(app, client, TEXT)
            assert first["state"] == "ready", first
            second = await _remember(app, client, TEXT)

            # HY really deduplicated: reconcile saw the first memory and wrote nothing new.
            assert len(model.reconcile_prompts) == 1 and TEXT in model.reconcile_prompts[0]
            assert first["memory_ids"][0] in model.reconcile_prompts[0]
            assert not hy_calls[-1][1]["raw"].get("memory_ids")
            assert second["state"] == "ready", second
            assert second["memory_ids"] == first["memory_ids"]
            memories = app.state.memory_governance_store.list(user_id="local")
            assert [item.memory_id for item in memories] == first["memory_ids"]
            assert {first["evidence_id"], second["evidence_id"]} <= set(memories[0].evidence_ids)

    asyncio.run(scenario())


def test_note_saved_before_t04_is_imported_too_and_keeps_all_apps_scope(tmp_path):
    """Notes stored before T04 are not migrated; retrying one must still write it as said."""
    calls = []

    class Bridge:
        async def ingest(self, request):
            pytest.fail("A note the person asked to keep must be imported, not re-extracted")

        async def ingest_extracted(self, request, contents):
            calls.append(list(contents))
            return HyMemoryIngestResult(
                success=True,
                memories=(
                    HyMemoryOutput(
                        memory_id="legacy-note-memory",
                        content=TEXT,
                        owner="user",
                        layer="L2_FACT",
                    ),
                ),
            )

        async def profile_memories(self, **kwargs):
            return ()

        async def close(self):
            pass

    async def scenario():
        app, model = _application(tmp_path), ScriptedModel()
        async with app.router.lifespan_context(app), _client(app) as client:
            await _configure(app, model)
            request_id = str(uuid4())
            record = app.state.evidence_store.put(
                EvidenceInput(
                    source_id=SOURCE_ID,
                    conversation_id=f"note:{request_id}",
                    message_id=request_id,
                    role="user",
                    written_by=WrittenBy.PERSON,
                    created_at=datetime.now(UTC),
                    verbatim=TEXT,
                    origin_meta={
                        EvidenceOrigin.MANUAL_IMPORT: OriginMetadata(
                            raw_meta={
                                "input_mode": "direct_memory_note",
                                "recall_scope": "all-apps",
                            }
                        )
                    },
                )
            ).record
            database = app.state.evidence_store.database_path
            with closing(sqlite3.connect(database)) as db, db:
                db.execute(
                    "INSERT INTO source_grants VALUES ('local',?,?,'default',?,NULL)",
                    (SOURCE_ID, str(database.parent), datetime.now(UTC).isoformat()),
                )
            await confirm_model_content(
                client,
                source_ids=(SOURCE_ID,),
                evidence_ids=(record.evidence_id,),
                future_additions=False,
            )
            processor = app.state.memory_processing_supervisor

            @asynccontextmanager
            async def isolated(lease, _secret, _data):
                lease.recheck()
                yield Bridge()

            processor.permission_bridge_factory = isolated
            app.state.memory_write_drain.stop()
            await app.state.memory_write_drain_task
            started = await client.post(f"/api/v1/memory-notes/{record.evidence_id}/process")
            assert started.status_code == 200, started.text
            assert await app.state.memory_write_drain.run_one()
            deadline = asyncio.get_running_loop().time() + 5
            while True:
                await asyncio.gather(*tuple(processor._workers.values()))
                receipt = (await client.get(f"/api/v1/memory-notes/{record.evidence_id}")).json()
                if receipt["state"] != "running" or asyncio.get_running_loop().time() >= deadline:
                    break
                await asyncio.sleep(0.01)
            assert calls == [[TEXT]]
            assert receipt["state"] == "ready" and receipt["memory_ids"] == ["legacy-note-memory"]
            memory = app.state.memory_governance_store.get("legacy-note-memory", user_id="local")
            assert memory.scopes == ("all-apps",) and memory.content == TEXT
            assert record.evidence_id in memory.evidence_ids

    asyncio.run(scenario())


def test_model_outage_keeps_the_original_words_and_a_retry_writes_them(tmp_path, hy_calls):
    async def scenario():
        app, model = _application(tmp_path), ScriptedModel()
        async with app.router.lifespan_context(app), _client(app) as client:
            await _configure(app, model)
            model.embeddings_down = True
            failed = await _remember(app, client, TEXT)
            assert failed["state"] in {"waiting_for_model", "failed"}, failed
            assert failed["memory_ids"] == []
            assert app.state.evidence_store.get(failed["evidence_id"]).verbatim == TEXT
            assert app.state.memory_governance_store.list(user_id="local") == []

            model.embeddings_down = False
            retried = await client.post(
                "/api/v1/memory-processing/retry-unknown", json={"confirm": True}
            )
            assert retried.status_code == 200, retried.text
            await _wait_for_note_job(app, failed["evidence_id"])
            receipt = (await client.get(f"/api/v1/memory-notes/{failed['evidence_id']}")).json()
            assert receipt["state"] == "ready", receipt
            assert hy_calls[-1][0] == "add_extracted" and hy_calls[-1][1]["contents"] == [TEXT]
            memory = app.state.memory_governance_store.get(
                receipt["memory_ids"][0], user_id="local"
            )
            assert memory.content == TEXT and failed["evidence_id"] in memory.evidence_ids

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("raw_meta", "imported"),
    [
        ({"input_mode": "direct_memory_note", "recall_scope": "all-apps"}, True),
        ({"matter_id": "matter-1"}, False),
    ],
    ids=["memory-note", "matter-note"],
)
def test_only_a_note_saved_as_memory_skips_extraction(tmp_path, raw_meta, imported):
    """Matter notes share the manual source but are context for a matter, not a memory."""
    stores = _setup(tmp_path)
    evidence, _profiles, tasks, governance, _secrets, profile = stores
    record = evidence.put(
        EvidenceInput(
            source_id=SOURCE_ID,
            conversation_id=f"note:{uuid4()}",
            message_id=str(uuid4()),
            role="user",
            written_by=WrittenBy.PERSON,
            created_at=datetime.now(UTC),
            verbatim=TEXT,
            origin_meta={EvidenceOrigin.MANUAL_IMPORT: OriginMetadata(raw_meta=raw_meta)},
        )
    ).record
    bridge = FakeBridge()

    asyncio.run(_runner(tmp_path, stores, bridge).run(_task(tasks, profile, record).task_id))

    assert len(bridge.requests) == 1
    assert bridge.extracted_contents == ([(TEXT,)] if imported else [])
    if imported:
        (memory,) = governance.list(user_id="local")
        assert memory.content == TEXT and record.evidence_id in memory.evidence_ids
