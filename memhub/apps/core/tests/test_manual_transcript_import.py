import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from one_memory.api import create_app
from one_memory.config import Settings
from one_memory.file_imports import FileImportStore
from one_memory.imports.chat_exports import parse_chat_export
from one_memory.imports.models import MAX_VERBATIM_CHARACTERS
from one_memory.secrets import InMemorySecretStore
from tests.source_consent import connection_request


@pytest.mark.parametrize("filename", ["chat.json", "renamed.json"])
def test_manual_json_keeps_authors_unknown_time_and_exact_text(filename):
    value = [
        {
            "role": "user",
            "content": "  Budget<5; this is not approval.\nKeep the words verbatim.",
            "time": None,
        },
        {
            "role": "assistant",
            "content": "Suggest assessing it first.",
            "time": "2026-09-12T09:00:00+08:00",
        },
    ]
    result = parse_chat_export(json.dumps(value).encode(), filename)
    assert result.vendor == "generic" and len(result.messages) == 2
    assert result.messages[0].verbatim == value[0]["content"]
    assert result.messages[0].created_at is None
    assert result.messages[1].created_at.utcoffset().total_seconds() == 8 * 3600
    assert [m.written_by.value for m in result.messages] == ["person", "agent"]


def test_marked_text_keeps_code_fences_and_never_invents_timestamps():
    text = (
        "# Design discussion\n\n## User\nBudget>5, not <5.\n```text\n"
        "Assistant: text inside code\n```\n\n## Assistant\nThis is still a suggestion.\n"
    )
    result = parse_chat_export(text.encode(), "chat.md")
    assert len(result.messages) == 2 and result.skipped_records == 0
    assert (
        result.messages[0].verbatim
        == "Budget>5, not <5.\n```text\nAssistant: text inside code\n```"
    )
    assert result.messages[1].verbatim == "This is still a suggestion."
    assert all(m.created_at is None and m.project == "Design discussion" for m in result.messages)


@pytest.mark.parametrize(
    "text,filename",
    [
        ("Ordinary work material, not a conversation with clear roles", "notes.txt"),
        ("User: first part\nAssistant:\nUser: last part", "chat.txt"),
        ("User: first part\nTool: not the user\nAssistant: last part", "chat.txt"),
        (
            json.dumps(
                [
                    {"role": "user", "content": "readable"},
                    {"role": "tool", "content": "unattributable"},
                ]
            ),
            "chat.json",
        ),
        (
            json.dumps(
                [{"role": "user", "content": "original words", "time": "2026-09-12T09:00:00"}]
            ),
            "chat.json",
        ),
        (
            json.dumps([{"role": "user", "content": "original words", "time": "yesterday"}]),
            "chat.json",
        ),
        (
            json.dumps([{"role": "user", "content": "x" * (MAX_VERBATIM_CHARACTERS + 1)}]),
            "chat.json",
        ),
    ],
)
def test_ambiguous_partial_or_invalid_manual_data_never_leaves_partial_import(
    tmp_path, text, filename
):
    files = FileImportStore(tmp_path / "imports")
    with pytest.raises(ValueError):
        files.prepare(text.encode(), filename)
    assert not files.root.exists()


def test_manual_import_reports_segment_then_follows_normal_authorize_and_dedup(tmp_path):
    async def scenario():
        app = create_app(
            Settings(
                data_dir=tmp_path / "data",
                source_home=tmp_path / "home",
                api_token=SecretStr("test"),
                automatic_memory_processing=False,
                live_sync_enabled=False,
            ),
            secret_store=InMemorySecretStore(),
        )
        async with (
            app.router.lifespan_context(app),
            AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
                headers={"Authorization": "Bearer test"},
            ) as client,
        ):
            bad = await client.post(
                "/api/v1/import-files?filename=chat.txt",
                content=b"User: readable\nTool: pending",
            )
            assert bad.status_code == 422
            assert bad.json()["detail"] == "generic_import_invalid"
            assert bad.json()["issue"]["location"] == "line:2"
            assert bad.json()["issue"]["excerpt"] == "Tool: pending"
            payload = (
                "User: Keep source citations before release.\nAssistant: Suggest running a check."
            )
            uploaded = await client.post(
                "/api/v1/import-files?filename=chat.txt", content=payload.encode()
            )
            assert uploaded.status_code == 201
            receipt = uploaded.json()
            assert receipt["source_id"] == "generic-export"
            assert receipt["human_messages"] == receipt["agent_messages"] == 1
            assert receipt["preview"][0]["text"] == "Keep source citations before release."
            assert app.state.evidence_store.count() == 0
            await client.post(
                f"/api/v1/import-files/{receipt['upload_id']}/commit", json={"confirmed": True}
            )
            connected = await client.post(
                "/api/v1/sources/connect",
                json=await connection_request(client, [receipt["source_id"]], enable_sync=True),
            )
            assert connected.status_code == 200 and connected.json()["failed_sources"] == {}
            job = await app.state.import_supervisor.wait(connected.json()["jobs"][0]["job_id"])
            assert job.stored == 2 and job.errors == 0
            found = (
                await client.get(
                    "/api/v1/conversations/search", params={"query": "source citations"}
                )
            ).json()
            assert found["hits"][0]["created_at"] is None
            again = await client.post(
                "/api/v1/import-jobs", json={"source_id": receipt["source_id"]}
            )
            job = await app.state.import_supervisor.wait(again.json()["job_id"])
            assert job.stored == 0 and app.state.evidence_store.count() == 2

    asyncio.run(scenario())
