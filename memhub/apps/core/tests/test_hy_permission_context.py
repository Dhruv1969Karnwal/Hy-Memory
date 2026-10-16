"""Whole native HY client with synthetic content and HTTP MockTransport, never cloud."""

import asyncio
import json
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from one_memory.hy_permission_context import permission_bridge_factory
from one_memory.imports.grants import SourceGrantStore
from one_memory.imports.models import (
    EvidenceContentKind,
    EvidenceInput,
    EvidenceOrigin,
    OriginMetadata,
    WrittenBy,
)
from one_memory.imports.sources.registry import RootCandidateKind
from one_memory.imports.storage import EvidenceStore
from one_memory.memory_governance import MemoryGovernanceStore
from one_memory.memory_processing import MemoryProcessingSupervisor
from one_memory.memory_tasks import MemoryTaskStore
from one_memory.model_content_permissions import ModelContentPermissionStore
from one_memory.model_profiles import (
    ModelCapabilities,
    ModelProfileInput,
    ModelProfileStore,
    ProviderKind,
)
from one_memory.secrets import InMemorySecretStore


@contextmanager
def whole_client(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    database = tmp_path / "one-memory.sqlite3"
    grants = SourceGrantStore(database, home=tmp_path)
    grants.grant(
        user_id="local", source_id="fixture", root=root, candidate_kind=RootCandidateKind.DEFAULT
    )
    evidence = EvidenceStore(database)
    text = "Synthetic supplied memory: review permissions before release."
    record = evidence.put(
        EvidenceInput(
            source_id="fixture",
            conversation_id="session",
            message_id="1",
            role="assistant",
            written_by=WrittenBy.UNKNOWN,
            content_kind=EvidenceContentKind.MEMORY_CONTENT,
            verbatim=text,
            origin_meta={EvidenceOrigin.MANUAL_IMPORT: OriginMetadata()},
        )
    ).record
    profiles, secrets = ModelProfileStore(database), InMemorySecretStore()
    profile = profiles.create(
        ModelProfileInput(
            name="Stub only",
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
    permissions = ModelContentPermissionStore(database)
    snapshot = permissions.snapshot(
        user_id="local",
        source_ids=("fixture",),
        profile=profiles.snapshot(profile.profile_id, profile.revision),
        evidence=(record,),
    )
    permission = permissions.grant(snapshot, user_confirmed=True, expected_revision=0)
    supervisor = MemoryProcessingSupervisor(
        tasks=MemoryTaskStore(database),
        profiles=profiles,
        secrets=secrets,
        evidence=evidence,
        governance=MemoryGovernanceStore(database),
        data_dir=tmp_path,
        model_content_permissions=permissions,
    )
    yield supervisor, record, permissions, permission


def test_native_memory_content_uses_guarded_sdk_then_returns_real_native_memory(tmp_path):
    async def scenario():
        with whole_client(tmp_path) as (supervisor, record, _, _):
            sends = []

            def handler(request: httpx.Request):
                body = json.loads(request.content)
                sends.append({"path": request.url.path, "body": body})
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
                        "id": "fixture-response",
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
                                                "content": record.verbatim,
                                                "layer": "L2_FACT",
                                                "owner": "agent",
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
            tasks = supervisor.prepare_conversation(record)
            try:
                result = await supervisor._create_runner().run(tasks[0].task_id)
                assert result.status == "completed", result.last_error_code
                assert sends and all(
                    item["path"].endswith(("embeddings", "chat/completions")) for item in sends
                )
                memories = supervisor.governance.list(user_id="local")
                assert len(memories) == 1 and memories[0].content == record.verbatim
                assert memories[0].status == "confirmed"
                assert supervisor.evidence.get(record.evidence_id).verbatim == record.verbatim
            finally:
                await supervisor.shutdown()

    asyncio.run(scenario())


def test_revoked_lease_never_creates_an_sdk_client_or_sends(tmp_path):
    async def scenario():
        with whole_client(tmp_path) as (supervisor, record, permissions, permission):
            tasks = supervisor.prepare_conversation(record)
            permissions.revoke(
                user_id="local", expected_revision=permission.revision, user_confirmed=True
            )
            supervisor.permission_bridge_factory = permission_bridge_factory(
                supervisor._bridge,
                http_transport=httpx.MockTransport(lambda _: pytest.fail("unauthorized request")),
            )
            result = await supervisor._create_runner().run(tasks[0].task_id)
            assert result.last_error_code == "model_content_permission_revoked"
            assert supervisor._bridges == {}
            assert not supervisor.governance.list(user_id="local")
            await supervisor.shutdown()

    asyncio.run(scenario())
