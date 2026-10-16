"""D04 authority and workspace scope through the real continuous permission guard.

Only HY/model transport is controlled. These assert the documented boundary,
not that a confirmed machine projection constitutes human factual acceptance.
"""

import asyncio
import json
from datetime import UTC, datetime

import pytest

from one_memory.hy_memory_bridge import HyMemoryIngestResult, HyMemoryOutput
from one_memory.imports.models import EvidenceInput, EvidenceOrigin, OriginMetadata, WrittenBy
from one_memory.memory_governance import MemoryDraft, MemoryKind, MemoryOrigin, MemoryStatus
from one_memory.memory_tasks import MemoryTaskStatus
from tests.test_continuous_processing_integration import IntegrationCore

WORKSPACE = "workspace:v1:" + "a" * 64
OTHER_WORKSPACE = "workspace:v1:" + "b" * 64
AGENT_TEXT = "authorityneedle assistant proposes project delivery without user acceptance"
PROJECT_TEXT = "projectneedle this project requires a staging approval before deployment"


def envelope(scopes, tools=("om_recall",)):
    # Simulated authenticated Broker envelope, not a real host installation/grant.
    return {
        "x-one-memory-read-context": "external",
        "x-one-memory-installation-id": "00000000-0000-0000-0000-000000000001",
        "x-one-memory-installation-revision": "1",
        "x-one-memory-allowed-scopes": json.dumps(scopes),
        "x-one-memory-allowed-tools": json.dumps(tools),
    }


async def search(core, query, scopes, tools=("om_recall",)):
    return await core.client.post(
        "/api/v1/context/search",
        json={"query": query, "kinds": ["memory"]},
        headers=envelope(scopes, tools),
    )


def put(core, key, text, author):
    return core.app.state.evidence_store.put(
        EvidenceInput(
            user_id="local",
            source_id="cursor",
            conversation_id="authority-review",
            message_id=key,
            workspace=WORKSPACE,
            role="assistant" if author is WrittenBy.AGENT else "user",
            written_by=author,
            created_at=datetime.now(UTC),
            verbatim=text,
            origin_meta={EvidenceOrigin.FILE: OriginMetadata()},
        )
    ).record


async def prepare(core):
    core.grant_source()
    core.app.state.config_store.set_sync("cursor", True)
    await core.save_and_probe()
    await core.enable_api()
    runner = core.app.state.memory_processing_supervisor._create_runner()
    assert runner._permission_guard is not None


async def process(core, record, outputs):
    async def ingest(request):
        core.bridge.requests.append(request)
        assert record.evidence_id in request.evidence_ids
        return HyMemoryIngestResult(success=True, memories=tuple(outputs))

    core.bridge.ingest = ingest
    results = await core.run_prepared(record)
    assert results and all(task.status == MemoryTaskStatus.COMPLETED for task in results)
    assert core.bridge.requests and core.bridge.leases
    provenance = [
        output
        for task in results
        for output in core.app.state.memory_task_store.outputs(task.task_id)
    ]
    assert len(provenance) == len(outputs)
    return provenance


@pytest.mark.parametrize("existing", [False, True])
def test_agent_output_lands_usable_without_demoting_an_existing_memory(
    tmp_path, monkeypatch, existing
):
    async def scenario():
        core = IntegrationCore(tmp_path, monkeypatch)
        async with core.running():
            await prepare(core)
            person = put(core, "person", "Please review the project proposal", WrittenBy.PERSON)
            agent = put(core, "agent", AGENT_TEXT, WrittenBy.AGENT)
            governance = core.app.state.memory_governance_store
            if existing:
                previous = core.app.state.person_writes.create(
                    MemoryDraft(
                        user_id="local",
                        kind=MemoryKind.CHAT,
                        content=AGENT_TEXT,
                        scopes=(WORKSPACE,),
                        evidence_ids=(agent.evidence_id,),
                        origin=MemoryOrigin.AGENT,
                    ),
                    request_id="fixture-prior",
                    memory_id="prior-agent-memory",
                )
            await process(
                core,
                person,
                [
                    HyMemoryOutput(
                        memory_id="hy-agent-proposal",
                        content=AGENT_TEXT,
                        layer="L2_FACT",
                        owner="agent",
                        evidence_ids=(agent.evidence_id,),
                    )
                ],
            )
            # HY's id is mirrored as its own row; the ledger does not fold it
            # into an older row that happens to share the words.
            memory = governance.get("hy-agent-proposal", user_id="local")
            assert memory is not None and memory.origin is MemoryOrigin.AGENT
            assert memory.evidence_ids == (agent.evidence_id,)
            assert memory.status is MemoryStatus.CONFIRMED
            prior_visible = 0
            if existing:
                prior = governance.get("prior-agent-memory", user_id="local")
                assert prior.status is MemoryStatus.CONFIRMED
                assert prior.revision == previous.revision
                prior_visible = 1
            matching = await search(core, "authorityneedle", [WORKSPACE])
            assert matching.status_code == 200, matching.text
            assert len(matching.json()["items"]) == 1 + prior_visible
            assert memory.scopes == (WORKSPACE,) or WORKSPACE in memory.scopes
            denied = await search(core, "authorityneedle", [WORKSPACE], ("om_conversations",))
            assert denied.status_code == 403

    asyncio.run(scenario())


def test_project_fact_is_shared_across_apps_and_leaves_with_its_grant(tmp_path, monkeypatch):
    async def scenario():
        core = IntegrationCore(tmp_path, monkeypatch)
        async with core.running():
            await prepare(core)
            record = put(core, "project", PROJECT_TEXT, WrittenBy.PERSON)
            put(core, "project-reply", "Acknowledged.", WrittenBy.AGENT)
            await process(
                core,
                record,
                [
                    HyMemoryOutput(
                        memory_id="hy-project-rule",
                        content=PROJECT_TEXT,
                        layer="L2_FACT",
                        owner="user",
                        evidence_ids=(record.evidence_id,),
                    ),
                    *[
                        HyMemoryOutput(
                            memory_id="hy-profile-" + layer,
                            content="profileneedle " + layer + " user prefers concise explanations",
                            layer=layer,
                            owner="user",
                            evidence_ids=(record.evidence_id,),
                        )
                        for layer in ("L0_BASIC_INFO", "L6_SCHEMA")
                    ],
                ],
            )
            memory = core.app.state.memory_governance_store.get("hy-project-rule", user_id="local")
            assert memory is not None and memory.status is MemoryStatus.CONFIRMED
            assert memory.origin is MemoryOrigin.EXTRACTED
            workspace = await search(core, "projectneedle", [WORKSPACE])
            assert workspace.status_code == 200 and len(workspace.json()["items"]) == 1
            unrelated = await search(core, "projectneedle", [OTHER_WORKSPACE])
            assert unrelated.status_code == 200 and unrelated.json()["items"] == []
            global_read = await search(core, "projectneedle", ["all-apps", OTHER_WORKSPACE])
            assert global_read.status_code == 200 and len(global_read.json()["items"]) == 1
            assert memory.scopes == ("all-apps", WORKSPACE)
            profiles = await search(core, "profileneedle", ["all-apps"])
            assert profiles.status_code == 200 and len(profiles.json()["items"]) == 2
            denied = await search(core, "projectneedle", [WORKSPACE], ("om_conversations",))
            assert denied.status_code == 403
            core.app.state.source_grant_store.revoke(user_id="local", source_id="cursor")
            revoked = await search(core, "projectneedle", [WORKSPACE])
            assert revoked.status_code == 200 and revoked.json()["items"] == []
            revoked_global = await search(core, "projectneedle", ["all-apps"])
            assert revoked_global.status_code == 200 and revoked_global.json()["items"] == []
            revoked_profile = await search(core, "profileneedle", ["all-apps"])
            assert revoked_profile.status_code == 200 and revoked_profile.json()["items"] == []
            assert len(core.bridge.requests) == 1

    asyncio.run(scenario())
