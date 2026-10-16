"""Read invalidation review: real guard/provenance, controlled HY, no model/network."""

import asyncio
import sqlite3
from contextlib import closing

import pytest

from one_memory import context_routes
from one_memory.hy_memory_bridge import HyMemoryOutput
from one_memory.imports.models import WrittenBy
from one_memory.memory_governance import MemoryStatus
from one_memory.memory_tasks import evidence_ref
from tests.test_continuous_processing_integration import IntegrationCore
from tests.test_governance_authority_review import WORKSPACE, prepare, process, put, search

OLD = "revisionneedle the project deploys only on Windows"
NEW = "revisionneedle the project deploys only on macOS"


async def derived(core):
    await prepare(core)
    original = put(core, "edited-turn", OLD, WrittenBy.PERSON)
    put(core, "edited-turn-reply", "Acknowledged.", WrittenBy.AGENT)
    outputs = await process(
        core,
        original,
        [
            HyMemoryOutput(
                memory_id="review-old-claim",
                content=OLD,
                layer="L2_FACT",
                owner="user",
                evidence_ids=(original.evidence_id,),
            )
        ],
    )
    output = outputs[0]
    assert output.support_refs == (evidence_ref(original),)
    assert output.memory_revision == 1
    before = await search(core, "revisionneedle", [WORKSPACE])
    assert before.status_code == 200 and len(before.json()["items"]) == 1
    assert before.json()["items"][0]["content"] == OLD
    return original, output


@pytest.mark.parametrize("during_read", [False, True], ids=["before-query", "before-publish"])
def test_revised_evidence_does_not_supply_old_derived_claim(tmp_path, monkeypatch, during_read):
    async def scenario():
        core = IntegrationCore(tmp_path, monkeypatch)
        async with core.running():
            original, output = await derived(core)
            revised = None

            def revise():
                nonlocal revised
                revised = put(core, "edited-turn", NEW, WrittenBy.PERSON)
                assert revised.evidence_id == original.evidence_id
                assert revised.revision == original.revision + 1
                assert revised.content_hash != original.content_hash

            if during_read:
                read = context_routes.read_memory_context

                def revision_after_selection(*args, **kwargs):
                    result = read(*args, **kwargs)
                    assert len(result.items) == 1
                    revise()
                    return result

                monkeypatch.setattr(context_routes, "read_memory_context", revision_after_selection)
            else:
                revise()
            response = await search(core, "revisionneedle", [WORKSPACE])
            assert response.status_code == 200
            assert revised is not None
            stored = core.app.state.memory_task_store.outputs(output.task_id)[0]
            assert stored.support_refs == output.support_refs
            memory = core.app.state.memory_governance_store.get(output.memory_id, user_id="local")
            assert memory.status is MemoryStatus.CONFIRMED and memory.content == OLD
            assert len(core.bridge.requests) == 1
            with closing(sqlite3.connect(core.database)) as db:
                versions = db.execute(
                    "SELECT revision FROM evidence_versions WHERE evidence_id=? ORDER BY revision",
                    (original.evidence_id,),
                ).fetchall()
                events = db.execute(
                    "SELECT evidence_revision FROM evidence_ingest_events "
                    "WHERE evidence_id=? ORDER BY seq",
                    (original.evidence_id,),
                ).fetchall()
            assert versions == [(1,), (2,)] and events == [(1,), (2,)]
            print("revision review:", during_read, output.support_refs, response.json())
            assert response.json()["items"] == []
            library = await core.client.get("/api/v1/memory-library")
            # D22: the total skips the per-row freshness scan (product accepted a
            # small drift), but the stale claim itself must never be listed.
            assert library.status_code == 200 and library.json()["items"] == []
            home = await core.client.get("/api/v1/home-summary")
            # Same D22 drift for the homepage count; the latest card must not be it.
            assert home.status_code == 200 and home.json()["latest"] is None
            assert (await core.client.get("/api/v1/memories")).json() == []
            detail = await core.client.get("/api/v1/memories/" + output.memory_id)
            assert detail.status_code == 403

    asyncio.run(scenario())


def test_explicit_correction_is_distinguishable_from_unchanged_sdk_projection(
    tmp_path, monkeypatch
):
    async def scenario():
        core = IntegrationCore(tmp_path, monkeypatch)
        async with core.running():
            original, output = await derived(core)
            corrected = core.app.state.person_writes.correct(
                output.memory_id,
                user_id="local",
                request_id="explicit-user-correction",
                expected_revision=output.memory_revision,
                content=NEW,
            )
            assert corrected.revision > output.memory_revision
            assert corrected.origin.value == "extracted"
            assert corrected.evidence_ids == (original.evidence_id,)
            with closing(sqlite3.connect(core.database)) as db:
                operation = db.execute(
                    "SELECT operation FROM governed_memory_mutations WHERE request_id=?",
                    ("explicit-user-correction",),
                ).fetchone()
                version = db.execute(
                    "SELECT action,content FROM governed_memory_versions "
                    "WHERE memory_id=? AND revision=?",
                    (corrected.memory_id, corrected.revision),
                ).fetchone()
            assert operation == ("correct",) and version == ("correct", NEW)
            stored = core.app.state.memory_task_store.outputs(output.task_id)[0]
            assert stored.memory_revision == 1 and stored.support_refs == output.support_refs
            # Inspect metadata only: no new policy that old refs authorize a manual correction.
            print("correction distinction:", corrected.revision, stored.memory_revision, operation)

    asyncio.run(scenario())
