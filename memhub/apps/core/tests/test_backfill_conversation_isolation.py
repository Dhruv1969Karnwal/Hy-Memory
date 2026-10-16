"""One stale conversation cannot invalidate an entire backfill scan."""

import asyncio
from contextlib import suppress

from one_memory.hy_memory_bridge import HyMemoryBridgeError
from one_memory.imports.models import EvidenceInput
from tests.test_continuous_processing_integration import IntegrationCore


def test_one_changed_conversation_does_not_block_other_conversations(tmp_path, monkeypatch):
    core = IntegrationCore(tmp_path, monkeypatch)

    async def scenario():
        async with core.running(automatic_memory_processing=False):
            monkeypatch.setenv("ONE_MEMORY_DIAGNOSTICS", "1")
            core.grant_source()
            core.put_complete("conversation-good", conversation="good")
            changed = core.put_complete("conversation-changed", conversation="changed")
            await core.save_and_probe()
            core.kernel_enable("include")

            evidence_store = core.app.state.evidence_store
            changed_input = EvidenceInput.model_validate(
                {
                    **changed.model_dump(include=set(EvidenceInput.model_fields)),
                    "verbatim": changed.verbatim + " edited",
                }
            )
            evidence_store.put(changed_input)
            assert evidence_store.get(changed.evidence_id).revision > changed.revision

            # Production reaches completed-input reuse with the live chunk when
            # an earlier task already covered it; that reuse rechecks the
            # stored input against the row now in Evidence. Model the observed
            # stale-row refusal at the exact boundary that raises in production.
            guard = core.app.state.memory_processing_supervisor._permission_guard
            original_completed_lookup = guard.completed_input_task

            def refuse_changed_conversation(receipt, records, **kwargs):
                if any(record.conversation_id == "changed" for record in records):
                    raise HyMemoryBridgeError("model_content_evidence_changed")
                return original_completed_lookup(receipt, records, **kwargs)

            monkeypatch.setattr(guard, "completed_input_task", refuse_changed_conversation)

            supervisor = core.app.state.memory_processing_supervisor
            with suppress(Exception):
                await supervisor.backfill_off_thread()

            await core.wait_completed(timeout=10, minimum_tasks=1)
            assert len(core.bridge.requests) == 1

    asyncio.run(scenario())
