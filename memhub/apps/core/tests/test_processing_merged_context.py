import asyncio
from uuid import uuid4

import pytest

from one_memory.memory_governance import HyLink, MemoryDraft, MemoryKind
from one_memory.memory_tasks import MemoryTaskOutput, MemoryTaskStatus, evidence_ref
from tests.person_door import as_person
from tests.test_continuous_processing_bridge import enable
from tests.test_model_processing_permissions import h as h  # noqa: F401


@pytest.mark.parametrize("merged_support", [False, True])
def test_new_chunk_omits_known_memory_with_changed_support_but_keeps_original_history(
    h, merged_support
):
    enable(h)
    old = h.put("completed")
    supervisor = h.supervisor()
    h.bridge.fail = False

    async def scenario():
        try:
            first = supervisor._enqueue_chunk((old,), h.profile)
            assert (
                await supervisor._create_runner().run(first.task_id)
            ).status == MemoryTaskStatus.COMPLETED
            memory = as_person(h.governance).create(
                MemoryDraft(
                    user_id="local",
                    kind=MemoryKind.CHAT,
                    content="Synthetic shared preference",
                    evidence_ids=(old.evidence_id,),
                    scopes=("all-apps",),
                ),
                request_id=str(uuid4()),
            )
            h.tasks.save_output(
                MemoryTaskOutput(
                    task_id=first.task_id,
                    memory_id=memory.memory_id,
                    memory_revision=memory.revision,
                    hy_memory_id="synthetic-known-memory",
                    owner="user",
                    layer="L2_FACT",
                    engine_version="controlled-test",
                    support_refs=(evidence_ref(old),),
                )
            )
            outside = h.put("other-conversation", conversation="another")
            if merged_support:
                h.governance.write_from_hy(
                    HyLink(memory_id=memory.memory_id, evidence_id=outside.evidence_id),
                    user_id="local",
                    derivation=h.governance.hy_derivation((outside.evidence_id,)),
                )
                changed = h.governance.get(memory.memory_id, user_id="local")
                assert changed.revision == memory.revision
                assert set(changed.evidence_ids) == {old.evidence_id, outside.evidence_id}
            fresh = h.put("next")
            next_task = supervisor._enqueue_chunk((fresh,), h.profile)
            result = await supervisor._create_runner().run(next_task.task_id)
            assert result.status == MemoryTaskStatus.COMPLETED
            context = h.leases[-1].source_context
            assert context.current == (fresh,)
            assert context.history == (old,)
            assert len(context.known_memories) == (0 if merged_support else 1)
            assert outside.evidence_id not in {
                record.evidence_id for record in h.leases[-1].records
            }
        finally:
            await supervisor.shutdown()

    asyncio.run(scenario())
