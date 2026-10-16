"""HY decides deletions; the ledger mirrors them. Pinned SDK, scripted HTTP."""

import asyncio
import json

import httpx
from hy_memory.models.memory import MemoryStatus as HyStatus

from one_memory.memory_governance import DeletionStatus, MemoryKind, MemoryStatus
from tests.test_hy_coding_lease import (
    TOOL_TEXT,
    CodingOracle,
    by_kind,
    run_coding_turn,
)
from tests.test_hy_permission_raw_context import (
    AGENT_TEXT,
    USER_TEXT,
    ScriptedOracle,
    assert_outputs,
    bind_oracle,
    grant_records,
    native_client,
    on_native_loop,
    put_turn,
    raw_client,
)


class DeletingOracle(ScriptedOracle):
    """HY's reconciler withdraws the user's claim and keeps the agent's."""

    def respond(self, request):
        response = super().respond(request)
        if self.stages[-1] != "reconcile":
            return response
        body = json.loads(response.content)
        ops = json.loads(body["choices"][0]["message"]["content"])
        for op in ops:
            if op["content"] == USER_TEXT and "memory_id" in op:
                op.clear()
                op.update({"op": "DELETE", "memory_id": self.withdrawn})
        body["choices"][0]["message"]["content"] = json.dumps(ops)
        return httpx.Response(200, json=body)


def test_a_memory_hy_deletes_is_deleted_in_the_ledger_and_the_rest_stay(tmp_path):
    async def scenario():
        async with raw_client(tmp_path) as (supervisor, records, permission):
            first = supervisor.prepare_conversation(records[0])[0]
            client = await native_client(supervisor, first)
            bind_oracle(supervisor, ScriptedOracle(records))
            assert (await supervisor._create_runner().run(first.task_id)).status == "completed"
            await assert_outputs(supervisor, first, records, client)
            by_content = {m.content: m for m in supervisor.governance.list(user_id="local")}
            withdrawn, kept = by_content[USER_TEXT], by_content[AGENT_TEXT]

            added = put_turn(supervisor.evidence, "user-withdraw", "user", USER_TEXT, 3)
            records = (*records, added)
            grant_records(supervisor, records, permission.revision)
            second = supervisor.prepare_conversation(added)[0]
            oracle = DeletingOracle(records)
            oracle.withdrawn = withdrawn.memory_id
            bind_oracle(supervisor, oracle)
            result = await supervisor._create_runner().run(second.task_id)
            assert not oracle.errors, oracle.errors
            assert result.status == "completed", (result.last_error_code, oracle.stages)

            native = await on_native_loop(
                client, client._vector_store.get_by_id(withdrawn.memory_id)
            )
            assert native is not None and native.status == HyStatus.SHADOW
            mirrored = supervisor.governance.get(withdrawn.memory_id, user_id="local")
            assert mirrored.status is MemoryStatus.DELETED
            deletion = supervisor.governance.deletion(withdrawn.memory_id, user_id="local")
            assert deletion is not None and deletion.status is DeletionStatus.COMPLETED
            untouched = supervisor.governance.get(kept.memory_id, user_id="local")
            assert untouched.status is MemoryStatus.CONFIRMED

    asyncio.run(scenario())


class RetiringCodingOracle(CodingOracle):
    def __init__(self, records, *, retire, **kwargs):
        super().__init__(records, **kwargs)
        self.retire = retire

    def respond(self, request):
        response = super().respond(request)
        if self.stages[-1] != "reconcile":
            return response
        body = json.loads(response.content)
        body["choices"][0]["message"]["content"] = json.dumps(
            {
                "ops": [
                    {"action": "ADD", "draft_idx": 0},
                    {
                        "action": "DELETE",
                        "draft_idx": 0,
                        "target_experience_id": self.retire,
                        "reason": "the new run explicitly replaces the old procedure",
                    },
                ]
            }
        )
        return httpx.Response(200, json=body)


def test_an_experience_hy_deletes_is_deleted_in_the_ledger(tmp_path):
    async def scenario():
        async with raw_client(tmp_path, tool=TOOL_TEXT) as (supervisor, records, permission):
            _, first = await run_coding_turn(supervisor, records[0], CodingOracle(records))
            assert first.status == "completed", first.last_error_code
            (original,) = by_kind(supervisor)[MemoryKind.CODING]
            question = put_turn(supervisor.evidence, "user-again", "user", "Use tox now.", 4)
            tool = put_turn(supervisor.evidence, "tool-again", "tool", "[Bash] tox\nok", 5)
            grant_records(supervisor, (*records, question, tool), permission.revision)
            oracle = RetiringCodingOracle(
                (question, tool),
                retire=original.memory_id,
                solution="Run `tox` from the repository root.",
                statement="The agent switched the synthetic suite to tox.",
            )
            _, second = await run_coding_turn(supervisor, question, oracle)
            assert second.status == "completed", (second.last_error_code, oracle.stages)
            client = await native_client(supervisor, second)
            assert (
                await on_native_loop(client, client._coding_store.get_by_id(original.memory_id))
                is None
            )

            assert (
                supervisor.governance.get(original.memory_id, user_id="local").status
                is MemoryStatus.DELETED
            )
            live = [
                m
                for m in by_kind(supervisor)[MemoryKind.CODING]
                if m.status is MemoryStatus.CONFIRMED
            ]
            assert [m.content for m in live] == [
                "Run the synthetic test suite\nRun `tox` from the repository root."
            ]

    asyncio.run(scenario())
