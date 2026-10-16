"""HY decides supersessions; the ledger mirrors them. Pinned SDK, scripted HTTP."""

import asyncio
import json
from copy import deepcopy

import httpx
import pytest
from hy_memory.agent.tools.basic_profile import upsert_basic_profile
from hy_memory.coding.types import WorkingFact
from hy_memory.models.memory import MemoryLayer
from hy_memory.models.memory import MemoryStatus as HyStatus

import tests.test_hy_permission_raw_context as raw_context
from one_memory.hy_memory_bridge import HyMemoryBridgeError
from one_memory.hy_permission_coding_store import HyPermissionCodingStore, hy_coding_fingerprint
from one_memory.hy_permission_store import HyNodePermission
from one_memory.memory_governance import MemoryKind, MemoryStatus
from tests.test_hy_coding_lease import TOOL_TEXT, CodingOracle, by_kind, run_coding_turn
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
from tests.test_hy_permission_store import Backend, Lease, node, wrapper

NEW_TEXT = "The user now requires two permission reviews before the synthetic release."


def _vector(text):
    axis = {USER_TEXT: 1, AGENT_TEXT: 2, NEW_TEXT: 3}.get(text, 0)
    return [float(index == axis) for index in range(384)]


class SupersedingOracle(ScriptedOracle):
    """HY's reconciler replaces existing claims with the new one instead of updating them."""

    def __init__(self, records, *, chain):
        super().__init__(records)
        self.chain = chain

    def respond(self, request):
        response = super().respond(request)
        if self.stages[-1] != "reconcile":
            return response
        body = json.loads(response.content)
        ops = json.loads(body["choices"][0]["message"]["content"])
        by_content = {op["content"]: op for op in ops}
        supersede = {
            "op": "SUPERSEDE",
            "content": NEW_TEXT,
            "owner": "user",
            "layer": "L2_FACT",
            "tags": [],
            "supersede_reason": "the user raised the requirement",
        }
        if self.chain:
            supersede["memory_ids"] = [self.chain[0], self.chain[1]]
            replaced = {USER_TEXT, AGENT_TEXT}
        else:
            supersede["memory_id"] = by_content[USER_TEXT]["memory_id"]
            replaced = {USER_TEXT}
        ops = [op for op in ops if op["content"] not in replaced | {NEW_TEXT}] + [supersede]
        body["choices"][0]["message"]["content"] = json.dumps(ops)
        return httpx.Response(200, json=body)


async def _first_turn(supervisor, records):
    first = supervisor.prepare_conversation(records[0])[0]
    client = await native_client(supervisor, first)
    bind_oracle(supervisor, ScriptedOracle(records))
    assert (await supervisor._create_runner().run(first.task_id)).status == "completed"
    await assert_outputs(supervisor, first, records, client)
    by_content = {m.content: m for m in supervisor.governance.list(user_id="local")}
    return client, by_content[USER_TEXT], by_content[AGENT_TEXT]


async def _superseding_turn(supervisor, records, permission, *, chain=()):
    added = put_turn(supervisor.evidence, "user-raise", "user", NEW_TEXT, 3)
    records = (*records, added)
    grant_records(supervisor, records, permission.revision)
    second = supervisor.prepare_conversation(added)[0]
    oracle = SupersedingOracle(records, chain=chain)
    bind_oracle(supervisor, oracle)
    result = await supervisor._create_runner().run(second.task_id)
    assert not oracle.errors, oracle.errors
    assert result.status == "completed", (result.last_error_code, oracle.stages)
    return oracle


def _head(supervisor):
    (head,) = [
        m for m in supervisor.governance.list(user_id="local", limit=500) if m.content == NEW_TEXT
    ]
    return head


def test_a_memory_hy_supersedes_is_superseded_in_the_ledger_with_its_replacement(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(raw_context, "vector", _vector)

    async def scenario():
        async with raw_client(tmp_path) as (supervisor, records, permission):
            client, replaced, kept = await _first_turn(supervisor, records)
            monkeypatch.setitem(raw_context.CLAIMS, NEW_TEXT, "user")
            await _superseding_turn(supervisor, records, permission)

            native = await on_native_loop(
                client, client._vector_store.get_by_id(replaced.memory_id)
            )
            assert native.status == HyStatus.SUPERSEDED
            head = _head(supervisor)
            assert native.superseded_by == [head.memory_id]

            old = supervisor.governance.get(replaced.memory_id, user_id="local")
            assert old.status is MemoryStatus.SUPERSEDED
            assert old.deleted_at is None and old.content == USER_TEXT
            assert old.superseded_by == (head.memory_id,)
            assert supervisor.governance.deletion(old.memory_id, user_id="local") is None
            assert head.status is MemoryStatus.CONFIRMED
            assert head.supersedes == (old.memory_id,)
            untouched = supervisor.governance.get(kept.memory_id, user_id="local")
            assert untouched.status is MemoryStatus.CONFIRMED
            assert untouched.supersedes == untouched.superseded_by == ()
            assert {m.memory_id for m in supervisor.governance.list(user_id="local")} == {
                head.memory_id,
                kept.memory_id,
            }

    asyncio.run(scenario())


def test_a_chain_fold_supersedes_every_folded_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(raw_context, "vector", _vector)

    async def scenario():
        async with raw_client(tmp_path) as (supervisor, records, permission):
            client, user, agent = await _first_turn(supervisor, records)
            monkeypatch.setitem(raw_context.CLAIMS, NEW_TEXT, "user")
            await _superseding_turn(
                supervisor, records, permission, chain=(agent.memory_id, user.memory_id)
            )

            head = _head(supervisor)
            oldest = supervisor.governance.get(agent.memory_id, user_id="local")
            newer = supervisor.governance.get(user.memory_id, user_id="local")
            assert oldest.status is newer.status is MemoryStatus.SUPERSEDED
            assert oldest.superseded_by == (user.memory_id,)
            assert newer.superseded_by == (head.memory_id,)
            assert newer.supersedes == (agent.memory_id,)
            assert head.supersedes == (user.memory_id,)
            assert head.status is MemoryStatus.CONFIRMED
            assert [m.memory_id for m in supervisor.governance.list(user_id="local")] == [
                head.memory_id
            ]
            for key in (agent.memory_id, user.memory_id):
                native = await on_native_loop(client, client._vector_store.get_by_id(key))
                assert native.status == HyStatus.SUPERSEDED

    asyncio.run(scenario())


async def _turn_after_supersede(supervisor, records, permission):
    """The user repeats the new requirement one turn after HY superseded the old one."""
    added = next(
        r
        for r in supervisor.evidence.list_conversation(records[0].evidence_id)
        if r.verbatim == NEW_TEXT
    )
    again = put_turn(supervisor.evidence, "user-again", "user", NEW_TEXT, 4)
    records = (*records, added, again)
    grant_records(supervisor, records, permission.revision + 1)
    third = supervisor.prepare_conversation(again)[0]
    oracle = ScriptedOracle(records)
    bind_oracle(supervisor, oracle)
    result = await supervisor._create_runner().run(third.task_id)
    assert not oracle.errors, oracle.errors
    return result, oracle


def test_the_turn_after_a_supersede_still_completes(tmp_path, monkeypatch):
    monkeypatch.setattr(raw_context, "vector", _vector)

    async def scenario():
        async with raw_client(tmp_path) as (supervisor, records, permission):
            _, replaced, _ = await _first_turn(supervisor, records)
            monkeypatch.setitem(raw_context.CLAIMS, NEW_TEXT, "user")
            await _superseding_turn(supervisor, records, permission)

            # The old memory's evidence is in this turn's window; being
            # superseded by HY is not the user changing what was said.
            result, oracle = await _turn_after_supersede(supervisor, records, permission)

            assert result.status == "completed", (result.last_error_code, oracle.stages)
            old = supervisor.governance.get(replaced.memory_id, user_id="local")
            assert old.status is MemoryStatus.SUPERSEDED

    asyncio.run(scenario())


def test_hy_still_sees_the_head_of_a_supersede_chain_on_the_next_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(raw_context, "vector", _vector)

    async def scenario():
        async with raw_client(tmp_path) as (supervisor, records, permission):
            await _first_turn(supervisor, records)
            monkeypatch.setitem(raw_context.CLAIMS, NEW_TEXT, "user")
            await _superseding_turn(supervisor, records, permission)
            head = _head(supervisor)

            result, oracle = await _turn_after_supersede(supervisor, records, permission)

            assert result.status == "completed", (result.last_error_code, oracle.stages)
            existing = {item["memory_id"] for batch in oracle.reconciles for item in batch}
            assert head.memory_id in existing, (head.memory_id, oracle.reconciles)
            repeated = [
                m
                for m in supervisor.governance.list(user_id="local", limit=500)
                if m.content == NEW_TEXT
            ]
            assert [m.memory_id for m in repeated] == [head.memory_id]

    asyncio.run(scenario())


class SupersedingFactOracle(CodingOracle):
    """HY's fact reconciler judges the new session to continue the old task."""

    def __init__(self, records, *, old_fact, **kwargs):
        super().__init__(records, **kwargs)
        self.old_fact = old_fact

    def respond(self, request):
        body = json.loads(request.content)
        if request.url.path != "/v1/embeddings":
            prompt = body["messages"][-1]["content"]
            if prompt.startswith("A user task in a working-memory store has progressed"):
                self.requests.append(body)
                self.stages.append("chain_summary")
                return _completion(body, "The suite moved from pytest to tox.")
        response = super().respond(request)
        if self.stages[-1] != "fact_reconcile":
            return response
        return _completion(
            body,
            json.dumps(
                {
                    "ops": [
                        {
                            "draft_idx": 0,
                            "supersedes": [self.old_fact],
                            "reason": "continues the same test-suite task",
                        }
                    ]
                }
            ),
        )


def _completion(body, content):
    return httpx.Response(
        200,
        json={
            "id": "synthetic-coding-response",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
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


def test_a_working_fact_hy_supersedes_is_superseded_in_the_ledger(tmp_path):
    async def scenario():
        async with raw_client(tmp_path, tool=TOOL_TEXT) as (supervisor, records, permission):
            _, first = await run_coding_turn(supervisor, records[0], CodingOracle(records))
            assert first.status == "completed", first.last_error_code
            (old_fact,) = by_kind(supervisor)[MemoryKind.WORKING]
            question = put_turn(supervisor.evidence, "user-again", "user", "Use tox now.", 4)
            tool = put_turn(supervisor.evidence, "tool-again", "tool", "[Bash] tox\nok", 5)
            grant_records(supervisor, (*records, question, tool), permission.revision)
            oracle = SupersedingFactOracle(
                (question, tool),
                old_fact=old_fact.memory_id,
                solution="Run `tox` from the repository root.",
                statement="The agent switched the synthetic suite to tox.",
            )
            _, second = await run_coding_turn(supervisor, question, oracle)
            assert second.status == "completed", (second.last_error_code, oracle.stages)
            assert "fact_reconcile" in oracle.stages
            client = await native_client(supervisor, second)
            (native_old,) = await on_native_loop(
                client, client._coding_store.get_facts([old_fact.memory_id], user_id="local")
            )
            assert native_old.status == "superseded"

            old = supervisor.governance.get(old_fact.memory_id, user_id="local")
            assert old.status is MemoryStatus.SUPERSEDED
            assert old.superseded_by == (native_old.fact_link,)
            new = supervisor.governance.get(native_old.fact_link, user_id="local")
            assert new is not None and new.status is MemoryStatus.CONFIRMED
            assert new.kind is MemoryKind.WORKING
            assert new.supersedes == (old.memory_id,)
            working = [
                m.memory_id
                for m in supervisor.governance.list(user_id="local")
                if m.kind is MemoryKind.WORKING
            ]
            assert working == [new.memory_id]

    asyncio.run(scenario())


def test_the_chat_store_reports_what_hy_superseded_and_nothing_else():
    head = node("head", supersedes=["old"])
    old = node("old")
    kept = node("kept")
    backend = Backend(old, kept)
    store = wrapper(backend, approved=(old, kept))

    async def scenario():
        # Positive control and negative in one store: an ordinary update is not reported.
        await store.update_payload("kept", {"content": "Synthetic kept, refined"})
        assert store.superseded_nodes == {}
        await store.update_payload("old", {"is_latest": False, "status": "superseded"})
        await store.upsert(head)
        await store.update_payload("old", {"superseded_by": ["head"]})
        assert store.superseded_nodes == {"old": ("head",)}
        assert "old" not in store.retired_node_ids

    asyncio.run(scenario())


def test_a_superseded_link_written_only_on_the_head_is_still_reported():
    old = node("old")
    backend = Backend(old)
    store = wrapper(backend, approved=(old,))

    async def scenario():
        await store.update_payload("old", {"is_latest": False, "status": "superseded"})
        await store.upsert(node("head", supersedes=["old"]))
        assert store.superseded_nodes == {"old": ("head",)}

    asyncio.run(scenario())


def test_a_basic_profile_supersede_is_reported_for_the_ledger():
    old = node(
        "old-profile",
        agent_id="",
        session_id="",
        owner=None,
        layer=MemoryLayer.L0_BASIC_INFO,
        content="name: Before",
        custom={"basic_info_kv": {"name": "Before"}},
    )
    backend = Backend(old)
    store = wrapper(backend, approved=(old,))

    class Embedder:
        async def embed_queued(self, text):
            return [1.0, 0.0]

    async def scenario():
        result = await upsert_basic_profile(
            user_id="local",
            kv={"name": "After"},
            vector_store=store,
            embed_service=Embedder(),
            allowed_fields=["name"],
        )
        assert result.success and result.changed
        assert store.superseded_nodes == {"old-profile": (result.node_id,)}

    asyncio.run(scenario())


def _fact(key, **kwargs):
    return WorkingFact(
        fact_id=key, user_id="local", agent_id="agent", session_id="session", **kwargs
    )


class _NativeFacts:
    def __init__(self, *facts):
        self.facts = {fact.fact_id: deepcopy(fact) for fact in facts}

    async def get_facts(self, keys, user_id=None):
        return [deepcopy(self.facts[key]) for key in keys if key in self.facts]

    async def supersede_fact(self, old_id, new_id):
        row = self.facts[old_id]
        row.status, row.fact_link = "superseded", new_id
        return True

    async def insert_fact(self, fact):
        self.facts[fact.fact_id] = deepcopy(fact)


def _coding_store(native, *approved):
    return HyPermissionCodingStore(
        native,
        Lease(),
        agent_id="agent",
        session_id="session",
        node_permissions={
            fact.fact_id: HyNodePermission(hy_coding_fingerprint(fact), ("current", "support"))
            for fact in approved
        },
    )


def test_the_coding_store_lets_hy_supersede_a_fact_before_inserting_its_successor():
    old = _fact("old-fact")
    store = _coding_store(_NativeFacts(old), old)

    async def scenario():
        assert store.superseded_ids == {}
        assert await store.supersede_fact("old-fact", "new-fact")
        await store.insert_fact(_fact("new-fact"))
        store.recheck()
        assert store.superseded_ids == {"old-fact": ("new-fact",)}
        assert "old-fact" not in store.retired_ids

    asyncio.run(scenario())


def test_the_coding_store_never_links_a_fact_to_an_existing_unapproved_one():
    old, foreign = _fact("old-fact"), _fact("foreign-fact", statement="DENIED")
    native = _NativeFacts(old, foreign)
    store = _coding_store(native, old)

    async def scenario():
        with pytest.raises(HyMemoryBridgeError, match="target_denied"):
            await store.supersede_fact("old-fact", "foreign-fact")

    asyncio.run(scenario())
    assert native.facts["old-fact"].status == "active"
