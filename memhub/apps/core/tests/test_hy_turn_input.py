"""HY gets one turn at a time, exactly as it happened, and picks its own route.

A turn is one question from the person plus everything that answered it:
assistant replies and the tool calls made along the way. HY's `add` decides
chat or coding itself when it sees tool messages (`scene=None`); forcing
`scene="chat"` made it strip them before it ever looked.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime

from one_memory.hy_memory_bridge import (
    HyMemoryBridge,
    HyMemoryIngestRequest,
    HyMemoryOutput,
    HyMemoryTurn,
)
from one_memory.hy_source_context import HySourceContext, HyWriteBindings
from one_memory.imports.grants import GRANT_SCHEMA
from one_memory.imports.models import WrittenBy
from one_memory.memory_processing import MemoryProcessingSupervisor, _conversation_chunks
from one_memory.memory_tasks import MemoryTaskKind, MemoryTaskStatus
from one_memory.model_profiles import ProviderKind
from tests.test_hy_memory_bridge import FakeHyMemoryClient, snapshot
from tests.test_memory_processing import FakeBridge, _put, _runner, _setup, _task
from tests.test_model_processing_permissions import Harness


def _turn(evidence, index: int, *, tool: str | None = None):
    records = [
        _put(
            evidence,
            message_id=f"u{index}",
            role="user",
            written_by=WrittenBy.PERSON,
            content=f"第{index}个问题。",
        ),
        _put(
            evidence,
            message_id=f"a{index}",
            role="assistant",
            written_by=WrittenBy.AGENT,
            content=f"第{index}个回答。",
        ),
    ]
    if tool is not None:
        records.append(
            _put(
                evidence,
                message_id=f"t{index}",
                role="tool",
                written_by=WrittenBy.TOOL,
                content=tool,
            )
        )
        records.append(
            _put(
                evidence,
                message_id=f"a{index}-after-tool",
                role="assistant",
                written_by=WrittenBy.AGENT,
                content=f"第{index}个结论。",
            )
        )
    return records


def _supervisor(tmp_path, stores, bridge=None):
    evidence, profiles, tasks, governance, secrets, _profile = stores
    return MemoryProcessingSupervisor(
        tasks=tasks,
        profiles=profiles,
        secrets=secrets,
        evidence=evidence,
        governance=governance,
        data_dir=tmp_path,
        runner_factory=lambda: _runner(tmp_path, stores, bridge or FakeBridge(outputs=())),
    )


def _sent_turns(tmp_path, stores, created):
    bridge = FakeBridge(outputs=())
    runner = _runner(tmp_path, stores, bridge)
    for task in created:
        asyncio.run(runner.run(task.task_id))
    return [[turn.content for turn in request.turns] for request in bridge.requests]


def test_each_turn_goes_to_hy_on_its_own(tmp_path) -> None:
    evidence = _setup(tmp_path)[0]
    first = _turn(evidence, 1, tool="[Bash] ls")
    second = _turn(evidence, 2)

    chunks = _conversation_chunks(evidence.list_conversation(first[0].evidence_id))

    assert [[r.evidence_id for r in chunk] for chunk in chunks] == [
        [r.evidence_id for r in first],
        [r.evidence_id for r in second],
    ]


def test_what_came_before_the_first_question_joins_the_turn_that_answers_it(tmp_path) -> None:
    """An opener with no reply yet is not a turn of its own.

    C3's real-authority drift showed splitting `assistant` proposal from the
    user turn that adopts it sends HY an answer without what it answers, so an
    unanswered assistant prefix joins the following turn. The prefix is still
    sent, once, and in order — it is just no longer a window of its own.
    """
    evidence = _setup(tmp_path)[0]
    opener = _put(
        evidence,
        message_id="a0",
        role="assistant",
        written_by=WrittenBy.AGENT,
        content="你好，我能帮你做什么？",
    )
    first = _turn(evidence, 1)

    chunks = _conversation_chunks(evidence.list_conversation(opener.evidence_id))

    assert [[r.evidence_id for r in chunk] for chunk in chunks] == [
        [opener.evidence_id, *[r.evidence_id for r in first]]
    ]


def test_a_turn_too_long_for_one_window_is_split_but_every_window_keeps_its_question(
    tmp_path, monkeypatch
) -> None:
    # D74 sends this turn whole; the invariant still holds on the split path.
    monkeypatch.setenv("ONE_MEMORY_SPLIT_TURNS", "1")
    evidence = _setup(tmp_path)[0]
    question = _put(
        evidence,
        message_id="u1",
        role="user",
        written_by=WrittenBy.PERSON,
        content="把这个仓库的测试跑一遍。",
    )
    calls = [
        _put(
            evidence,
            message_id=f"t{index}",
            role="tool",
            written_by=WrittenBy.TOOL,
            content=f"[Bash] {index} " + "x" * 1500,
        )
        for index in range(12)
    ]

    chunks = _conversation_chunks(evidence.list_conversation(question.evidence_id))

    assert len(chunks) > 1
    assert all(chunk[0].evidence_id == question.evidence_id for chunk in chunks)
    assert all(sum(len(r.verbatim) for r in chunk) <= 12_000 for chunk in chunks)
    assert {r.evidence_id for chunk in chunks for r in chunk} == {
        question.evidence_id,
        *(call.evidence_id for call in calls),
    }


def test_a_split_turn_never_sends_a_tool_only_window(tmp_path, monkeypatch) -> None:
    """WorkBuddy/Codex can answer, then emit many tool calls.

    If the answer fills the first window, the remaining calls used to become a
    tool-only window; HY rejected it with ``content or messages is required``.
    The tool calls must stay with the preceding non-tool window instead.
    D74 sends this turn whole; the invariant still holds on the split path.
    """
    monkeypatch.setenv("ONE_MEMORY_SPLIT_TURNS", "1")
    evidence = _setup(tmp_path)[0]
    question = _put(
        evidence,
        message_id="u1",
        role="user",
        written_by=WrittenBy.PERSON,
        content="先回答，再调用很多工具。",
    )
    answer = _put(
        evidence,
        message_id="a1",
        role="assistant",
        written_by=WrittenBy.AGENT,
        content="好。" + "z" * 7_000,
    )
    calls = [
        _put(
            evidence,
            message_id=f"t{index}",
            role="tool",
            written_by=WrittenBy.TOOL,
            content=f"[Bash] {index} " + "x" * 1_500,
        )
        for index in range(4)
    ]

    chunks = _conversation_chunks(evidence.list_conversation(question.evidence_id))

    assert chunks
    assert all(any(record.role != "tool" for record in chunk) for chunk in chunks)
    assert all(sum(len(record.verbatim) for record in chunk) <= 12_000 for chunk in chunks)
    assert list(dict.fromkeys(record.evidence_id for chunk in chunks for record in chunk)) == [
        question.evidence_id,
        answer.evidence_id,
        *(call.evidence_id for call in calls),
    ]
    assert any(record.role == "tool" for chunk in chunks for record in chunk)


def test_tool_calls_travel_with_their_turn_redacted_and_as_tool_messages(tmp_path) -> None:
    stores = _setup(tmp_path)
    evidence, _profiles, tasks, _governance, _secrets, profile = stores
    records = _turn(
        evidence, 1, tool="[Bash] docker login --username kristinhe --password exposed-token-value"
    )
    bridge = FakeBridge()

    asyncio.run(_runner(tmp_path, stores, bridge).run(_task(tasks, profile, records[0]).task_id))

    request = bridge.requests[0]
    assert [turn.role for turn in request.turns] == ["user", "assistant", "tool", "assistant"]
    assert "exposed-token-value" not in request.turns[2].content
    assert "<redacted-secret>" in request.turns[2].content
    assert records[2].evidence_id in request.evidence_ids


def test_the_next_turn_reaches_hy_with_the_previous_turn_as_history(tmp_path) -> None:
    stores = _setup(tmp_path)
    evidence, _profiles, _tasks, _governance, _secrets, profile = stores
    with sqlite3.connect(tmp_path / "one-memory.sqlite3") as db:
        db.executescript(GRANT_SCHEMA)
        db.execute(
            "INSERT INTO source_grants VALUES ('local','cursor','/authorized','default',?,NULL)",
            (datetime.now(UTC).isoformat(),),
        )
    first = _turn(evidence, 1, tool="[Bash] 12 passed")
    second = _turn(evidence, 2)
    supervisor = _supervisor(tmp_path, stores)
    bridge = FakeBridge(outputs=())
    runner = _runner(tmp_path, stores, bridge)

    for turn in (first, second):
        task = supervisor._enqueue_chunk(tuple(turn), profile, freeze_inputs=True)
        asyncio.run(runner.run(task.task_id))

    context = bridge.requests[1].source_context
    assert context is not None
    assert [r.evidence_id for r in context.current] == [r.evidence_id for r in second]
    assert [r.evidence_id for r in context.history] == [r.evidence_id for r in first]


def test_each_later_turn_carries_the_earlier_turns_as_history(tmp_path) -> None:
    """Later windows in one conversation must not arrive at HY alone.

    The frozen live pair expects later batches to grow because a window is sent
    with the turns behind it as history. A run where every request holds only the
    current turn loses the context the authority/adoption gate reads, which is
    what the cross-window real run showed.
    """
    stores = _setup(tmp_path)
    evidence, _profiles, _tasks, _governance, _secrets, profile = stores
    with sqlite3.connect(tmp_path / "one-memory.sqlite3") as db:
        db.executescript(GRANT_SCHEMA)
        db.execute(
            "INSERT INTO source_grants VALUES ('local','cursor','/authorized','default',?,NULL)",
            (datetime.now(UTC).isoformat(),),
        )
    turns = [_turn(evidence, index) for index in (1, 2, 3)]
    supervisor = _supervisor(tmp_path, stores)
    bridge = FakeBridge(outputs=())
    runner = _runner(tmp_path, stores, bridge)

    for turn in turns:
        task = supervisor._enqueue_chunk(tuple(turn), profile, freeze_inputs=True)
        asyncio.run(runner.run(task.task_id))

    assert len(bridge.requests) == 3
    for index, request in enumerate(bridge.requests):
        context = request.source_context
        assert context is not None
        assert [r.evidence_id for r in context.current] == [r.evidence_id for r in turns[index]]
        # Every turn after the first is sent with the ones behind it.
        expected = [r.evidence_id for turn in turns[:index] for r in turn]
        assert [r.evidence_id for r in context.history] == expected


def test_the_bridge_hands_hy_the_turn_and_lets_hy_choose_the_route(tmp_path) -> None:
    client = FakeHyMemoryClient({"success": True, "memory_ids": []})
    bridge = HyMemoryBridge(
        profile=snapshot(ProviderKind.OLLAMA, base_url="http://127.0.0.1:11434", has_api_key=False),
        api_key=None,
        data_dir=tmp_path,
        client=client,
    )
    request = HyMemoryIngestRequest(
        request_id="turn-1",
        user_id="local",
        source_id="codex",
        session_id="conversation-1",
        turns=(
            HyMemoryTurn(role="user", content="跑一下测试"),
            HyMemoryTurn(role="tool", content="[Bash] 12 passed"),
            HyMemoryTurn(role="assistant", content="全部通过。"),
        ),
        evidence_ids=("u", "t", "a"),
    )

    asyncio.run(bridge.ingest(request))
    asyncio.run(bridge.close())

    assert client.calls[0]["scene"] is None
    assert [item["role"] for item in client.calls[0]["data"]] == ["user", "tool", "assistant"]


def test_write_verification_accepts_the_tool_messages_hy_extracts_from(tmp_path) -> None:
    evidence = _setup(tmp_path)[0]
    question, _reply, call, _conclusion = _turn(evidence, 1, tool="[Bash] 12 passed")
    bindings = HyWriteBindings(HySourceContext(current=(question, call)))

    bindings.check_content(
        json.dumps(
            [
                {"role": "user", "content": question.verbatim},
                {"role": "tool", "content": call.verbatim},
            ],
            ensure_ascii=False,
        )
    )

    assert [source["role"] for source in bindings.sources] == ["user", "tool"]


def test_every_turn_gets_its_own_task_when_nothing_was_queued_before(tmp_path) -> None:
    stores = _setup(tmp_path)
    evidence = stores[0]
    turns = [_turn(evidence, index) for index in (1, 2, 3)]

    created = _supervisor(tmp_path, stores).prepare_conversation(turns[2][0])

    assert _sent_turns(tmp_path, stores, created) == [[r.verbatim for r in turn] for turn in turns]


def _earlier_policy_window(stores, records, *, succeeded: bool):
    _evidence, _profiles, tasks, _governance, _secrets, profile = stores
    task = tasks.create(
        request_id=f"conversation:hy-native-v22:cursor:{records[0].evidence_id}",
        user_id=records[0].user_id,
        kind=MemoryTaskKind.EXTRACT,
        input_ref=records[0].evidence_id,
        profile=profile,
        input_records=tuple(records),
    )
    _claimed, token = tasks.claim(task.task_id)
    if succeeded:
        tasks.complete(task.task_id, runner_token=token)
    else:
        tasks.fail(task.task_id, runner_token=token, error_code="hy_memory_unavailable")
    return tasks.require(task.task_id)


def test_turns_an_earlier_policy_already_extracted_are_not_sent_again(tmp_path) -> None:
    stores = _setup(tmp_path)
    evidence = stores[0]
    old = [*_turn(evidence, 1), *_turn(evidence, 2)]
    assert _earlier_policy_window(stores, old, succeeded=True).status is (
        MemoryTaskStatus.COMPLETED
    )
    new = _turn(evidence, 3)

    created = _supervisor(tmp_path, stores).prepare_conversation(new[0])

    assert _sent_turns(tmp_path, stores, created) == [[r.verbatim for r in new]]


def test_renewed_consent_still_prepares_a_turn_the_current_policy_completed(tmp_path) -> None:
    h = Harness(tmp_path)
    h.bridge.fail = False
    turn = h.put_closed_turn("selected", conversation="renewed-consent")
    h.grant(turn)
    supervisor = h.supervisor()

    async def scenario():
        try:
            first = supervisor.prepare_conversation(turn[0])[0]
            done = await supervisor._create_runner().run(first.task_id)
            assert done.status is MemoryTaskStatus.COMPLETED
            h.grant(turn)
            return first, supervisor.prepare_conversation(turn[0])
        finally:
            await supervisor.shutdown()

    first, again = asyncio.run(scenario())

    assert len(again) == 1 and again[0].task_id != first.task_id


def test_a_failed_earlier_window_does_not_count_as_already_sent(tmp_path) -> None:
    stores = _setup(tmp_path)
    old = _turn(stores[0], 1)
    assert _earlier_policy_window(stores, old, succeeded=False).status is MemoryTaskStatus.FAILED

    created = _supervisor(tmp_path, stores).prepare_conversation(old[0])

    assert _sent_turns(tmp_path, stores, created) == [[r.verbatim for r in old]]


def test_a_tool_heavy_turn_reaches_hy_once_and_still_becomes_a_memory(tmp_path) -> None:
    """D74's full criterion: one call to HY, and a memory comes out of it.

    The half that was only ever checked by hand is the second one. On 17:14
    the acceptance run for this ended with `completed=false` and zero
    evidence, so nothing proved the memory half either way; the window count
    alone leaves it open.
    """
    stores = _setup(tmp_path)
    evidence, _profiles, tasks, governance, _secrets, profile = stores
    records = _turn(evidence, 1, tool="x" * 50_000)
    task = _task(tasks, profile, records[0])
    bridge = FakeBridge(
        outputs=(HyMemoryOutput(memory_id="hy-tool-1", content="用户跑过一次全量测试。"),)
    )

    asyncio.run(_runner(tmp_path, stores, bridge).run(task.task_id))

    # One window, so one call.
    assert len(_conversation_chunks(records)) == 1
    assert len(bridge.requests) == 1
    # And the memory that call produced actually landed.
    assert [item.content for item in governance.list(user_id="local")] == ["用户跑过一次全量测试。"]
