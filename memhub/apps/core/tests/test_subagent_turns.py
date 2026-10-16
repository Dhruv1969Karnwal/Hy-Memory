"""D98: a sub agent's turn is ordinary user-assistant data.

The host parser marks a sub agent's question as written by the agent, and the
turn assembly then kept only `user` + `PERSON` — so Claude Code sent just the
first question of a transcript, and a turn holding only a sub agent's question
was dropped whole.

The `role` field cannot say who wrote a message, so filtering on the author
threw away real turns. Author stays: it labels the display and decides owner.
It no longer decides whether a message is sent.
"""

import asyncio

from one_memory.imports.models import WrittenBy
from tests.test_hy_turn_input import (
    FakeBridge,
    _runner,
    _setup,
    _task,
)
from tests.test_memory_processing import _put


def test_a_sub_agent_question_reaches_hy_with_its_answer(tmp_path) -> None:
    """Acceptance case 1: the sub agent's Q and A are both sent."""
    stores = _setup(tmp_path)
    evidence, _profiles, tasks, _governance, _secrets, profile = stores
    bridge = FakeBridge()

    person = _put(
        evidence,
        message_id="u0",
        role="user",
        written_by=WrittenBy.PERSON,
        content="帮我看下这个仓库的测试为什么挂了",
    )
    sub_question = _put(
        evidence,
        message_id="u1",
        role="user",
        written_by=WrittenBy.AGENT,
        content="读取 tests/test_import.py 并找出失败原因",
    )
    sub_answer = _put(
        evidence,
        message_id="a1",
        role="assistant",
        written_by=WrittenBy.AGENT,
        content="失败原因是 fixture 没清理",
    )

    asyncio.run(_runner(tmp_path, stores, bridge).run(_task(tasks, profile, person).task_id))

    request = bridge.requests[0]
    roles = [turn.role for turn in request.turns]
    contents = [turn.content for turn in request.turns]
    assert roles == ["user", "user", "assistant"], roles
    assert any("读取 tests/test_import.py" in content for content in contents)
    assert sub_question.evidence_id in request.evidence_ids
    assert sub_answer.evidence_id in request.evidence_ids


def test_a_turn_of_only_a_sub_agent_question_is_not_dropped(tmp_path) -> None:
    """Acceptance case 2: other hosts lost the whole turn here."""
    stores = _setup(tmp_path)
    evidence, _profiles, tasks, _governance, _secrets, profile = stores
    bridge = FakeBridge()

    sub_question = _put(
        evidence,
        message_id="u0",
        role="user",
        written_by=WrittenBy.AGENT,
        content="这条是子 agent 的追问",
    )
    _put(
        evidence,
        message_id="a0",
        role="assistant",
        written_by=WrittenBy.AGENT,
        content="这条是它的回答",
    )

    asyncio.run(_runner(tmp_path, stores, bridge).run(_task(tasks, profile, sub_question).task_id))

    assert bridge.requests, "a turn with no question by the person was dropped"
    assert [turn.role for turn in bridge.requests[0].turns] == ["user", "assistant"]


def test_a_turn_by_the_person_is_unchanged(tmp_path) -> None:
    """Acceptance case 3: regression — the ordinary case sends the same."""
    stores = _setup(tmp_path)
    evidence, _profiles, tasks, _governance, _secrets, profile = stores
    bridge = FakeBridge()

    person = _put(
        evidence,
        message_id="u0",
        role="user",
        written_by=WrittenBy.PERSON,
        content="我以后都用中文记",
    )
    _put(
        evidence,
        message_id="a0",
        role="assistant",
        written_by=WrittenBy.AGENT,
        content="好的，记下了",
    )

    asyncio.run(_runner(tmp_path, stores, bridge).run(_task(tasks, profile, person).task_id))

    request = bridge.requests[0]
    assert [turn.role for turn in request.turns] == ["user", "assistant"]
    assert [turn.content for turn in request.turns] == ["我以后都用中文记", "好的，记下了"]
