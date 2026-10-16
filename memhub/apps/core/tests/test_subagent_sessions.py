"""D98: a sub agent's transcript is its own conversation, not part of the main one.

Claude Code writes sidechain records into the same jsonl as the main session.
Mixing them in gives HY one long conversation whose questions are not the
person's, and dropping them loses the sub agent's work. Each sidechain has to
become a conversation of its own; the main one keeps only its own records.
"""

import asyncio
import json

from one_memory.imports.sources.claude_code import create_claude_code_adapter
from tests.test_primary_source_adapters import evidence, scan
from tests.test_tool_evidence_hy_turns import _grant


def test_a_sidechain_record_gets_its_own_conversation(tmp_path) -> None:
    root = tmp_path / "claude"
    (root / "project").mkdir(parents=True)
    transcript = root / "project" / "session.jsonl"

    def line(**overrides: object) -> dict[str, object]:
        record: dict[str, object] = {
            "type": "user",
            "sessionId": "session",
            "cwd": "/private/work/acme",
            "timestamp": "2026-09-01T00:00:00Z",
            "message": {"content": [{"type": "text", "text": "一句话"}]},
        }
        record.update(overrides)
        return record

    transcript.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False)
            for record in (
                line(uuid="m1"),
                line(uuid="m2", type="assistant"),
                # Real sidechain records carry the agent they belong to; the
                # fixture has to, or the parser cannot group them.
                line(uuid="s1", isSidechain=True, agentId="agent-a"),
                line(uuid="s2", type="assistant", isSidechain=True, agentId="agent-a"),
                line(uuid="m3", type="assistant"),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    records = evidence(asyncio.run(scan(create_claude_code_adapter(_grant("claude-code", root)))))
    by_message = {item.evidence.message_id: item.evidence for item in records}

    main_ids = {"m1", "m2", "m3"}
    sub_ids = {"s1", "s2"}
    assert by_message.keys() >= main_ids | sub_ids, sorted(by_message)

    main_conversations = {by_message[message_id].conversation_id for message_id in main_ids}
    sub_conversations = {by_message[message_id].conversation_id for message_id in sub_ids}
    assert len(main_conversations) == 1
    assert len(sub_conversations) == 1
    # Separate, so HY never sees the sub agent's turns as part of the person's.
    assert main_conversations.isdisjoint(sub_conversations)


def test_the_switch_off_skips_sub_agent_transcripts(tmp_path, monkeypatch) -> None:
    """`ONE_MEMORY_IMPORT_SUBAGENT_TRACE=0`: the sub agent's trace is not imported.

    The privacy-preserving option — the sub-agent trace need not be sent at all. Off
    means the whole transcript is skipped at the scan, not dropped later.
    """
    monkeypatch.setenv("ONE_MEMORY_IMPORT_SUBAGENT_TRACE", "0")
    root = tmp_path / "claude"
    (root / "project").mkdir(parents=True)
    (root / "project" / "subagents").mkdir()

    def write(path, records):
        path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
            encoding="utf-8",
        )

    write(
        root / "project" / "session.jsonl",
        [
            {
                "type": "user",
                "sessionId": "session",
                "uuid": "m1",
                "cwd": "/private/work",
                "timestamp": "2026-09-01T00:00:00Z",
                "message": {"content": [{"type": "text", "text": "本人问的"}]},
            },
            {
                "type": "user",
                "sessionId": "session",
                "uuid": "s1",
                "isSidechain": True,
                "message": {"content": [{"type": "text", "text": "子 agent 的"}]},
            },
        ],
    )
    write(
        root / "project" / "subagents" / "child.jsonl",
        [
            {
                "type": "user",
                "sessionId": "child",
                "uuid": "child-u1",
                "message": {"content": [{"type": "text", "text": "委派任务"}]},
            }
        ],
    )

    records = evidence(asyncio.run(scan(create_claude_code_adapter(_grant("claude-code", root)))))
    ids = {item.evidence.message_id for item in records}

    assert "m1" in ids, "the main transcript must still import"
    assert "s1" not in ids, "a sidechain record was imported with the switch off"
    assert "child-u1" not in ids, "a subagents/ transcript was imported with the switch off"


def test_two_sub_agents_do_not_share_a_conversation(tmp_path) -> None:
    """Each delegation is its own thread.

    One main session can delegate more than once. A single shared suffix would
    splice every sub agent into one conversation and hand HY a transcript that
    never happened — which is worse than the old behaviour of dropping them.
    """
    root = tmp_path / "claude"
    (root / "project").mkdir(parents=True)
    transcript = root / "project" / "session.jsonl"

    def line(**overrides: object) -> dict[str, object]:
        record: dict[str, object] = {
            "type": "user",
            "sessionId": "session",
            "cwd": "/private/work",
            "timestamp": "2026-09-01T00:00:00Z",
            "message": {"content": [{"type": "text", "text": "一句话"}]},
        }
        record.update(overrides)
        return record

    transcript.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False)
            for record in (
                line(uuid="m1"),
                line(uuid="a1-1", isSidechain=True, agentId="agent-a"),
                line(uuid="b1-1", isSidechain=True, agentId="agent-b"),
                line(uuid="a1-2", type="assistant", isSidechain=True, agentId="agent-a"),
                line(uuid="m2", type="assistant"),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    records = evidence(asyncio.run(scan(create_claude_code_adapter(_grant("claude-code", root)))))
    by_message = {item.evidence.message_id: item.evidence for item in records}

    assert by_message["m1"].conversation_id == by_message["m2"].conversation_id
    main = by_message["m1"].conversation_id
    agent_a = {by_message["a1-1"].conversation_id, by_message["a1-2"].conversation_id}
    agent_b = {by_message["b1-1"].conversation_id}

    # One thread per delegation, and neither is the main one.
    assert len(agent_a) == 1
    assert agent_a != agent_b
    assert main not in agent_a | agent_b


def test_sidechain_records_without_an_agent_share_one_conversation(tmp_path) -> None:
    """Known limit, pinned: no `agentId` means one `{session}:subagent` thread.

    Two delegations that both lack the field would be spliced together. No
    real transcript omits it, so this is not guessed at from `parentUuid`
    chains — it is left as the one conversation and recorded here, so a change
    in behaviour has to say why.
    """
    root = tmp_path / "claude"
    (root / "project").mkdir(parents=True)
    transcript = root / "project" / "session.jsonl"

    def line(**overrides: object) -> dict[str, object]:
        record: dict[str, object] = {
            "type": "user",
            "sessionId": "session",
            "cwd": "/private/work",
            "timestamp": "2026-09-01T00:00:00Z",
            "message": {"content": [{"type": "text", "text": "一句话"}]},
        }
        record.update(overrides)
        return record

    transcript.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False)
            for record in (
                line(uuid="m1"),
                line(uuid="s1", isSidechain=True),
                line(uuid="s2", type="assistant", isSidechain=True),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    records = evidence(asyncio.run(scan(create_claude_code_adapter(_grant("claude-code", root)))))
    by_message = {item.evidence.message_id: item.evidence for item in records}

    assert by_message["s1"].conversation_id == by_message["s2"].conversation_id
    assert by_message["s1"].conversation_id != by_message["m1"].conversation_id
