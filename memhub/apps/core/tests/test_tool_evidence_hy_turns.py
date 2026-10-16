"""Every adapter's tool records reach HY inside the turn they happened in.

Each case starts from the source's own on-disk shape, imports it through
`ImportRunner`, and reads back what the processing path hands HY. Adapter
tests stop at the evidence row and turn tests start from hand-built rows;
this is the seam between them: a call filed under the wrong conversation,
dropped from a mixed block, or ordered by something other than when it
happened would pass both and still reach HY in the wrong turn.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

from one_memory.imports.access import AuthorizedSourceRoot
from one_memory.imports.adapters import SourceAdapter
from one_memory.imports.jobs import ImportJobStore, ImportStatus
from one_memory.imports.runner import ImportRunner
from one_memory.imports.sources.claude_code import create_claude_code_adapter
from one_memory.imports.sources.cursor_unified import CursorUnifiedSourceAdapter
from one_memory.imports.sources.deepseek_harness import create_deepseek_harness_adapter
from one_memory.imports.sources.hermes import create_hermes_adapter
from one_memory.imports.sources.openclaw import (
    OPENCLAW_DESCRIPTOR,
    OpenClawSourceAdapter,
)
from one_memory.imports.sources.opencode import create_opencode_adapter
from one_memory.imports.sources.pi import create_pi_adapter
from one_memory.imports.sources.qwen_code import create_qwen_code_adapter
from one_memory.memory_processing import MemoryTaskRunner, _conversation_chunks
from tests.test_hy_turn_input import _supervisor
from tests.test_memory_processing import FakeBridge, _runner, _setup

SECRET = "sk-abcdefghijklmnopqrstu"

#: (role, a marker the content must contain), in the order HY must see them.
Expected = list[tuple[str, str]]

FIRST_TURN: Expected = [
    ("user", "第一个问题"),
    ("tool", "pytest --token"),
    ("tool", "3 failed"),
    ("assistant", "第一个结论"),
]
SECOND_TURN: Expected = [
    ("user", "第二个问题"),
    ("tool", "git status"),
    ("tool", "working tree clean"),
    ("assistant", "第二个结论"),
]
FIRST_ONLY = ("pytest --token", "3 failed", "第一个")
SECOND_ONLY = ("git status", "working tree clean", "第二个")


def _grant(source_id: str, root: Path) -> AuthorizedSourceRoot:
    return AuthorizedSourceRoot.grant(
        source_id=source_id, root=root, user_confirmed=True, home=root.parent / "other-home"
    )


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(r, ensure_ascii=False)}\n" for r in records), "utf-8")


def _import(tmp_path: Path, adapter: SourceAdapter, source_id: str):
    stores = _setup(tmp_path)
    jobs = ImportJobStore(tmp_path / "one-memory.sqlite3")
    job = jobs.create(user_id="local", source_id=source_id)
    done = asyncio.run(
        ImportRunner(evidence_store=stores[0], job_store=jobs).run(
            adapter=adapter, job_id=job.job_id
        )
    )
    assert done.status is ImportStatus.COMPLETED
    return stores


def _requests_to_hy(tmp_path: Path, adapter: SourceAdapter, source_id: str) -> list[Expected]:
    """What the processing path sends HY for every imported conversation, call by call."""
    stores = _import(tmp_path, adapter, source_id)
    evidence = stores[0]
    supervisor = _supervisor(tmp_path, stores)
    bridge = FakeBridge(outputs=())
    runner: MemoryTaskRunner = _runner(tmp_path, stores, bridge)
    for seed_id in evidence.list_conversation_seed_ids():
        seed = evidence.get(seed_id)
        assert seed is not None
        for task in supervisor.prepare_conversation(seed):
            asyncio.run(runner.run(task.task_id))
    return [[(turn.role, turn.content) for turn in request.turns] for request in bridge.requests]


def _assert_turns(sent: list[Expected], *expected: Expected) -> None:
    assert len(sent) == len(expected), sent
    for turns, wanted in zip(sent, expected, strict=True):
        assert [role for role, _ in turns] == [role for role, _ in wanted], turns
        for (_, content), (_, marker) in zip(turns, wanted, strict=True):
            assert marker in content, (marker, turns)
        assert all(SECRET not in content for _, content in turns)


def _assert_two_turns(sent: list[Expected]) -> None:
    _assert_turns(sent, FIRST_TURN, SECOND_TURN)
    first, second = ("\n".join(content for _, content in turns) for turns in sent)
    assert not any(marker in first for marker in SECOND_ONLY)
    assert not any(marker in second for marker in FIRST_ONLY)


def test_claude_code_tool_calls_reach_hy_inside_their_turn(tmp_path: Path) -> None:
    root = tmp_path / "claude"
    at = iter(f"2026-09-20T10:00:{second:02d}.000Z" for second in range(60))

    def line(uuid: str, kind: str, content: object) -> dict[str, object]:
        return {
            "type": kind,
            "uuid": uuid,
            "sessionId": "s",
            "cwd": "/private/work/acme",
            "timestamp": next(at),
            "message": {"role": kind, "content": content},
        }

    def call(uuid: str, command: str, *, saying: str | None = None) -> dict[str, object]:
        block = {"type": "tool_use", "id": uuid, "name": "Bash", "input": {"command": command}}
        said = [{"type": "text", "text": saying}] if saying else []
        return line(uuid, "assistant", [*said, block])

    def result(uuid: str, text: str) -> dict[str, object]:
        block = {"type": "tool_result", "tool_use_id": uuid, "content": text}
        return line(uuid, "user", [block])

    _write_jsonl(
        root / "project" / "s.jsonl",
        [
            line("u1", "user", "第一个问题：跑测试"),
            call("c1", f"pytest --token {SECRET}"),
            result("r1", "3 failed"),
            line("a1", "assistant", [{"type": "text", "text": "第一个结论"}]),
            line("u2", "user", "第二个问题：看状态"),
            # Said something and called a tool in one record.
            call("c2", "git status", saying="先看一下。"),
            result("r2", "working tree clean"),
            line("a2", "assistant", [{"type": "text", "text": "第二个结论"}]),
        ],
    )

    sent = _requests_to_hy(
        tmp_path, create_claude_code_adapter(_grant("claude-code", root)), "claude-code"
    )

    second = [("user", "第二个问题"), ("assistant", "先看一下"), *SECOND_TURN[1:]]
    _assert_turns(sent, FIRST_TURN, second)
    assert sent[0][1][1].startswith("[Bash] ")
    first, later = ("\n".join(content for _, content in turns) for turns in sent)
    assert not any(marker in first for marker in SECOND_ONLY)
    assert not any(marker in later for marker in FIRST_ONLY)


def test_cursor_transcript_tool_calls_reach_hy_inside_their_turn(tmp_path: Path) -> None:
    """Cursor transcripts carry no timestamps; file order is the only order there is.

    An assistant message routinely says something and calls a tool in the same
    content list; the call must not be lost because the text was kept.
    """
    root = tmp_path / ".cursor"

    def user(text: str) -> dict[str, object]:
        block = {"type": "text", "text": f"<user_query>{text}</user_query>"}
        return {"role": "user", "message": {"content": [block]}}

    def assistant(*blocks: dict[str, object]) -> dict[str, object]:
        return {"role": "assistant", "message": {"content": list(blocks)}}

    def call(command: str) -> dict[str, object]:
        return {"type": "tool_use", "name": "Shell", "input": {"command": command}}

    def result(text: str) -> dict[str, object]:
        block = {"type": "tool_result", "tool_use_id": "t", "content": text}
        return {"role": "user", "message": {"content": [block]}}

    _write_jsonl(
        root / "projects" / "workspace" / "agent-transcripts" / "chat" / "chat.jsonl",
        [
            user("第一个问题：跑测试"),
            assistant(call(f"pytest --token {SECRET}")),
            result("3 failed"),
            assistant({"type": "text", "text": "第一个结论"}),
            user("第二个问题：看状态"),
            assistant({"type": "text", "text": "先看一下。"}, call("git status")),
            result("working tree clean"),
            assistant({"type": "text", "text": "第二个结论"}),
        ],
    )

    sent = _requests_to_hy(
        tmp_path, CursorUnifiedSourceAdapter(grant=_grant("cursor", root)), "cursor"
    )

    second = [("user", "第二个问题"), ("assistant", "先看一下"), *SECOND_TURN[1:]]
    _assert_turns(sent, FIRST_TURN, second)


def test_pi_tool_calls_reach_hy_inside_their_turn(tmp_path: Path) -> None:
    root = tmp_path / "pi"
    root.mkdir()
    at = iter(f"2026-09-20T10:00:{second:02d}.000Z" for second in range(60))

    def message(entry_id: str, role: str, content: object, **extra: object) -> dict[str, object]:
        body = {"role": role, "content": content, **extra}
        return {"type": "message", "id": entry_id, "timestamp": next(at), "message": body}

    def call(call_id: str, command: str) -> dict[str, object]:
        return {
            "type": "toolCall",
            "id": call_id,
            "name": "bash",
            "arguments": {"command": command},
        }

    def result(entry_id: str, call_id: str, text: str) -> dict[str, object]:
        return message(
            entry_id,
            "toolResult",
            [{"type": "text", "text": text}],
            toolCallId=call_id,
            toolName="bash",
        )

    _write_jsonl(
        root / "session-t.jsonl",
        [
            {"type": "session", "id": "session-t", "cwd": "/Users/example/work/atlas"},
            message("u1", "user", [{"type": "text", "text": "第一个问题：跑测试"}]),
            message("c1", "assistant", [call("k1", f"pytest --token {SECRET}")]),
            result("r1", "k1", "3 failed"),
            message("a1", "assistant", [{"type": "text", "text": "第一个结论"}]),
            message("u2", "user", [{"type": "text", "text": "第二个问题：看状态"}]),
            message(
                "c2",
                "assistant",
                [{"type": "text", "text": "先看一下。"}, call("k2", "git status")],
            ),
            result("r2", "k2", "working tree clean"),
            message("a2", "assistant", [{"type": "text", "text": "第二个结论"}]),
        ],
    )

    sent = _requests_to_hy(tmp_path, create_pi_adapter(_grant("pi", root)), "pi")

    second = [("user", "第二个问题"), ("assistant", "先看一下"), *SECOND_TURN[1:]]
    _assert_turns(sent, FIRST_TURN, second)


def test_qwen_code_tool_calls_reach_hy_inside_their_turn(tmp_path: Path) -> None:
    root = tmp_path / "qwen-code"
    session = "a1b2c3d4-0000-4000-8000-000000000003"
    at = iter(f"2026-09-20T10:00:{second:02d}.000Z" for second in range(60))
    base = {"sessionId": session, "cwd": "/Users/example/work/atlas"}

    def user(uuid: str, text: str) -> dict[str, object]:
        return {**base, "uuid": uuid, "timestamp": next(at), "type": "user", "text": text}

    def model(uuid: str, *parts: dict[str, object]) -> dict[str, object]:
        return {
            **base,
            "uuid": uuid,
            "timestamp": next(at),
            "type": "assistant",
            "provenance": "assistant_output",
            "message": {"role": "model", "parts": list(parts)},
        }

    def call(command: str) -> dict[str, object]:
        return {"functionCall": {"name": "run_shell_command", "args": {"command": command}}}

    def result(uuid: str, text: str) -> dict[str, object]:
        response = {"name": "run_shell_command", "response": {"output": text}}
        return {
            **base,
            "uuid": uuid,
            "timestamp": next(at),
            "type": "tool_result",
            "provenance": "tool_result",
            "message": {"role": "user", "parts": [{"functionResponse": response}]},
        }

    _write_jsonl(
        root / "-Users-example-work-atlas" / "chats" / f"{session}.jsonl",
        [
            user("u1", "第一个问题：跑测试"),
            model("c1", call(f"pytest --token {SECRET}")),
            result("r1", "3 failed"),
            model("a1", {"text": "第一个结论"}),
            user("u2", "第二个问题：看状态"),
            model("c2", {"text": "先看一下。"}, call("git status")),
            result("r2", "working tree clean"),
            model("a2", {"text": "第二个结论"}),
        ],
    )

    sent = _requests_to_hy(
        tmp_path, create_qwen_code_adapter(_grant("qwen-code", root)), "qwen-code"
    )

    second = [("user", "第二个问题"), ("assistant", "先看一下"), *SECOND_TURN[1:]]
    _assert_turns(sent, FIRST_TURN, second)


def test_deepseek_harness_tool_calls_reach_hy_inside_their_turn(tmp_path: Path) -> None:
    root = tmp_path / "dsh"
    seq = iter(range(1, 100))

    def event(kind: str, data: dict[str, object]) -> dict[str, object]:
        number = next(seq)
        return {
            "type": kind,
            "seq": number,
            "time": f"2026-09-20T10:00:{number:02d}Z",
            "data": data,
        }

    def user(message_id: str, text: str) -> dict[str, object]:
        content = [{"type": "text", "text": text}]
        body = {"id": message_id, "role": "user", "source": {"kind": "user"}, "content": content}
        return event("user/message", body)

    def assistant(message_id: str, *blocks: dict[str, object]) -> dict[str, object]:
        body = {"id": message_id, "role": "assistant", "content": list(blocks)}
        return event("assistant/message", {"message": body})

    def call(call_id: str, command: str) -> list[dict[str, object]]:
        arguments = json.dumps({"command": command})
        block = {"type": "tool-call", "id": call_id, "name": "shell", "arguments": arguments}
        # Upstream stores the call twice: inside the assistant message and as its own event.
        return [
            assistant(f"m-{call_id}", block),
            event("tool/call", {"callId": call_id, "name": "shell", "arguments": arguments}),
        ]

    def result(call_id: str, text: str) -> dict[str, object]:
        block = {
            "type": "tool-result",
            "toolCallId": call_id,
            "content": [{"type": "text", "text": text}],
            "isError": False,
        }
        body = {
            "id": f"result-{call_id}",
            "role": "user",
            "source": {"kind": "tool", "callId": call_id},
            "content": [block],
        }
        return event("tool/result", {"message": body})

    _write_jsonl(
        root / "run-t" / "session.jsonl",
        [
            {"type": "session", "id": "run-t", "cwd": "/private/project"},
            user("u1", "第一个问题：跑测试"),
            *call("k1", f"pytest --token {SECRET}"),
            result("k1", "3 failed"),
            assistant("a1", {"type": "text", "text": "第一个结论"}),
            user("u2", "第二个问题：看状态"),
            *call("k2", "git status"),
            result("k2", "working tree clean"),
            assistant("a2", {"type": "text", "text": "第二个结论"}),
        ],
    )

    sent = _requests_to_hy(
        tmp_path,
        create_deepseek_harness_adapter(_grant("deepseek-harness", root)),
        "deepseek-harness",
    )

    _assert_two_turns(sent)


def test_hermes_tool_calls_reach_hy_inside_their_turn(tmp_path: Path) -> None:
    """Hermes keeps the call on the assistant row (`tool_calls`) and the result as a `tool` row.

    `timestamp` is whole seconds here, so rows written in the same second tie;
    the row id is the order Hermes wrote them in.
    """
    root = tmp_path / "hermes"
    root.mkdir()
    connection = sqlite3.connect(root / "state.db")
    connection.executescript(
        """
        CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, cwd TEXT);
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,
            tool_call_id TEXT, tool_calls TEXT, tool_name TEXT, reasoning TEXT,
            timestamp INTEGER, active INTEGER
        );
        INSERT INTO sessions VALUES ('s', 'cli', '/private/project');
        """
    )

    def calls(command: str) -> str:
        arguments = json.dumps({"command": command})
        return json.dumps([{"id": "c", "function": {"name": "terminal", "arguments": arguments}}])

    connection.executemany(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "s", "user", "第一个问题：跑测试", None, None, None, None, 1000, 1),
            (2, "s", "assistant", "", None, calls(f"pytest --token {SECRET}"), None, None, 1001, 1),
            (3, "s", "tool", "3 failed", "c", None, "terminal", None, 1001, 1),
            (4, "s", "assistant", "第一个结论", None, None, None, None, 1001, 1),
            (5, "s", "user", "第二个问题：看状态", None, None, None, None, 1002, 1),
            (6, "s", "assistant", "", None, calls("git status"), None, None, 1003, 1),
            (7, "s", "tool", "working tree clean", "c", None, "terminal", None, 1003, 1),
            (8, "s", "assistant", "第二个结论", None, None, None, None, 1003, 1),
        ],
    )
    connection.commit()
    connection.close()

    sent = _requests_to_hy(tmp_path, create_hermes_adapter(_grant("hermes", root)), "hermes")

    _assert_two_turns(sent)
    assert sent[0][1][1].startswith("[terminal] ")


def _assembled_for_hy(tmp_path: Path, adapter: SourceAdapter, source_id: str) -> list[Expected]:
    """The turns HY would get for each imported conversation, assembled the processing way.

    For sources whose `user` rows are not attributed to the person, the
    processing path does not schedule the conversation at all, so the
    assembly is read directly.
    """
    evidence = _import(tmp_path, adapter, source_id)[0]
    with sqlite3.connect(tmp_path / "one-memory.sqlite3") as db:
        members = [
            row[0]
            for row in db.execute("SELECT MIN(evidence_id) FROM evidence GROUP BY conversation_id")
        ]
    requests: list[Expected] = []
    for member in members:
        for chunk in _conversation_chunks(evidence.list_conversation(member)):
            request = MemoryTaskRunner._request(_FakeTask, chunk)
            requests.append([(turn.role, turn.content) for turn in request.turns])
    return requests


class _FakeTask:
    request_id = "assembly-only"


def _assert_in_source_order(sent: list[Expected], expected: Expected) -> None:
    assert len(sent) == 1, sent
    _assert_turns(sent, expected)


def test_openclaw_tool_rows_reach_hy_in_the_order_they_happened(tmp_path: Path) -> None:
    """`created_at` is whole seconds; a call and its result share one.

    Row order is what breaks the tie, not the hash of an id.
    """
    root = tmp_path / "openclaw"
    root.mkdir()
    connection = sqlite3.connect(root / "conversations.db")
    connection.executescript(
        """
        CREATE TABLE conversations (id TEXT PRIMARY KEY, workspace_path TEXT, git_root TEXT);
        CREATE TABLE messages (
            id TEXT PRIMARY KEY, conversation_id TEXT, role TEXT, content TEXT, created_at TEXT
        );
        INSERT INTO conversations VALUES ('c', '/private/project', NULL);
        """
    )
    connection.executemany(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
        [
            ("m1", "c", "user", "第一个问题：跑测试", "2026-09-20T10:00:00Z"),
            ("m2", "c", "tool", f"pytest --token {SECRET}", "2026-09-20T10:00:01Z"),
            ("m3", "c", "tool", "3 failed", "2026-09-20T10:00:01Z"),
            ("m4", "c", "assistant", "第一个结论", "2026-09-20T10:00:01Z"),
            ("m5", "c", "user", "第二个问题：看状态", "2026-09-20T10:00:02Z"),
            ("m6", "c", "tool", "git status", "2026-09-20T10:00:03Z"),
            ("m7", "c", "tool", "working tree clean", "2026-09-20T10:00:03Z"),
            ("m8", "c", "assistant", "第二个结论", "2026-09-20T10:00:03Z"),
        ],
    )
    connection.commit()
    connection.close()

    # Reader-shape coverage only. The public factory is paused; the negative
    # control lives in test_c1_openclaw_paused.py.
    adapter = OpenClawSourceAdapter(OPENCLAW_DESCRIPTOR, _grant("openclaw", root))
    sent = _assembled_for_hy(tmp_path, adapter, "openclaw")

    # D98: a `user` row reaches HY whoever the parser says wrote it, so the
    # questions are here now — what must hold is that everything is still in
    # the order it happened.
    _assert_in_source_order(sent, FIRST_TURN + SECOND_TURN)


def test_opencode_tool_parts_reach_hy_in_the_order_they_happened(tmp_path: Path) -> None:
    """A message's tool parts carry the message's own time, so they tie with its text.

    OpenCode keeps one assistant message across several steps: text, a tool
    call, more text. The call must stay between the two.
    """
    root = tmp_path / "opencode"
    root.mkdir()
    connection = sqlite3.connect(root / "opencode.db")
    connection.executescript(
        """
        CREATE TABLE session (id TEXT PRIMARY KEY, directory TEXT);
        CREATE TABLE message (
            id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER, data TEXT
        );
        CREATE TABLE part (
            id TEXT PRIMARY KEY, message_id TEXT, time_created INTEGER, data TEXT
        );
        INSERT INTO session VALUES ('s', '/private/project');
        """
    )
    at = iter(range(1_758_000_000_000, 1_758_000_100_000, 1000))

    def message(message_id: str, role: str, *parts: dict[str, object]) -> None:
        connection.execute(
            "INSERT INTO message VALUES (?, 's', ?, ?)",
            (message_id, next(at), json.dumps({"role": role})),
        )
        for index, data in enumerate(parts):
            connection.execute(
                "INSERT INTO part VALUES (?, ?, ?, ?)",
                (f"{message_id}-p{index}", message_id, next(at), json.dumps(data)),
            )

    def tool(command: str, output: str) -> dict[str, object]:
        state = {"status": "completed", "input": {"command": command}, "output": output}
        return {"type": "tool", "tool": "bash", "callID": command, "state": state}

    message("u1", "user", {"type": "text", "text": "第一个问题：跑测试"})
    message(
        "a1",
        "assistant",
        tool(f"pytest --token {SECRET}", "3 failed"),
        {"type": "text", "text": "第一个结论"},
    )
    message("u2", "user", {"type": "text", "text": "第二个问题：看状态"})
    message(
        "a2",
        "assistant",
        {"type": "text", "text": "先看一下。"},
        tool("git status", "working tree clean"),
        {"type": "text", "text": "第二个结论"},
    )
    connection.commit()
    connection.close()

    sent = _assembled_for_hy(
        tmp_path, create_opencode_adapter(_grant("opencode", root)), "opencode"
    )

    # One tool part is one row: OpenCode keeps a call and its output together.
    # D98: the questions are here now too — a `user` row reaches HY whoever
    # the parser says wrote it — and everything stays in the order it happened.
    _assert_in_source_order(
        sent,
        [
            ("user", "第一个问题：跑测试"),
            ("tool", "[bash] 3 failed"),
            ("assistant", "第一个结论"),
            ("user", "第二个问题：看状态"),
            ("assistant", "先看一下。"),
            ("tool", "[bash] working tree clean"),
            ("assistant", "第二个结论"),
        ],
    )


def test_source_positions_compare_as_numbers_and_leave_the_file_key_alone() -> None:
    from one_memory.imports.sources.common import sortable_ordinal

    assert sortable_ordinal("f0a9:byte:900") < sortable_ordinal("f0a9:byte:1000")
    assert sortable_ordinal("f0a9:row:12:1") < sortable_ordinal("f0a9:row:12:10")
    assert sortable_ordinal("f0a9:byte:900").startswith("f0a9:byte:")
