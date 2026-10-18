"""Manual conversation import keeps original text, role, fences, and unknown time.

The contract (`manual-conversation-import.md`) forbids guessing an author or a
message time. These tests pin the observable shape: time stays null, role markers
drive authorship, code fences survive verbatim, and an unsupported marker is a
located rejection rather than a silent rewrite.
"""

import json

import pytest

from one_memory.imports.chat_exports import parse_chat_export
from one_memory.imports.manual_transcript import (
    ManualTranscriptError,
    parse_manual_json,
    parse_marked_text,
)


def test_unknown_time_stays_null_and_is_never_guessed_from_text():
    payload = json.dumps(
        [
            {
                "role": "user",
                "content": "The budget we agreed on yesterday afternoon was 50k.",
                "time": None,
            }
        ]
    ).encode()
    export = parse_chat_export(payload, "chat.json")
    assert len(export.messages) == 1
    assert export.messages[0].created_at is None
    assert export.messages[0].verbatim == "The budget we agreed on yesterday afternoon was 50k."


def test_natural_language_time_is_rejected_not_inferred():
    payload = json.dumps(
        [{"role": "user", "content": "Keep the source text.", "time": "yesterday afternoon"}]
    ).encode()
    with pytest.raises(ManualTranscriptError) as caught:
        parse_chat_export(payload, "chat.json")
    assert caught.value.issue["code"] == "invalid_time"
    assert caught.value.issue["location"] == "message:1"


def test_naive_time_without_timezone_is_rejected_not_defaulted():
    payload = json.dumps(
        [{"role": "user", "content": "Keep the source text.", "time": "2026-09-12T09:00:00"}]
    ).encode()
    with pytest.raises(ManualTranscriptError) as caught:
        parse_chat_export(payload, "chat.json")
    assert caught.value.issue["code"] == "invalid_time"


def test_role_markers_drive_authorship_and_unsupported_markers_are_located():
    text = (
        "# Session name\n"
        "User: Source text citations must be kept before release.\n"
        "\n"
        "Assistant: Suggest adding item-by-item checks.\n"
        "```python\n"
        "User: This is not a new role segment\n"
        "```\n"
    )
    parsed = parse_marked_text(text)
    assert [message.role for message in parsed] == ["user", "assistant"]
    assert parsed[0].project == "Session name"
    assert "```python" in parsed[1].verbatim
    assert parsed[1].verbatim.endswith("```")
    assert "This is not a new role segment" in parsed[1].verbatim
    assert all(message.created_at is None for message in parsed)

    with pytest.raises(ManualTranscriptError) as caught:
        parse_marked_text("User: readable\nTool: pending")
    assert caught.value.issue["code"] == "unknown_role"
    assert caught.value.issue["location"] == "line:2"


def test_unclosed_code_fence_is_rejected_with_its_starting_line():
    with pytest.raises(ManualTranscriptError) as caught:
        parse_marked_text("User: look at this code\n```python\nx = 1\n")
    assert caught.value.issue["code"] == "unclosed_code"
    assert caught.value.issue["location"] == "line:1"


def test_the_same_snapshot_twice_stays_one_conversation_identity():
    payload = json.dumps(
        [
            {"role": "user", "content": "Keep the source text."},
            {"role": "assistant", "content": "Suggest item-by-item checks."},
        ]
    ).encode()
    first = parse_chat_export(payload, "chat.json").messages
    second = parse_chat_export(payload, "chat.json").messages
    assert [message.message_id for message in first] == ["1", "2"]
    assert [message.conversation_id for message in first] == [
        message.conversation_id for message in second
    ]
    assert [message.verbatim for message in first] == [message.verbatim for message in second]


def test_manual_json_rejects_unknown_fields_and_empty_messages():
    with pytest.raises(ManualTranscriptError) as caught:
        parse_manual_json([{"role": "user", "content": "x", "author": "person"}], "manual:1")
    assert caught.value.issue["code"] == "invalid_record"

    with pytest.raises(ManualTranscriptError) as caught:
        parse_manual_json([{"role": "user", "content": "   "}], "manual:1")
    assert caught.value.issue["code"] == "empty_message"
