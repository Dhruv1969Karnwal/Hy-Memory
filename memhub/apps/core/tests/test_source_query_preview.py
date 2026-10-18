import json

import pytest

from one_memory.source_query_preview import display_query_projection


def test_preserves_exact_user_text_and_span():
    text = "  Please keep this exact quote.  "
    result = json.loads(display_query_projection(text, "user", "person"))
    assert result["excerpt"] == text[result["query_start"] : result["query_end"]]
    assert result["projection_version"] == 1 and not result["truncated"]


def test_supported_outer_wrapper_is_stripped_without_paraphrasing():
    text = "<user_info>system background</user_info><user_query>my question this time</user_query>"
    result = json.loads(display_query_projection(text, "user", "person"))
    assert result["excerpt"] == "my question this time"
    assert text[result["query_start"] : result["query_end"]] == result["excerpt"]


@pytest.mark.parametrize(
    "prefix",
    [
        "<timestamp>Wednesday, Sep 23, 2026, 7:34 AM (UTC+8)</timestamp>\n",
        (
            "[Image]\n"
            "<image_files>\n"
            "The following images were provided by the user:\n"
            "1. /private/project/assets/image.png\n"
            "</image_files>\n"
            "<timestamp>Wednesday, Sep 23, 2026, 7:34 AM (UTC+8)</timestamp>\n"
        ),
    ],
)
def test_cursor_host_prefixes_are_not_shown_as_the_user_query(prefix):
    text = f"{prefix}<user_query>keep only this question</user_query>"
    result = json.loads(display_query_projection(text, "user", "person"))
    assert result["excerpt"] == "keep only this question"
    assert text[result["query_start"] : result["query_end"]] == result["excerpt"]


@pytest.mark.parametrize(
    "text",
    [
        "<user_query>incomplete",
        "<user_query>one</user_query><user_query>two</user_query>",
        "<user_info>only the system</user_info>",
        "<subagent_notification>task finished</subagent_notification>",
        "<codex_delegation>execute instructions</codex_delegation>",
        "<dynamic_tool_catalog>tool list",
        "<user_info>legacy host context wrongly persisted</user_info>",
        "<dynamic_tools>legacy tool catalog wrongly persisted</dynamic_tools>",
        "[Image] legacy image message lacks an explicit query wrapper",
        "# AGENTS.md instructions for /private/workspace\nproject context",
        "# Files mentioned by the user:\nattachment context",
        "Briefly inform the user about the task result and perform any follow-up actions",
        "<rules>duplicate</rules><user_query>duplicate</user_query>",
        "x" * 65537,
    ],
)
def test_ambiguous_or_generated_content_is_withheld(text):
    assert display_query_projection(text, "user", "person") is None


@pytest.mark.parametrize(
    "role,author",
    [("assistant", "agent"), ("user", "agent"), ("user", "unknown"), ("tool", "tool")],
)
def test_role_user_alone_is_not_human_authorship(role, author):
    assert display_query_projection("the task is done", role, author) is None


def test_long_display_is_truncated_without_changing_full_span():
    text = "the real question of a real user" * 120
    result = json.loads(display_query_projection(text, "user", "person"))
    assert len(result["excerpt"]) == 600 and result["truncated"]
    assert result["query_end"] == len(text)


def test_receipt_qualifies_user_queries_before_sampling_and_preserves_originals(tmp_path):
    import hashlib

    from one_memory.imports.models import WrittenBy
    from one_memory.imports.storage import EvidenceStore
    from one_memory.source_import_receipts import build_source_import_receipt
    from tests.test_source_import_receipts import record

    path = tmp_path / "receipt.sqlite3"
    store = EvidenceStore(path)
    valid = record(1, text="<user_query>the real question</user_query>")
    store.put(valid)
    for index in range(2, 102):
        store.put(record(index, author=WrittenBy.AGENT, text="assistant message"))
    store.put(record(200, text="<user_query>truncated"))
    store.put(record(201, user="other", text="a different user"))
    store.put(record(202, source="other", text="a different source"))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = build_source_import_receipt(path, user_id="local", source_id="claude-code")
    assert len(receipt.query_examples) == 1
    item = receipt.query_examples[0]
    assert item.evidence_id == valid.evidence_id and item.evidence_revision == 1
    assert item.excerpt == "the real question" and item.written_by == "person"
    assert store.get(valid.evidence_id).verbatim == valid.verbatim
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_receipt_examples_strip_cursor_host_prefixes_before_sampling(tmp_path):
    from one_memory.imports.models import WrittenBy
    from one_memory.imports.storage import EvidenceStore
    from one_memory.source_import_receipts import build_source_import_receipt
    from tests.test_source_import_receipts import record

    path = tmp_path / "receipt-cursor.sqlite3"
    store = EvidenceStore(path)
    valid = record(
        1,
        text=(
            "[Image]\n<image_files>/private/image.png</image_files>\n"
            "<timestamp>Wednesday, Sep 23, 2026, 7:34 AM (UTC+8)</timestamp>\n"
            "<user_query>the real question in history</user_query>"
        ),
    )
    store.put(valid)
    for index in range(2, 102):
        store.put(record(index, author=WrittenBy.AGENT, text="assistant message"))

    receipt = build_source_import_receipt(path, user_id="local", source_id="claude-code")

    assert [item.excerpt for item in receipt.query_examples] == ["the real question in history"]
