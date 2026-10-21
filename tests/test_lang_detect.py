"""Language routing for prompts, including code-heavy agent inputs."""

import pytest

from hy_memory.utils.lang_detect import (
    _leading_language_window,
    detect_language,
    is_chinese,
)
from hy_memory.agent.reconciler import build_reconcile_prompt


@pytest.mark.parametrize(
    "text,expected",
    [
        ("\u4f60\u597d", "zh"),  # "hello"
        ("\u7528\u6237\u559c\u6b22\u79d1\u5e7b\u7535\u5f71\uff0c\u5c24\u5176\u662f\u661f\u9645\u7a7f\u8d8a\u3002", "zh"),  # "the user likes sci-fi movies, especially Interstellar."
        ("Hello", "en"),
        ("The user likes science fiction movies.", "en"),
        ("\u8bf7\u4fee\u590d search_client.py \u91cc\u7684 timeout bug", "zh"),  # "please fix the timeout bug in search_client.py"
        ("\u4fee\u590d bug: TypeError in function parse_response(request_id)", "zh"),  # "fix bug: ..."
        ("\u6267\u884c pytest tests/test_search.py\uff0c\u7136\u540e\u4fee\u590d\u5931\u8d25\u7528\u4f8b", "zh"),  # "run pytest tests/test_search.py, then fix the failing cases"
        ("[user]: \u8bf7\u68c0\u67e5 api_client.py \u7684 timeout", "zh"),  # "please check the timeout in api_client.py"
        ("user: \u8bf7\u68c0\u67e5 api_client.py \u7684 timeout", "zh"),
        ("Please update the \u4e2d\u6587 README file.", "en"),  # "Chinese"
        ("def search(query): return client.search(query)", "en"),
        ("Check https://example.com/docs and update config", "en"),
    ],
)
def test_detect_language_uses_leading_instruction(text, expected):
    detected = detect_language(text)
    routed = "zh" if detected == "zh" else "en"
    assert routed == expected


def test_leading_window_ignores_role_prefix_and_punctuation():
    assert _leading_language_window("  [user]: \u8bf7\u4fee\u590d api_client.py") == "\u8bf7\u4fee\u590dapiclie"


def test_messages_route_from_message_content_not_role_names():
    messages = [
        {"role": "user", "content": "\u8bf7\u4fee\u590d search_client.py \u7684\u8d85\u65f6\u95ee\u9898"},  # "please fix the timeout problem in search_client.py"
        {"role": "assistant", "content": "I will inspect the implementation."},
    ]
    assert is_chinese(messages=messages) is True


def test_reconcile_uses_chinese_prompt_for_code_heavy_chinese_instruction():
    memory = (
        "\u8bf7\u4fee\u590d search_client.py \u91cc\u7684 timeout bug: "
        "TypeError in parse_response(request_id)"
    )
    prompt = build_reconcile_prompt(
        new_memories=[memory],
        new_memories_block=memory,
        existing_memories_block="[]",
        current_date="2026-10-12 16:00",
    )
    assert prompt.startswith("You are a memory reconciliation system")


def test_reconcile_keeps_english_prompt_when_chinese_term_appears_later():
    memory = "Please update the \u4e2d\u6587 README file."
    prompt = build_reconcile_prompt(
        new_memories=[memory],
        new_memories_block=memory,
        existing_memories_block="[]",
        current_date="2026-10-12 16:00",
    )
    assert prompt.startswith("You are a memory management system.")
