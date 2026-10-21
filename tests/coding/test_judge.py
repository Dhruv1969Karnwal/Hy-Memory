# -*- coding: utf-8 -*-
"""Judge module tests"""

import pytest

from hy_memory.coding.judge import (
    classify_messages_scene,
    classify_queries_scene,
    classify_messages_is_coding,
    classify_queries_is_coding,
    _parse_is_coding_json,
)
from hy_memory.pipelines.base import ChatMessage, ToolCall

from tests.coding._helpers import MockLLMProvider, FailingLLMProvider


class TestParseIsCodingJSON:
    def test_pure_json(self):
        assert _parse_is_coding_json('{"is_coding": true}') == {"is_coding": True}

    def test_with_reason(self):
        out = _parse_is_coding_json('{"is_coding": false, "reason": "x"}')
        assert out["is_coding"] is False and out["reason"] == "x"

    def test_embedded_json(self):
        out = _parse_is_coding_json('foo {"is_coding": true} bar')
        assert out["is_coding"] is True

    def test_garbage(self):
        assert _parse_is_coding_json("garbage") is None
        assert _parse_is_coding_json("") is None
        assert _parse_is_coding_json("{}") is None  # missing the is_coding field

    def test_markdown_fence(self):
        out = _parse_is_coding_json('```json\n{"is_coding": true}\n```')
        assert out["is_coding"] is True


@pytest.mark.asyncio
class TestClassifyMessages:
    async def test_three_way_coding(self):
        msgs = [
            ChatMessage(role="user", content="Write a Python script to generate a quarterly financial report"),
            ChatMessage(role="assistant", content="", tool_calls=[
                ToolCall(id="1", name="Edit", arguments={"path": "report.py"}),
            ]),
        ]
        llm = MockLLMProvider('{"scene":"coding","reason":"writes code"}')
        assert await classify_messages_scene(msgs, llm) == "coding"

    async def test_three_way_working(self):
        msgs = [
            ChatMessage(role="user", content="Organize the quarterly financial data and make a PPT report"),
            ChatMessage(role="assistant", content="", tool_calls=[
                ToolCall(id="1", name="Spreadsheet", arguments={}),
                ToolCall(id="2", name="Slides", arguments={}),
            ]),
        ]
        llm = MockLLMProvider('{"scene":"working","reason":"office deliverable"}')
        assert await classify_messages_scene(msgs, llm) == "working"

    async def test_search_tool_smalltalk_stays_chat(self):
        msgs = [
            ChatMessage(role="user", content="Search Shanghai's weather today, and let's casually chat about what to wear"),
            ChatMessage(role="assistant", content="", tool_calls=[
                ToolCall(id="1", name="web_search", arguments={"query": "Shanghai weather"}),
            ]),
            ChatMessage(role="tool", content="Sunny, 25 degrees", tool_name="web_search"),
        ]
        llm = MockLLMProvider('{"scene":"chat","reason":"incidental lookup"}')
        assert await classify_messages_scene(msgs, llm) == "chat"
        prompt = llm.calls[-1]["prompt"]
        assert "Search/browse tools do NOT make a task working" in prompt

    async def test_coding_path(self):
        msgs = [
            ChatMessage(role="user", content="Refactor auth to use JWT"),
            ChatMessage(role="assistant", content="", tool_calls=[
                ToolCall(id="1", name="Read", arguments={"path": "a"}),
            ]),
            ChatMessage(role="tool", content="ok", tool_call_id="1", tool_name="Read"),
        ]
        llm = MockLLMProvider('{"is_coding": true, "reason": "engineering"}')
        result = await classify_messages_is_coding(msgs, llm)
        assert result is True

    async def test_chat_path(self):
        msgs = [ChatMessage(role="user", content="Hello")]
        llm = MockLLMProvider('{"is_coding": false}')
        result = await classify_messages_is_coding(msgs, llm)
        assert result is False

    async def test_failsafe_on_garbage(self):
        msgs = [ChatMessage(role="user", content="x")]
        llm = MockLLMProvider("not json")
        result = await classify_messages_is_coding(msgs, llm)
        assert result is False

    async def test_failsafe_on_llm_error(self):
        msgs = [ChatMessage(role="user", content="x")]
        result = await classify_messages_is_coding(msgs, FailingLLMProvider())
        assert result is False

    async def test_empty_messages(self):
        result = await classify_messages_is_coding([], MockLLMProvider("{}"))
        assert result is False

    async def test_prompt_contains_turn_summary(self):
        msgs = [
            ChatMessage(role="user", content="Refactor auth"),
            ChatMessage(role="assistant", content="", tool_calls=[
                ToolCall(id="1", name="Read", arguments={"path": "a"}),
            ]),
        ]
        llm = MockLLMProvider('{"is_coding": true}')
        await classify_messages_is_coding(msgs, llm)
        last_prompt = llm.calls[-1]["prompt"]
        assert "Read" in last_prompt
        assert "Refactor auth" in last_prompt or "user=" in last_prompt


@pytest.mark.asyncio
class TestClassifyQueries:
    async def test_three_way_query_scene(self):
        llm = MockLLMProvider('{"scene":"coding"}')
        assert await classify_queries_scene(["Help me fix this Python traceback"], llm) == "coding"

    async def test_target_is_last(self):
        llm = MockLLMProvider('{"is_coding": true}')
        result = await classify_queries_is_coding(
            ["Where are the credentials", "last release command", "why is it slow"], llm
        )
        assert result is True
        prompt = llm.calls[-1]["prompt"]
        assert "why is it slow" in prompt
        assert "target" in prompt

    async def test_single_query(self):
        llm = MockLLMProvider('{"is_coding": false}')
        result = await classify_queries_is_coding(["What does the user like"], llm)
        assert result is False

    async def test_empty(self):
        result = await classify_queries_is_coding([], MockLLMProvider("{}"))
        assert result is False

    async def test_failsafe(self):
        result = await classify_queries_is_coding(["x"], FailingLLMProvider())
        assert result is False
