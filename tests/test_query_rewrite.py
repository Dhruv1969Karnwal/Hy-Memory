# -*- coding: utf-8 -*-
"""Unit tests for analyze_queries (merged classification + rewrite + time span).

A fake LLM provider returns exact text so parsing, time conversion and fail-safe can be verified.
"""

import asyncio
from datetime import datetime

import pytest

from hy_memory.coding.judge import analyze_queries, _since_to_created_after
from hy_memory.agent.llm_provider import LLMResponse


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _FakeLLM:
    def __init__(self, content, raise_exc=False):
        self._content = content
        self._raise = raise_exc
        self.last_prompt = None

    async def complete(self, *, prompt, max_tokens, temperature=None, **kw):
        self.last_prompt = prompt
        if self._raise:
            raise RuntimeError("llm down")
        return LLMResponse(content=self._content, tokens_used=1,
                           prompt_tokens=1, completion_tokens=0)


CT = "2026-07-01T10:00:00"


class TestClassifyPlusRewritePlusSince:
    def test_full_output_parsed(self):
        llm = _FakeLLM("1\nhow did we configure the qdrant client\nSINCE: 2026-06-24")
        r = run(analyze_queries(
            ["the vector database config from last week", "right, that is the one I just mentioned"],
            llm, want_classify=True, want_rewrite=True, current_time=CT))
        assert r["ok"] is True
        assert r["is_coding"] is True
        assert r["rewrite_query"] == "how did we configure the qdrant client"
        assert r["created_after"] == datetime(2026, 6, 24).timestamp()

    def test_chat_class_since_none(self):
        llm = _FakeLLM("0\nwhat was the important thing the user mentioned yesterday\nSINCE: NONE")
        r = run(analyze_queries(
            ["you still remember the thing I told you yesterday, right"],
            llm, want_classify=True, want_rewrite=True, current_time=CT))
        assert r["ok"] is True
        assert r["is_coding"] is False
        assert r["created_after"] is None
        assert "important thing" in r["rewrite_query"]


class TestRewriteOnly:
    def test_no_classify_line(self):
        # want_classify=False -> no 0/1 header in the output, the first line already is the rewrite
        llm = _FakeLLM("what important thing did the user mention last week\nSINCE: 2026-06-24")
        r = run(analyze_queries(
            ["what important thing did I tell you last week"],
            llm, want_classify=False, want_rewrite=True, current_time=CT))
        assert r["ok"] is True
        assert r["is_coding"] is False  # always False when classification is off
        assert r["rewrite_query"].startswith("what important thing")
        assert r["created_after"] == datetime(2026, 6, 24).timestamp()

    def test_prompt_has_no_classify_task_when_off(self):
        llm = _FakeLLM("rewritten\nSINCE: NONE")
        run(analyze_queries(["q"], llm, want_classify=False,
                           want_rewrite=True, current_time=CT))
        # with classification disabled the prompt must not contain "Classify"
        assert "Classify the target" not in llm.last_prompt


class TestFailSafe:
    def test_llm_exception_returns_ok_false(self):
        llm = _FakeLLM("", raise_exc=True)
        r = run(analyze_queries(["a", "b"], llm, want_classify=True,
                               want_rewrite=True, current_time=CT))
        assert r["ok"] is False
        assert r["rewrite_query"] == "b"  # fallback = target
        assert r["created_after"] is None

    def test_empty_content_fallback(self):
        llm = _FakeLLM("")
        r = run(analyze_queries(["a"], llm, want_classify=True,
                               want_rewrite=True, current_time=CT))
        assert r["ok"] is False

    def test_unrecognized_class_head(self):
        llm = _FakeLLM("maybe\nrewritten\nSINCE: NONE")
        r = run(analyze_queries(["a"], llm, want_classify=True,
                               want_rewrite=True, current_time=CT))
        assert r["ok"] is False

    def test_want_rewrite_false_short_circuits(self):
        llm = _FakeLLM("should not be called")
        r = run(analyze_queries(["a"], llm, want_classify=True,
                               want_rewrite=False, current_time=CT))
        assert r["ok"] is False
        assert llm.last_prompt is None  # the LLM was never called


class TestSinceParse:
    def test_iso_date(self):
        assert _since_to_created_after("2026-06-24", CT) == datetime(2026, 6, 24).timestamp()

    def test_slash_date(self):
        assert _since_to_created_after("2026/06/24", CT) == datetime(2026, 6, 24).timestamp()

    def test_none_variants(self):
        for v in ["NONE", "none", "", "N/A", "-", "null"]:
            assert _since_to_created_after(v, CT) is None

    def test_quoted_and_trailing(self):
        assert _since_to_created_after('"2026-06-24".', CT) == datetime(2026, 6, 24).timestamp()

    def test_date_with_time_truncated(self):
        assert _since_to_created_after("2026-06-24T08:00:00", CT) == datetime(2026, 6, 24).timestamp()

    def test_garbage_returns_none(self):
        assert _since_to_created_after("last tuesday", CT) is None
