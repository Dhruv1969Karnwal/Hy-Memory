# -*- coding: utf-8 -*-
"""
Reasoning diagnostics fields for reasoning models (hy3 / kimi-k2.5 / deepseek-v4).

Background: a reasoning model shares max_tokens between its chain of thought (reasoning_content) and
its answer (content). Once the chain of thought exhausts the budget, content comes back as an empty
string, EXTRACT reports EMPTY_RESPONSE and silently drops the memory, and the logs give no hint why.
This covers reasoning field extraction on LLMResponse so triage can read straight from the logs that the chain of thought ate the whole output budget.
"""

from hy_memory.agent.llm_provider import LLMProvider, LLMResponse


class _Details:
    def __init__(self, reasoning_tokens):
        self.reasoning_tokens = reasoning_tokens


class _Usage:
    def __init__(self, reasoning_tokens=None, flat_reasoning_tokens=None):
        self.total_tokens = 2000
        self.prompt_tokens = 0
        self.completion_tokens = 2000
        if reasoning_tokens is not None:
            self.completion_tokens_details = _Details(reasoning_tokens)
        if flat_reasoning_tokens is not None:
            self.reasoning_tokens = flat_reasoning_tokens


class _Msg:
    def __init__(self, content="", reasoning_content=None, provider_specific_fields=None):
        self.content = content
        if reasoning_content is not None:
            self.reasoning_content = reasoning_content
        if provider_specific_fields is not None:
            self.provider_specific_fields = provider_specific_fields


def test_reasoning_from_message_field():
    """Standard position for hy3 / kimi: message.reasoning_content."""
    msg = _Msg(content="", reasoning_content="Let me think about this...")
    usage = _Usage(reasoning_tokens=2000)

    rc, rt = LLMProvider._extract_reasoning(msg, usage)

    assert rc == "Let me think about this..."
    assert rt == 2000


def test_reasoning_from_provider_specific_fields():
    """Some gateways only return the chain of thought inside provider_specific_fields."""
    msg = _Msg(content="", provider_specific_fields={"reasoning_content": "reasoning process"})
    usage = _Usage(reasoning_tokens=123)

    rc, rt = LLMProvider._extract_reasoning(msg, usage)

    assert rc == "reasoning process"
    assert rt == 123


def test_reasoning_tokens_flattened_on_usage():
    """LiteLLM gateways flatten reasoning_tokens onto usage instead of into details."""
    msg = _Msg(content="")
    usage = _Usage(flat_reasoning_tokens=888)

    _rc, rt = LLMProvider._extract_reasoning(msg, usage)

    assert rt == 888


def test_no_reasoning_is_zero():
    """Non-reasoning models / no_think mode: both fields are empty and nothing should raise."""
    msg = _Msg(content="ok")
    usage = _Usage()

    rc, rt = LLMProvider._extract_reasoning(msg, usage)

    assert rc == ""
    assert rt == 0


def test_missing_usage_is_tolerated():
    """Missing usage (some error responses) must not raise."""
    rc, rt = LLMProvider._extract_reasoning(_Msg(content="ok"), None)

    assert rc == ""
    assert rt == 0


def test_llm_response_defaults():
    """The new fields carry defaults, so existing LLMResponse call sites are unaffected."""
    resp = LLMResponse(content="ok")

    assert resp.reasoning_content == ""
    assert resp.reasoning_tokens == 0
