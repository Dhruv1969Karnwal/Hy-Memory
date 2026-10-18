# -*- coding: utf-8 -*-
"""Unit tests for custom extractor system prompt overrides (client-level zh/en).

Coverage:
  - Chinese input uses extract_system_prompt_zh; English input uses extract_system_prompt_en
  - None (the default) falls back to the built-in EXTRACT_SYSTEM_PROMPT[_ZH]
  - a custom prompt suppresses the built-in few-shot, even with few_shot_enabled=True
  - the user template (data injection section) is unaffected
"""

import asyncio

from hy_memory.config import LLMConfig
from hy_memory.agent.extractor import (
    Extractor,
    EXTRACT_SYSTEM_PROMPT,
    EXTRACT_FEW_SHOT_EN,
)
from hy_memory.agent.prompts_zh import EXTRACT_SYSTEM_PROMPT_ZH
from hy_memory.agent.llm_provider import LLMResponse


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _CapturingLLM:
    """Records the messages of the last complete_messages call and returns a valid empty extraction."""

    def __init__(self):
        self.last_messages = None

    async def complete_messages(self, *, messages, max_tokens, temperature=None,
                                tools=None, tool_choice=None):
        self.last_messages = messages
        return LLMResponse(
            content='{"memory": [], "intentions": [], "basic_info": {}}',
            tokens_used=1, prompt_tokens=1, completion_tokens=0, tool_calls=None,
        )

    def _system(self):
        return next(m["content"] for m in self.last_messages if m["role"] == "system")

    def _user(self):
        return next(m["content"] for m in self.last_messages if m["role"] == "user")


ZH_INPUT = "\u6211\u53eb\u5c0f\u660e\uff0c\u4f4f\u5728\u5317\u4eac\uff0c\u559c\u6b22\u559d\u7f8e\u5f0f\u5496\u5561\u3002"
EN_INPUT = "My name is Bob, I live in Seattle and I love hiking."


def _extract(llm, cfg, content):
    ex = Extractor(llm, cfg)
    run(ex.extract(content=content, current_time="2026-07-01"))


def test_default_uses_builtin_system_prompt():
    llm = _CapturingLLM()
    _extract(llm, LLMConfig(), EN_INPUT)
    assert llm._system() == EXTRACT_SYSTEM_PROMPT
    llm2 = _CapturingLLM()
    _extract(llm2, LLMConfig(), ZH_INPUT)
    assert llm2._system() == EXTRACT_SYSTEM_PROMPT_ZH


def test_code_heavy_chinese_instruction_uses_chinese_prompt():
    llm = _CapturingLLM()
    _extract(
        llm,
        LLMConfig(),
        "\u8bf7\u4fee\u590d search_client.py \u91cc\u7684 timeout bug: TypeError in parse_response(request_id)",
    )
    assert llm._system() == EXTRACT_SYSTEM_PROMPT_ZH


def test_english_instruction_with_later_chinese_term_uses_english_prompt():
    llm = _CapturingLLM()
    _extract(llm, LLMConfig(), "Please update the \u4e2d\u6587 README file.")
    assert llm._system() == EXTRACT_SYSTEM_PROMPT


def test_custom_en_prompt_used_for_english():
    custom = "You are a CUSTOM English extractor. Output JSON with {\"memory\": []}."
    llm = _CapturingLLM()
    _extract(llm, LLMConfig(extract_system_prompt_en=custom), EN_INPUT)
    assert llm._system() == custom
    # The user template still injects the original content
    assert EN_INPUT in llm._user()


def test_custom_zh_prompt_used_for_chinese():
    custom = "You are a custom Chinese extractor. Output JSON only."
    llm = _CapturingLLM()
    _extract(llm, LLMConfig(extract_system_prompt_zh=custom), ZH_INPUT)
    assert llm._system() == custom
    assert "\u5c0f\u660e" in llm._user()


def test_language_selects_correct_override():
    """Only a zh override: Chinese uses the custom prompt, English stays built-in."""
    cfg = LLMConfig(extract_system_prompt_zh="Chinese-only custom prompt")
    llm_zh = _CapturingLLM()
    _extract(llm_zh, cfg, ZH_INPUT)
    assert llm_zh._system() == "Chinese-only custom prompt"

    llm_en = _CapturingLLM()
    _extract(llm_en, cfg, EN_INPUT)
    assert llm_en._system() == EXTRACT_SYSTEM_PROMPT  # No English override -> built-in


def test_custom_prompt_suppresses_few_shot():
    custom = "CUSTOM en prompt"
    cfg = LLMConfig(extract_system_prompt_en=custom, few_shot_enabled=True)
    llm = _CapturingLLM()
    _extract(llm, cfg, EN_INPUT)
    # A custom prompt does not append the built-in few-shot
    assert llm._system() == custom
    assert EXTRACT_FEW_SHOT_EN not in llm._system()


def test_few_shot_still_appends_to_builtin_when_no_override():
    cfg = LLMConfig(few_shot_enabled=True)
    llm = _CapturingLLM()
    _extract(llm, cfg, EN_INPUT)
    assert llm._system() == EXTRACT_SYSTEM_PROMPT + EXTRACT_FEW_SHOT_EN
