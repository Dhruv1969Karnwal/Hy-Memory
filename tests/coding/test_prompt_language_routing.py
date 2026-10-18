"""Working/coding prompt language routing and format-string contracts."""

import json
from string import Formatter

import pytest

from hy_memory.coding.extractor import (
    EXTRACT_PROMPT,
    EXTRACT_PROMPT_CODING,
    CodingMemoryExtractor,
)
from hy_memory.coding.fact_reconciler import (
    CHAIN_SUMMARY_PROMPT,
    FACT_RECONCILE_PROMPT,
    build_chain_summary_prompt,
    build_fact_reconcile_prompt,
    WorkingFactReconciler,
)
from hy_memory.coding.reconciler import (
    RECONCILE_PROMPT,
    CodingMemoryReconciler,
    build_experience_reconcile_prompt,
)
from hy_memory.coding.types import CodingMemory, CodingMemoryDraft, WorkingFact, WorkingFactDraft
from hy_memory.pipelines.base import ChatMessage

from tests.coding._helpers import MockLLMProvider


def _fields(template: str) -> set[str]:
    return {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name
    }


@pytest.fixture
def zh_prompts():
    from hy_memory.coding.prompts_zh import (
        CHAIN_SUMMARY_PROMPT_ZH,
        EXTRACT_PROMPT_CODING_ZH,
        EXTRACT_PROMPT_ZH,
        FACT_RECONCILE_PROMPT_ZH,
        RECONCILE_PROMPT_ZH,
    )

    return {
        "extract": EXTRACT_PROMPT_ZH,
        "extract_coding": EXTRACT_PROMPT_CODING_ZH,
        "reconcile": RECONCILE_PROMPT_ZH,
        "fact_reconcile": FACT_RECONCILE_PROMPT_ZH,
        "chain_summary": CHAIN_SUMMARY_PROMPT_ZH,
    }


def test_chinese_templates_match_english_placeholder_sets(zh_prompts):
    pairs = {
        "extract": EXTRACT_PROMPT,
        "extract_coding": EXTRACT_PROMPT_CODING,
        "reconcile": RECONCILE_PROMPT,
        "fact_reconcile": FACT_RECONCILE_PROMPT,
        "chain_summary": CHAIN_SUMMARY_PROMPT,
    }
    for name, english in pairs.items():
        assert _fields(zh_prompts[name]) == _fields(english), name


def test_all_chinese_templates_format_without_errors(zh_prompts):
    rendered = {
        "extract": zh_prompts["extract"].format(
            workspace_present="true",
            branch_present="false",
            conversation_block="User: please tidy up the report",
        ),
        "extract_coding": zh_prompts["extract_coding"].format(
            workspace_present="true",
            branch_present="true",
            conversation_block="User: please fix api.py",
        ),
        "reconcile": zh_prompts["reconcile"].format(
            drafts_block="draft[0]",
            candidates_block="(no candidates)",
        ),
        "fact_reconcile": zh_prompts["fact_reconcile"].format(
            drafts_block="fact[0]",
            candidates_block="(no candidates)",
        ),
        "chain_summary": zh_prompts["chain_summary"].format(
            chain_block="Phase 1\nPhase 2",
        ),
    }
    assert all(rendered.values())


def test_chinese_templates_preserve_core_behavior_contracts(zh_prompts):
    working = zh_prompts["extract"]
    coding = zh_prompts["extract_coding"]
    experience_reconcile = zh_prompts["reconcile"]
    fact_reconcile = zh_prompts["fact_reconcile"]
    chain_summary = zh_prompts["chain_summary"]

    assert "waiting on the user" in working
    assert '"experiences": []' in working
    assert "Do not manufacture an experience to satisfy this instruction" in working
    assert "evidence" in working and "never" in working
    assert "subject" in working and "different" in working

    assert "HARD CONSTRAINT" in coding
    assert "BRANCH / WORKING LOCATION" in coding
    assert "WHICH VERSION WON" in coding
    assert "EXACT COORDINATES, NOT PASTED CODE" in coding
    assert "VERIFIED vs MERELY WRITTEN" in coding

    for action in ("ADD", "UPDATE", "DELETE", "SKIP"):
        assert action in experience_reconcile
    assert "clear evidence" in experience_reconcile
    assert "Never" in experience_reconcile

    assert "SUPERSEDE" in fact_reconcile
    assert "subject" in fact_reconcile
    assert "the same entity" in fact_reconcile
    assert "conservative" in fact_reconcile

    assert "Report only" in chain_summary
    assert "evidence states" in chain_summary
    assert "Never" in chain_summary
    assert "current status" in chain_summary


@pytest.mark.parametrize("scene", ["working", "coding"])
def test_extractor_routes_chinese_new_user_message_to_chinese_prompt(scene, zh_prompts):
    extractor = CodingMemoryExtractor(MockLLMProvider("{}"), scene=scene)
    template, language = extractor._select_prompt_template([
        ChatMessage(role="user", content="\u8bf7\u4fee\u590d api_client.py \u7684 timeout \u95ee\u9898"),  # "fix the timeout issue in api_client.py", zh input
        ChatMessage(role="tool", content="Traceback: TypeError at line 100"),
    ])
    expected = zh_prompts["extract_coding" if scene == "coding" else "extract"]
    assert language == "zh"
    assert template is expected


@pytest.mark.parametrize("scene", ["working", "coding"])
def test_extractor_keeps_english_prompt_when_chinese_term_appears_later(scene):
    extractor = CodingMemoryExtractor(MockLLMProvider("{}"), scene=scene)
    template, language = extractor._select_prompt_template([
        ChatMessage(role="user", content="Please update the \u4e2d\u6587 README file."),  # "Chinese", a zh term inside an otherwise English sentence
    ])
    expected = EXTRACT_PROMPT_CODING if scene == "coding" else EXTRACT_PROMPT
    assert language == "en"
    assert template is expected


def test_experience_reconcile_routes_from_new_drafts_not_candidates():
    drafts = [CodingMemoryDraft(
        task="\u4fee\u590d\u641c\u7d22\u8d85\u65f6",  # "fix the search timeout", zh input
        solution="\u4fee\u6539 search_client.py \u7684 timeout \u5904\u7406",  # "change the timeout handling in search_client.py", zh input
        user_id="u1",
    )]
    candidates = [[CodingMemory(
        experience_id="old",
        user_id="u1",
        task="Maintain the search client",
        solution="Use the legacy retry helper",
    )]]
    prompt, language = build_experience_reconcile_prompt(drafts, candidates)
    assert language == "zh"
    assert prompt.startswith("You decide how to merge the new")


def test_experience_reconcile_keeps_english_for_english_leading_draft():
    drafts = [CodingMemoryDraft(
        task="Please update the \u4e2d\u6587 README file",  # "Chinese", a zh term inside an otherwise English sentence
        solution="Keep the existing structure",
        user_id="u1",
    )]
    prompt, language = build_experience_reconcile_prompt(drafts, [[]])
    assert language == "en"
    assert prompt.startswith("You")


def test_fact_reconcile_routes_from_new_fact_statement():
    drafts = [WorkingFactDraft(
        subject="api_client.py",
        statement="\u8bf7\u4fee\u590d timeout \u95ee\u9898\uff1b\u5df2\u5b9a\u4f4d\u5230 retry \u903b\u8f91\u3002",  # "please fix the timeout issue; the retry logic has been located", zh input
    )]
    candidates = [[WorkingFact(
        fact_id="old",
        user_id="u1",
        statement="The previous session inspected retry behavior.",
    )]]
    prompt, language = build_fact_reconcile_prompt(drafts, candidates)
    assert language == "zh"
    assert prompt.startswith("You maintain session-level WORKING FACTS. Each statement summarizes an entire")


def test_chain_summary_routes_from_current_statement():
    from hy_memory.coding.prompts_zh import CHAIN_SUMMARY_PROMPT_ZH

    prompt, language = build_chain_summary_prompt(
        "(2026-10-11) The previous session inspected timeout behavior.\n"
        "(now) \u7528\u6237\u7ee7\u7eed\u4fee\u590d timeout\uff0c\u5e76\u5b8c\u6210\u9a8c\u8bc1\u3002",  # "the user keeps fixing the timeout and has finished verifying", zh input
        current_statement="\u7528\u6237\u7ee7\u7eed\u4fee\u590d timeout\uff0c\u5e76\u5b8c\u6210\u9a8c\u8bc1\u3002",  # "the user keeps fixing the timeout and has finished verifying", zh input
    )
    assert language == "zh"
    assert prompt == CHAIN_SUMMARY_PROMPT_ZH.format(
        chain_block=(
            "(2026-10-11) The previous session inspected timeout behavior.\n"
            "(now) \u7528\u6237\u7ee7\u7eed\u4fee\u590d timeout\uff0c\u5e76\u5b8c\u6210\u9a8c\u8bc1\u3002"
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("scene", ["working", "coding"])
async def test_chinese_extractor_prompt_runs_and_parses_json(scene):
    response = json.dumps({
        "experiences": [{
            "task": "Fix the search timeout",
            "search_keys": ["search timeout"],
            "solution": "Modify api_client.py and run the tests",
            "boundary_envs": "",
            "confidence": 0.9,
        }],
        "fact": {
            "subject": "api_client.py",
            "statement": "Fixed the search timeout and verified with tests.",
            "files": ["api_client.py"],
            "details": "pytest passed",
        },
    }, ensure_ascii=False)
    llm = MockLLMProvider(response)
    extractor = CodingMemoryExtractor(llm, scene=scene)

    drafts, facts = await extractor.extract(
        [ChatMessage(role="user", content="\u8bf7\u4fee\u590d api_client.py \u7684 timeout \u95ee\u9898")],  # "fix the timeout issue in api_client.py", zh input
        user_id="u1",
    )

    assert len(drafts) == 1
    assert drafts[0].task == "Fix the search timeout"
    assert len(facts) == 1
    assert facts[0].statement == "Fixed the search timeout and verified with tests."
    assert "A memory must satisfy ALL of the following at once" in llm.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_experience_reconciler_sends_chinese_prompt_to_llm():
    llm = MockLLMProvider('{"ops": []}')
    reconciler = CodingMemoryReconciler(None, None, llm)
    drafts = [CodingMemoryDraft(
        task="\u4fee\u590d\u641c\u7d22\u8d85\u65f6",  # "fix the search timeout", zh input
        solution="\u4fee\u6539 api_client.py",  # "modify api_client.py", zh input
        user_id="u1",
    )]

    await reconciler._llm_decide(drafts, [[]])

    assert llm.calls[0]["prompt"].startswith("You decide how to merge the new")


@pytest.mark.asyncio
async def test_fact_reconciler_sends_chinese_prompt_to_llm():
    llm = MockLLMProvider('{"ops": []}')
    reconciler = WorkingFactReconciler(None, None, llm)
    drafts = [WorkingFactDraft(
        subject="api_client.py",
        statement="\u8bf7\u4fee\u590d\u641c\u7d22\u8d85\u65f6\u5e76\u5b8c\u6210\u9a8c\u8bc1\u3002",  # "please fix the search timeout and complete the verification", zh input
    )]

    await reconciler._llm_decide(drafts, [[WorkingFact(
        fact_id="old",
        user_id="u1",
        statement="Previously inspected timeout behavior.",
    )]])

    assert llm.calls[0]["prompt"].startswith("You maintain session-level WORKING FACTS. Each statement summarizes an entire")
