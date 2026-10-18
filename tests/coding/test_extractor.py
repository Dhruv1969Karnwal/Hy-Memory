# -*- coding: utf-8 -*-
"""Extractor tests (two-layer extraction)."""

import json

import pytest

from hy_memory.coding.extractor import CodingMemoryExtractor, EXTRACT_PROMPT
from hy_memory.pipelines.base import ChatMessage, ToolCall

from tests.coding._helpers import MockLLMProvider, FailingLLMProvider


def _llm_resp(experiences, fact=None):
    return json.dumps({"experiences": experiences, "fact": fact})


def _legacy_llm_resp(experiences, facts=None):
    return json.dumps({"experiences": experiences, "facts": facts or []})


def _msgs():
    return [
        ChatMessage(role="user", content="Refactor auth to JWT"),
        ChatMessage(role="assistant", content="", tool_calls=[
            ToolCall(id="1", name="Read", arguments={"path": "auth/middleware.ts"}),
        ]),
        ChatMessage(role="tool", content="42 lines", tool_call_id="1", tool_name="Read"),
    ]


def _assert_verbatim_fallback(facts):
    """Assert that, once explicitly enabled, the fact layer used the source-text fallback.

    The criteria are the statement landing on the user's own words and details keeping the full trace.
    """
    assert len(facts) == 1, f"expected exactly one verbatim fact, got {facts!r}"
    f = facts[0]
    assert f.statement == "Refactor auth to JWT"
    assert f.subject == ""          # the fallback has no subject to judge, so it stays empty and coverage falls back to pure semantics
    assert "[0] user: Refactor auth to JWT" in f.details
    assert "[2] tool(Read): 42 lines" in f.details


class TestFactPromptContract:
    """Static protection for the evidence-fidelity contract; it does not claim any LLM always complies."""

    def test_only_executed_commands_without_invented_flags(self):
        assert "only actions actually executed" in EXTRACT_PROMPT
        assert "NEVER complete a" in EXTRACT_PROMPT
        assert "command with customary flags" in EXTRACT_PROMPT
        assert '"ran Get-ChildItem -Force" → "ran Get-ChildItem"' in EXTRACT_PROMPT

    def test_partial_negative_findings_keep_scope_and_limits(self):
        assert "Scope every negative finding" in EXTRACT_PROMPT
        assert "enumerate EVERY explicit coverage limit" in EXTRACT_PROMPT
        assert "Never promote a partial search to" in EXTRACT_PROMPT
        assert "Access denied" in EXTRACT_PROMPT
        assert "depth limits" in EXTRACT_PROMPT

    def test_inference_stays_uncertain_and_success_requires_evidence(self):
        assert "inferred, and unresolved" in EXTRACT_PROMPT

        assert "a cause was established" in EXTRACT_PROMPT
        assert "not yet verified" in EXTRACT_PROMPT

    def test_files_contract_includes_investigated_directories(self):
        assert "FILE OR DIRECTORY paths" in EXTRACT_PROMPT
        assert "roots such as `C:\\` and `D:\\`" in EXTRACT_PROMPT
        assert "not paths merely suggested or planned" in EXTRACT_PROMPT

    def test_statement_requires_specific_goal_and_ending_state(self):
        assert "Start with the" in EXTRACT_PROMPT
        assert "specific goal and object" in EXTRACT_PROMPT
        assert "partially completed, failed, blocked, or" in EXTRACT_PROMPT
        assert "still awaiting verification" in EXTRACT_PROMPT

    def test_command_evidence_window_keeps_material_late_flags(self):
        """Protects the command evidence window in the prompt input; it does not claim the model always writes flags into the fact."""
        command = (
            'Get-ChildItem -Path "C:\\\\" -Directory -Filter "new_file_search" '
            "-Recurse -Depth 3 -ErrorAction SilentlyContinue "
            "| Select-Object -First 5 FullName"
        )
        rendered = CodingMemoryExtractor._format_conversation([
            ChatMessage(role="assistant", tool_calls=[
                ToolCall(id="1", name="shell", arguments={"command": command}),
            ]),
        ])
        assert "-Depth 3" in rendered
        assert "-ErrorAction SilentlyContinue" in rendered

    def test_asking_the_user_is_not_a_reusable_experience(self):
        """Regression protection for the clarification stall scenario.

        If the Agent produces no plan and only asks back for the missing fields, and the extractor
        distills that into experiences such as "how to ask for departure city and travel dates"
        / "eliciting required fields for travel planning", later retrieval keeps amplifying the
        counter-question instead of producing a parseable plan. A stall written down as a skill
        replicates itself, so clarification behavior must stay in the fact and never enter experiences.
        """
        assert "The act of ASKING THE USER for missing information" in EXTRACT_PROMPT
        assert "conversational" in EXTRACT_PROMPT
        assert '"how to ask for X"' in EXTRACT_PROMPT
        assert "eliciting required fields" in EXTRACT_PROMPT
        # A stalled turn writes only the fact, never an experience
        assert 'emit "experiences": [] and write the FACT alone' in EXTRACT_PROMPT
        # The criteria come from the extractor's own read of the conversation state, not from any eval dataset field
        assert "stalled waiting on the" in EXTRACT_PROMPT
        assert "Judge substance by what the tools actually did" in EXTRACT_PROMPT

    def test_mandatory_extraction_exempts_the_stall_case(self):
        """The trailing MANDATORY line used to be the direct cause of fabricated experiences.

        The original read "You MUST output at least ONE experience. Do NOT return an empty
        list." -- last position, no exceptions. A counter-question turn has no method to distill yet
        was barred from returning an empty list, so the model had to invent one. The prohibition above
        sits in the middle of the prompt and gets overridden by that trailing line, so it must explicitly yield for the stall case; extraction stays the default, and the exemption only covers stalls so it cannot become an excuse to slack off.
        """
        assert "DEFAULT TO EXTRACTING" in EXTRACT_PROMPT
        assert "The ONE exception is the stall case above" in EXTRACT_PROMPT
        assert "Do not manufacture an experience" in EXTRACT_PROMPT
        assert "never as a shortcut when work was actually done" in EXTRACT_PROMPT
        # The old unconditional prohibition must not come back
        assert "Do NOT return an empty list." not in EXTRACT_PROMPT


@pytest.mark.asyncio
class TestExtractor:
    async def test_new_messages_all_kept_history_only_last_five_user_turns(self):
        llm = MockLLMProvider(_llm_resp([], {"statement": "done"}))
        ext = CodingMemoryExtractor(llm)

        current = []
        for i in range(10):
            current.extend([
                ChatMessage(role="user", content=f"new-user-{i}"),
                ChatMessage(role="assistant", content="A" * 8000),
                ChatMessage(role="tool", content="T" * 5000, tool_name="Read"),
            ])
        history = []
        for i in range(7):
            history.extend([
                ChatMessage(role="user", content=f"history-user-{i}"),
                ChatMessage(role="assistant", content="history answer"),
            ])

        await ext.extract(current, history_messages=history, user_id="u1")
        prompt = llm.calls[0]["prompt"]
        history_block, new_block = prompt.split(
            "NEW MESSAGES (extract only from this complete write):", 1
        )

        assert "history-user-0" not in history_block
        assert "history-user-1" not in history_block
        for i in range(2, 7):
            assert f"history-user-{i}" in history_block

        for i in range(10):
            assert f"new-user-{i}" in new_block
        assert "chars omitted" in new_block
        assert "bytes omitted" in new_block

    async def test_normal_extract(self):
        llm = MockLLMProvider(_llm_resp([
            {
                "task": "Auth refactor",
                "search_keys": ["JWT implementation", "why not use sessions"],
                "solution": "Switched to JWT, file auth/middleware.ts",
                "boundary_envs": "node>=18",
                "confidence": 0.9,
            },
        ]))
        ext = CodingMemoryExtractor(llm, verbatim_fallback_enabled=True)
        drafts, facts = await ext.extract(
            _msgs(), user_id="u1", agent_id="a1",
            workspace_id="github.com/x/y", branch="main",
        )
        assert len(drafts) == 1
        d = drafts[0]
        assert d.task == "Auth refactor"
        assert d.search_keys == ["JWT implementation", "why not use sessions"]
        assert d.solution.startswith("Switched to JWT")
        assert d.boundary_envs == "node>=18"
        assert d.confidence == 0.9
        assert d.workspace_id == "github.com/x/y"
        assert d.files == ["auth/middleware.ts"]
        # The LLM returned experiences but no fact, so the fact layer uses the source-text fallback
        _assert_verbatim_fallback(facts)

    async def test_kept_without_workspace_or_branch(self):
        """A missing workspace_id / branch no longer rejects the draft.

        Boundaries no longer participate in recall filtering, so rejecting on them only loses experiences for nothing.
        """
        llm = MockLLMProvider(_llm_resp([
            {"task": "T1", "solution": "S", "confidence": 0.9},
            {"task": "T2", "solution": "S", "confidence": 0.9},
        ]))
        ext = CodingMemoryExtractor(llm)
        drafts, _ = await ext.extract(_msgs(), user_id="u1", workspace_id=None, branch=None)
        assert [d.task for d in drafts] == ["T1", "T2"]

    async def test_boundary_envs_defaults_empty(self):
        llm = MockLLMProvider(_llm_resp([{"task": "T", "solution": "S"}]))
        ext = CodingMemoryExtractor(llm)
        drafts, _ = await ext.extract(_msgs(), user_id="u1")
        assert drafts[0].boundary_envs == ""

    async def test_search_keys_dedup_and_truncate(self):
        llm = MockLLMProvider(_llm_resp([{
            "task": "T", "solution": "S",
            "search_keys": ["T", "A", "A", "B", "C", "D", "E", "F"],
            "confidence": 0.8,
        }]))
        ext = CodingMemoryExtractor(llm)
        drafts, _ = await ext.extract(_msgs(), user_id="u1")
        # T matches the task name and is dropped; the duplicate A is dropped; the first 5 are kept
        assert drafts[0].search_keys == ["A", "B", "C", "D", "E"]

    async def test_confidence_clamp(self):
        llm = MockLLMProvider(_llm_resp([
            {"task": "T1", "solution": "S", "confidence": 1.5},
            {"task": "T2", "solution": "S", "confidence": -0.5},
            {"task": "T3", "solution": "S", "confidence": "bad"},
        ]))
        ext = CodingMemoryExtractor(llm)
        drafts, _ = await ext.extract(_msgs(), user_id="u1")
        confs = [d.confidence for d in drafts]
        assert confs == [1.0, 0.0, 0.7]

    async def test_drop_missing_task_or_solution(self):
        llm = MockLLMProvider(_llm_resp([
            {"task": "", "solution": "S"},
            {"task": "T", "solution": ""},
            {"task": "T2", "solution": "S"},
        ]))
        ext = CodingMemoryExtractor(llm)
        drafts, _ = await ext.extract(_msgs(), user_id="u1")
        assert [d.task for d in drafts] == ["T2"]

    async def test_files_auto_extracted(self):
        llm = MockLLMProvider(_llm_resp([
            {"task": "T", "solution": "S", "confidence": 0.8},
        ]))
        msgs = [
            ChatMessage(role="assistant", content="", tool_calls=[
                ToolCall(id="1", name="Read", arguments={"path": "src/auth.py"}),
                ToolCall(id="2", name="Edit", arguments={"file_path": "src/jwt.py"}),
            ]),
        ]
        ext = CodingMemoryExtractor(llm)
        drafts, _ = await ext.extract(msgs, user_id="u1")
        assert drafts[0].files == ["src/auth.py", "src/jwt.py"]

    async def test_empty_memories(self):
        llm = MockLLMProvider(_llm_resp([]))
        ext = CodingMemoryExtractor(llm)
        drafts, facts = await ext.extract(_msgs(), user_id="u1", workspace_id="ws")
        assert drafts == []
        assert facts == []

    async def test_empty_memories_fallback_can_be_enabled(self):
        llm = MockLLMProvider(_llm_resp([]))
        ext = CodingMemoryExtractor(llm, verbatim_fallback_enabled=True)
        drafts, facts = await ext.extract(_msgs(), user_id="u1", workspace_id="ws")
        assert drafts == []
        _assert_verbatim_fallback(facts)

    async def test_failsafe_on_llm_error(self):
        ext = CodingMemoryExtractor(FailingLLMProvider())
        drafts, facts = await ext.extract(_msgs(), user_id="u1")
        assert drafts == []
        assert facts == []

    async def test_failsafe_on_llm_error_fallback_can_be_enabled(self):
        ext = CodingMemoryExtractor(
            FailingLLMProvider(), verbatim_fallback_enabled=True,
        )
        drafts, facts = await ext.extract(_msgs(), user_id="u1")
        assert drafts == []
        _assert_verbatim_fallback(facts)

    async def test_markdown_fence_parse(self):
        llm = MockLLMProvider(
            "```json\n" + _llm_resp([
                {"task": "T", "solution": "S"},
            ]) + "\n```"
        )
        ext = CodingMemoryExtractor(llm, verbatim_fallback_enabled=True)
        drafts, _ = await ext.extract(_msgs(), user_id="u1")
        assert len(drafts) == 1

    async def test_empty_messages(self):
        llm = MockLLMProvider(_llm_resp([]))
        ext = CodingMemoryExtractor(llm)
        drafts, facts = await ext.extract([], user_id="u1")
        assert drafts == []
        # The trace itself is empty: the fallback criterion is "is there any content", so with none nothing at all is stored.
        assert facts == []

    async def test_legacy_memories_key_still_parsed(self):
        """Before the two-layer split the prompt used "memories"; older models may still answer that way."""
        llm = MockLLMProvider(json.dumps({
            "memories": [{"task": "T", "solution": "S"}],
        }))
        ext = CodingMemoryExtractor(llm, verbatim_fallback_enabled=True)
        drafts, facts = await ext.extract(_msgs(), user_id="u1")
        assert [d.task for d in drafts] == ["T"]
        _assert_verbatim_fallback(facts)


@pytest.mark.asyncio
class TestFactExtraction:
    async def test_single_fact_object_extracted(self):
        llm = MockLLMProvider(_llm_resp(
            [{"task": "T", "solution": "S"}],
            {
                "statement": (
                    "The user wanted to locate new_file_search; after checking src/search.py the implementation was confirmed there, "
                    "and foo_config was searched for on C: and D: but not found."
                ),
                "files": ["src/search.py"],
            },
        ))
        ext = CodingMemoryExtractor(llm)
        drafts, facts = await ext.extract(_msgs(), user_id="u1")
        assert len(drafts) == 1
        assert [f.statement for f in facts] == [
            "The user wanted to locate new_file_search; after checking src/search.py the implementation was confirmed there, "
            "and foo_config was searched for on C: and D: but not found.",
        ]
        assert facts[0].files == ["src/search.py"]

    async def test_facts_only_without_experiences(self):
        """Facts are extracted without a threshold, so empty experiences alongside a non-empty fact is a legal state."""
        llm = MockLLMProvider(_llm_resp([], {"statement": "Build artifacts land in dist/"}))
        ext = CodingMemoryExtractor(llm)
        drafts, facts = await ext.extract(_msgs(), user_id="u1")
        assert drafts == []
        assert [f.statement for f in facts] == ["Build artifacts land in dist/"]

    async def test_fact_null_falls_back_to_verbatim(self):
        """When explicitly enabled, fact:null triggers the source-text fallback."""
        llm = MockLLMProvider(_llm_resp([], None))
        ext = CodingMemoryExtractor(llm, verbatim_fallback_enabled=True)
        _, facts = await ext.extract(_msgs(), user_id="u1")
        _assert_verbatim_fallback(facts)

    async def test_fact_missing_statement_falls_back_to_verbatim(self):
        """When explicitly enabled, a fact without a statement triggers the source-text fallback."""
        llm = MockLLMProvider(_llm_resp([], {"files": ["b.py"]}))
        ext = CodingMemoryExtractor(llm, verbatim_fallback_enabled=True)
        _, facts = await ext.extract(_msgs(), user_id="u1")
        _assert_verbatim_fallback(facts)

    async def test_tool_only_trace_does_not_create_fallback_by_default(self):
        messages = [
            ChatMessage(role="assistant", content="", tool_calls=[
                ToolCall(id="1", name="exec_command", arguments={"cmd": "pwd"}),
            ]),
            ChatMessage(
                role="tool", content="/workspace", tool_call_id="1",
                tool_name="exec_command",
            ),
        ]
        ext = CodingMemoryExtractor(MockLLMProvider(_llm_resp([], None)))
        _, facts = await ext.extract(messages, user_id="u1")
        assert facts == []

    async def test_tool_only_trace_fallback_can_be_enabled(self):
        messages = [
            ChatMessage(role="assistant", content="", tool_calls=[
                ToolCall(id="1", name="exec_command", arguments={"cmd": "pwd"}),
            ]),
            ChatMessage(
                role="tool", content="/workspace", tool_call_id="1",
                tool_name="exec_command",
            ),
        ]
        ext = CodingMemoryExtractor(
            MockLLMProvider(_llm_resp([], None)),
            verbatim_fallback_enabled=True,
        )
        _, facts = await ext.extract(messages, user_id="u1")
        assert len(facts) == 1
        assert facts[0].statement.startswith(
            '[0] assistant:   [tools: exec_command({"cmd":"pwd"})]'
        )
        assert "[1] tool(exec_command): /workspace" in facts[0].details

    async def test_legacy_three_facts_keeps_first_and_warns(self, caplog):
        llm = MockLLMProvider(_legacy_llm_resp([], [
            {"statement": "first conversation summary", "files": ["a.py"]},
            {"statement": "second atomic fact"},
            {"statement": "third atomic fact"},
        ]))
        ext = CodingMemoryExtractor(llm)
        with caplog.at_level("WARNING"):
            _, facts = await ext.extract(_msgs(), user_id="u1")
        assert [f.statement for f in facts] == ["first conversation summary"]
        assert "legacy 'facts' array with 3 entries" in caplog.text
        assert "keeping the first, dropping 2" in caplog.text

    async def test_fact_files_deduped(self):
        llm = MockLLMProvider(_llm_resp([],
            {
                "statement": "S",
                "files": ["C:\\", "C:\\", " D:\\ ", "src/a.py", ""],
            },
        ))
        ext = CodingMemoryExtractor(llm)
        _, facts = await ext.extract(_msgs(), user_id="u1")
        assert facts[0].files == ["C:\\", "D:\\", "src/a.py"]

    async def test_fact_files_non_list_tolerated(self):
        llm = MockLLMProvider(_llm_resp([], {"statement": "S", "files": "a.py"}))
        ext = CodingMemoryExtractor(llm)
        _, facts = await ext.extract(_msgs(), user_id="u1")
        assert facts[0].files == []
