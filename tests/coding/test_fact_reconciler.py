# -*- coding: utf-8 -*-
"""Session-level fact reconciler tests: INSERT / same-task SUPERSEDE / experience chain inheritance"""

import json
import re
from datetime import datetime

import pytest

from hy_memory.coding.fact_reconciler import (
    CHAIN_SUMMARY_PROMPT,
    FACT_RECONCILE_PROMPT,
    WorkingFactReconciler,
)
from hy_memory.coding.types import CodingMemory, WorkingFact, WorkingFactDraft

from tests.coding._helpers import FailingLLMProvider, MockLLMProvider


def _ops_resp(ops):
    return json.dumps({"ops": ops})


# On SUPERSEDE the reconciler asks the LLM once more to write an evolution summary, so the mock
# needs a second reply. Without it the summary call receives the raw ops JSON and the test outcome is confusing.
_SUMMARY = "To fix the CI build, the Node 18 error was located first and the workflow was then upgraded; the build now passes."


def _llm(ops, summary=_SUMMARY):
    return MockLLMProvider([_ops_resp(ops), summary])


def _draft(statement, files=None):
    return WorkingFactDraft(statement=statement, files=files or [])


class TestPromptEvidenceContract:
    """Checks only the reconciler prompt contract, without passing it off as proof of LLM behaviour."""

    def test_candidate_judgement_does_not_upgrade_conjecture(self):
        assert "evidence-bounded text" in FACT_RECONCILE_PROMPT
        assert "shared conjecture/error" in FACT_RECONCILE_PROMPT
        assert "only about task continuation" in FACT_RECONCILE_PROMPT

    def test_chain_summary_preserves_uncertainty_and_blockers(self):
        assert "This is compression only" in CHAIN_SUMMARY_PROMPT
        assert "inferred is not established" in CHAIN_SUMMARY_PROMPT
        assert "unresolved stays unresolved" in CHAIN_SUMMARY_PROMPT
        assert "search/permission/coverage limits" in CHAIN_SUMMARY_PROMPT
        assert "Never upgrade it" in CHAIN_SUMMARY_PROMPT
        assert "directly says that cause was verified" in CHAIN_SUMMARY_PROMPT

    def test_chain_summary_preserves_stages_and_current_status(self):
        assert "every important stage in chronological order" in CHAIN_SUMMARY_PROMPT
        assert "actual action or change" in CHAIN_SUMMARY_PROMPT
        assert "direct result" in CHAIN_SUMMARY_PROMPT
        assert "unresolved issue or limiting condition" in CHAIN_SUMMARY_PROMPT
        assert "explicit current status" in CHAIN_SUMMARY_PROMPT
        assert "chains of 2-3 generations" in CHAIN_SUMMARY_PROMPT
        assert "2-4 compact paragraphs" in CHAIN_SUMMARY_PROMPT
        assert "Do not mechanically copy the full history" in CHAIN_SUMMARY_PROMPT
        assert "Do not invent stage headings or dates" in CHAIN_SUMMARY_PROMPT
        assert "earlier history was omitted" in CHAIN_SUMMARY_PROMPT
        assert "important numeric values" in CHAIN_SUMMARY_PROMPT


@pytest.mark.asyncio
class TestInsert:
    async def test_insert_all_drafts(self, coding_store, mock_embed):
        llm = MockLLMProvider(_ops_resp([]))
        rec = WorkingFactReconciler(coding_store, mock_embed, llm)

        facts, ops = await rec.reconcile(
            [_draft("Fact A", ["a.py"]), _draft("Fact B")],
            user_id="u1", agent_id="ag1", session_id="s1", workspace_id="ws/a",
        )

        assert [f.statement for f in facts] == ["Fact A", "Fact B"]
        assert [o.action for o in ops] == ["INSERT", "INSERT"]
        stored = await coding_store.get_fact(facts[0].fact_id, user_id="u1")
        assert stored is not None
        assert stored.status == "active"
        assert stored.files == ["a.py"]
        assert stored.agent_id == "ag1"
        assert stored.session_id == "s1"
        assert stored.workspace_id == "ws/a"
        assert stored.fact_link is None

    async def test_no_llm_call_without_candidates(self, coding_store, mock_embed):
        """With no candidate in the store there is nothing to judge, so an LLM call should not be wasted."""
        llm = MockLLMProvider(_ops_resp([]))
        rec = WorkingFactReconciler(coding_store, mock_embed, llm)

        await rec.reconcile([_draft("The first fact")], user_id="u1")
        assert len(llm.calls) == 0

    async def test_empty_drafts(self, coding_store, mock_embed):
        llm = MockLLMProvider(_ops_resp([]))
        rec = WorkingFactReconciler(coding_store, mock_embed, llm)
        facts, ops = await rec.reconcile([], user_id="u1")
        assert facts == []
        assert ops == []


@pytest.mark.asyncio
class TestSupersede:
    async def _seed_old(self, store, *, derived=None):
        old = WorkingFact(
            fact_id="f_old", user_id="u1",
            statement=(
                "The user wants to fix npm run build in CI; after checking .github/workflows/ci.yml,"
                " the Node 18 ERR_OSSL_EVP_UNSUPPORTED error was located, nothing changed yet."
            ),
            derived_experience_ids=derived or [],
        )
        await store.insert_fact(old)
        return old

    async def test_supersede_marks_old_and_links(self, coding_store, mock_embed):
        await self._seed_old(coding_store)
        llm = _llm([{
            "draft_idx": 0,
            "supersedes": ["f_old"],
            "reason": "Continuing the same CI build fix",
        }])
        rec = WorkingFactReconciler(coding_store, mock_embed, llm)

        facts, ops = await rec.reconcile(
            [_draft(
                "The user keeps fixing the same CI build; .github/workflows/ci.yml was raised to Node 20"
                " and npm run build now passes."
            )], user_id="u1",
        )

        new_id = facts[0].fact_id
        old = await coding_store.get_fact("f_old", user_id="u1")
        assert old.status == "superseded"
        assert old.fact_link == new_id
        assert [o.action for o in ops] == ["SUPERSEDE", "INSERT"]

    async def test_experience_chain_inherited(self, coding_store, mock_embed):
        """The experience chain the old fact accumulated must be handed to the new fact, patched on both ends."""
        await coding_store.insert(CodingMemory(
            experience_id="e1", user_id="u1", task="edit config", solution="S",
            source_fact_ids=["f_old"],
        ))
        await self._seed_old(coding_store, derived=["e1"])
        llm = _llm([{"draft_idx": 0, "supersedes": ["f_old"]}])
        rec = WorkingFactReconciler(coding_store, mock_embed, llm)

        facts, _ = await rec.reconcile(
            [_draft(
                "The user keeps fixing the same CI build; .github/workflows/ci.yml was raised to Node 20"
                " and npm run build now passes."
            )], user_id="u1",
        )
        new_id = facts[0].fact_id

        # Forward: the new fact receives the old fact's experience chain
        assert facts[0].derived_experience_ids == ["e1"]
        stored_new = await coding_store.get_fact(new_id, user_id="u1")
        assert stored_new.derived_experience_ids == ["e1"]
        # Backward: the experience gains the new fact while the old one is kept (the chain stays traceable)
        exp = await coding_store.get_by_id("e1", user_id="u1")
        assert exp.source_fact_ids == ["f_old", new_id]

    async def test_link_chain_stays_walkable(self, coding_store, mock_embed):
        """History is reconstructed by walking fact_link backwards, so consecutive supersessions must not break a single hop."""
        await self._seed_old(coding_store)          # generation 1
        gens = [
            (
                "The user keeps fixing the same CI build; .github/workflows/ci.yml was raised to Node 20,"
                " but npm run build is still affected by the stale cache.",
                "f_old",
            ),
            (
                "The user keeps verifying the same CI fix; after clearing the npm cache npm run build passes.",
                None,
            ),
        ]
        prev_id = None
        for stmt, target in gens:
            rec = WorkingFactReconciler(
                coding_store, mock_embed,
                _llm([{"draft_idx": 0, "supersedes": [target or prev_id]}]),
            )
            facts, _ = await rec.reconcile([_draft(stmt)], user_id="u1")
            prev_id = facts[0].fact_id

        lineage = await coding_store.get_fact_ancestors(prev_id, user_id="u1")
        assert [f.statement for f in lineage] == [
            "The user wants to fix npm run build in CI; after checking .github/workflows/ci.yml,"
            " the Node 18 ERR_OSSL_EVP_UNSUPPORTED error was located, nothing changed yet.",
            "The user keeps fixing the same CI build; .github/workflows/ci.yml was raised to Node 20,"
            " but npm run build is still affected by the stale cache.",
        ]

    async def test_reconcile_prompt_requires_same_task_not_same_file(
        self, coding_store, mock_embed
    ):
        await self._seed_old(coding_store)
        llm = _llm([])
        rec = WorkingFactReconciler(coding_store, mock_embed, llm)

        await rec.reconcile(
            [_draft(
                "The user wants to add a publish job to .github/workflows/ci.yml; twine upload was added"
                " and the setup is complete, but nothing has been published yet."
            )],
            user_id="u1",
        )

        prompt = llm.calls[0]["prompt"]
        assert "the same user recorded by an existing session fact" in prompt
        assert "merely mentions the same file" in prompt
        assert "pursues a different user goal" in prompt
        assert "INSERT is the safe default" in prompt
        assert "ERR_OSSL_EVP_UNSUPPORTED" in prompt
        assert "twine upload" in prompt

    async def test_summary_empty_when_nothing_superseded(self, coding_store, mock_embed):
        """Nothing was superseded, so there is no evolution to report."""
        llm = MockLLMProvider(_ops_resp([]))
        rec = WorkingFactReconciler(coding_store, mock_embed, llm)
        facts, _ = await rec.reconcile([_draft("A brand new conclusion")], user_id="u1")

        stored = await coding_store.get_fact(facts[0].fact_id, user_id="u1")
        assert stored.fact_chain_summary == ""

    async def test_old_statement_not_indexed_for_retrieval(self, coding_store, mock_embed):
        """A superseded wording stays in the store but leaves the vector database, so it must not be recallable again."""
        await self._seed_old(coding_store)
        rec = WorkingFactReconciler(
            coding_store, mock_embed, _llm([{"draft_idx": 0, "supersedes": ["f_old"]}]),
        )
        await rec.reconcile(
            [_draft("config is at /etc/app/config.yaml")], user_id="u1",
        )

        # Searching with the old wording only lands on the new fact (because the two are similar),
        # and the recalled statement is always the current value, never the one kept in the lineage
        emb = await coding_store.embed_service.embed("config is at /opt/app/config.yaml")
        hits = await coding_store.search_facts(emb, user_id="u1", top=5)
        assert all(f.statement != "config is at /opt/app/config.yaml" for f, _ in hits)

    async def test_ignores_id_outside_candidates(self, coding_store, mock_embed):
        """If the LLM hallucinates a fact_id that is not among the candidates, it must not actually touch it."""
        await self._seed_old(coding_store)
        llm = _llm([{"draft_idx": 0, "supersedes": ["f_hallucinated"]}])
        rec = WorkingFactReconciler(coding_store, mock_embed, llm)

        _, ops = await rec.reconcile(
            [_draft("config is at /etc/app/config.yaml")], user_id="u1",
        )
        assert [o.action for o in ops] == ["INSERT"]
        old = await coding_store.get_fact("f_old", user_id="u1")
        assert old.status == "active"

    async def test_old_fact_superseded_only_once(self, coding_store, mock_embed):
        """When two new facts both claim to supersede the same old fact, only the first one takes effect.

        Otherwise fact_link would be overwritten by the later write and the earlier one's inherited chain would break.
        """
        await self._seed_old(coding_store)
        llm = _llm([
            {"draft_idx": 0, "supersedes": ["f_old"]},
            {"draft_idx": 1, "supersedes": ["f_old"]},
        ])
        rec = WorkingFactReconciler(coding_store, mock_embed, llm)

        facts, ops = await rec.reconcile(
            [
                _draft("config is at /etc/app/config.yaml"),
                _draft("config is at /srv/app/config.yaml"),
            ],
            user_id="u1",
        )
        assert len(facts) == 2
        assert sum(1 for o in ops if o.action == "SUPERSEDE") == 1
        old = await coding_store.get_fact("f_old", user_id="u1")
        assert old.fact_link == facts[0].fact_id


@pytest.mark.asyncio
class TestChainSummary:
    """Progress on one task is summarised by the LLM; history is read from the fact_link chain on demand instead of being stored twice."""

    async def _seed_old(self, store):
        await store.insert_fact(WorkingFact(
            fact_id="f_old", user_id="u1",
            statement=(
                "The user wants to migrate the app config; it currently reads /opt/app/config.yaml,"
                " and the config loading logic has not been changed yet."
            ),
            created_at=datetime(2026, 1, 1),
        ))

    async def test_llm_summary_stored_and_used(self, coding_store, mock_embed):
        await self._seed_old(coding_store)
        written = (
            "To migrate the app config, /opt/app/config.yaml was confirmed first and the switch to"
            " /etc/app/config.yaml followed; the loading test passes now."
        )
        rec = WorkingFactReconciler(
            coding_store, mock_embed,
            _llm([{"draft_idx": 0, "supersedes": ["f_old"]}], summary=written),
        )

        facts, _ = await rec.reconcile(
            [_draft(
                "The user continues migrating the app config; the load path now points at /etc/app/config.yaml"
                " and the config loading test passes."
            )], user_id="u1",
        )

        stored = await coding_store.get_fact(facts[0].fact_id, user_id="u1")
        assert stored.fact_chain_summary == written

    async def test_prompt_carries_history_and_current(self, coding_store, mock_embed):
        """The summary call must see the old session and the current session together to write task progress."""
        await self._seed_old(coding_store)
        llm = _llm([{"draft_idx": 0, "supersedes": ["f_old"]}])
        rec = WorkingFactReconciler(coding_store, mock_embed, llm)

        await rec.reconcile(
            [_draft(
                "The user continues migrating the app config; the load path now points at /etc/app/config.yaml"
                " and the config loading test passes."
            )], user_id="u1",
        )

        assert len(llm.calls) == 2, "SUPERSEDE should trigger a second call"
        prompt = llm.calls[1]["prompt"]
        assert "/opt/app/config.yaml" in prompt
        assert "/etc/app/config.yaml" in prompt
        assert "goal" in prompt
        assert "stage-by-stage progression" in prompt
        assert "current status" in prompt
        assert llm.calls[1]["max_tokens"] == 600

    async def test_no_extra_call_without_supersede(self, coding_store, mock_embed):
        """No supersede means no evolution to summarise, so no extra call should be spent."""
        await self._seed_old(coding_store)
        llm = _llm([])
        rec = WorkingFactReconciler(coding_store, mock_embed, llm)

        await rec.reconcile([_draft("A conclusion about another topic")], user_id="u1")
        assert len(llm.calls) == 1

    async def test_prompt_carries_full_lineage(self, coding_store, mock_embed):
        """The third generation's summary must see the first two generations, not only the entry that was just superseded."""
        await self._seed_old(coding_store)
        rec = WorkingFactReconciler(
            coding_store, mock_embed, _llm([{"draft_idx": 0, "supersedes": ["f_old"]}]),
        )
        facts, _ = await rec.reconcile(
            [_draft(
                "The user continues migrating the app config; the load path now points at /etc/app/config.yaml"
                " and the config loading test passes."
            )], user_id="u1",
        )

        llm2 = _llm([{"draft_idx": 0, "supersedes": [facts[0].fact_id]}])
        rec2 = WorkingFactReconciler(coding_store, mock_embed, llm2)
        await rec2.reconcile(
            [_draft(
                "The user keeps improving the same config migration; the path now comes from the APP_CONFIG env var"
                " and the regression test passes."
            )], user_id="u1",
        )

        prompt = llm2.calls[1]["prompt"]
        assert "/opt/app/config.yaml" in prompt, "generation 1 must not be lost"
        assert "/etc/app/config.yaml" in prompt
        assert "APP_CONFIG" in prompt
        assert prompt.index("/opt/") < prompt.index("/etc/"), "chronological order"

    async def test_summary_failure_does_not_block_write(self, coding_store, mock_embed):
        """If the summary fails, supersede and persist proceed as usual; history stays in the store and the next supersede re-summarises it."""
        await self._seed_old(coding_store)

        class _SummaryFails(MockLLMProvider):
            async def complete(self, prompt, **kwargs):
                if "ENTRIES" in prompt:
                    raise RuntimeError("summary boom")
                return await super().complete(prompt, **kwargs)

        rec = WorkingFactReconciler(
            coding_store, mock_embed,
            _SummaryFails(_ops_resp([{"draft_idx": 0, "supersedes": ["f_old"]}])),
        )
        facts, ops = await rec.reconcile(
            [_draft(
                "The user continues migrating the app config; the load path now points at /etc/app/config.yaml"
                " and the config loading test passes."
            )], user_id="u1",
        )

        assert sum(1 for o in ops if o.action == "SUPERSEDE") == 1
        stored = await coding_store.get_fact(facts[0].fact_id, user_id="u1")
        assert stored.fact_chain_summary == ""
        # Key point: this generation has no summary written, but the lineage is intact and still walkable
        lineage = await coding_store.get_fact_ancestors(stored.fact_id, user_id="u1")
        assert [f.statement for f in lineage] == [
            "The user wants to migrate the app config; it currently reads /opt/app/config.yaml,"
            " and the config loading logic has not been changed yet."
        ]

    async def test_summary_recovers_after_a_failed_generation(
        self, coding_store, mock_embed
    ):
        """If one generation writes no summary, the next pass over the chain recovers the generation that was missed."""
        await self._seed_old(coding_store)

        class _SummaryFails(MockLLMProvider):
            async def complete(self, prompt, **kwargs):
                if "ENTRIES" in prompt:
                    raise RuntimeError("summary boom")
                return await super().complete(prompt, **kwargs)

        rec = WorkingFactReconciler(
            coding_store, mock_embed,
            _SummaryFails(_ops_resp([{"draft_idx": 0, "supersedes": ["f_old"]}])),
        )
        facts, _ = await rec.reconcile(
            [_draft(
                "The user continues migrating the app config; the load path now points at /etc/app/config.yaml"
                " and the config loading test passes."
            )], user_id="u1",
        )
        assert facts[0].fact_chain_summary == ""

        llm2 = _llm([{"draft_idx": 0, "supersedes": [facts[0].fact_id]}])
        rec2 = WorkingFactReconciler(coding_store, mock_embed, llm2)
        await rec2.reconcile(
            [_draft(
                "The user keeps improving the same config migration; the path now comes from the APP_CONFIG env var"
                " and the regression test passes."
            )], user_id="u1",
        )

        prompt = llm2.calls[1]["prompt"]
        assert "/opt/app/config.yaml" in prompt, "history the previous generation missed must be recovered"

    async def test_deep_lineage_truncated_and_flagged(self, coding_store, mock_embed):
        """A long-lived fact must not stuff its whole chain into the prompt, but it has to say how many generations were omitted."""
        from hy_memory.coding.types import MAX_FACT_CHAIN_ENTRIES

        prev = None
        total = MAX_FACT_CHAIN_ENTRIES + 3
        for i in range(total):
            await coding_store.insert_fact(WorkingFact(
                fact_id=f"g{i}", user_id="u1", statement=f"Generation {i} conclusion",
                created_at=datetime(2026, 1, i + 1),
            ))
            if prev:
                await coding_store.supersede_fact(prev, f"g{i}")
            prev = f"g{i}"

        llm = _llm([{"draft_idx": 0, "supersedes": [prev]}])
        rec = WorkingFactReconciler(coding_store, mock_embed, llm)
        await rec.reconcile([_draft("The latest conclusion")], user_id="u1")

        prompt = llm.calls[1]["prompt"]
        assert "Generation 0 conclusion" not in prompt, "the oldest generations should be cut"
        assert f"Generation {total - 1} conclusion" in prompt, "the most recent one must stay"
        assert "omitted" in prompt, "truncation has to be flagged"
        # Do not report a generation count. The reverse walk itself only goes MAX+1 steps, so the real depth is not countable and printing a number would lie
        # downstream: a generation-30 supersede would still compute "1 generation" while more than twenty were actually skipped.
        assert not re.search(r"\d+\s+earlier", prompt), "the truncation note must not carry a generation count"

    async def test_no_truncation_flag_on_short_lineage(self, coding_store, mock_embed):
        """With no truncation there must be no omission note, otherwise the LLM assumes there is history it was not given."""
        await self._seed_old(coding_store)
        llm = _llm([{"draft_idx": 0, "supersedes": ["f_old"]}])
        rec = WorkingFactReconciler(coding_store, mock_embed, llm)
        await rec.reconcile(
            [_draft(
                "The user continues migrating the app config; the load path now points at /etc/app/config.yaml"
                " and the config loading test passes."
            )], user_id="u1",
        )
        entries = llm.calls[1]["prompt"].split(
            "ENTRIES\n═══════════════════════════════════════════════════════════════════\n",
            1,
        )[1]
        assert "older revisions omitted" not in entries

    async def test_summary_rewritten_each_generation(self, coding_store, mock_embed):
        """When the second generation supersedes, the summary is rewritten instead of keeping the first generation's sentence."""
        await self._seed_old(coding_store)
        rec = WorkingFactReconciler(
            coding_store, mock_embed,
            _llm([{"draft_idx": 0, "supersedes": ["f_old"]}], summary="Summary draft 1"),
        )
        facts, _ = await rec.reconcile(
            [_draft(
                "The user continues migrating the app config; the load path now points at /etc/app/config.yaml"
                " and the config loading test passes."
            )], user_id="u1",
        )

        rec2 = WorkingFactReconciler(
            coding_store, mock_embed,
            _llm([{"draft_idx": 0, "supersedes": [facts[0].fact_id]}],
                 summary="Summary draft 2"),
        )
        facts2, _ = await rec2.reconcile(
            [_draft(
                "The user keeps improving the same config migration; the path now comes from the APP_CONFIG env var"
                " and the regression test passes."
            )], user_id="u1",
        )

        stored = await coding_store.get_fact(facts2[0].fact_id, user_id="u1")
        assert stored.fact_chain_summary == "Summary draft 2"

    async def test_disabled_skips_the_call(self, coding_store, mock_embed):
        """Once it is disabled no extra call should be spent; the supersede itself still happens."""
        await self._seed_old(coding_store)
        llm = _llm([{"draft_idx": 0, "supersedes": ["f_old"]}])
        rec = WorkingFactReconciler(
            coding_store, mock_embed, llm, enable_chain_summary=False,
        )
        facts, ops = await rec.reconcile(
            [_draft(
                "The user continues migrating the app config; the load path now points at /etc/app/config.yaml"
                " and the config loading test passes."
            )], user_id="u1",
        )

        assert len(llm.calls) == 1
        assert sum(1 for o in ops if o.action == "SUPERSEDE") == 1
        stored = await coding_store.get_fact(facts[0].fact_id, user_id="u1")
        assert stored.fact_chain_summary == ""


class TestCleanSummary:
    """Models like to decorate summaries; strip that before the text reaches the store."""

    @pytest.mark.parametrize("raw,expected", [
        ("```\nOne sentence summary.\n```", "One sentence summary."),
        ("```markdown\nOne sentence summary.\n```", "One sentence summary."),
        ("Summary:One sentence summary.", "One sentence summary."),
        ("Summary: One sentence summary.", "One sentence summary."),
        ("'One sentence summary.'", "One sentence summary."),
        ('"One sentence summary."', "One sentence summary."),
        # The Chinese prefix and the CJK quote pair are still matched in
        # `fact_reconciler._clean_summary`, and model output in that language is
        # what this stripper exists for, so they stay as the bytes they match on.
        # "Summary: One sentence summary." / "One sentence summary." in CJK quotes.
        ("\u603b\u7ed3\uff1a\u4e00\u53e5\u603b\u7ed3\u3002", "\u4e00\u53e5\u603b\u7ed3\u3002"),
        ("Summary: \u4e00\u53e5\u603b\u7ed3\u3002", "\u4e00\u53e5\u603b\u7ed3\u3002"),
        ("\u603b\u7ed3:\u4e00\u53e5\u603b\u7ed3\u3002", "\u4e00\u53e5\u603b\u7ed3\u3002"),
        ("\u201c\u4e00\u53e5\u603b\u7ed3\u3002\u201d", "\u4e00\u53e5\u603b\u7ed3\u3002"),
        ("One sentence\nsummary.", "One sentence summary."),
        ("  One sentence summary.  ", "One sentence summary."),
        ("", ""),
        ("```", ""),
    ])
    def test_strips_decoration(self, raw, expected):
        assert WorkingFactReconciler._clean_summary(raw) == expected

    def test_keeps_inner_quotes(self):
        """Only the outermost paired quotes are stripped; a path quoted inside the sentence must not be shaved off."""
        s = WorkingFactReconciler._clean_summary('It now reads the "APP_CONFIG" variable.')
        assert s == 'It now reads the "APP_CONFIG" variable.'


@pytest.mark.asyncio
class TestFailSafe:
    async def test_llm_failure_still_inserts(self, coding_store, mock_embed):
        """If the LLM dies, keeping one stale fact is better than losing a conclusion over a single failed call."""
        await coding_store.insert_fact(WorkingFact(
            fact_id="f_old", user_id="u1", statement="config is at /opt/app/config.yaml",
        ))
        rec = WorkingFactReconciler(coding_store, mock_embed, FailingLLMProvider())

        facts, ops = await rec.reconcile(
            [_draft("config is at /etc/app/config.yaml")], user_id="u1",
        )
        assert len(facts) == 1
        assert [o.action for o in ops] == ["INSERT"]
        old = await coding_store.get_fact("f_old", user_id="u1")
        assert old.status == "active"

    async def test_unparsable_llm_output(self, coding_store, mock_embed):
        await coding_store.insert_fact(WorkingFact(
            fact_id="f_old", user_id="u1", statement="config is at /opt/app/config.yaml",
        ))
        rec = WorkingFactReconciler(
            coding_store, mock_embed, MockLLMProvider("A paragraph that is not JSON")
        )

        facts, ops = await rec.reconcile(
            [_draft("config is at /etc/app/config.yaml")], user_id="u1",
        )
        assert len(facts) == 1
        assert [o.action for o in ops] == ["INSERT"]

    async def test_markdown_fence_parsed(self, coding_store, mock_embed):
        await coding_store.insert_fact(WorkingFact(
            fact_id="f_old", user_id="u1", statement="config is at /opt/app/config.yaml",
        ))
        llm = MockLLMProvider([
            "```json\n" + _ops_resp([{"draft_idx": 0, "supersedes": ["f_old"]}]) + "\n```",
            _SUMMARY,
        ])
        rec = WorkingFactReconciler(coding_store, mock_embed, llm)

        _, ops = await rec.reconcile(
            [_draft("config is at /etc/app/config.yaml")], user_id="u1",
        )
        assert sum(1 for o in ops if o.action == "SUPERSEDE") == 1


@pytest.mark.asyncio
class TestSubjectGateSwitch:
    """Switch for the mechanical subject gate.

    The gate is exact string equality, so its correctness depends entirely on whether subject is a stable entity name.
    For Travel it is (person names; measured 0 clear false positives among 933 blocked pairs); for Progressive Search it is not
    (59.2% are positional labels like 'Subquery N', and two positional labels can never be equal, so they always block each other).
    So it has to be switchable per dataset instead of being fixed one way for everything.
    """

    async def _seed(self, store, fact_id, subject, statement):
        await store.insert_fact(WorkingFact(
            fact_id=fact_id, user_id="u1", subject=subject, statement=statement,
        ))

    async def test_gate_on_blocks_cross_subject_supersede(self, coding_store, mock_embed):
        """When enabled, candidates with a different subject never reach the LLM list, so a supersede cannot happen."""
        await self._seed(coding_store, "f_emma", "Emma", "Emma's three-day itinerary is settled, budget still open.")
        rec = WorkingFactReconciler(
            coding_store, mock_embed,
            _llm([{"draft_idx": 0, "supersedes": ["f_emma"]}]),
            subject_gate=True,
        )
        _, ops = await rec.reconcile(
            [WorkingFactDraft(subject="Bart", statement="Bart's three-day itinerary is settled, budget still open.")],
            user_id="u1",
        )
        assert [o.action for o in ops] == ["INSERT"]
        assert (await coding_store.get_fact("f_emma", user_id="u1")).status == "active"

    async def test_gate_off_allows_cross_subject_supersede(self, coding_store, mock_embed):
        """When disabled, candidates go to the LLM as they are, and if it judges supersede the supersede really happens."""
        await self._seed(coding_store, "f_emma", "Emma", "Emma's three-day itinerary is settled, budget still open.")
        rec = WorkingFactReconciler(
            coding_store, mock_embed,
            _llm([{"draft_idx": 0, "supersedes": ["f_emma"]}]),
            subject_gate=False,
        )
        facts, ops = await rec.reconcile(
            [WorkingFactDraft(subject="Bart", statement="Bart's three-day itinerary is settled, budget still open.")],
            user_id="u1",
        )
        assert [o.action for o in ops] == ["SUPERSEDE", "INSERT"]
        old = await coding_store.get_fact("f_emma", user_id="u1")
        assert old.status == "superseded"
        assert old.fact_link == facts[0].fact_id

    async def test_gate_off_skips_pre_execute_recheck(self, coding_store, mock_embed):
        """When disabled, the recheck before execution has to be skipped too, otherwise the gate is not fully closed.

        Calling _execute directly bypasses recall, simulating the leak where the candidate gate let things through but the recheck still blocks.
        """
        await self._seed(coding_store, "f_emma", "Emma", "Emma's itinerary.")
        old = await coding_store.get_fact("f_emma", user_id="u1")
        rec = WorkingFactReconciler(
            coding_store, mock_embed, _llm([]), subject_gate=False,
        )
        _, ops = await rec._execute(
            [WorkingFactDraft(subject="Bart", statement="Bart's itinerary.")],
            [[old]],
            {0: ["f_emma"]},
            user_id="u1", agent_id="a1", session_id=None, workspace_id=None,
        )
        assert "SUPERSEDE" in [o.action for o in ops]

    async def test_empty_subject_never_blocked_either_way(self, coding_store, mock_embed):
        """When either side's subject is empty, no switch setting should block.

        Empty does not mean "different subject", it means "no subject was resolved this time" (fallback-stored source text, rows written before the change).
        Using an unknown to veto a judgement is wrong; hand it back to the LLM to decide on semantics.
        """
        for gate in (True, False):
            await self._seed(coding_store, f"f_noneed_{gate}", "", "Conclusion about the old deploy script.")
            old = await coding_store.get_fact(f"f_noneed_{gate}", user_id="u1")
            rec = WorkingFactReconciler(
                coding_store, mock_embed, _llm([]), subject_gate=gate,
            )
            _, ops = await rec._execute(
                [WorkingFactDraft(subject="deploy.sh", statement="Conclusion about the new deploy script.")],
                [[old]],
                {0: [f"f_noneed_{gate}"]},
                user_id="u1", agent_id="a1", session_id=None, workspace_id=None,
            )
            assert "SUPERSEDE" in [o.action for o in ops], f"empty subject wrongly blocked when gate={gate}"

    async def test_env_var_controls_default(self, coding_store, mock_embed, monkeypatch):
        """An env var switches the default, and an explicit constructor argument wins over it."""
        monkeypatch.setenv("HY_FACT_SUBJECT_GATE", "0")
        assert WorkingFactReconciler(coding_store, mock_embed, _llm([])).subject_gate is False
        assert WorkingFactReconciler(
            coding_store, mock_embed, _llm([]), subject_gate=True
        ).subject_gate is True
        monkeypatch.setenv("HY_FACT_SUBJECT_GATE", "1")
        assert WorkingFactReconciler(coding_store, mock_embed, _llm([])).subject_gate is True


@pytest.mark.asyncio
class TestDetailsInheritance:
    """Inheriting details from superseded facts.

    fact_chain_summary is an LLM compression product and its measured fidelity decays with chain distance (in one 3-generation chain gen2
    kept 5/5 slots while gen1 kept 0/2, and what was lost were exactly the concrete numeric values). Inheritance copies rather than regenerates, so
    within budget it is 100% faithful; also details never enter the embedding, so appending does not change retrieval scoring.
    """

    async def test_ancestor_details_copied_verbatim(self, coding_store, mock_embed):
        await coding_store.insert_fact(WorkingFact(
            fact_id="f_old", user_id="u1", subject="Emma",
            statement="Emma's itinerary draft is out, budget pending verification.",
            details="Average daily food cost $32.4-$57.6\nHotel rating 3.2-3.8",
        ))
        rec = WorkingFactReconciler(
            coding_store, mock_embed,
            _llm([{"draft_idx": 0, "supersedes": ["f_old"]}]),
            subject_gate=False, inherit_details=True,
        )
        facts, _ = await rec.reconcile(
            [WorkingFactDraft(subject="Emma", statement="Emma's itinerary is finalised.",
                              details="Flight CA1234")],
            user_id="u1",
        )
        stored = await coding_store.get_fact(facts[0].fact_id, user_id="u1")
        # This generation's details are there, and the ancestors' concrete values are present word for word too
        assert "Flight CA1234" in stored.details
        assert "$32.4-$57.6" in stored.details
        assert "3.2-3.8" in stored.details
        assert "gen1" in stored.details and "Emma" in stored.details

    async def test_no_exponential_growth_down_the_chain(self, coding_store, mock_embed):
        """Across three consecutive supersessions, details must not double with each generation.

        An ancestor's details already contain the blocks it inherited into its own generation, so appending directly would grow exponentially.
        _collect_history walks the whole lineage once, so each generation contributes only its own share.
        """
        await coding_store.insert_fact(WorkingFact(
            fact_id="f_g1", user_id="u1", subject="T",
            statement="Generation 1 conclusion.", details="V1=" + "a" * 300,
        ))
        prev = "f_g1"
        sizes = []
        for gen in range(2, 5):
            rec = WorkingFactReconciler(
                coding_store, mock_embed,
                _llm([{"draft_idx": 0, "supersedes": [prev]}]),
                subject_gate=False, inherit_details=True,
            )
            facts, _ = await rec.reconcile(
                [WorkingFactDraft(subject="T", statement=f"Generation {gen} conclusion.",
                                  details=f"V{gen}=" + "a" * 300)],
                user_id="u1",
            )
            prev = facts[0].fact_id
            stored = await coding_store.get_fact(prev, user_id="u1")
            sizes.append(len(stored.details))
            # Each generation's value marker appears exactly once, never inherited twice
            for g in range(1, gen + 1):
                assert stored.details.count(f"V{g}=") == 1, (
                    f"generation {gen} contains V{g}= {stored.details.count(f'V{g}=')} times, inherited more than once"
                )
        # Linear growth: 4 generations cost at most about 4x(300+marker); exponential growth would exceed 300*2^3
        assert sizes[-1] < 2200, f"details growing too fast: {sizes}"

    async def test_total_budget_enforced_and_loss_reported(self, coding_store, mock_embed):
        """The budget is hard, but the generations that got dropped must be written out instead of being silently truncated."""
        await coding_store.insert_fact(WorkingFact(
            fact_id="f_old", user_id="u1", subject="T",
            statement="Old conclusion.", details="X=" + "b" * 8000,
        ))
        rec = WorkingFactReconciler(
            coding_store, mock_embed,
            _llm([{"draft_idx": 0, "supersedes": ["f_old"]}]),
            subject_gate=False, inherit_details=True,
        )
        facts, _ = await rec.reconcile(
            [WorkingFactDraft(subject="T", statement="New conclusion.", details="Y=1")],
            user_id="u1",
        )
        stored = await coding_store.get_fact(facts[0].fact_id, user_id="u1")
        assert "…(truncated)" in stored.details
        assert len(stored.details) < 6000

    async def test_switch_off_keeps_details_untouched(self, coding_store, mock_embed):
        await coding_store.insert_fact(WorkingFact(
            fact_id="f_old", user_id="u1", subject="T",
            statement="Old conclusion.", details="Ancestor value Z=9",
        ))
        rec = WorkingFactReconciler(
            coding_store, mock_embed,
            _llm([{"draft_idx": 0, "supersedes": ["f_old"]}]),
            subject_gate=False, inherit_details=False,
        )
        facts, ops = await rec.reconcile(
            [WorkingFactDraft(subject="T", statement="New conclusion.", details="Y=1")],
            user_id="u1",
        )
        # The supersede still happens, only the inheritance is skipped
        assert "SUPERSEDE" in [o.action for o in ops]
        stored = await coding_store.get_fact(facts[0].fact_id, user_id="u1")
        assert stored.details == "Y=1"
        assert "Z=9" not in stored.details

    async def test_inheritance_does_not_change_recall_vector(self, coding_store, mock_embed):
        """details never enter the embedding, so however much is inherited it must not change retrieval scoring.

        This is a direct check of the claim "appending does not hurt recall quality", not an inference.
        """
        await coding_store.insert_fact(WorkingFact(
            fact_id="f_old", user_id="u1", subject="T",
            statement="Old conclusion.", details="Irrelevant noise " + "q" * 2000,
        ))
        rec = WorkingFactReconciler(
            coding_store, mock_embed,
            _llm([{"draft_idx": 0, "supersedes": ["f_old"]}]),
            subject_gate=False, inherit_details=True,
        )
        facts, _ = await rec.reconcile(
            [WorkingFactDraft(subject="T", statement="New conclusion.", details="Y=1")],
            user_id="u1",
        )
        new_id = facts[0].fact_id
        emb = await coding_store.embed_service.embed("T New conclusion.")
        hits = await coding_store.search_facts(emb, user_id="u1", top=5)
        got = {f.fact_id: s for f, s in hits}
        assert new_id in got
        # Same statement+subject but without inheritance: the scores should match
        await coding_store.insert_fact(WorkingFact(
            fact_id="f_clean", user_id="u1", subject="T",
            statement="New conclusion.", details="Y=1",
        ))
        hits2 = await coding_store.search_facts(emb, user_id="u1", top=5)
        got2 = {f.fact_id: s for f, s in hits2}
        assert abs(got2["f_clean"] - got2[new_id]) < 1e-6, (
            f"inheritance changed recall scoring: clean={got2['f_clean']} inherited={got2[new_id]}"
        )
