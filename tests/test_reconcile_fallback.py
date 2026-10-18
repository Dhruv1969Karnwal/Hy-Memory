"""When reconcile fails it must fall back to "all ADD" and never drop the extraction results.

In one staging regression scenario extraction had already succeeded, but the reconcile LLM failed after retries,
so a whole batch of good facts was dropped and `memory_ops` returned 0 ops.
reconcile only decides how new facts merge with old memories; it being unavailable does not mean the extraction results are worthless.
Better to keep duplicates (later dedup / reconcile can still clean them up) than to lose content.
"""

from hy_memory.agent.reconciler import MemoryReconciler, ReconcileResult


NEW = [
    "The user took a road trip in July 2023 to Banff, where he went skiing.",
    "The user is in a new romantic relationship as of August 2023.",
    "Sam offered to give the user tips on organizing a book club.",
]
META = [
    {"content": NEW[0], "layer": "L2_FACT", "tags": ["travel", "skiing"], "owner": "user"},
    {"content": NEW[1], "layer": "L2_FACT", "tags": ["relationship"], "owner": "user"},
    {"content": NEW[2], "layer": "L2_FACT", "tags": ["friends"], "owner": "agent"},
]


class TestFallbackAddOps:
    def test_builds_one_add_per_extracted_memory(self):
        ops = MemoryReconciler._fallback_add_ops(NEW, META)
        assert len(ops) == 3
        assert all(o.op == "ADD" for o in ops)
        assert [o.content for o in ops] == NEW

    def test_preserves_layer_tags_owner(self):
        """The fallback must not drop meta, or owner/tags all degrade."""
        ops = MemoryReconciler._fallback_add_ops(NEW, META)
        assert [o.layer for o in ops] == ["L2_FACT"] * 3
        assert ops[0].tags == ["travel", "skiing"]
        assert ops[2].owner == "agent"

    def test_reason_marks_degradation(self):
        ops = MemoryReconciler._fallback_add_ops(NEW, META)
        assert "degraded" in ops[0].reason

    def test_works_without_meta(self):
        """The fallback must also work when meta is missing (just without tags/owner)."""
        ops = MemoryReconciler._fallback_add_ops(NEW, None)
        assert len(ops) == 3 and all(o.op == "ADD" for o in ops)

    def test_skips_empty_content(self):
        ops = MemoryReconciler._fallback_add_ops(["", None, "real fact"], None)
        assert [o.content for o in ops] == ["real fact"]

    def test_no_memories_yields_nothing(self):
        assert MemoryReconciler._fallback_add_ops([], None) == []


class TestReconcileResultDegradedFlag:
    def test_default_is_not_degraded(self):
        r = ReconcileResult()
        assert r.degraded is False and r.degraded_reason is None

    def test_degraded_result_still_succeeds(self):
        """Key semantics: after the fallback, success must be True, or the writer would still discard everything.

        In the writer this is `if not recon_result.success: return [], ...` -- whenever success=False,
        whatever is in ops never gets persisted.
        """
        r = ReconcileResult(
            ops=MemoryReconciler._fallback_add_ops(NEW, META),
            success=True, degraded=True, degraded_reason="LLM_UNAVAILABLE",
        )
        assert r.success is True and r.degraded is True
        assert len(r.ops) == 3


class TestWriterHonorsDegradedResult:
    def test_writer_stores_ops_when_success_true(self):
        """Pinned from the writer source: with success=True the early-return discard branch is not taken."""
        import inspect

        from hy_memory.pipelines.writer import MemoryWriter

        src = inspect.getsource(MemoryWriter._reconcile_and_store_locked)
        assert "if not recon_result.success:" in src
        # The fallback path must be observable
        assert "degraded" in src


class TestLLMFailureIsNotPropagated:
    """LLM exceptions must not bubble up -- bubbling would discard the whole batch of extraction results."""

    def test_llm_call_is_wrapped(self):
        import inspect

        from hy_memory.agent.reconciler import MemoryReconciler as R

        src = inspect.getsource(R.reconcile)
        # llm.complete must be inside try, and on failure it must fall back to ADD-all
        assert "_llm_error" in src
        assert "_fallback_add_ops" in src
        assert "LLM_UNAVAILABLE" in src

    def test_failure_log_tolerates_none_response(self):
        """When the LLM is down, response is None; if the failure branch reads response.prompt_tokens directly
        it raises AttributeError and disables the fallback itself too."""
        import inspect

        from hy_memory.agent.reconciler import MemoryReconciler as R

        src = inspect.getsource(R.reconcile)
        # Only inspect the failure branch (on the success path response is necessarily non-None; reading attributes directly is fine)
        start = src.index("if not _parse_ok:")
        end = src.index("_fallback_add_ops", start)
        fail_branch = src[start:end]
        assert 'getattr(response, "prompt_tokens", 0)' in fail_branch
        assert "response.prompt_tokens," not in fail_branch


class TestExtractorPromptRules:
    """The two new extractor rules (each mapped directly to a production badcase)."""

    def _prompts(self):
        from hy_memory.agent.extractor import EXTRACT_SYSTEM_PROMPT
        from hy_memory.agent import prompts_zh

        zh = next(
            v for k, v in vars(prompts_zh).items()
            if isinstance(v, str) and "Time rules" in v and "Quality requirements" in v
        )
        return EXTRACT_SYSTEM_PROMPT, zh

    def test_just_happened_events_rule_present_both_languages(self):
        """"I just got back from a trip": the return date is the answer to a LoCoMo temporal question, and it used to be dropped."""
        en, zh = self._prompts()
        assert "just got back" in en
        assert "just got back" in zh or "just happened" in zh

    def test_no_generalisation_rule_present_both_languages(self):
        """Banff was generalised into "the Rocky Mountains", losing the specific place name."""
        en, zh = self._prompts()
        assert "Banff" in en and "Banff" in zh
        assert "Rocky Mountains" in en

    def test_cross_speaker_detail_rule_present(self):
        """The detail is given in the other speaker's turn and confirmed by the user -- the subject rules must not cause detail loss."""
        en, zh = self._prompts()
        assert "other speaker" in en
        assert "other speaker's turn" in zh
