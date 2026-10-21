# -*- coding: utf-8 -*-
"""Lexical (BM25) channel for the fact layer (coding/lexical.py + the rerank path of store.search_facts).

MockEmbedService returns md5-hash vectors that carry no semantics -- which suits this path well: dense order is
essentially random, so lexical hits become the only ranking signal with content, and whether hybrid retrieval helped is obvious at a glance.
"""

import pytest

from hy_memory.coding.lexical import rerank, tokenize
from hy_memory.coding.types import WorkingFact


class TestTokenize:
    def test_keeps_numbers_whole(self):
        """The identity of a quantity lives entirely in the intact number; shredding it ruins the signal."""
        assert "1e" not in tokenize("tolerance 1e-6")
        assert "64" in tokenize("step 3 input: n=64")
        assert "688.7" in tokenize("cost 688.7-721.3 per unit")

    def test_case_insensitive(self):
        assert tokenize("Get-ChildItem") == tokenize("get-childitem")

    def test_mixed_cjk_and_latin(self):
        toks = tokenize("\u7b2c3\u5929\u665a\u9910\u9884\u7b97 77-91 USD")  # "day 3 dinner budget"
        assert "77" in toks and "91" in toks and "usd" in toks
        assert any("\u665a\u9910" in t for t in toks)  # "dinner"

    def test_empty(self):
        assert tokenize("") == []
        assert tokenize("!!! ???") == []


class TestRerank:
    def test_exact_token_beats_dense_order(self):
        """Exactly what hybrid retrieval exists for: n=64 and n=32 mean almost the same thing, and the query wants one of them."""
        # dense puts a first (0.81 vs 0.80) -- semantically the two are near indistinguishable, exactly where lexical should speak
        cands = [("a", 0.81, "step 3 input: n=32, tolerance 1e-6"),
                 ("b", 0.80, "step 3 input: n=64, tolerance 1e-6")]
        order = rerank("what was n=64", cands)
        assert order[0] == "b"

    def test_returns_all_ids_exactly_once(self):
        cands = [(str(i), 0.9 - i * 0.01, f"fact {i} value={i * 7}") for i in range(6)]
        order = rerank("value=21", cands)
        assert sorted(order) == sorted(c[0] for c in cands)

    def test_no_lexical_hit_keeps_dense(self):
        """With lexical scores all zero its ranking is arbitrary, so blending it in perturbs dense order with noise."""
        cands = [("a", 0.9, "alpha beta"), ("b", 0.8, "gamma delta")]
        assert rerank("completely unrelated query text", cands) is None

    def test_single_candidate_is_noop(self):
        assert rerank("q", [("a", 0.9, "text")]) is None

    def test_empty_query_is_noop(self):
        assert rerank("!!!", [("a", 0.9, "alpha"), ("b", 0.8, "beta")]) is None

    def test_no_lexical_discrimination_is_noop(self):
        """When every lexical score is equal its order is arbitrary and must not be used to perturb dense."""
        # both texts identical => no lexical discrimination => no rerank, hand back to the caller as pure dense
        cands = [("a", 0.8, "same text here"), ("b", 0.9, "same text here")]
        assert rerank("same text", cands) is None

    def test_lex_cannot_override_a_large_dense_gap(self):
        """Lexical is a patch, not the primary index: a 0.4 cosine gap must not be flipped by one lexical hit.

        This pins the semantic boundary of lex_weight -- it is exactly "how large a cosine gap lexical may still flip".
        """
        cands = [("far", 0.90, "a completely different topic"), ("near", 0.50, "retry budget = 5")]
        order = rerank("retry budget 5", cands, lex_weight=0.05)
        assert order[0] == "far"


@pytest.mark.asyncio
class TestSearchFactsRerank:
    async def _seed(self, store, statements):
        for fid, stmt in statements:
            await store.insert_fact(WorkingFact(
                fact_id=fid, user_id="u1", statement=stmt,
            ))

    async def test_query_text_can_change_the_top(self, coding_store):
        """With more candidates than slots, the lexically matched one must get in."""
        await self._seed(coding_store, [
            (f"f{i}", f"step {i} input: n={2 ** i}, tolerance 1e-6") for i in range(8)
        ])
        emb = await coding_store.embed_service.embed("step input tolerance")

        dense = await coding_store.search_facts(emb, user_id="u1", top=3)
        fused = await coding_store.search_facts(
            emb, user_id="u1", top=3, query_text="what was n=64",
        )
        assert len(fused) == len(dense) == 3
        assert "f6" in {f.fact_id for f, _ in fused}  # n=64

    async def test_scores_stay_dense_not_rrf(self, coding_store):
        """Fused scores are a different scale; returning them would silently void the 0.30 gate at client.py:851."""
        await self._seed(coding_store, [
            (f"f{i}", f"Release pipeline step {i}, timeout {i * 10}s") for i in range(8)
        ])
        emb = await coding_store.embed_service.embed("Release pipeline step 3, timeout 30s")
        fused = await coding_store.search_facts(
            emb, user_id="u1", top=3, query_text="timeout 30s",
        )
        assert fused and all(s > 0.5 for _, s in fused), [s for _, s in fused]

    async def test_rerank_never_returns_fewer_than_dense(self, coding_store):
        """This case is the foundation of the whole design.

        Reranking before the gate lets BM25 push a low-cosine fact into the top slice, which the upper gate then
        drops -- hybrid would recall less than pure dense. So min_score must cut the candidate pool before reranking.
        """
        await self._seed(coding_store, [
            (f"f{i}", f"Task {i}: set the index baseline to {i}, retry budget {i + 1}") for i in range(10)
        ])
        emb = await coding_store.embed_service.embed("Task 7: set the index baseline to 7, retry budget 8")
        for ms in (0.0, 0.3, 0.5, 0.7):
            dense = await coding_store.search_facts(emb, user_id="u1", top=4, min_score=ms)
            fused = await coding_store.search_facts(
                emb, user_id="u1", top=4, min_score=ms, query_text="retry budget 8",
            )
            assert len(fused) == len(dense), f"min_score={ms}"
            assert all(s >= ms for _, s in fused)

    async def test_no_query_text_is_pure_dense(self, coding_store):
        """Without query_text results must match the pre-change behaviour item by item -- an optional patch, not a new default."""
        await self._seed(coding_store, [
            (f"f{i}", f"fact {i}") for i in range(6)
        ])
        emb = await coding_store.embed_service.embed("fact 0")
        a = await coding_store.search_facts(emb, user_id="u1", top=3)
        b = await coding_store.search_facts(emb, user_id="u1", top=3, query_text="   ")
        assert [f.fact_id for f, _ in a] == [f.fact_id for f, _ in b]

    async def test_min_score_filters(self, coding_store):
        await self._seed(coding_store, [("f1", "Publishing the build to the private package index")])
        emb = await coding_store.embed_service.embed("Publishing the build to the private package index")
        assert len(await coding_store.search_facts(emb, user_id="u1", min_score=0.9)) == 1
        assert await coding_store.search_facts(emb, user_id="u1", min_score=1.01) == []

    async def test_superseded_still_excluded_under_rerank(self, coding_store):
        """Reranking must not become a bypass for superseded conclusions."""
        await self._seed(coding_store, [
            (f"f{i}", f"Conclusion {i}, threshold {i * 5}") for i in range(8)
        ])
        assert await coding_store.supersede_fact("f3", "f_new") is True
        emb = await coding_store.embed_service.embed("Conclusion 3, threshold 15")
        fused = await coding_store.search_facts(
            emb, user_id="u1", top=5, query_text="threshold 15",
        )
        assert "f3" not in {f.fact_id for f, _ in fused}


class TestKillSwitch:
    """Ablation must be switchable without code changes; the gate taught us that "env var only, code default opposite"
    falls back silently. Here it is inverted -- on by default in code, the env var only disables it or retunes weights."""

    def test_env_off_disables(self, monkeypatch):
        from hy_memory.coding import lexical
        cands = [("a", 0.81, "n=32"), ("b", 0.80, "n=64")]
        assert lexical.rerank("n=64", cands)[0] == "b"
        monkeypatch.setenv(lexical.ENV_DISABLE, "1")
        assert lexical.rerank("n=64", cands) is None

    def test_env_weight_overrides(self, monkeypatch):
        from hy_memory.coding import lexical
        cands = [("far", 0.90, "a completely different topic"), ("near", 0.50, "retry budget = 5")]
        assert lexical.rerank("retry budget 5", cands)[0] == "far"
        monkeypatch.setenv(lexical.ENV_WEIGHT, "9.0")   # weight large enough to swamp the cosine gap
        assert lexical.rerank("retry budget 5", cands)[0] == "near"

    def test_bad_weight_falls_back(self, monkeypatch):
        from hy_memory.coding import lexical
        monkeypatch.setenv(lexical.ENV_WEIGHT, "not-a-number")
        cands = [("a", 0.81, "n=32"), ("b", 0.80, "n=64")]
        assert lexical.rerank("n=64", cands) is not None   # no raise, falls back to the default weight
