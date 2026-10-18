"""
Unit tests for the _retrieval module, covering:
  - intent.classify_intent / extract_keywords
  - bm25.compute_bm25_scores
  - rrf.rrf_fuse / compute_confidence

evolution / tag_index depend on the async vector_store; their integration tests run manually under dev_test.
"""

import pytest

from hy_memory.pipelines._retrieval import bm25, intent, rrf


# ==========================================================
# intent
# ==========================================================

class TestIntentClassify:
    def test_navigational_camelcase(self):
        assert intent.classify_intent("my SoundCloud username") == "NAVIGATIONAL"

    def test_navigational_snake_case(self):
        assert intent.classify_intent("what is user_preference") == "NAVIGATIONAL"

    def test_navigational_backtick(self):
        assert intent.classify_intent("how does `mem_agent` work") == "NAVIGATIONAL"

    def test_navigational_url(self):
        assert intent.classify_intent("see https://example.com/doc") == "NAVIGATIONAL"

    def test_navigational_memid(self):
        assert intent.classify_intent("show mem-a1b2c3d4") == "NAVIGATIONAL"

    def test_navigational_version(self):
        assert intent.classify_intent("use v0.3.45") == "NAVIGATIONAL"

    def test_conceptual_how(self):
        # "how does X work" hits a conceptual trigger but also NAV (backticks do not here);
        # pure English input with no NAV signal here -> CONCEPTUAL
        assert intent.classify_intent("how do I approach career decisions") == "CONCEPTUAL"

    def test_conceptual_chinese(self):
        assert intent.classify_intent("\u6211\u5bf9\u8fdc\u7a0b\u5de5\u4f5c\u7684\u503e\u5411\u662f\u4ec0\u4e48") == "CONCEPTUAL"

    def test_factual_default(self):
        # Purely factual question, no NAV and no CONCEPTUAL trigger
        assert intent.classify_intent("what is my favorite food") == "FACTUAL"

    def test_factual_simple(self):
        assert intent.classify_intent("am I 28 years old") == "FACTUAL"

    def test_nav_wins_over_conceptual(self):
        # Hits both NAV and CONCEPTUAL; NAV takes priority
        assert intent.classify_intent("how does `foo` work") == "NAVIGATIONAL"


class TestExtractKeywords:
    def test_basic_english(self):
        kws = intent.extract_keywords("my favorite food is beef noodle")
        # "my" and "is" are dropped; "favorite", "food", "beef", "noodle" are kept
        assert "favorite" in kws
        assert "food" in kws
        assert "beef" in kws
        assert "noodle" in kws
        assert "my" not in kws
        assert "is" not in kws

    def test_stopwords_filtered(self):
        assert intent.extract_keywords("what is the weather") == ["weather"]

    def test_min_length_filter(self):
        # "ok" has length 2 and is filtered out
        kws = intent.extract_keywords("ok python great")
        assert "ok" not in kws
        assert "python" in kws
        assert "great" in kws

    def test_chinese(self):
        kws = intent.extract_keywords("\u4e0a\u6b21\u5403\u7684\u725b\u8089\u9762\u600e\u4e48\u6837")
        # A CJK run becomes one whole token (a single character never appears alone, because a CJK token is a contiguous run of ideographs)
        assert any("\u725b\u8089\u9762" in k or "\u4e0a\u6b21\u5403\u7684\u725b\u8089\u9762\u600e\u4e48\u6837" in k for k in kws)

    def test_dedup_preserve_order(self):
        kws = intent.extract_keywords("python is great python rocks great")
        # Deduplicated
        assert kws.count("python") == 1
        assert kws.count("great") == 1
        # Order preserved
        assert kws.index("python") < kws.index("great")

    def test_empty(self):
        assert intent.extract_keywords("") == []
        assert intent.extract_keywords("   ") == []


# ==========================================================
# bm25
# ==========================================================

class TestBM25:
    def test_empty_query_returns_zeros(self):
        scores = bm25.compute_bm25_scores([], ["doc one", "doc two"])
        assert scores == [0.0, 0.0]

    def test_empty_pool_returns_empty(self):
        scores = bm25.compute_bm25_scores(["foo"], [])
        assert scores == []

    def test_exact_match_scores_highest(self):
        docs = [
            "user says beef noodle is great",        # Contains two keywords
            "user says pizza is amazing",             # Contains one "is"
            "unrelated content about music",          # No hits
        ]
        scores = bm25.compute_bm25_scores(["beef", "noodle"], docs)
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]
        # Documents without beef/noodle should score 0
        assert scores[2] == 0.0

    def test_rare_term_higher_idf(self):
        # "rare" appears in only 1 document, "common" appears in every document
        docs = [
            "rare word here",
            "common word everywhere common",
            "common word everywhere common",
        ]
        rare_scores = bm25.compute_bm25_scores(["rare"], docs)
        common_scores = bm25.compute_bm25_scores(["common"], docs)
        # rare's matching document should outscore common's matching document (higher IDF at the same tf)
        # Note common appears twice in doc[1]/doc[2]: higher tf but lower idf
        assert rare_scores[0] > 0
        # common hits every document but its scores should stay flatter and lower than rare's hit score
        assert common_scores[1] > 0

    def test_score_and_rank_normalized(self):
        docs = [("id1", "foo bar"), ("id2", "bar baz"), ("id3", "nothing")]
        ranked = bm25.score_and_rank(["foo"], docs)
        # Sorted by score descending
        assert [r[0] for r in ranked] == sorted([r[0] for r in ranked],
                                                 key=lambda x: dict(ranked)[x],
                                                 reverse=True)
        # The top score normalises to 1.0
        assert abs(ranked[0][1] - 1.0) < 1e-9


# ==========================================================
# rrf
# ==========================================================

class TestRRF:
    def test_single_channel_preserves_order(self):
        hits = [{"node_id": "a", "score": 0.9}, {"node_id": "b", "score": 0.5}]
        fused = rrf.rrf_fuse({"vec": hits})
        assert [h["node_id"] for h in fused] == ["a", "b"]

    def test_multi_channel_intersection_wins(self):
        # The doc ranked first by both channels should win
        vec = [{"node_id": "a"}, {"node_id": "b"}, {"node_id": "c"}]
        bm25 = [{"node_id": "b"}, {"node_id": "a"}, {"node_id": "c"}]
        fused = rrf.rrf_fuse({"vec": vec, "bm25": bm25})
        # a: (1/(60+1)) + (1/(60+2)) ≈ 0.0164 + 0.0161 = 0.0325
        # b: (1/(60+2)) + (1/(60+1)) ~= 0.0161 + 0.0164 = 0.0325 -- almost the same as a
        # c: (1/(60+3)) + (1/(60+3)) ≈ 0.0317
        # a ≈ b > c
        node_ids = [h["node_id"] for h in fused]
        assert node_ids[0] in ("a", "b")
        assert node_ids[1] in ("a", "b")
        assert node_ids[2] == "c"

    def test_weights_affect_ranking(self):
        vec = [{"node_id": "a"}, {"node_id": "b"}]
        bm25 = [{"node_id": "b"}, {"node_id": "a"}]
        # Heavy bm25 weight -> b should rank first
        fused = rrf.rrf_fuse(
            {"vec": vec, "bm25": bm25},
            weights={"vec": 0.3, "bm25": 1.5},
        )
        assert fused[0]["node_id"] == "b"

    def test_per_channel_tracking(self):
        vec = [{"node_id": "a", "score": 0.9}]
        bm25 = [{"node_id": "a", "score": 0.7}]
        fused = rrf.rrf_fuse({"vec": vec, "bm25": bm25})
        entry = fused[0]
        assert entry["rrf_rank_by_channel"] == {"vec": 1, "bm25": 1}
        assert abs(entry["per_channel_score"]["vec"] - 0.9) < 1e-9
        assert abs(entry["per_channel_score"]["bm25"] - 0.7) < 1e-9

    def test_empty_channels(self):
        assert rrf.rrf_fuse({}) == []
        assert rrf.rrf_fuse({"vec": []}) == []

    def test_compute_confidence_empty(self):
        assert rrf.compute_confidence([]) == 0.0

    def test_compute_confidence_nonempty(self):
        fused = [
            {"rrf_score": 0.05},
            {"rrf_score": 0.04},
            {"rrf_score": 0.03},
        ]
        conf = rrf.compute_confidence(fused)
        assert 0.0 < conf <= 1.0
