# -*- coding: utf-8 -*-
"""
Chinese BM25 symmetry tests - keeps the query/candidate tokenization pipeline consistent.

Covers:
1. search_text produced by _node_to_payload is jieba-segmented (space separated)
2. the lemmatize candidate path inside ReconcileHybridRetriever
"""

import pytest

from hy_memory.data.vector_store_base import VectorStoreBase
from hy_memory.models.memory import MemoryNode, MemoryLayer, MemoryStatus, SourceType
from hy_memory.pipelines._retrieval import bm25 as rbm25
from hy_memory.pipelines._retrieval.lemmatize import lemmatize_for_bm25


def _make_node(content: str, tags=None) -> MemoryNode:
    return MemoryNode(
        node_id="n1",
        user_id="u1",
        agent_id="a1",
        session_id="s1",
        layer=MemoryLayer.L2_FACT,
        content=content,
        status=MemoryStatus.ACTIVE,
        confidence=1.0,
        source_type=SourceType.EXPLICIT,
        tags=tags or [],
    )


class TestNodeToPayloadSearchText:
    """payload.search_text produced by _node_to_payload must be jieba-segmented."""

    def test_chinese_content_is_jieba_segmented(self):
        node = _make_node("\u7528\u6237\u503e\u5411\u4e8e\u63a5\u53d7\u52a1\u5b9e\u7684\u89e3\u51b3\u65b9\u6848")  # "the user prefers pragmatic solutions"
        payload = VectorStoreBase._node_to_payload(node)
        st = payload["search_text"]
        # segmentation output is space separated
        assert " " in st, f"expected jieba-segmented (with spaces), got {st!r}"
        # the content field is left untouched
        assert payload["content"] == "\u7528\u6237\u503e\u5411\u4e8e\u63a5\u53d7\u52a1\u5b9e\u7684\u89e3\u51b3\u65b9\u6848"
        # search_text contains the keywords jieba segmented out
        tokens = st.split()
        assert "\u7528\u6237" in tokens  # "user"
        assert "\u503e\u5511" in tokens  # "prefers"
        assert "\u89e3\u51b3\u65b9\u6848" in tokens or ("\u89e3\u51b3" in tokens and "\u65b9\u6848" in tokens)  # "solution" / "resolve" + "plan"

    def test_english_content_lemmatized(self):
        node = _make_node("The user prefers pragmatic solutions for deployment")
        payload = VectorStoreBase._node_to_payload(node)
        st = payload["search_text"]
        # after English lemmatization everything is lowercase and keywords survive
        assert "user" in st
        assert "pragmatic" in st
        # the content field keeps its original casing
        assert payload["content"] == "The user prefers pragmatic solutions for deployment"

    def test_tags_concatenated_then_lemmatized(self):
        node = _make_node("\u7528\u6237\u559c\u6b22\u732b", tags=["\u5ba0\u7269", "\u52a8\u7269"])  # "the user likes cats" / "pets" / "animals"
        payload = VectorStoreBase._node_to_payload(node)
        st = payload["search_text"]
        tokens = st.split()
        assert "\u5ba0\u7269" in tokens  # "pets"
        assert "\u52a8\u7269" in tokens  # "animals"
        assert "\u7528\u6237" in tokens  # "user"

    def test_empty_content_safe(self):
        node = _make_node("")
        payload = VectorStoreBase._node_to_payload(node)
        assert payload["search_text"] == ""

    def test_idempotent_already_segmented(self):
        """re-segmenting already segmented input is stable (same length, same token set)"""
        once = lemmatize_for_bm25("\u7528\u6237\u503e\u5411\u4e8e\u63a5\u53d7\u52a1\u5b9e\u7684\u89e3\u51b3\u65b9\u6848")
        twice = lemmatize_for_bm25(once)
        assert sorted(once.split()) == sorted(twice.split())


class TestReconcileBM25Symmetric:
    """compute_bm25_scores straight up - the candidate side must lemmatize"""

    def test_unfixed_pattern_is_zero(self):
        """regression: jieba query + raw Chinese candidate -> all-zero scores"""
        query_terms = rbm25.tokenize(lemmatize_for_bm25("\u7528\u6237\u503e\u5411\u4e8e\u63a5\u53d7"))
        # candidate not lemmatized (the old behaviour)
        contents_raw = ["\u7528\u6237\u503e\u5411\u4e8e\u63a5\u53d7\u52a1\u5b9e\u7684\u89e3\u51b3\u65b9\u6848"]
        raw = rbm25.compute_bm25_scores(query_terms, contents_raw)
        assert raw == [0.0], f"expected old buggy zero, got {raw}"

    def test_fixed_pattern_is_nonzero(self):
        """new pipeline: both sides lemmatize -> match"""
        query = "\u7528\u6237\u503e\u5411\u4e8e\u63a5\u53d7"
        cand = "\u7528\u6237\u503e\u5411\u4e8e\u63a5\u53d7\u52a1\u5b9e\u7684\u89e3\u51b3\u65b9\u6848"
        q_terms = rbm25.tokenize(lemmatize_for_bm25(query))
        contents = [lemmatize_for_bm25(cand)]
        raw = rbm25.compute_bm25_scores(q_terms, contents)
        assert raw[0] > 0.0, f"expected non-zero after fix, got {raw}"
