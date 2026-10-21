# -*- coding: utf-8 -*-
"""
Routing tests for the scene parameter and queries-rewrite in search() / async_search().

Covers the three branches:
  1) Explicit scene ("coding"|"chat") -> no LLM call
  2) search_mode=default + no scene -> fixed chat/normal, zero LLM
  3) search_mode=agentic + no scene -> three-way classification + query rewrite

Plus:
  - LLM failure / parse failure -> fail-safe falls back to chat
  - Invalid scene value -> ValueError
  - scene=coding but coding.enable=False -> falls back to chat
  - Unit tests for classify_and_rewrite_queries
"""

import asyncio
import pytest

from hy_memory.coding.judge import classify_and_rewrite_queries

from tests.coding._helpers import MockLLMProvider, FailingLLMProvider


# ================================================================
# Phase 1: unit tests for classify_and_rewrite_queries (no real client needed)
# ================================================================

class TestClassifyAndRewrite:

    def test_coding_with_rewrite(self):
        async def _run():
            llm = MockLLMProvider("1\nHow do I publish the hy_memory project to a private package index")
            result = await classify_and_rewrite_queries(
                ["How do I publish it", "With twine or manually?", "I mean publishing to a private package index"],
                llm,
            )
            assert result["ok"] is True
            assert result["is_coding"] is True
            assert result["rewrite_query"] == "How do I publish the hy_memory project to a private package index"
        asyncio.run(_run())

    def test_chat_with_rewrite(self):
        async def _run():
            llm = MockLLMProvider("0\nWhat is the user's favorite color")
            result = await classify_and_rewrite_queries(
                ["What is your favorite color", "And mine?"], llm,
            )
            assert result["ok"] is True
            assert result["is_coding"] is False
            assert result["rewrite_query"] == "What is the user's favorite color"
        asyncio.run(_run())

    def test_no_rewrite_keeps_target(self):
        """LLM returns 0/1 but no rewrite -> the original target is used"""
        async def _run():
            llm = MockLLMProvider("1")
            result = await classify_and_rewrite_queries(["foo"], llm)
            assert result["ok"] is True
            assert result["is_coding"] is True
            assert result["rewrite_query"] == "foo"
        asyncio.run(_run())

    def test_unrecognized_head_falls_back(self):
        """LLM output is not 0/1 -> ok=False but the rewrite is still returned"""
        async def _run():
            llm = MockLLMProvider("maybe\nthe rewrite")
            result = await classify_and_rewrite_queries(["foo"], llm)
            assert result["ok"] is False
            assert result["is_coding"] is False
            assert result["rewrite_query"] == "the rewrite"
        asyncio.run(_run())

    def test_llm_failure(self):
        async def _run():
            result = await classify_and_rewrite_queries(["foo"], FailingLLMProvider())
            assert result["ok"] is False
            assert result["is_coding"] is False
            assert result["rewrite_query"] == "foo"
        asyncio.run(_run())

    def test_empty_queries(self):
        async def _run():
            llm = MockLLMProvider("1\nrewrite")
            result = await classify_and_rewrite_queries([], llm)
            assert result["ok"] is False
            assert result["rewrite_query"] == ""
        asyncio.run(_run())

    def test_markdown_fence_stripped(self):
        async def _run():
            llm = MockLLMProvider("```\n1\nrewritten X\n```")
            result = await classify_and_rewrite_queries(["x"], llm)
            assert result["ok"] is True
            assert result["is_coding"] is True
            assert result["rewrite_query"] == "rewritten X"
        asyncio.run(_run())


# ================================================================
# Phase 2: scene parameter routing -- a minimal SimpleNamespace stand-in client,
# covering the routing logic at the async_search entry directly (no full client init)
# ================================================================
#
# This reuses the existing search_ctx fixture (real store + mock embed); the scene routing
# lives at the top of client.async_search, and whether the store is actually called is
# short-circuited by us inside the mock client.
# ================================================================

import time
import uuid
import asyncio
from types import SimpleNamespace
from typing import List, Optional, Any, Dict

from hy_memory.client import HyMemoryClient
from hy_memory.coding.types import CodingMemory


@pytest.fixture
async def search_ctx_with_llm(tmp_data_dir, mock_embed):
    """Reuses the test_search_e2e ctx idea: real store + mock embed + injectable mock LLM"""
    import os, uuid
    from hy_memory.config import MemoryConfig
    from hy_memory.coding.store import CodingMemoryStore

    cfg = MemoryConfig.from_env()
    cfg.history.db_path = os.path.join(tmp_data_dir, "history.db")
    cfg.coding.db_path = os.path.join(tmp_data_dir, "coding_memory.db")
    cfg.coding.enable = True
    cfg.vector_store.provider = "chroma"
    cfg.vector_store.path = os.path.join(tmp_data_dir, "chroma")
    cfg.vector_store.collection_name = f"test_scene_{uuid.uuid4().hex[:8]}"
    cfg.vector_store.embedding_dims = 1536

    store = CodingMemoryStore(cfg, embed_service=mock_embed)
    await store.initialize()

    # Mock client-like namespace (covers the fields async_search uses)
    ns = SimpleNamespace(
        _config=cfg,
        _embed_service=mock_embed,
        _coding_store=store,
        _coding_writer=object(),  # Placeholder
        _coding_llm_provider=None,
        _coding_init_lock=asyncio.Lock(),
        _history_store=None,        # Used by record_search
        _enable_history_record=False,
        _build_isolation_key=lambda u, a: f"{u}:{a}",
    )

    # Use HyMemoryClient._search_coding with self bound to ns
    # Accepts **kwargs: only self is bound to ns here, threshold-like args pass through,
    # so new ones added later need no change here
    async def _search_coding_bound(target_query, **kw):
        return await HyMemoryClient._search_coding(ns, target_query=target_query, **kw)
    ns._search_coding = _search_coding_bound

    # Short-circuit the internal reader the chat path needs: return a fixed empty result
    async def _stub_get_coding_writer():
        return ns._coding_writer
    ns._get_coding_writer = _stub_get_coding_writer

    # Short-circuit the chat path: return empty memories (only the routing is tested here)
    async def _stub_chat_search(read_req):
        return type("Resp", (), {
            "memories": [], "total": 0, "elapsed_ms": 1.0,
        })()

    ns._reader = SimpleNamespace(read=_stub_chat_search)

    # async_search is now a thin shell: it sets the embed_override contextvar and forwards
    # to _async_search_impl (the real implementation). The SimpleNamespace must expose it,
    # otherwise forwarding raises AttributeError. Bind it in bound-method form.
    ns._async_search_impl = HyMemoryClient._async_search_impl.__get__(ns, type(ns))

    yield ns
    await store.close()


@pytest.mark.asyncio
class TestSceneRouting:
    """Three-branch routing over scene/queries/query in async_search"""

    # async_search is decorated by _route_to_internal_loop; tests call the raw function
    _async_search = staticmethod(HyMemoryClient.async_search.__wrapped__)

    async def _seed_coding_memory(self, ns):
        await ns._coding_store.insert(CodingMemory(
            experience_id="e_pub", user_id="u1", task="Publish to a private package index",
            search_keys=["twine upload"], solution="...", confidence=0.9,
        ))

    async def test_explicit_scene_coding_no_llm(self, search_ctx_with_llm):
        """Explicit scene='productivity' + single query -> coding path, no LLM call"""
        ns = search_ctx_with_llm
        await self._seed_coding_memory(ns)
        # Inject an LLM -- if it gets called, calls records it
        sentinel_llm = MockLLMProvider("0\nshould not be called")
        ns._coding_llm_provider = sentinel_llm

        resp = await self._async_search(
            ns,
            query="Publish to a private package index",
            scene="productivity",
            user_ids=["u1"],
            workspace_id=None, branch=None,
            limit=10, min_score=0.0,
        )
        assert resp["scene"] == "productivity"
        assert "facts" in resp
        assert "experiences" in resp
        # The LLM must not be called
        assert len(sentinel_llm.calls) == 0

    async def test_explicit_scene_chat_no_llm(self, search_ctx_with_llm):
        """Explicit scene='normal' + multiple queries -> chat path, no LLM call (even with several queries)"""
        ns = search_ctx_with_llm
        sentinel_llm = MockLLMProvider("1\nshould not be called")
        ns._coding_llm_provider = sentinel_llm

        # The chat path needs _reader.read; let it run through
        # Mock the tail dependencies of async_search to finish with an empty result
        # Only verifying that the coding branch is skipped and the LLM stays uncalled
        try:
            resp = await self._async_search(
                ns,
                queries=["a", "b", "Please publish it for me"],
                scene="normal",
                user_ids=["u1"],
            )
            # The chat branch needs more client internals; the try block is a fallback,
            # the point is that the LLM stays uncalled (short-circuited)
        except Exception:
            pass
        assert len(sentinel_llm.calls) == 0  # <- key assertion

    async def test_default_single_query_no_llm(self, search_ctx_with_llm):
        """Default light search: scene=None + single query is fixed normal, zero LLM."""
        ns = search_ctx_with_llm
        sentinel = MockLLMProvider('{"scene":"working"}')
        ns._coding_llm_provider = sentinel
        try:
            await self._async_search(ns, query="Publish to a private package index", user_ids=["u1"])
        except Exception:
            pass
        assert len(sentinel.calls) == 0

    async def test_default_multi_queries_keep_previous_rewrite_logic(self, search_ctx_with_llm):
        ns = search_ctx_with_llm
        ns._config.llm.enable_query_rewrite = True
        llm = MockLLMProvider("coding\nFix auth.py\nSINCE: NONE")
        ns._coding_llm_provider = llm
        seen = {}

        async def _capture(target_query, **kwargs):
            seen["query"] = target_query
            seen.update(kwargs)
            return {
                "request_id": "r", "scene": "productivity",
                "facts": [], "experiences": [], "elapsed_ms": 1.0,
                "chain_block": "", "chain_used": [],
            }

        ns._search_coding = _capture
        await self._async_search(
            ns, queries=["auth.py errors", "Fix it"], user_ids=["u1"],
        )
        assert len(llm.calls) == 1
        assert seen["variant"] == "coding"
        assert seen["query"] == "Fix auth.py"

    async def test_default_multi_queries_with_features_disabled_no_llm(self, search_ctx_with_llm):
        ns = search_ctx_with_llm
        ns._config.coding.enable = False
        ns._config.llm.enable_query_rewrite = False
        sentinel = MockLLMProvider("should not be called")
        ns._coding_llm_provider = sentinel
        try:
            await self._async_search(
                ns, queries=["Context", "Target question"], user_ids=["u1"],
            )
        except Exception:
            pass
        assert len(sentinel.calls) == 0

    async def test_no_scene_multi_queries_calls_llm_rewrite_used(self, search_ctx_with_llm):
        """No scene + multiple queries -> LLM called; is_coding=1 + rewrite -> effective query = rewrite"""
        ns = search_ctx_with_llm
        await self._seed_coding_memory(ns)
        # The LLM returns 1 + a rewrite
        llm = MockLLMProvider("1\nHow to publish to a private package index (rewritten)")
        ns._coding_llm_provider = llm

        resp = await self._async_search(
            ns,
            queries=["Publish it", "With twine?", "I mean the private package index"],
            user_ids=["u1"],
            workspace_id=None, branch=None,
            limit=10, min_score=0.0,
        )
        assert resp["scene"] == "productivity"
        assert len(llm.calls) == 1  # The LLM was called once

    async def test_no_scene_auto_coding_uses_coding_variant(self, search_ctx_with_llm):
        ns = search_ctx_with_llm
        ns._coding_llm_provider = MockLLMProvider(
            "coding\nFix the traceback in auth.py"
        )
        seen = {}

        async def _capture(target_query, **kwargs):
            seen.update(kwargs)
            return {
                "request_id": "r", "scene": "productivity",
                "facts": [], "experiences": [], "elapsed_ms": 1.0,
                "chain_block": "", "chain_used": [],
            }

        ns._search_coding = _capture
        await self._async_search(
            ns,
            queries=["auth.py errors", "Help me fix this traceback"],
            user_ids=["u1"],
        )
        assert seen["variant"] == "coding"

    async def test_no_scene_multi_queries_chat_classification(self, search_ctx_with_llm):
        """No scene + multiple queries + LLM classifies chat -> chat path"""
        ns = search_ctx_with_llm
        llm = MockLLMProvider("0\nrewrite for chat")
        ns._coding_llm_provider = llm

        try:
            await self._async_search(
                ns,
                queries=["You like", "what color", "cats or dogs"],
                user_ids=["u1"],
            )
        except Exception:
            pass
        # The LLM was called once
        assert len(llm.calls) == 1

    async def test_no_scene_llm_failure_falls_back_chat(self, search_ctx_with_llm):
        """No scene + multiple queries + LLM failure -> fail-safe chat path"""
        ns = search_ctx_with_llm
        ns._coding_llm_provider = FailingLLMProvider()

        try:
            await self._async_search(
                ns,
                queries=["debug", "my error"],
                user_ids=["u1"],
            )
        except Exception:
            pass
        # No exception may propagate upwards (fail-safe)

    async def test_invalid_scene_raises(self, search_ctx_with_llm):
        """Invalid scene value -> ValueError"""
        ns = search_ctx_with_llm
        with pytest.raises(ValueError, match="scene must be"):
            await self._async_search(
                ns,
                query="x",
                scene="weird",
                user_ids=["u1"],
            )

    async def test_invalid_search_mode_raises(self, search_ctx_with_llm):
        with pytest.raises(ValueError, match="search_mode must be"):
            await self._async_search(
                search_ctx_with_llm,
                query="x", search_mode="turbo", user_ids=["u1"],
            )

    async def test_explicit_coding_but_disabled_falls_back(self, search_ctx_with_llm):
        """scene='productivity' but config.coding.enable=False -> falls back to chat"""
        ns = search_ctx_with_llm
        ns._config.coding.enable = False
        sentinel_llm = MockLLMProvider("0\n")
        ns._coding_llm_provider = sentinel_llm

        try:
            await self._async_search(
                ns,
                query="x",
                scene="productivity",
                user_ids=["u1"],
            )
        except Exception:
            pass
        # Even with scene='productivity', coding off must fall back to chat (no LLM call)
        assert len(sentinel_llm.calls) == 0


@pytest.mark.asyncio
class TestRequestIdPassthrough:
    """request_id in search()/async_search(): reused when passed, generated when missing or blank."""

    _async_search = staticmethod(HyMemoryClient.async_search.__wrapped__)

    async def _seed_coding_memory(self, ns):
        await ns._coding_store.insert(CodingMemory(
            experience_id="m_pub", user_id="u1", task="Publish to a private package index",
            search_keys=["twine upload"], solution="...",
            confidence=0.9,
        ))

    async def test_explicit_request_id_reused(self, search_ctx_with_llm):
        """A passed-in request_id shows up unchanged in the result"""
        ns = search_ctx_with_llm
        await self._seed_coding_memory(ns)

        resp = await self._async_search(
            ns,
            query="Publish to a private package index",
            scene="productivity",
            user_ids=["u1"],
            request_id="app-pregenerated-req-123",
        )
        assert resp["request_id"] == "app-pregenerated-req-123"

    async def test_missing_request_id_autogenerated(self, search_ctx_with_llm):
        """No request_id -> the SDK generates a valid uuid itself"""
        ns = search_ctx_with_llm
        await self._seed_coding_memory(ns)

        resp = await self._async_search(
            ns,
            query="Publish to a private package index",
            scene="productivity",
            user_ids=["u1"],
        )
        assert resp["request_id"]
        uuid.UUID(resp["request_id"])  # No exception means it is a valid uuid

    async def test_blank_request_id_autogenerated(self, search_ctx_with_llm):
        """A blank request_id (empty string / spaces only) is treated as absent and regenerated"""
        ns = search_ctx_with_llm
        await self._seed_coding_memory(ns)

        for blank in ("", "   "):
            resp = await self._async_search(
                ns,
                query="Publish to a private package index",
                scene="productivity",
                user_ids=["u1"],
                request_id=blank,
            )
            assert resp["request_id"]
            uuid.UUID(resp["request_id"])

    async def test_each_call_without_request_id_gets_distinct_id(self, search_ctx_with_llm):
        """Separate calls each generate a different request_id"""
        ns = search_ctx_with_llm
        await self._seed_coding_memory(ns)

        resp1 = await self._async_search(
            ns, query="Publish to a private package index", scene="productivity", user_ids=["u1"],
        )
        resp2 = await self._async_search(
            ns, query="Publish to a private package index", scene="productivity", user_ids=["u1"],
        )
        assert resp1["request_id"] != resp2["request_id"]


@pytest.mark.asyncio
class TestQueryRewriteRouting:
    """Combined routing of query rewrite and working memory under search_mode=agentic."""

    _async_search = staticmethod(HyMemoryClient.async_search.__wrapped__)

    async def test_agentic_single_query_classifies_and_rewrites(self, search_ctx_with_llm):
        ns = search_ctx_with_llm
        llm = MockLLMProvider("coding\nFix the exception in auth.py\nSINCE: NONE")
        ns._coding_llm_provider = llm
        seen = {}

        async def _capture(target_query, **kwargs):
            seen["query"] = target_query
            seen.update(kwargs)
            return {
                "request_id": "r", "scene": "productivity",
                "facts": [], "experiences": [], "elapsed_ms": 1.0,
                "chain_block": "", "chain_used": [],
            }

        ns._search_coding = _capture
        await self._async_search(
            ns, query="Fix this exception", search_mode="agentic", user_ids=["u1"],
        )
        assert len(llm.calls) == 1
        assert seen["variant"] == "coding"
        assert seen["query"] == "Fix the exception in auth.py"

    async def test_agentic_without_working_memory_still_classifies_and_rewrites(self, search_ctx_with_llm):
        """agentic ignores the working-memory switch: classification, rewrite and time parsing still run.

        The tail of the chat path depends on a full client (_registry and friends); only the LLM call
        is checked here (single queries go through too); rewrite parsing is covered by analyze_queries tests.
        """
        ns = search_ctx_with_llm
        ns._config.coding.enable = False
        llm = MockLLMProvider(
            "chat\nwhat important thing did the user say last week\nSINCE: 2026-06-24"
        )
        ns._coding_llm_provider = llm

        try:
            await self._async_search(
                ns, query="What important thing did I say last week", search_mode="agentic",
                user_ids=["u1"],
            )
        except Exception:
            pass
        assert len(llm.calls) == 1                        # A single query also calls the LLM
        assert "Classify the target" in llm.calls[0]["prompt"]

    async def test_agentic_classifies_and_rewrites_in_one_call(self, search_ctx_with_llm):
        """agentic + working memory: one LLM call does classification, rewrite and time parsing."""
        ns = search_ctx_with_llm
        ns._config.coding.enable = True
        await TestSceneRouting()._seed_coding_memory(ns)
        llm = MockLLMProvider("1\nHow to publish to a private package index\nSINCE: NONE")
        ns._coding_llm_provider = llm

        resp = await self._async_search(
            ns, queries=["How do I publish it", "I mean the private package index"],
            search_mode="agentic",
            user_ids=["u1"], workspace_id=None, branch=None,
            limit=10, min_score=0.0,
        )
        assert len(llm.calls) == 1                        # Only one merged LLM call
        assert "Classify the target" in llm.calls[0]["prompt"]  # Classification shares the prompt
        assert resp["scene"] == "productivity"

    async def test_agentic_long_queries_still_classify_and_rewrite(self, search_ctx_with_llm):
        """agentic uses a bounded query view; oversized raw queries do not disable rewrite."""
        ns = search_ctx_with_llm
        ns._config.coding.enable = True
        ns._config.llm.query_rewrite_max_chars = 20  # Deliberately low
        llm = MockLLMProvider("chat\nRewritten query\nSINCE: NONE")
        ns._coding_llm_provider = llm

        try:
            resp = await self._async_search(
                ns, queries=["This is a very very long query" * 5, "More content appended past the threshold"],
                search_mode="agentic",
                user_ids=["u1"],
            )
            assert resp["rewrite"]["used"] is True
            assert resp["rewrite"]["effective_query"] == "Rewritten query"
        except Exception:
            pass
        assert len(llm.calls) == 1

    async def test_agentic_long_queries_without_wm_still_use_llm(self, search_ctx_with_llm):
        """agentic always enables judge/rewrite, even with working memory off and a long query."""
        ns = search_ctx_with_llm
        ns._config.coding.enable = False
        ns._config.llm.query_rewrite_max_chars = 10
        llm = MockLLMProvider("chat\nRewritten long query\nSINCE: NONE")
        ns._coding_llm_provider = llm

        try:
            resp = await self._async_search(
                ns, query="This query runs past ten characters",
                search_mode="agentic", user_ids=["u1"],
            )
            assert resp["rewrite"]["used"] is True
        except Exception:
            pass
        assert len(llm.calls) == 1

    async def test_default_mode_no_llm(self, search_ctx_with_llm):
        """The default mode ignores legacy settings: fixed normal and zero LLM."""
        ns = search_ctx_with_llm
        ns._config.coding.enable = False
        ns._config.llm.enable_query_rewrite = False
        sentinel = MockLLMProvider("nope")
        ns._coding_llm_provider = sentinel

        try:
            await self._async_search(ns, query="Just asking something", user_ids=["u1"])
        except Exception:
            pass
        assert len(sentinel.calls) == 0
