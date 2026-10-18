# -*- coding: utf-8 -*-
"""
Retrieval-side end to end for scene="coding" -- really exercising _search_coding / async_search,
not a hand-rolled stand-in.

Division of labor with test_scene_coding_variant.py:
  that file covers the components -- route resolution, prompt selection, the chain layer's own observe/decorate.
  this file covers the wiring -- real embed + real store + the real _search_coding return path,
                     confirming chain_block travels from the store all the way into the response.

Reuses the existing tests/coding/ fixtures (tmp_data_dir / mock_embed / search_ctx),
so the store here is real sqlite with real vector retrieval; only embed is mocked.
"""
import asyncio
import os
import tempfile
import time

import pytest

from hy_memory.client import HyMemoryClient
from hy_memory.coding.chain_store import HyCodeMem
from hy_memory.coding.types import WorkingFact

# the search_ctx / tmp_data_dir / mock_embed fixtures come from these two places
from tests.coding.test_search_e2e import search_ctx  # noqa: F401
from tests.coding.conftest import *  # noqa: F401,F403


_STATEMENT = ("On branch feat/q4, in qutebrowser/config/configtypes.py "
              "QtColor._parse_value: settled on the hue multiplier.")
_PATH = "qutebrowser/config/configtypes.py"


class _Msg:
    def __init__(self, role, content, tool_name=None):
        self.role, self.content, self.tool_name = role, content, tool_name
        self.tool_calls = None

    def is_tool_message(self):
        return self.role == "tool"


def _seed_chain(tmpdir, branch="feat/q4", path=_PATH):
    """Load two versions into the chain layer. Version 2 carries "scratch that" -- the revocation marker must be recognized."""
    H = HyCodeMem(os.path.join(tmpdir, "chain"))
    for said, old, new in [
        ("On branch %s, first pass: 255.0 everywhere." % branch,
         "mult = 255.0\n", "mult = 360.0\n"),
        ("Hold on, scratch that — special-case only hue.",
         "mult = 360.0\n", "mult = 360.0 if kind == 'h' else 255.0\n"),
    ]:
        H.observe(
            HyMemoryClient._chain_text([
                _Msg("user", said),
                _Msg("tool",
                     "path='%s'\nold='''%s'''\nnew='''%s'''" % (path, old, new),
                     tool_name="edit"),
            ]),
            meta={"branch": branch},
        )
    return H


def _bind_chain(ctx, store):
    """Attach the chain layer to the search_ctx SimpleNamespace.

    search_ctx is a bare namespace, not a real client, so the _decorate_with_chain
    bound method must be attached by hand (it calls self._get_chain_store internally).
    """
    ctx._chain_store = store
    ctx._chain_store_failed = False
    ctx._get_chain_store = lambda owner="": store
    ctx._decorate_with_chain = HyMemoryClient._decorate_with_chain.__get__(
        ctx, type(ctx))
    return ctx


@pytest.fixture
def chain_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.mark.asyncio
class TestSearchCodingVariantEndToEnd:
    """Call HyMemoryClient._search_coding for real, verifying each variant path once."""

    async def _search(self, ctx, query, *, variant, workspace_id=None):
        return await HyMemoryClient._search_coding(
            ctx, target_query=query,
            user_ids=["u1"], workspace_id=workspace_id, branch=None,
            limit=10, request_id="r", t0=time.perf_counter(),
            variant=variant,
        )

    async def _seed_fact(self, ctx):
        await ctx._coding_store.insert_fact(WorkingFact(
            fact_id="f1", user_id="u1",
            statement=_STATEMENT, files=[_PATH], branch="feat/q4",
        ))

    async def test_working_variant_has_no_chain_fields(self, search_ctx, chain_dir):
        """Regression gate: the variant=working response must match the pre-change shape exactly.

        Even with data in the chain store, chain_block must not appear -- otherwise working callers
        would suddenly receive one extra field.
        """
        await self._seed_fact(search_ctx)
        _bind_chain(search_ctx, _seed_chain(chain_dir))

        resp = await self._search(search_ctx, _STATEMENT, variant="working")
        assert "chain_block" not in resp
        assert "chain_used" not in resp
        assert len(resp["facts"]) == 1      # the vector dual-layer still works normally

    async def test_coding_variant_returns_chain_block(self, search_ctx, chain_dir):
        """variant=coding: the recalled fact gets the chain attached, with the branch and all versions in the block."""
        await self._seed_fact(search_ctx)
        _bind_chain(search_ctx, _seed_chain(chain_dir))

        resp = await self._search(search_ctx, _STATEMENT, variant="coding")
        assert len(resp["facts"]) == 1
        assert resp["chain_used"] == [0]
        blk = resp["chain_block"]
        # branch name -- code edited on the wrong branch is silently dropped, so this one token has top priority
        assert "feat/q4" in blk
        # both versions stay, not just the last: abandoned-version code must stay recognizable
        assert "v1" in blk and "v2" in blk
        assert "255.0" in blk and "360.0" in blk

    async def test_facts_not_deduped(self, search_ctx, chain_dir):
        """A chain-attached fact stays in facts -- deliberately not deduplicated.

        The SDK never quietly withholds data; a "block + remaining facts" layout is the caller's job via chain_used.
        """
        await self._seed_fact(search_ctx)
        _bind_chain(search_ctx, _seed_chain(chain_dir))

        resp = await self._search(search_ctx, _STATEMENT, variant="coding")
        assert resp["chain_used"] == [0]
        assert len(resp["facts"]) == 1, "the fact was swallowed by the chain layer"
        assert resp["facts"][0]["fact_id"] == "f1"

    async def test_empty_chain_store_gives_empty_block(self, search_ctx, chain_dir):
        """Empty chain store -> the block is "", no error, and the vector dual-layer still returns."""
        await self._seed_fact(search_ctx)
        _bind_chain(search_ctx, HyCodeMem(os.path.join(chain_dir, "empty")))

        resp = await self._search(search_ctx, _STATEMENT, variant="coding")
        assert resp["chain_block"] == ""
        assert resp["chain_used"] == []
        assert len(resp["facts"]) == 1

    async def test_chain_unavailable_degrades_to_vector_only(self, search_ctx):
        """Chain init failure -> degrade to the pure vector dual-layer without failing the whole search."""
        await self._seed_fact(search_ctx)
        _bind_chain(search_ctx, None)

        resp = await self._search(search_ctx, _STATEMENT, variant="coding")
        assert resp["chain_block"] == ""
        assert len(resp["facts"]) == 1

    async def test_no_fact_hit_gives_no_block(self, search_ctx, chain_dir):
        """The vector layer recalls nothing -> no mount point, so the block must be empty (the chain layer does no retrieval itself)."""
        _bind_chain(search_ctx, _seed_chain(chain_dir))

        resp = await self._search(search_ctx, "a completely unrelated query", variant="coding")
        assert resp["chain_used"] == []
        assert resp["chain_block"] == ""


@pytest.mark.asyncio
class TestEmptyPathKeysConsistent:
    """The fail-safe empty response must carry the same key set as the success path."""

    async def test_empty_carries_chain_keys_for_coding(self, search_ctx, chain_dir):
        # store init failure -> _empty() runs. The coding variant must still carry both chain_* keys,
        # otherwise callers need a second accessor path just for the fail-safe shape.
        async def _boom():
            raise RuntimeError("store down")
        search_ctx._get_coding_writer = _boom

        resp = await HyMemoryClient._search_coding(
            search_ctx, target_query="x", user_ids=["u1"],
            workspace_id=None, branch=None, limit=10,
            request_id="r", t0=time.perf_counter(), variant="coding",
        )
        assert resp["facts"] == [] and resp["experiences"] == []
        assert resp["chain_block"] == ""
        assert resp["chain_used"] == []

    async def test_empty_has_no_chain_keys_for_working(self, search_ctx):
        async def _boom():
            raise RuntimeError("store down")
        search_ctx._get_coding_writer = _boom

        resp = await HyMemoryClient._search_coding(
            search_ctx, target_query="x", user_ids=["u1"],
            workspace_id=None, branch=None, limit=10,
            request_id="r", t0=time.perf_counter(), variant="working",
        )
        assert "chain_block" not in resp


@pytest.mark.asyncio
class TestAsyncSearchPassesVariant:
    """The async_search(scene=...) -> _search_coding(variant=...) hop itself.

    The batch above calls _search_coding directly and bypasses argument mapping; this covers the mapping itself.
    """

    _async_search = staticmethod(HyMemoryClient.async_search.__wrapped__)

    async def _capture(self, scene, coding_config):
        """Capture the variant async_search actually passes down.

        Uses a real MemoryConfig (the coding_config fixture) rather than a hand-made namespace:
        _async_search_impl reads _config.llm / _config.search and several more layers, and a hand-made
        one missing any layer would raise AttributeError -- a test scaffold gap, not a bug in the code under test.
        """
        from types import SimpleNamespace
        seen = {}

        async def _spy(target_query, **kw):
            seen.update(kw)
            return {"request_id": "r", "scene": "productivity",
                    "facts": [], "experiences": [], "elapsed_ms": 1.0}

        coding_config.coding.enable = True
        ns = SimpleNamespace(
            _config=coding_config, _cache=None, _search_coding=_spy,
            _coding_llm_provider=None, _history_store=None,
            _enable_history_record=False,
        )
        ns._async_search_impl = HyMemoryClient._async_search_impl.__get__(
            ns, type(ns))
        await self._async_search(
            ns, query="q", scene=scene, search_mode="agentic", user_ids=["u1"],
            workspace_id=None, branch=None, limit=10, min_score=0.0,
        )
        return seen.get("variant")

    async def test_scene_coding_passes_coding_variant(self, coding_config):
        assert await self._capture("coding", coding_config) == "coding"

    async def test_scene_working_passes_working_variant(self, coding_config):
        assert await self._capture("working", coding_config) == "working"

    async def test_legacy_productivity_alias_stays_working(self, coding_config):
        # the legacy alias must keep the old behavior -- old callers must not get chain blocks from a new variant.
        assert await self._capture("productivity", coding_config) == "working"
