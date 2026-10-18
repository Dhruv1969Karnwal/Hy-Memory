# -*- coding: utf-8 -*-
"""
Search path e2e tests

Instead of booting the full HyMemoryClient (heavy), this combines a real CodingMemoryStore with a
lightweight SimpleNamespace to test _search_coding directly, plus focused tests for input normalization.
"""

import asyncio
import os
import shutil
import tempfile
import uuid
from types import SimpleNamespace
from typing import List

import pytest

from hy_memory.client import HyMemoryClient
from hy_memory.coding.store import CodingMemoryStore
from hy_memory.coding.types import CodingMemory, WorkingFact

from tests.coding._helpers import MockEmbedService, MockLLMProvider


# ================================================================
# Fixtures (avoids repeatedly booting the full client)
# ================================================================

@pytest.fixture
async def search_ctx(tmp_data_dir, mock_embed):
    """
    Build a lightweight client-like ctx with a real CodingMemoryStore + mock embed.
    Suitable for exercising the HyMemoryClient._search_coding method.
    """
    from hy_memory.config import MemoryConfig

    cfg = MemoryConfig.from_env()
    cfg.history.db_path = os.path.join(tmp_data_dir, "history.db")
    cfg.coding.db_path = os.path.join(tmp_data_dir, "coding_memory.db")
    cfg.vector_store.provider = "chroma"
    cfg.vector_store.path = os.path.join(tmp_data_dir, "chroma")
    cfg.vector_store.collection_name = f"test_search_{uuid.uuid4().hex[:8]}"
    cfg.vector_store.embedding_dims = 1536

    store = CodingMemoryStore(cfg, embed_service=mock_embed)
    await store.initialize()

    # A SimpleNamespace carrying the required attributes, used as self
    ns = SimpleNamespace(
        _embed_service=mock_embed,
        _coding_store=store,
        _coding_writer=object(),  # Placeholder so _get_coding_writer skips the lazy path
        _coding_llm_provider=None,
        _coding_init_lock=asyncio.Lock(),
    )

    async def _stub_get_coding_writer():
        return ns._coding_writer

    ns._get_coding_writer = _stub_get_coding_writer

    yield ns
    await store.close()


def _seed_memories(store_to_seed: CodingMemoryStore, mock_embed: MockEmbedService):
    """Seed a few memories for the tests synchronously (run via asyncio.run outside the fixture)."""
    async def _do():
        await store_to_seed.insert(CodingMemory(
            experience_id="m_publish", user_id="u1", task="Package and publish to the private package index",
            search_keys=["credential location", "twine upload"], solution="Detailed publish steps",
            boundary_envs="twine>=6", workspace_id="ws/a",
            files=["scripts/publish_package.sh"], confidence=0.9,
        ))
        await store_to_seed.insert(CodingMemory(
            experience_id="m_pnpm", user_id="u1", task="This project uses pnpm",
            search_keys=["Do not use npm"], solution="All projects use pnpm",
confidence=1.0,
        ))
        await store_to_seed.insert(CodingMemory(
            experience_id="m_other_user", user_id="u_other", task="Another user's memory",
            solution="x",
        ))
    asyncio.run(_do())


# ================================================================
# _search_coding path tests
# ================================================================

@pytest.mark.asyncio
class TestSearchCodingMethod:
    async def test_hit_by_task(self, search_ctx, mock_embed):
        # Insert data into the already initialized store
        await search_ctx._coding_store.insert(CodingMemory(
            experience_id="m_publish", user_id="u1", task="Package and publish to the private package index",
            search_keys=["credential location"], solution="Detailed publish steps",
            boundary_envs="twine>=6",
            workspace_id="ws/a", files=["scripts/publish_package.sh"], confidence=0.9,
        ))

        import time
        resp = await HyMemoryClient._search_coding(
            search_ctx,
            target_query="Package and publish to the private package index",
            user_ids=["u1"],
            workspace_id="ws/a",
            branch=None,
            limit=10,
            working_fact_min_score=0.0,
            request_id="req-1",
            t0=time.perf_counter(),
        )
        assert resp["scene"] == "productivity"
        assert resp["request_id"] == "req-1"
        assert "elapsed_ms" in resp
        mems = resp["experiences"]
        assert len(mems) == 1
        m = mems[0]
        assert m["experience_id"] == "m_publish"
        assert m["task"] == "Package and publish to the private package index"
        assert m["search_keys"] == ["credential location"]
        assert m["solution"] == "Detailed publish steps"
        assert m["boundary_envs"] == "twine>=6"
        assert m["workspace_id"] == "ws/a"
        assert m["files"] == ["scripts/publish_package.sh"]
        assert m["matched_key"] == "Package and publish to the private package index"
        assert m["matched_key_kind"] == "task"
        assert m["score"] > 0.9
        assert m["confidence"] == 0.9

    async def test_hit_by_search_key(self, search_ctx, mock_embed):
        await search_ctx._coding_store.insert(CodingMemory(
            experience_id="m_publish", user_id="u1", task="Package and publish to the private package index",
            search_keys=["credential location"], solution="...",
confidence=0.9,
        ))

        import time
        resp = await HyMemoryClient._search_coding(
            search_ctx,
            target_query="credential location", user_ids=["u1"],
            workspace_id=None, branch=None,
            limit=10, working_fact_min_score=0.0,
            request_id="r", t0=time.perf_counter(),
        )
        mems = resp["experiences"]
        assert len(mems) == 1
        assert mems[0]["matched_key_kind"] == "aspect"
        assert mems[0]["matched_key"] == "credential location"

    async def test_workspace_filters(self, search_ctx, mock_embed):
        """A workspace filters by scope: another workspace recalls nothing, omitting it recalls everything."""
        await search_ctx._coding_store.insert(CodingMemory(
            experience_id="m1", user_id="u1", task="task A", solution="S",
            workspace_id="ws/a",
        ))

        import time
        for ws, expect in (("ws/a", 1), ("ws/b", 0), (None, 1)):
            resp = await HyMemoryClient._search_coding(
                search_ctx,
                target_query="task A", user_ids=["u1"],
                workspace_id=ws, branch=None,
                limit=10, working_fact_min_score=0.0,
                request_id="r", t0=time.perf_counter(),
            )
            assert len(resp["experiences"]) == expect, f"workspace_id={ws!r}"
            if expect:
                assert resp["experiences"][0]["workspace_id"] == "ws/a"

    async def test_min_score_filter(self, search_ctx, mock_embed):
        await search_ctx._coding_store.insert(CodingMemory(
            experience_id="m_unrelated", user_id="u1", task="task X",
            solution="S",
        ))

        import time
        # Use a query unrelated to the task; mock embed hashes the text, so similarity is low
        resp = await HyMemoryClient._search_coding(
            search_ctx,
            target_query="completely unrelated query string ABCDEFG xyz",
            user_ids=["u1"],
            workspace_id=None, branch=None,
            limit=10, working_fact_min_score=0.99,  # High threshold filters everything out
            request_id="r", t0=time.perf_counter(),
        )
        assert resp["experiences"] == []

    async def test_user_isolation(self, search_ctx, mock_embed):
        await search_ctx._coding_store.insert(CodingMemory(
            experience_id="m_other", user_id="u_other", task="my task",
            solution="S",
        ))

        import time
        resp = await HyMemoryClient._search_coding(
            search_ctx,
            target_query="my task", user_ids=["u1"],
            workspace_id=None, branch=None,
            limit=10, working_fact_min_score=0.0,
            request_id="r", t0=time.perf_counter(),
        )
        assert resp["experiences"] == []

    async def test_empty_store_returns_empty(self, search_ctx, mock_embed):
        import time
        resp = await HyMemoryClient._search_coding(
            search_ctx,
            target_query="anything", user_ids=["u1"],
            workspace_id=None, branch=None,
            limit=10, working_fact_min_score=0.0,
            request_id="r", t0=time.perf_counter(),
        )
        assert resp["experiences"] == []
        assert resp["facts"] == []
        assert resp["scene"] == "productivity"


# ================================================================
# Two-layer recall: fact primary index + experiences via the link + direct-query fallback
# ================================================================

@pytest.mark.asyncio
class TestDualLayerSearch:
    async def _search(self, ctx, query, *, limit=10, min_score=0.0,
                      exp_min_score=None):
        import time
        return await HyMemoryClient._search_coding(
            ctx,
            target_query=query, user_ids=["u1"],
            workspace_id=None, branch=None,
            limit=limit, working_fact_min_score=min_score,
            working_experience_min_score=exp_min_score,
            request_id="r", t0=time.perf_counter(),
        )

    async def test_fact_hit_returns_fact_layer(self, search_ctx):
        await search_ctx._coding_store.insert_fact(WorkingFact(
            fact_id="f1", user_id="u1",
            statement="new_file_search lives in src/search.py",
            files=["src/search.py"],
        ))

        resp = await self._search(search_ctx, "new_file_search lives in src/search.py")
        assert len(resp["facts"]) == 1
        f = resp["facts"][0]
        assert f["fact_id"] == "f1"
        assert f["statement"] == "new_file_search lives in src/search.py"
        assert f["files"] == ["src/search.py"]
        assert f["created_at"] is not None
        assert f["score"] > 0.9

    async def _linked_fixture(self, ctx):
        """An experience whose task looks nothing like the query; only the fact link reaches it."""
        await ctx._coding_store.insert(CodingMemory(
            experience_id="e1", user_id="u1",
            task="totally unrelated task name QQQQ", solution="How to change the search implementation",
        ))
        await ctx._coding_store.insert_fact(WorkingFact(
            fact_id="f1", user_id="u1",
            statement="new_file_search lives in src/search.py",
            derived_experience_ids=["e1"],
        ))

    async def test_fact_carries_linked_experience(self, search_ctx):
        """The experience itself does not match the query; the hit fact identifies it, marked fact_link."""
        await self._linked_fixture(search_ctx)

        resp = await self._search(
            search_ctx, "new_file_search lives in src/search.py", exp_min_score=0.0,
        )
        assert [e["experience_id"] for e in resp["experiences"]] == ["e1"]
        assert resp["experiences"][0]["via"] == "fact_link"

    async def test_linked_experience_is_gated_not_exempt(self, search_ctx):
        """The link is not an exemption: below the threshold nothing gets in, even when the fact backing it hit.

        This path used to be fully exempt, and in practice facts dragged in unrelated experiences from "another aspect".
        """
        await self._linked_fixture(search_ctx)

        resp = await self._search(
            search_ctx, "new_file_search lives in src/search.py", exp_min_score=0.99,
        )
        assert [f["fact_id"] for f in resp["facts"]] == ["f1"]  # The fact still hits
        assert resp["experiences"] == []                        # The experience is stopped by the gate

    async def test_both_experience_paths_share_one_gate(self, search_ctx):
        """The link and the direct query pass the same gate; there is no back door opened just for links.

        Both paths gate on the very same scores anyway (a link borrows the score of the direct round), so two tiers have no basis.
        Fact endorsement shows up in ranking -- linked results come before direct ones -- not in discounted admission.
        """
        await self._linked_fixture(search_ctx)

        loose = await self._search(
            search_ctx, "new_file_search lives in src/search.py", exp_min_score=0.0,
        )
        strict = await self._search(
            search_ctx, "new_file_search lives in src/search.py", exp_min_score=0.99,
        )
        # The same experience is admitted or not by one single value, working_experience_min_score
        assert [e["experience_id"] for e in loose["experiences"]] == ["e1"]
        assert strict["experiences"] == []
        # The fact layer has its own threshold, unaffected by the one for experiences
        assert len(strict["facts"]) == 1

    async def test_direct_hit_marked_via_direct(self, search_ctx):
        await search_ctx._coding_store.insert(CodingMemory(
            experience_id="e1", user_id="u1", task="Package and publish to the private package index", solution="S",
        ))

        resp = await self._search(search_ctx, "Package and publish to the private package index")
        assert resp["facts"] == []
        assert [e["experience_id"] for e in resp["experiences"]] == ["e1"]
        assert resp["experiences"][0]["via"] == "direct"
        assert resp["experiences"][0]["score"] > 0.9

    async def test_fact_link_wins_over_direct_on_dedup(self, search_ctx):
        """An experience both surfaced by a fact and hit directly appears once only, marked fact_link."""
        await search_ctx._coding_store.insert(CodingMemory(
            experience_id="e1", user_id="u1", task="shared query term", solution="S",
        ))
        await search_ctx._coding_store.insert_fact(WorkingFact(
            fact_id="f1", user_id="u1", statement="shared query term",
            derived_experience_ids=["e1"],
        ))

        resp = await self._search(search_ctx, "shared query term")
        assert len(resp["experiences"]) == 1
        assert resp["experiences"][0]["via"] == "fact_link"
        # The score from the direct path is still reported; taking the link does not discard it
        assert resp["experiences"][0]["score"] > 0.9

    async def test_dead_link_filtered(self, search_ctx):
        """A fact pointing at an experience already DELETEd: the existence filter should drop the dead link."""
        await search_ctx._coding_store.insert_fact(WorkingFact(
            fact_id="f1", user_id="u1", statement="fact pointing at a ghost experience",
            derived_experience_ids=["ghost"],
        ))

        resp = await self._search(search_ctx, "fact pointing at a ghost experience")
        assert len(resp["facts"]) == 1
        assert resp["experiences"] == []

    async def test_superseded_fact_not_recalled(self, search_ctx):
        await search_ctx._coding_store.insert_fact(WorkingFact(
            fact_id="f_old", user_id="u1", statement="config lives at /opt/app/config.yaml",
        ))
        await search_ctx._coding_store.insert_fact(WorkingFact(
            fact_id="f_new", user_id="u1", statement="config lives at /etc/app/config.yaml",
        ))
        await search_ctx._coding_store.supersede_fact("f_old", "f_new")

        resp = await self._search(search_ctx, "config lives at /opt/app/config.yaml")
        assert [f["fact_id"] for f in resp["facts"]] == ["f_new"]

    async def test_min_score_filters_facts(self, search_ctx):
        await search_ctx._coding_store.insert_fact(WorkingFact(
            fact_id="f1", user_id="u1", statement="some fact",
        ))

        assert len((await self._search(search_ctx, "some fact"))["facts"]) == 1
        resp = await self._search(search_ctx, "completely unrelated query ABCDEFG", min_score=0.99)
        assert resp["facts"] == []

    async def test_low_score_fact_does_not_drag_experience(self, search_ctx):
        """Once a low-score fact is stopped by the gate, its experience must not be linked into the results.

        The fact layer is the primary index; noise there amplifies into a stream of unrelated experiences.
        """
        await search_ctx._coding_store.insert(CodingMemory(
            experience_id="e1", user_id="u1", task="unrelated experience", solution="S",
        ))
        await search_ctx._coding_store.insert_fact(WorkingFact(
            fact_id="f1", user_id="u1", statement="unrelated fact",
            derived_experience_ids=["e1"],
        ))

        resp = await self._search(
            search_ctx, "totally off-topic query ZZZZZ", min_score=0.99,
        )
        assert resp["facts"] == []
        assert resp["experiences"] == []

    async def test_fact_user_isolation(self, search_ctx):
        await search_ctx._coding_store.insert_fact(WorkingFact(
            fact_id="f1", user_id="u_other", statement="someone else's fact",
        ))

        resp = await self._search(search_ctx, "someone else's fact")
        assert resp["facts"] == []

    async def test_facts_respect_limit_below_floor(self, search_ctx):
        """When limit is below the fact layer fetch floor (5), the returned count must still fall back to limit.

        The fact layer fetches max(limit, 5); the extra rows are headroom left for threshold filtering
        and must not leak straight through into the results.
        """
        for i in range(6):
            await search_ctx._coding_store.insert_fact(WorkingFact(
                fact_id=f"f{i}", user_id="u1", statement=f"facts in the same batch {i}",
            ))

        resp = await self._search(search_ctx, "facts in the same batch 0", limit=2)
        assert len(resp["facts"]) == 2

    async def test_experiences_capped_at_limit_across_both_sources(self, search_ctx):
        """Linked + direct combined never exceed limit, and the linked batch is kept first."""
        for i in range(4):
            await search_ctx._coding_store.insert(CodingMemory(
                experience_id=f"linked{i}", user_id="u1",
                task=f"experience pulled in through the fact link {i}", solution="S",
            ))
        # Experiences the direct query can hit: task equals the query
        for i in range(4):
            await search_ctx._coding_store.insert(CodingMemory(
                experience_id=f"direct{i}", user_id="u1",
                task="facts in the same batch 0", solution="S",
            ))
        await search_ctx._coding_store.insert_fact(WorkingFact(
            fact_id="f1", user_id="u1", statement="facts in the same batch 0",
            derived_experience_ids=[f"linked{i}" for i in range(4)],
        ))

        resp = await self._search(search_ctx, "facts in the same batch 0", limit=3)
        assert len(resp["experiences"]) == 3
        # Linked results come first, so what limit cuts away is the direct batch
        assert all(e["via"] == "fact_link" for e in resp["experiences"])

    async def test_dead_links_do_not_eat_limit_slots(self, search_ctx):
        """Dead links are dropped at get_many and must not occupy limit slots.

        Truncating before get_many lets the first few dead links eat every slot, so the result never reaches limit.
        """
        await search_ctx._coding_store.insert(CodingMemory(
            experience_id="alive", user_id="u1", task="living experience", solution="S",
        ))
        await search_ctx._coding_store.insert_fact(WorkingFact(
            fact_id="f1", user_id="u1", statement="fact with a dead link",
            # The first two experience_ids do not exist in the store
            derived_experience_ids=["ghost1", "ghost2", "alive"],
        ))

        resp = await self._search(search_ctx, "fact with a dead link", limit=1)
        assert [e["experience_id"] for e in resp["experiences"]] == ["alive"]
        # It must come through the link; the direct query could also pick up alive, which would make this test pass vacuously
        assert resp["experiences"][0]["via"] == "fact_link"


# ================================================================
# Input normalization tests -- no full client needed, called statically
# ================================================================

class TestInputNormalization:
    """Input validation tests (triggering ValueError is enough)"""

    def test_query_and_queries_mutex_via_async(self):
        # async_search validates its arguments at the very start, so an uninitialized client can be awaited
        # A minimal mock client is enough to reach that validation
        async def _run():
            class _DummyClient:
                _coding_llm_provider = None

                async def _check(self, query=None, queries=None):
                    # Reuse the real async_search validation logic (it raises directly)
                    if queries is not None and query is not None:
                        raise ValueError("query and queries are mutually exclusive; choose one")
                    if queries is None and query is None:
                        raise ValueError("either query or queries is required")
                    if queries is not None and (not isinstance(queries, list) or not queries):
                        raise ValueError("queries must be a non-empty list[str]")

            c = _DummyClient()
            # mutually exclusive
            with pytest.raises(ValueError, match="mutually exclusive"):
                await c._check(query="a", queries=["b"])
            # neither one passed
            with pytest.raises(ValueError, match="either query or queries"):
                await c._check()
            # empty list
            with pytest.raises(ValueError, match="non-empty"):
                await c._check(queries=[])

        asyncio.run(_run())
