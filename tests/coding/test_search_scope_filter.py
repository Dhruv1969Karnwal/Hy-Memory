# -*- coding: utf-8 -*-
"""
Scope filtering by workspace_id / branch in working (coding) retrieval.

The rule itself is documented in hy_memory.coding.store._scope_where; here we test how it behaves across
both recall layers. The most critical group is "no filter when nothing is passed" -- that was the behaviour
before the filter existed, and it must not change just because this capability was added.
"""

import asyncio
import os
import sqlite3
import time
import uuid
from types import SimpleNamespace

import pytest

from hy_memory.client import HyMemoryClient
from hy_memory.coding.store import (
    CodingMemoryStore,
    _scope_matches,
    _scope_overfetch,
    _scope_where,
)
from hy_memory.coding.types import CodingMemory, WorkingFact


# ================================================================
# Fixtures
# ================================================================

@pytest.fixture
async def scope_ctx(tmp_data_dir, mock_embed):
    """Real CodingMemoryStore + mock embed, bound into a lightweight ctx usable as self."""
    from hy_memory.config import MemoryConfig

    cfg = MemoryConfig.from_env()
    cfg.history.db_path = os.path.join(tmp_data_dir, "history.db")
    cfg.coding.db_path = os.path.join(tmp_data_dir, "coding_memory.db")
    cfg.vector_store.provider = "chroma"
    cfg.vector_store.path = os.path.join(tmp_data_dir, "chroma")
    cfg.vector_store.collection_name = f"test_scope_{uuid.uuid4().hex[:8]}"
    cfg.vector_store.embedding_dims = 1536

    store = CodingMemoryStore(cfg, embed_service=mock_embed)
    await store.initialize()

    ns = SimpleNamespace(
        _embed_service=mock_embed,
        _coding_store=store,
        _coding_writer=object(),
        _coding_llm_provider=None,
        _coding_init_lock=asyncio.Lock(),
    )

    async def _stub_get_coding_writer():
        return ns._coding_writer

    ns._get_coding_writer = _stub_get_coding_writer

    yield ns
    await store.close()


async def _search(ctx, query, *, workspace_id=None, branch=None, limit=10):
    """All score gates wide open so only the scope filter has an effect."""
    return await HyMemoryClient._search_coding(
        ctx,
        target_query=query, user_ids=["u1"],
        workspace_id=workspace_id, branch=branch,
        limit=limit,
        working_fact_min_score=0.0, working_experience_min_score=0.0,
        request_id="r", t0=time.perf_counter(),
    )


async def _seed_two_workspaces(store):
    """One experience + one fact per workspace, worded equivalently for the same query."""
    for ws, branch, suffix in (("ws/a", "main", "a"), ("ws/b", "dev", "b")):
        await store.insert(CodingMemory(
            experience_id=f"e_{suffix}", user_id="u1",
            task="Package and publish to the private package index", solution=f"S-{suffix}",
            workspace_id=ws, branch=branch,
        ))
        await store.insert_fact(WorkingFact(
            fact_id=f"f_{suffix}", user_id="u1",
            statement="Package and publish to the private package index",
            workspace_id=ws, branch=branch,
        ))


def _eids(resp):
    return sorted(e["experience_id"] for e in resp["experiences"])


def _fids(resp):
    return sorted(f["fact_id"] for f in resp["facts"])


# ================================================================
# The rule itself (pure functions, no database involved)
# ================================================================

class TestScopeRule:
    def test_no_workspace_means_no_filter(self):
        assert _scope_where(None, None) == ("1", [])
        # branch alone has no effect (consistent with the combined semantics of list_memories)
        assert _scope_where(None, "main") == ("1", [])
        assert _scope_where("", "main") == ("1", [])

    def test_workspace_only_is_exact(self):
        assert _scope_where("ws/a", None) == ("workspace_id = ?", ["ws/a"])

    def test_branch_stacks_and_tolerates_null(self):
        sql, params = _scope_where("ws/a", "main")
        assert sql == "workspace_id = ? AND (branch = ? OR branch IS NULL)"
        assert params == ["ws/a", "main"]

    def test_matches_mirrors_where(self):
        # Nothing passed -> everything passes
        assert _scope_matches(None, None, None, None)
        assert _scope_matches("ws/a", "main", None, None)
        # workspace must match exactly; NULL does not count as a hit
        assert _scope_matches("ws/a", None, "ws/a", None)
        assert not _scope_matches("ws/b", None, "ws/a", None)
        assert not _scope_matches(None, None, "ws/a", None)
        # branch tolerates NULL, but not a row recorded on some other branch
        assert _scope_matches("ws/a", "main", "ws/a", "main")
        assert _scope_matches("ws/a", None, "ws/a", "main")
        assert not _scope_matches("ws/a", "dev", "ws/a", "main")

    def test_overfetch_scales_by_selectivity(self):
        # The scope covers everything -> no overfetch
        assert _scope_overfetch(8, 10, 10) == 8
        # The scope covers half -> double the fetch
        assert _scope_overfetch(8, 5, 10) == 16
        # A single row -> scale up to nearly the whole table
        assert _scope_overfetch(8, 1, 21) == 168
        # When selectivity cannot be derived, fall back to the original value instead of inflating
        assert _scope_overfetch(8, 0, 10) == 8

    def test_overfetch_capped_but_never_shrinks(self):
        assert _scope_overfetch(8, 1, 10 ** 6) == 2000
        # When top already exceeds the cap, the cap must not shrink it
        assert _scope_overfetch(5000, 1, 10 ** 6) == 5000


# ================================================================
# Nothing passed = no filter (regression protection)
# ================================================================

@pytest.mark.asyncio
class TestNoScopeUnchanged:
    async def test_both_absent_recalls_everything(self, scope_ctx):
        await _seed_two_workspaces(scope_ctx._coding_store)

        resp = await _search(scope_ctx, "Package and publish to the private package index")
        assert _eids(resp) == ["e_a", "e_b"]
        assert _fids(resp) == ["f_a", "f_b"]

    async def test_branch_alone_is_ignored(self, scope_ctx):
        """branch alone does not filter. As in list_memories: branch only subdivides a workspace."""
        await _seed_two_workspaces(scope_ctx._coding_store)

        resp = await _search(scope_ctx, "Package and publish to the private package index", branch="main")
        assert _eids(resp) == ["e_a", "e_b"]
        assert _fids(resp) == ["f_a", "f_b"]

    async def test_untagged_rows_still_recalled(self, scope_ctx):
        """Experiences/facts with an empty workspace_id are still recalled when no scope is passed."""
        await scope_ctx._coding_store.insert(CodingMemory(
            experience_id="e_bare", user_id="u1", task="unscoped experience", solution="S",
        ))
        await scope_ctx._coding_store.insert_fact(WorkingFact(
            fact_id="f_bare", user_id="u1", statement="unscoped experience",
        ))

        resp = await _search(scope_ctx, "unscoped experience")
        assert _eids(resp) == ["e_bare"]
        assert _fids(resp) == ["f_bare"]


# ================================================================
# workspace_id only
# ================================================================

@pytest.mark.asyncio
class TestWorkspaceOnly:
    async def test_filters_both_layers(self, scope_ctx):
        await _seed_two_workspaces(scope_ctx._coding_store)

        resp = await _search(scope_ctx, "Package and publish to the private package index", workspace_id="ws/a")
        assert _eids(resp) == ["e_a"]
        assert _fids(resp) == ["f_a"]
        assert resp["experiences"][0]["workspace_id"] == "ws/a"

    async def test_ignores_branch_of_in_scope_rows(self, scope_ctx):
        """With only workspace_id, branch plays no role and every branch inside that workspace is recalled."""
        for i, br in enumerate(("main", "dev", None)):
            await scope_ctx._coding_store.insert(CodingMemory(
                experience_id=f"e{i}", user_id="u1", task="the same task", solution="S",
                workspace_id="ws/a", branch=br,
            ))

        resp = await _search(scope_ctx, "the same task", workspace_id="ws/a")
        assert _eids(resp) == ["e0", "e1", "e2"]

    async def test_untagged_rows_excluded(self, scope_ctx):
        """An empty workspace_id = belongs to no repo, so it is excluded when filtering by repo."""
        await scope_ctx._coding_store.insert(CodingMemory(
            experience_id="e_bare", user_id="u1", task="unscoped experience", solution="S",
        ))
        await scope_ctx._coding_store.insert(CodingMemory(
            experience_id="e_a", user_id="u1", task="unscoped experience", solution="S",
            workspace_id="ws/a",
        ))

        resp = await _search(scope_ctx, "unscoped experience", workspace_id="ws/a")
        assert _eids(resp) == ["e_a"]


# ================================================================
# workspace_id + branch
# ================================================================

@pytest.mark.asyncio
class TestWorkspaceAndBranch:
    async def test_branch_narrows_within_workspace(self, scope_ctx):
        for i, (ws, br) in enumerate(
            (("ws/a", "main"), ("ws/a", "dev"), ("ws/b", "main"))
        ):
            await scope_ctx._coding_store.insert(CodingMemory(
                experience_id=f"e{i}", user_id="u1", task="the same task", solution="S",
                workspace_id=ws, branch=br,
            ))
            await scope_ctx._coding_store.insert_fact(WorkingFact(
                fact_id=f"f{i}", user_id="u1", statement="the same task",
                workspace_id=ws, branch=br,
            ))

        resp = await _search(
            scope_ctx, "the same task", workspace_id="ws/a", branch="main",
        )
        assert _eids(resp) == ["e0"]
        assert _fids(resp) == ["f0"]


# ================================================================
# Scope passed but no matching data
# ================================================================

@pytest.mark.asyncio
class TestNoMatchInScope:
    async def test_unknown_workspace_returns_empty(self, scope_ctx):
        await _seed_two_workspaces(scope_ctx._coding_store)

        resp = await _search(scope_ctx, "Package and publish to the private package index", workspace_id="ws/nope")
        assert resp["experiences"] == []
        assert resp["facts"] == []
        assert resp["scene"] == "productivity"

    async def test_unknown_branch_returns_empty(self, scope_ctx):
        await _seed_two_workspaces(scope_ctx._coding_store)

        resp = await _search(
            scope_ctx, "Package and publish to the private package index", workspace_id="ws/a", branch="release",
        )
        assert resp["experiences"] == []
        assert resp["facts"] == []

    async def test_empty_scope_skips_vdb_entirely(self, scope_ctx):
        """With nothing in scope there is no point querying the vector store -- SQLite already has a certain answer."""
        await _seed_two_workspaces(scope_ctx._coding_store)
        store = scope_ctx._coding_store

        calls = []
        for vec in (store._vec, store._fact_vec):
            original = vec.search

            async def _counting(*a, _orig=original, **kw):
                calls.append(1)
                return await _orig(*a, **kw)

            vec.search = _counting

        resp = await _search(scope_ctx, "Package and publish to the private package index", workspace_id="ws/nope")
        assert resp["experiences"] == [] and resp["facts"] == []
        assert calls == []


# ================================================================
# Legacy data: branch is NULL
# ================================================================

@pytest.mark.asyncio
class TestLegacyNullBranch:
    async def test_null_branch_survives_branch_filter(self, scope_ctx):
        """Rows written without a branch must not be killed by accident when filtering by branch.

        The branch column of the fact table was added later, so in old databases the whole column is NULL; new rows
        written without branch are NULL too. Filtering NULL as "recorded on another branch" would leave this data permanently unreachable.
        """
        await scope_ctx._coding_store.insert(CodingMemory(
            experience_id="e_old", user_id="u1", task="legacy experience", solution="S",
            workspace_id="ws/a", branch=None,
        ))
        await scope_ctx._coding_store.insert_fact(WorkingFact(
            fact_id="f_old", user_id="u1", statement="legacy experience",
            workspace_id="ws/a", branch=None,
        ))

        resp = await _search(
            scope_ctx, "legacy experience", workspace_id="ws/a", branch="main",
        )
        assert _eids(resp) == ["e_old"]
        assert _fids(resp) == ["f_old"]

    async def test_other_branch_still_excluded(self, scope_ctx):
        """Tolerating NULL is not giving up on filtering: rows explicitly written to another branch are still excluded."""
        await scope_ctx._coding_store.insert(CodingMemory(
            experience_id="e_dev", user_id="u1", task="legacy experience", solution="S",
            workspace_id="ws/a", branch="dev",
        ))
        await scope_ctx._coding_store.insert(CodingMemory(
            experience_id="e_null", user_id="u1", task="legacy experience", solution="S",
            workspace_id="ws/a", branch=None,
        ))

        resp = await _search(
            scope_ctx, "legacy experience", workspace_id="ws/a", branch="main",
        )
        assert _eids(resp) == ["e_null"]


# Databases created before the fact table gained the branch column
_PRE_BRANCH_FACT_DDL = """
CREATE TABLE IF NOT EXISTS working_fact (
    fact_id                TEXT PRIMARY KEY,
    user_id                TEXT NOT NULL,
    agent_id               TEXT NOT NULL DEFAULT 'default_agent',
    session_id             TEXT,
    workspace_id           TEXT,
    statement              TEXT NOT NULL,
    files                  TEXT NOT NULL DEFAULT '[]',
    fact_link              TEXT,
    derived_experience_ids TEXT NOT NULL DEFAULT '[]',
    status                 TEXT NOT NULL DEFAULT 'active',
    fact_chain_summary     TEXT NOT NULL DEFAULT '',
    created_at             TEXT NOT NULL
);
"""


@pytest.mark.asyncio
class TestPreBranchFactSchema:
    async def test_add_column_and_old_rows_stay_reachable(
        self, coding_config, mock_embed
    ):
        """Adding branch via ADD COLUMN to an old fact table; the resulting NULL rows must stay filterable."""
        db_path = coding_config.coding.db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.executescript(_PRE_BRANCH_FACT_DDL)
        conn.execute(
            "INSERT INTO working_fact VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            ("f_old", "u1", "a1", "s1", "ws/a", "legacy conclusion", "[]", None, "[]",
             "active", "", "2026-01-01T00:00:00"),
        )
        conn.commit()
        conn.close()

        store = CodingMemoryStore(coding_config, embed_service=mock_embed)
        await store.initialize()
        try:
            cols = [
                r["name"] for r in
                store._conn.execute("PRAGMA table_info(working_fact)").fetchall()
            ]
            assert "branch" in cols

            got = await store.get_fact("f_old", user_id="u1")
            assert got is not None and got.branch is None

            # Old databases do have the vector (both sides are persisted together on write), but the raw SQL above
            # only built the SQLite half. search_facts starts from the vector store, so without this step the recall is empty and the scope rule is never exercised.
            await store._upsert_fact_vec(got)

            emb = await mock_embed.embed("legacy conclusion")
            hits = await store.search_facts(
                emb, user_id="u1", workspace_id="ws/a", branch="main",
            )
            assert [f.fact_id for f, _ in hits] == ["f_old"]

            # After the column is added, branch can still be written normally
            await store.insert_fact(WorkingFact(
                fact_id="f_new", user_id="u1", statement="new conclusion",
                workspace_id="ws/a", branch="main",
            ))
            back = await store.get_fact("f_new", user_id="u1")
            assert back.branch == "main"
        finally:
            await store.close()


# ================================================================
# Fact layer filtered empty -> the two layers degrade to a single one
# ================================================================

@pytest.mark.asyncio
class TestFactLinkDegradation:
    async def _seed_linked(self, store):
        """An experience whose task looks nothing like the query; only the fact link reaches it."""
        await store.insert(CodingMemory(
            experience_id="e1", user_id="u1", task="totally unrelated task name QQQQ",
            solution="S", workspace_id="ws/a", branch="main",
        ))
        await store.insert_fact(WorkingFact(
            fact_id="f1", user_id="u1", statement="new_file_search lives in src/search.py",
            workspace_id="ws/a", branch="main",
            derived_experience_ids=["e1"],
        ))

    async def test_in_scope_keeps_fact_link_path(self, scope_ctx):
        await self._seed_linked(scope_ctx._coding_store)

        resp = await _search(
            scope_ctx, "new_file_search lives in src/search.py",
            workspace_id="ws/a", branch="main",
        )
        assert _fids(resp) == ["f1"]
        assert _eids(resp) == ["e1"]
        assert resp["experiences"][0]["via"] == "fact_link"

    async def test_out_of_scope_collapses_to_single_layer(self, scope_ctx, caplog):
        """When the fact is filtered out of scope, the experience linked through it disappears too, and the degradation must warn.

        The experience itself is also in ws/a, so both layers come back empty here; the degradation is not
        visible in the result, only in the logs.
        """
        await self._seed_linked(scope_ctx._coding_store)

        with caplog.at_level("WARNING", logger="hy_memory.client"):
            resp = await _search(
                scope_ctx, "new_file_search lives in src/search.py", workspace_id="ws/b",
            )
        assert resp["facts"] == []
        assert resp["experiences"] == []
        assert any("fact_link path disabled" in r.message for r in caplog.records)

    async def test_experience_survives_when_only_fact_is_out_of_scope(self, scope_ctx):
        """The experience is in scope but the fact backing it is not, so the experience falls back to the direct path."""
        store = scope_ctx._coding_store
        await store.insert(CodingMemory(
            experience_id="e1", user_id="u1", task="shared query term", solution="S",
            workspace_id="ws/a",
        ))
        await store.insert_fact(WorkingFact(
            fact_id="f1", user_id="u1", statement="shared query term",
            workspace_id="ws/b", derived_experience_ids=["e1"],
        ))

        resp = await _search(scope_ctx, "shared query term", workspace_id="ws/a")
        assert resp["facts"] == []
        assert _eids(resp) == ["e1"]
        assert resp["experiences"][0]["via"] == "direct"


# ================================================================
# Filtering must not eat recall
# ================================================================

@pytest.mark.asyncio
class TestScopeDoesNotEatRecall:
    async def test_in_scope_hit_found_behind_many_out_of_scope_ones(self, scope_ctx):
        """The one in-scope row sits behind 20 more similar out-of-scope experiences and must still be recalled.

        Vector search takes top-N and returns: filtering the top-N as returned means this row never makes the quota,
        which is the same as zeroing out recall. Only scaling the fetch by selectivity can retrieve it.
        """
        store = scope_ctx._coding_store
        query = "Package and publish to the private package index"
        for i in range(20):
            await store.insert(CodingMemory(
                experience_id=f"e_other{i}", user_id="u1",
                task=query, solution="S", workspace_id="ws/other",
            ))
        # The in-scope row has different wording, so by score alone it ranks behind the 20 exact hits
        await store.insert(CodingMemory(
            experience_id="e_mine", user_id="u1",
            task="where credentials go in the publish flow", solution="S", workspace_id="ws/mine",
        ))

        resp = await _search(scope_ctx, query, workspace_id="ws/mine", limit=1)
        assert _eids(resp) == ["e_mine"]


# ================================================================
# list and search share the same rule
# ================================================================

@pytest.mark.asyncio
class TestListSharesTheSameRule:
    async def test_list_branch_tolerates_null(self, coding_store):
        for i, br in enumerate(("main", "dev", None)):
            await coding_store.insert(CodingMemory(
                experience_id=f"e{i}", user_id="u1", task=f"task {i}", solution="S",
                workspace_id="ws/a", branch=br,
            ))

        got = await coding_store.list_by_workspace("u1", "ws/a", branch="main")
        assert sorted(m.experience_id for m in got) == ["e0", "e2"]

        got = await coding_store.list_by_workspace("u1", "ws/a")
        assert sorted(m.experience_id for m in got) == ["e0", "e1", "e2"]
