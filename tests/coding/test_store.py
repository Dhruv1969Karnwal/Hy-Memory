# -*- coding: utf-8 -*-
"""CodingMemoryStore tests."""

import json
import os
import sqlite3
from datetime import datetime

import pytest

from hy_memory.coding.store import _vec_node_id, _derive_coding_collection_name
from hy_memory.coding.types import CodingMemory, WorkingFact


# ================================================================
# Helpers
# ================================================================

class TestHelpers:
    def test_vec_node_id_format(self):
        assert _vec_node_id("e1", 0) == "e1:0"
        assert _vec_node_id("e1", 5) == "e1:5"

    def test_derive_collection_name(self):
        assert _derive_coding_collection_name("agent_memories") == "agent_memories_coding_keys"
        assert _derive_coding_collection_name(None) == "agent_memories_coding_keys"
        assert _derive_coding_collection_name("custom") == "custom_coding_keys"


# ================================================================
# Store e2e (real chroma + mock embed)
# ================================================================

@pytest.mark.asyncio
class TestStoreCRUD:
    async def test_insert_get(self, coding_store):
        m = CodingMemory(
            experience_id="e1", user_id="u1", agent_id="a1",
            task="Packaging and publishing", search_keys=["credentials location"], solution="...",
            boundary_envs="twine>=6", workspace_id="ws/a",
            files=["s.py"], confidence=0.9,
        )
        await coding_store.insert(m)
        got = await coding_store.get_by_id("e1", user_id="u1")
        assert got is not None
        assert got.task == "Packaging and publishing" and got.search_keys == ["credentials location"]
        assert got.boundary_envs == "twine>=6"
        assert got.files == ["s.py"]

    async def test_get_by_id_wrong_user(self, coding_store):
        m = CodingMemory(experience_id="e1", user_id="u1", task="T", solution="S")
        await coding_store.insert(m)
        # A wrong user_id must not retrieve anything
        got = await coding_store.get_by_id("e1", user_id="u_wrong")
        assert got is None

    async def test_list_user_tasks(self, coding_store):
        await coding_store.insert(CodingMemory(experience_id="e1", user_id="u1",
                                               task="T1", solution="S"))
        await coding_store.insert(CodingMemory(experience_id="e2", user_id="u1",
                                               task="T2", solution="S"))
        await coding_store.insert(CodingMemory(experience_id="e3", user_id="u_other",
                                               task="OTHER", solution="S"))
        tasks = await coding_store.list_user_tasks("u1", limit=10)
        assert sorted(tasks) == ["T1", "T2"]
        assert "OTHER" not in tasks

    async def test_update_overrides(self, coding_store, mock_embed):
        m = CodingMemory(experience_id="e1", user_id="u1", task="T",
                         search_keys=["old1", "old2"], solution="old")
        await coding_store.insert(m)

        # The old aspect should still match
        emb = (await mock_embed.embed_batch(["old1"]))[0]
        hits = await coding_store.search_by_query_embedding(emb, user_id="u1", top=5)
        assert any(h["experience_id"] == "e1" for h in hits)

        # update
        m.search_keys = ["new1"]
        m.solution = "new"
        await coding_store.update(m)

        # The new aspect should match
        emb_new = (await mock_embed.embed_batch(["new1"]))[0]
        hits = await coding_store.search_by_query_embedding(emb_new, user_id="u1", top=5)
        assert any(h["experience_id"] == "e1" for h in hits)

        # SQLite side is correct
        got = await coding_store.get_by_id("e1", user_id="u1")
        assert got.solution == "new"
        assert got.search_keys == ["new1"]

    async def test_delete(self, coding_store):
        m = CodingMemory(experience_id="e1", user_id="u1", task="T",
                         search_keys=["k"], solution="S")
        await coding_store.insert(m)
        assert await coding_store.delete("e1", user_id="u1") is True
        assert await coding_store.get_by_id("e1", user_id="u1") is None
        # Deleting a missing row returns False
        assert await coding_store.delete("nope", user_id="u1") is False

    async def test_delete_wrong_user(self, coding_store):
        m = CodingMemory(experience_id="e1", user_id="u1", task="T", solution="S")
        await coding_store.insert(m)
        # A wrong user_id must not delete anything
        deleted = await coding_store.delete("e1", user_id="u_wrong")
        assert deleted is False
        # The row is still retrievable
        got = await coding_store.get_by_id("e1", user_id="u1")
        assert got is not None


@pytest.mark.asyncio
class TestStoreSearch:
    async def test_search_hit_task_key(self, coding_store, mock_embed):
        m = CodingMemory(experience_id="e1", user_id="u1", task="Packaging and publishing",
                         search_keys=["credentials location", "upload endpoint"], solution="S",
                         workspace_id="ws/a")
        await coding_store.insert(m)
        emb = (await mock_embed.embed_batch(["Packaging and publishing"]))[0]
        hits = await coding_store.search_by_query_embedding(
            emb, user_id="u1", workspace_id="ws/a", top=10,
        )
        assert len(hits) == 1
        assert hits[0]["experience_id"] == "e1"
        assert hits[0]["matched_key_kind"] == "task"

    async def test_search_hit_aspect_key(self, coding_store, mock_embed):
        m = CodingMemory(experience_id="e1", user_id="u1", task="Packaging and publishing",
                         search_keys=["credentials location"], solution="S")
        await coding_store.insert(m)
        emb = (await mock_embed.embed_batch(["credentials location"]))[0]
        hits = await coding_store.search_by_query_embedding(emb, user_id="u1", top=10)
        assert hits[0]["matched_key_kind"] == "aspect"
        assert hits[0]["matched_key_text"] == "credentials location"

    async def test_search_filters_by_workspace(self, coding_store, mock_embed):
        """When workspace is passed it filters by scope; when omitted it recalls everything."""
        m = CodingMemory(experience_id="e1", user_id="u1", task="T", solution="S",
                         workspace_id="ws/a")
        await coding_store.insert(m)
        emb = (await mock_embed.embed_batch(["T"]))[0]
        for ws, expect in (("ws/a", 1), ("ws/wrong", 0), (None, 1)):
            hits = await coding_store.search_by_query_embedding(
                emb, user_id="u1", workspace_id=ws, top=5,
            )
            assert len(hits) == expect, f"workspace_id={ws!r}"

    async def test_search_isolates_by_user(self, coding_store, mock_embed):
        await coding_store.insert(CodingMemory(experience_id="e1", user_id="u1",
                                               task="T", solution="S"))
        await coding_store.insert(CodingMemory(experience_id="e2", user_id="u_other",
                                               task="T", solution="S"))
        emb = (await mock_embed.embed_batch(["T"]))[0]
        hits = await coding_store.search_by_query_embedding(emb, user_id="u1", top=10)
        assert [h["experience_id"] for h in hits] == ["e1"]

    async def test_search_dedup_by_experience_id(self, coding_store, mock_embed):
        # Several keys of the same experience may all match; dedup must keep at most one row per experience
        m = CodingMemory(experience_id="e1", user_id="u1", task="aaa",
                         search_keys=["aaa", "aaa"], solution="S")
        await coding_store.insert(m)
        emb = (await mock_embed.embed_batch(["aaa"]))[0]
        hits = await coding_store.search_by_query_embedding(emb, user_id="u1", top=10)
        unique = {h["experience_id"] for h in hits}
        assert len(hits) == len(unique)
        assert len(hits) <= 1


# ================================================================
# Legacy schema migration
# ================================================================

_LEGACY_DDL = """
CREATE TABLE coding_memory_meta (
    memory_id        TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    agent_id         TEXT NOT NULL DEFAULT 'default_agent',
    task             TEXT NOT NULL,
    search_keys      TEXT NOT NULL DEFAULT '[]',
    solution         TEXT NOT NULL,
    boundary_envs    TEXT NOT NULL DEFAULT '',
    boundary_scope   TEXT NOT NULL,
    workspace_id     TEXT,
    branch           TEXT,
    session_id       TEXT,
    files            TEXT NOT NULL DEFAULT '[]',
    confidence       REAL NOT NULL DEFAULT 0.7,
    source           TEXT NOT NULL DEFAULT 'auto_extract',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
"""


@pytest.mark.asyncio
class TestLegacySchemaMigration:
    async def test_migrates_and_keeps_rows(self, coding_config, mock_embed):
        from hy_memory.coding.store import CodingMemoryStore

        db_path = coding_config.coding.db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.executescript(_LEGACY_DDL)
        conn.execute("CREATE INDEX idx_coding_user ON coding_memory_meta(user_id)")
        conn.execute(
            "INSERT INTO coding_memory_meta VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("m_old", "u1", "a1", "old task", json.dumps(["k1"]), "old solution",
             "twine>=6", "project", "ws/a", "main", "s1", json.dumps(["f.py"]),
             0.9, "auto_extract", "2026-01-01T00:00:00", "2026-01-02T00:00:00"),
        )
        conn.commit()
        conn.close()

        store = CodingMemoryStore(coding_config, embed_service=mock_embed)
        await store.initialize()
        try:
            got = await store.get_by_id("m_old", user_id="u1")
            assert got is not None
            # The id keeps its old value, so the VDB node_id is unaffected
            assert got.experience_id == "m_old"
            assert got.task == "old task"
            assert got.search_keys == ["k1"]
            assert got.boundary_envs == "twine>=6"
            assert got.files == ["f.py"]
            # Writes still work after migration
            await store.insert(CodingMemory(experience_id="e_new", user_id="u1",
                                            task="new task", solution="S"))
            assert sorted(await store.list_user_tasks("u1")) == ["new task", "old task"]
        finally:
            await store.close()

    async def test_migration_is_idempotent(self, coding_config, mock_embed):
        from hy_memory.coding.store import CodingMemoryStore

        for _ in range(2):
            store = CodingMemoryStore(coding_config, embed_service=mock_embed)
            await store.initialize()
            await store.close()

        conn = sqlite3.connect(coding_config.coding.db_path)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(coding_memory_meta)")]
        conn.close()
        assert "experience_id" in cols
        assert "memory_id" not in cols
        assert "boundary_scope" not in cols

    async def test_legacy_rows_get_empty_source_fact_ids(self, coding_config, mock_embed):
        """A legacy DB has no source_fact_ids column; after migration it should hold an empty chain instead of blowing up."""
        from hy_memory.coding.store import CodingMemoryStore

        db_path = coding_config.coding.db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.executescript(_LEGACY_DDL)
        conn.execute(
            "INSERT INTO coding_memory_meta VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("m_old", "u1", "a1", "old task", "[]", "old solution",
             "", "project", "ws/a", "main", "s1", "[]",
             0.9, "auto_extract", "2026-01-01T00:00:00", "2026-01-02T00:00:00"),
        )
        conn.commit()
        conn.close()

        store = CodingMemoryStore(coding_config, embed_service=mock_embed)
        await store.initialize()
        try:
            got = await store.get_by_id("m_old", user_id="u1")
            assert got.source_fact_ids == []
            await store.set_experience_source_facts("m_old", ["f1"])
            got = await store.get_by_id("m_old", user_id="u1")
            assert got.source_fact_ids == ["f1"]
        finally:
            await store.close()

    async def test_adds_source_fact_ids_to_renamed_schema(self, coding_config, mock_embed):
        """A DB that was already renamed but predates source_fact_ids gets the column added via ADD COLUMN."""
        from hy_memory.coding.store import CodingMemoryStore

        db_path = coding_config.coding.db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.executescript(_PRE_FACT_DDL)
        conn.execute(
            "INSERT INTO coding_memory_meta VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("e1", "u1", "a1", "task", "[]", "solution", "", "ws/a", "main", "s1",
             "[]", 0.9, "auto_extract", "2026-01-01T00:00:00", "2026-01-02T00:00:00"),
        )
        conn.commit()
        conn.close()

        store = CodingMemoryStore(coding_config, embed_service=mock_embed)
        await store.initialize()
        try:
            got = await store.get_by_id("e1", user_id="u1")
            assert got.source_fact_ids == []
        finally:
            await store.close()

    async def test_adds_chain_summary_to_pre_chain_schema(self, coding_config, mock_embed):
        """A DB created before the chain summary existed gets the column added via ADD COLUMN, and legacy facts get an empty summary."""
        from hy_memory.coding.store import CodingMemoryStore
        from hy_memory.coding.types import WorkingFact

        db_path = coding_config.coding.db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.executescript(_PRE_CHAIN_FACT_DDL)
        conn.execute(
            "INSERT INTO working_fact VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("f_old", "u1", "a1", "s1", "ws/a", "old conclusion", "[]", None, "[]",
             "active", "2026-01-01T00:00:00"),
        )
        conn.commit()
        conn.close()

        store = CodingMemoryStore(coding_config, embed_service=mock_embed)
        await store.initialize()
        try:
            got = await store.get_fact("f_old", user_id="u1")
            assert got is not None
            assert got.fact_chain_summary == ""
            # Summaries can be written normally once the column exists
            await store.insert_fact(WorkingFact(
                fact_id="f_new", user_id="u1", statement="new conclusion",
                fact_chain_summary="Changed from the old conclusion to the new conclusion.",
            ))
            back = await store.get_fact("f_new", user_id="u1")
            assert back.fact_chain_summary == "Changed from the old conclusion to the new conclusion."
        finally:
            await store.close()

    async def test_writes_ignore_abandoned_fact_chain_column(
        self, coding_config, mock_embed
    ):
        """The short-lived fact_chain column is abandoned, but old DBs still carry it and it is NOT NULL.

        The current INSERT omits that column and relies on its DEFAULT to cover it; without that
        default every write would fail outright with NOT NULL constraint failed.
        """
        from hy_memory.coding.store import CodingMemoryStore
        from hy_memory.coding.types import WorkingFact

        db_path = coding_config.coding.db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.executescript(_ABANDONED_CHAIN_FACT_DDL)
        conn.commit()
        conn.close()

        store = CodingMemoryStore(coding_config, embed_service=mock_embed)
        await store.initialize()
        try:
            await store.insert_fact(WorkingFact(
                fact_id="f1", user_id="u1", statement="new conclusion",
                fact_chain_summary="A one-sentence summary.",
            ))
            got = await store.get_fact("f1", user_id="u1")
            assert got.statement == "new conclusion"
            assert got.fact_chain_summary == "A one-sentence summary."
            # upsert takes a different SQL branch and must not be tripped by that column either
            await store.supersede_fact("f1", "f2")
            assert (await store.get_fact("f1", user_id="u1")).status == "superseded"
        finally:
            await store.close()


# Fact table from before the chain summary existed
_PRE_CHAIN_FACT_DDL = """
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
    created_at             TEXT NOT NULL
);
"""


# Fact table version that briefly had a fact_chain column (now abandoned, still present in old DBs)
_ABANDONED_CHAIN_FACT_DDL = """
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
    fact_chain             TEXT NOT NULL DEFAULT '[]',
    fact_chain_summary     TEXT NOT NULL DEFAULT '',
    created_at             TEXT NOT NULL
);
"""


# ================================================================
# Fact layer storage
# ================================================================

_PRE_FACT_DDL = """
CREATE TABLE IF NOT EXISTS coding_memory_meta (
    experience_id    TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    agent_id         TEXT NOT NULL DEFAULT 'default_agent',
    task             TEXT NOT NULL,
    search_keys      TEXT NOT NULL DEFAULT '[]',
    solution         TEXT NOT NULL,
    boundary_envs    TEXT NOT NULL DEFAULT '',
    workspace_id     TEXT,
    branch           TEXT,
    session_id       TEXT,
    files            TEXT NOT NULL DEFAULT '[]',
    confidence       REAL NOT NULL DEFAULT 0.7,
    source           TEXT NOT NULL DEFAULT 'auto_extract',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
"""


class TestFactCollectionName:
    def test_derive(self):
        from hy_memory.coding.store import _derive_fact_collection_name
        assert _derive_fact_collection_name("agent_memories") == "agent_memories_working_facts"
        assert _derive_fact_collection_name(None) == "agent_memories_working_facts"

    def test_separate_from_experience_collection(self):
        """Each layer has its own collection; the names must not collide."""
        from hy_memory.coding.store import (
            _derive_coding_collection_name, _derive_fact_collection_name,
        )
        assert _derive_coding_collection_name("x") != _derive_fact_collection_name("x")


@pytest.mark.asyncio
class TestFactStore:
    async def test_insert_and_get(self, coding_store):
        await coding_store.insert_fact(WorkingFact(
            fact_id="f1", user_id="u1", agent_id="a1", session_id="s1",
            workspace_id="ws/a", statement="fact body text", files=["a.py"],
            derived_experience_ids=["e1"],
        ))
        got = await coding_store.get_fact("f1", user_id="u1")
        assert got.statement == "fact body text"
        assert got.files == ["a.py"]
        assert got.derived_experience_ids == ["e1"]
        assert got.status == "active"
        assert got.fact_link is None
        # created_at is filled in automatically
        assert got.created_at is not None

    async def test_get_fact_user_isolation(self, coding_store):
        await coding_store.insert_fact(WorkingFact(
            fact_id="f1", user_id="u1", statement="S",
        ))
        assert await coding_store.get_fact("f1", user_id="u_other") is None

    async def test_get_facts_preserves_order(self, coding_store):
        for i in range(3):
            await coding_store.insert_fact(WorkingFact(
                fact_id=f"f{i}", user_id="u1", statement=f"fact {i}",
            ))
        got = await coding_store.get_facts(["f2", "f0", "f_missing"], user_id="u1")
        assert [f.fact_id for f in got] == ["f2", "f0"]

    async def test_search_facts_scores_and_isolation(self, coding_store):
        await coding_store.insert_fact(WorkingFact(
            fact_id="f1", user_id="u1", statement="Packaging and publishing to a private package index",
        ))
        await coding_store.insert_fact(WorkingFact(
            fact_id="f2", user_id="u_other", statement="Packaging and publishing to a private package index",
        ))
        emb = await coding_store.embed_service.embed("Packaging and publishing to a private package index")

        hits = await coding_store.search_facts(emb, user_id="u1", top=5)
        assert [f.fact_id for f, _ in hits] == ["f1"]
        assert hits[0][1] > 0.9

    async def test_supersede_soft_deletes(self, coding_store):
        await coding_store.insert_fact(WorkingFact(
            fact_id="f_old", user_id="u1", statement="old conclusion",
        ))
        assert await coding_store.supersede_fact("f_old", "f_new") is True

        got = await coding_store.get_fact("f_old", user_id="u1")
        # The row remains, so the chain stays traceable
        assert got.status == "superseded"
        assert got.fact_link == "f_new"
        # But it no longer occupies a recall slot
        emb = await coding_store.embed_service.embed("old conclusion")
        assert await coding_store.search_facts(emb, user_id="u1") == []

    async def test_search_respects_top(self, coding_store):
        for i in range(6):
            await coding_store.insert_fact(WorkingFact(
                fact_id=f"f{i}", user_id="u1", statement=f"fact {i}",
            ))
        emb = await coding_store.embed_service.embed("fact 0")
        assert len(await coding_store.search_facts(emb, user_id="u1", top=3)) == 3

    async def test_orphan_index_rows_dont_starve_recall(self, coding_store):
        """Orphans -- vector row still there, SQLite row gone -- must not consume candidate slots.

        The two sides are not written in one transaction, so a crash in between leaves these rows
        behind; with sampling, retrieval comes back empty whenever the first top rows are all orphans.
        """
        await coding_store.insert_fact(WorkingFact(
            fact_id="f_real", user_id="u1", statement="a real fact",
        ))
        # Build 5 orphans: still in the VDB, deleted from SQLite
        for i in range(5):
            await coding_store.insert_fact(WorkingFact(
                fact_id=f"f_orphan{i}", user_id="u1", statement="a real fact",
            ))
        with coding_store._lock:
            coding_store._conn.execute(
                "DELETE FROM working_fact WHERE fact_id LIKE 'f_orphan%'"
            )

        emb = await coding_store.embed_service.embed("a real fact")
        hits = await coding_store.search_facts(emb, user_id="u1", top=5)
        assert [f.fact_id for f, _ in hits] == ["f_real"]

    async def test_supersede_twice_is_noop(self, coding_store):
        await coding_store.insert_fact(WorkingFact(
            fact_id="f_old", user_id="u1", statement="old conclusion",
        ))
        assert await coding_store.supersede_fact("f_old", "f_a") is True
        assert await coding_store.supersede_fact("f_old", "f_b") is False
        got = await coding_store.get_fact("f_old", user_id="u1")
        assert got.fact_link == "f_a"

    async def test_supersede_missing_fact(self, coding_store):
        assert await coding_store.supersede_fact("nope", "f_new") is False


@pytest.mark.asyncio
class TestFactAncestors:
    """History is not stored separately; it is rebuilt by walking fact_link backwards."""

    async def _chain(self, store, n, *, user_id="u1"):
        """Build n generations: f0 superseded by f1, f1 by f2 ... returns ids with the newest last."""
        ids = []
        for i in range(n):
            fid = f"f{i}"
            await store.insert_fact(WorkingFact(
                fact_id=fid, user_id=user_id, statement=f"generation {i}",
                created_at=datetime(2026, 1, i + 1),
            ))
            if ids:
                await store.supersede_fact(ids[-1], fid)
            ids.append(fid)
        return ids

    async def test_walks_back_in_time_order(self, coding_store):
        ids = await self._chain(coding_store, 4)
        got = await coding_store.get_fact_ancestors(ids[-1], user_id="u1")
        assert [f.fact_id for f in got] == ["f0", "f1", "f2"], "chronological order, excluding itself"
        assert [f.statement for f in got] == ["generation 0", "generation 1", "generation 2"]

    async def test_no_ancestors_for_fresh_fact(self, coding_store):
        await coding_store.insert_fact(WorkingFact(
            fact_id="f_solo", user_id="u1", statement="a brand new conclusion",
        ))
        assert await coding_store.get_fact_ancestors("f_solo", user_id="u1") == []

    async def test_respects_max_depth(self, coding_store):
        """A long-lived fact must not drag its entire chain into the prompt."""
        ids = await self._chain(coding_store, 6)
        got = await coding_store.get_fact_ancestors(
            ids[-1], user_id="u1", max_depth=2
        )
        assert [f.fact_id for f in got] == ["f3", "f4"], "truncation keeps the most recent generations"

    async def test_scoped_by_user(self, coding_store):
        """Someone else's facts must not show up in my lineage."""
        await coding_store.insert_fact(WorkingFact(
            fact_id="f_theirs", user_id="u2", statement="someone else's old conclusion",
        ))
        await coding_store.insert_fact(WorkingFact(
            fact_id="f_mine", user_id="u1", statement="my conclusion",
        ))
        await coding_store.supersede_fact("f_theirs", "f_mine")
        assert await coding_store.get_fact_ancestors("f_mine", user_id="u1") == []

    async def test_cycle_does_not_hang(self, coding_store):
        """A fact_link cycle should never happen, and must not spin forever if it does."""
        for fid in ("f_a", "f_b"):
            await coding_store.insert_fact(WorkingFact(
                fact_id=fid, user_id="u1", statement=fid,
            ))
        await coding_store.supersede_fact("f_a", "f_b")
        await coding_store.supersede_fact("f_b", "f_a")

        got = await coding_store.get_fact_ancestors("f_a", user_id="u1")
        assert [f.fact_id for f in got] == ["f_b"], "stops once it hits the starting point again"

    async def test_set_derived_experiences(self, coding_store):
        await coding_store.insert_fact(WorkingFact(
            fact_id="f1", user_id="u1", statement="S",
        ))
        await coding_store.set_fact_derived_experiences("f1", ["e1", "e2"])
        got = await coding_store.get_fact("f1", user_id="u1")
        assert got.derived_experience_ids == ["e1", "e2"]

    async def test_experience_and_fact_dont_cross_contaminate(self, coding_store):
        """Each layer has its own collection: fact search must not return experiences, and vice versa."""
        await coding_store.insert(CodingMemory(
            experience_id="e1", user_id="u1", task="the same text", solution="S",
        ))
        await coding_store.insert_fact(WorkingFact(
            fact_id="f1", user_id="u1", statement="the same text",
        ))
        emb = await coding_store.embed_service.embed("the same text")

        fact_hits = await coding_store.search_facts(emb, user_id="u1", top=10)
        assert [f.fact_id for f, _ in fact_hits] == ["f1"]

        exp_hits = await coding_store.search_by_query_embedding(emb, user_id="u1", top=10)
        assert [h["experience_id"] for h in exp_hits] == ["e1"]
