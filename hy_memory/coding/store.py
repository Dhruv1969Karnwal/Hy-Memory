# -*- coding: utf-8 -*-
"""
Coding Memory Store - two-layer physical storage

- SQLite (coding_memory.db)        — coding_memory_meta table, one row per memory
- VDB (separate collection)              — one vector row per search key, isolated by the _vec_collection name
                                       the separate collection name is derived from the chat collection

Each CodingMemory occupies 1 + N rows in the VDB (task + N search_keys).
SQLite holds 1 row (with a search_keys JSON field, so an update knows how many key rows exist).

Write flow:
  1. SQLite upsert
  2. embed_batch(task + search_keys)
  3. VDB upsert_batch (one MemoryNode per row; the custom field carries coding_experience_id and so on)

Update flow (overwrite):
  1. Read the old search_keys length → delete the old keys' VDB node_ids
  2. SQLite update
  3. Repeat write steps 2-3

Delete flow:
  1. Read the search_keys length → delete all VDB node_ids
  2. SQLite delete

See docs/scene_coding_usage.md for the public usage.
"""

import concurrent.futures
import functools
import json
import logging
import math
import os
import sqlite3
import threading
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from ..config import MemoryConfig
from ..core.embed_service import EmbedService
from ..data.vector_store import create_vector_store
from ..data.vector_store_base import VectorStoreBase
from ..models.memory import MemoryLayer, MemoryNode, MemoryStatus
from ..utils.async_executor import run_in_executor_guarded
from .types import CodingMemory, WorkingFact

logger = logging.getLogger(__name__)


# ================================================================
# SQLite thread pool
# ================================================================

_CODING_SQLITE_POOL_SIZE = 16
_coding_sqlite_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=_CODING_SQLITE_POOL_SIZE,
    thread_name_prefix="coding-sqlite",
)


async def _run_in_pool(func, *args, **kwargs):
    call = functools.partial(func, *args, **kwargs) if args or kwargs else func
    return await run_in_executor_guarded(_coding_sqlite_executor, call)


# ================================================================
# DDL
# ================================================================

_CREATE_TABLE_SQL = """
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
    source_fact_ids  TEXT NOT NULL DEFAULT '[]',
    confidence       REAL NOT NULL DEFAULT 0.7,
    source           TEXT NOT NULL DEFAULT 'default',
    extraction_source TEXT NOT NULL DEFAULT 'auto_extract',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
"""

# Fact layer (episodic). Same database as the experience table but a different one; no TTL, freshness comes from SUPERSEDE soft deletes.
_CREATE_FACT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS working_fact (
    fact_id                TEXT PRIMARY KEY,
    user_id                TEXT NOT NULL,
    agent_id               TEXT NOT NULL DEFAULT 'default_agent',
    session_id             TEXT,
    workspace_id           TEXT,
    branch                 TEXT,
    source                 TEXT NOT NULL DEFAULT 'default',
    statement              TEXT NOT NULL,
    files                  TEXT NOT NULL DEFAULT '[]',
    subject                TEXT NOT NULL DEFAULT '',
    details                TEXT NOT NULL DEFAULT '',
    fact_link              TEXT,
    derived_experience_ids TEXT NOT NULL DEFAULT '[]',
    status                 TEXT NOT NULL DEFAULT 'active',
    fact_chain_summary     TEXT NOT NULL DEFAULT '',
    created_at             TEXT NOT NULL
);
-- Looking up predecessors (whose fact_link points at me) uses this index; writing the evolution summary walks the chain backwards
CREATE INDEX IF NOT EXISTS idx_working_fact_link ON working_fact(fact_link);
"""

_CREATE_TRACE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS coding_trace (
    request_id    TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    agent_id      TEXT NOT NULL DEFAULT 'default_agent',
    session_id    TEXT,
    workspace_id  TEXT,
    branch        TEXT,
    source        TEXT NOT NULL DEFAULT 'default',
    messages_json TEXT NOT NULL DEFAULT '[]',
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trace_user_created
    ON coding_trace(user_id, agent_id, source, created_at DESC);
"""

_INDEX_NAMES = ("idx_coding_user", "idx_coding_user_workspace", "idx_coding_updated")

_CREATE_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_coding_user ON coding_memory_meta(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_coding_user_workspace ON coding_memory_meta(user_id, workspace_id);",
    "CREATE INDEX IF NOT EXISTS idx_coding_user_source ON coding_memory_meta(user_id, source);",
    "CREATE INDEX IF NOT EXISTS idx_coding_updated ON coding_memory_meta(updated_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_fact_user ON working_fact(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_fact_user_status ON working_fact(user_id, status);",
    "CREATE INDEX IF NOT EXISTS idx_fact_user_workspace ON working_fact(user_id, workspace_id);",
    "CREATE INDEX IF NOT EXISTS idx_fact_user_source ON working_fact(user_id, source);",
    "CREATE INDEX IF NOT EXISTS idx_fact_created ON working_fact(created_at DESC);",
]

# Legacy schema: the primary key was named memory_id, plus a boundary_scope column that is no longer used.
_MIGRATE_COPY_SQL = """
INSERT INTO coding_memory_meta (
    experience_id, user_id, agent_id, task, search_keys, solution,
    boundary_envs, workspace_id, branch, session_id,
    files, confidence, source, created_at, updated_at
)
SELECT
    memory_id, user_id, agent_id, task, search_keys, solution,
    boundary_envs, workspace_id, branch, session_id,
    files, confidence, source, created_at, updated_at
FROM coding_memory_meta_legacy;
"""

# Columns added later: legacy databases get them via ADD COLUMN, no table rebuild needed
_ADDED_COLUMNS = {
    "coding_memory_meta": {
        "source_fact_ids": "TEXT NOT NULL DEFAULT '[]'",
        "extraction_source": "TEXT NOT NULL DEFAULT 'auto_extract'",
    },
    # Early versions stored a fact_chain column (mechanically accumulated wordings over time), now obsolete: the same information
    # is available by looking it up backwards along fact_link. Legacy databases keep that column untouched; the default keeps inserts working.
    "working_fact": {
        "fact_chain_summary": "TEXT NOT NULL DEFAULT ''",
        # The fact layer got this column later than the experience layer, so rows already in legacy databases are NULL. Scope filtering
        # treats "NULL = branch not recorded", not "a different branch recorded"; see _scope_where.
        "branch": "TEXT",
        # Hard anchor for the override gate, plus the addressable source text paired with it. Legacy rows hold empty strings: an empty subject
        # falls back to pure semantics plus an LLM override decision (the behavior before this change), and empty details leaves only the summary readable.
        "subject": "TEXT NOT NULL DEFAULT ''",
        "details": "TEXT NOT NULL DEFAULT ''",
        "source": "TEXT NOT NULL DEFAULT 'default'",
    },
}


# ================================================================
# workspace / branch scope filtering
#
# One rule set, shared by the experience and fact layers and by recall (search) and enumeration (list):
#   - empty workspace_id → no filtering, full recall. A caller that passes nothing keeps the current behavior.
#   - non-empty workspace_id → exact match. NULL means "belongs to no repo" and is excluded.
#   - branch is only layered on when workspace_id is non-empty (consistent with the existing combined
#     semantics of list_memories); passing branch alone has no effect.
#   - branch tolerates NULL. A row with branch IS NULL means "branch not recorded", not "a different branch recorded",
#     and filtering those out would wrongly drop rows in two places: the fact table got this column late, so legacy databases are all NULL,
#     and new rows written without a branch are NULL too. branch is a subdivision inside a workspace and most
#     experiences are branch-independent anyway; better to return a few uncertain ones than to make them unretrievable.
# ================================================================

def _scope_where(
    workspace_id: Optional[str], branch: Optional[str]
) -> Tuple[str, List[Any]]:
    """Translate the scope condition into a SQL boolean expression plus bound parameters; returns the always-true "1" when not filtering."""
    if not workspace_id:
        return "1", []
    sql = "workspace_id = ?"
    params: List[Any] = [workspace_id]
    if branch:
        sql += " AND (branch = ? OR branch IS NULL)"
        params.append(branch)
    return sql, params


def _scope_matches(
    row_workspace_id: Optional[str],
    row_branch: Optional[str],
    workspace_id: Optional[str],
    branch: Optional[str],
) -> bool:
    """Python equivalent of _scope_where, applying the same rules to objects that were already loaded."""
    if not workspace_id:
        return True
    if row_workspace_id != workspace_id:
        return False
    if branch and row_branch is not None and row_branch != branch:
        return False
    return True


# The scope filter cannot be pushed down into the vector database: chroma (the default backend) serializes all of custom into a single
# metadata field, so a condition on custom.coding_workspace_id is impossible; qdrant could do it, but a hard filter
# that really takes effect on only some backends is more dangerous than none. So filtering lives in SQLite -- it is the authority for these two
# columns -- while the vector side over-fetches by the filter's selectivity so filtering does not eat recall.
#
# The overfetch factor is not a guess: total/scoped is the inverse selectivity counted from the authority table, and fetching that many
# keeps an expected top entries inside the scope among the ANN top-N. When the scope covers most of this user's data the factor
# approaches 1; when it covers only a sliver it grows until this user's rows are nearly all fetched, at which point the filter is exact.
# The upper bound only guards against one combination: pathologically large data for a single user plus a very narrow scope.
_SCOPE_OVERFETCH_MAX_ROWS = 2000


def _scope_overfetch(top: int, scoped: int, total: int) -> int:
    """Scale the number of vector rows fetched according to the scope selectivity."""
    if scoped <= 0 or scoped >= total:
        return top
    scaled = math.ceil(top * total / scoped)
    return max(top, min(scaled, _SCOPE_OVERFETCH_MAX_ROWS))


# ================================================================
# Helpers
# ================================================================

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _vec_node_id(experience_id: str, key_idx: int) -> str:
    return f"{experience_id}:{key_idx}"


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        # Strip the trailing Z
        return datetime.fromisoformat(s.rstrip("Z"))
    except Exception:
        return None


def _row_to_memory(row: sqlite3.Row) -> CodingMemory:
    """SQLite row → CodingMemory"""
    return CodingMemory(
        experience_id=row["experience_id"],
        user_id=row["user_id"],
        agent_id=row["agent_id"],
        task=row["task"],
        search_keys=json.loads(row["search_keys"] or "[]"),
        solution=row["solution"],
        boundary_envs=row["boundary_envs"],
        workspace_id=row["workspace_id"],
        branch=row["branch"],
        session_id=row["session_id"],
        files=json.loads(row["files"] or "[]"),
        source_fact_ids=json.loads(row["source_fact_ids"] or "[]"),
        confidence=row["confidence"],
        source=row["source"],
        extraction_source=row["extraction_source"],
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


def _row_to_fact(row: sqlite3.Row) -> WorkingFact:
    """SQLite row → WorkingFact"""
    return WorkingFact(
        fact_id=row["fact_id"],
        user_id=row["user_id"],
        agent_id=row["agent_id"],
        session_id=row["session_id"],
        workspace_id=row["workspace_id"],
        branch=(row["branch"] if "branch" in row.keys() else None),
        source=row["source"],
        statement=row["statement"],
        files=json.loads(row["files"] or "[]"),
        # Right after ADD COLUMN on a legacy database these two columns may be missing, so keys() guards them (same as fact_chain_summary)
        subject=(row["subject"] or "" if "subject" in row.keys() else ""),
        details=(row["details"] or "" if "details" in row.keys() else ""),
        fact_link=row["fact_link"],
        derived_experience_ids=json.loads(row["derived_experience_ids"] or "[]"),
        status=row["status"],
        # Right after ADD COLUMN on a legacy database this column may be missing, so keys() guards it
        fact_chain_summary=(row["fact_chain_summary"] or ""
                            if "fact_chain_summary" in row.keys() else ""),
        created_at=_parse_dt(row["created_at"]),
    )


def _derive_coding_collection_name(base: str, suffix: str = "") -> str:
    """Derive the coding_keys collection from the chat collection.

    suffix is the **variant suffix** (scene="coding" passes "_cv", working passes ""). When empty the return value is
    byte-for-byte what it was before this parameter existed -- the working collection name must not change, or databases already on disk become unreadable.
    """
    base = (base or "agent_memories").strip()
    # The dimension suffix is appended automatically by VectorStoreBase; only _coding_keys is added to base_name here
    return f"{base}_coding_keys{suffix}"


def _derive_fact_collection_name(base: str, suffix: str = "") -> str:
    """Derive the fact-layer collection from the chat collection. suffix behaves as above."""
    base = (base or "agent_memories").strip()
    return f"{base}_working_facts{suffix}"


def _variant_db_path(db_path: str, suffix: str) -> str:
    """Append a variant suffix to a sqlite path. Returned unchanged when suffix is empty.

    coding_memory.db → coding_memory_cv.db
    """
    if not suffix:
        return db_path
    root, ext = os.path.splitext(db_path)
    return f"{root}{suffix}{ext}"


def _clone_config_with_collection(config: MemoryConfig, derive, suffix: str = "") -> MemoryConfig:
    """Deep-copy config and replace collection_name with the name derived by derive()."""
    cfg = deepcopy(config)
    base = cfg.vector_store.collection_name or "agent_memories"
    cfg.vector_store.collection_name = derive(base, suffix)
    return cfg


# ================================================================
# Store
# ================================================================

class CodingMemoryStore:
    """
    Two-layer storage for Coding Memory.

    Await initialize() before use. close() releases the resources.
    """

    def __init__(
        self,
        config: MemoryConfig,
        embed_service: EmbedService,
        db_path: Optional[str] = None,
        variant: str = "working",
    ):
        self.config = config
        self.embed_service = embed_service
        # Variant suffix. working → "" (paths and collection names are byte-identical to before, so databases already on disk stay readable);
        # coding → "_cv" (its own sqlite plus its own two vector collections).
        #
        # Why separate databases instead of a variant column to filter on: retrieval **first** takes top-K from the vector collection and
        # **then** filters with sqlite. Once both variants' facts share one collection, the vector step is already mixed --
        # no later SQL condition can recover it, because the other variant has taken the top-K slots.
        # Moreover the two extraction specs produce differently styled statements, and recalling them mixed is harmful by itself.
        self.variant = "coding" if str(variant or "").strip().lower() == "coding" else "working"
        _suffix = "_cv" if self.variant == "coding" else ""

        # SQLite path
        if db_path is None:
            data_dir = os.path.dirname(config.history.db_path) if config.history.db_path else "./data"
            db_path = os.path.join(data_dir, "coding_memory.db")
        self._db_path = _variant_db_path(db_path, _suffix)
        # Raw traces are user/agent context and do not belong to any variant. The working and coding
        # business tables keep separate databases, but last-K history shares one separate SQLite so an automatic routing switch does not cause amnesia.
        self._trace_db_path = _variant_db_path(db_path, "_trace")

        # SQLite connections (matching the existing cache_sqlite style)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._trace_conn: Optional[sqlite3.Connection] = None
        self._trace_lock = threading.Lock()

        # VDB: the experience layer and the fact layer each get their own collection
        self._coding_config = _clone_config_with_collection(
            config, _derive_coding_collection_name, _suffix
        )
        self._fact_config = _clone_config_with_collection(
            config, _derive_fact_collection_name, _suffix
        )
        self._vec: Optional[VectorStoreBase] = None
        self._fact_vec: Optional[VectorStoreBase] = None

        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        # SQLite
        await _run_in_pool(self._init_sqlite_sync)
        await _run_in_pool(self._init_trace_sqlite_sync)
        # VDB
        self._vec = create_vector_store(self._coding_config)
        await self._vec.initialize()
        self._fact_vec = create_vector_store(self._fact_config)
        await self._fact_vec.initialize()
        self._initialized = True
        logger.info(
            f"[coding-store] initialized: variant={self.variant} sqlite={self._db_path} "
            f"vdb={self._coding_config.vector_store.collection_name} "
            f"facts={self._fact_config.vector_store.collection_name}"
        )

    def _init_sqlite_sync(self) -> None:
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(
            self._db_path, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._migrate_legacy_schema_sync()
        self._conn.executescript(_CREATE_TABLE_SQL)
        self._conn.executescript(_CREATE_FACT_TABLE_SQL)
        self._add_missing_columns_sync()
        for sql in _CREATE_INDEX_SQL:
            self._conn.execute(sql)

    def _init_trace_sqlite_sync(self) -> None:
        os.makedirs(os.path.dirname(self._trace_db_path) or ".", exist_ok=True)
        self._trace_conn = sqlite3.connect(
            self._trace_db_path, check_same_thread=False, isolation_level=None
        )
        self._trace_conn.row_factory = sqlite3.Row
        self._trace_conn.execute("PRAGMA journal_mode=WAL")
        self._trace_conn.execute("PRAGMA synchronous=NORMAL")
        self._trace_conn.executescript(_CREATE_TRACE_TABLE_SQL)

    def _add_missing_columns_sync(self) -> None:
        """Add the later-added columns to existing tables. Every SQLite version supports ADD COLUMN, so no rebuild is needed."""
        assert self._conn is not None
        for table, columns in _ADDED_COLUMNS.items():
            cur = self._conn.execute(f"PRAGMA table_info({table})")
            existing = {r["name"] for r in cur.fetchall()}
            for name, ddl in columns.items():
                if name not in existing:
                    logger.info(f"[coding-store] adding column {table}.{name}")
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")

    def _migrate_legacy_schema_sync(self) -> None:
        """Rebuild the legacy schema (memory_id primary key + boundary_scope column) in place into the new schema.

        It creates a new table, copies the data and drops the old one instead of using ALTER: RENAME COLUMN needs SQLite 3.25+
        and DROP COLUMN needs 3.35+, while a rebuild is version agnostic. experience_id reuses the old memory_id
        values directly, so the deterministic node_ids on the vector database side are unaffected.
        """
        assert self._conn is not None
        cur = self._conn.execute("PRAGMA table_info(coding_memory_meta)")
        cols = [r["name"] for r in cur.fetchall()]
        if not cols or "memory_id" not in cols:
            return

        logger.info("[coding-store] migrating legacy schema (memory_id → experience_id)")
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "ALTER TABLE coding_memory_meta RENAME TO coding_memory_meta_legacy"
            )
            # Indexes follow the old table and would clash by name, so drop them and let the new table recreate them
            for name in _INDEX_NAMES:
                self._conn.execute(f"DROP INDEX IF EXISTS {name}")
            # Uses execute rather than executescript: the latter implicitly COMMITs when a transaction is pending,
            # which would break this migration's atomicity
            self._conn.execute(_CREATE_TABLE_SQL)
            self._conn.execute(_MIGRATE_COPY_SQL)
            self._conn.execute("DROP TABLE coding_memory_meta_legacy")
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    async def close(self) -> None:
        for vec in (self._vec, self._fact_vec):
            if vec:
                try:
                    await vec.close()
                except Exception:
                    pass
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
        with self._trace_lock:
            if self._trace_conn:
                self._trace_conn.close()
                self._trace_conn = None
        self._initialized = False

    # ------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------

    async def insert(self, memory: CodingMemory) -> None:
        """Insert one coding memory (the full SQLite + VDB set)."""
        self._fill_timestamps(memory)
        await _run_in_pool(self._upsert_sql_sync, memory)
        await self._upsert_vec_keys(memory)
        logger.info(
            f"[coding-store] inserted experience_id={memory.experience_id} "
            f"task={memory.task!r} keys={1 + len(memory.search_keys)}"
        )

    async def update(self, memory: CodingMemory) -> None:
        """Overwrite an existing coding memory."""
        # Read the old keys length and delete the old VDB rows first
        prev = await self.get_by_id(memory.experience_id, user_id=memory.user_id)
        prev_key_count = (1 + len(prev.search_keys)) if prev else 0

        memory.updated_at = datetime.utcnow()
        if memory.created_at is None and prev is not None:
            memory.created_at = prev.created_at
        await _run_in_pool(self._upsert_sql_sync, memory)
        # Delete the old VDB rows
        await self._delete_vec_keys_by_count(memory.experience_id, prev_key_count)
        # Write the new VDB rows
        await self._upsert_vec_keys(memory)
        logger.info(
            f"[coding-store] updated experience_id={memory.experience_id} "
            f"old_keys={prev_key_count} new_keys={1 + len(memory.search_keys)}"
        )

    async def delete(self, experience_id: str, user_id: str) -> bool:
        """Delete one coding memory, returning whether anything was really deleted."""
        prev = await self.get_by_id(experience_id, user_id=user_id)
        if prev is None:
            logger.info(f"[coding-store] delete skipped (not found): {experience_id}")
            return False
        await self._delete_vec_keys_by_count(experience_id, 1 + len(prev.search_keys))
        await _run_in_pool(self._delete_sql_sync, experience_id, user_id)
        logger.info(f"[coding-store] deleted experience_id={experience_id}")
        return True

    # ------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------

    async def get_by_id(
        self, experience_id: str, user_id: Optional[str] = None
    ) -> Optional[CodingMemory]:
        return await _run_in_pool(self._get_by_id_sync, experience_id, user_id)

    async def get_many(
        self, experience_ids: List[str], user_id: Optional[str] = None,
        include_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
    ) -> List[CodingMemory]:
        if not experience_ids:
            return []
        return await _run_in_pool(
            self._get_many_sync, experience_ids, user_id,
            include_sources, exclude_sources,
        )

    async def list_user_tasks(
        self, user_id: str, limit: int = 50
    ) -> List[str]:
        """Return this user's existing task list (most recently modified first); referenced by the extractor prompt."""
        return await _run_in_pool(self._list_user_tasks_sync, user_id, limit)

    async def append_trace(
        self,
        messages: List[Any],
        *,
        request_id: str,
        user_id: str,
        agent_id: str = "default_agent",
        session_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        branch: Optional[str] = None,
        source: str = "default",
    ) -> None:
        """Store one length-bounded working/coding raw trace for later last-K context."""
        payload = [m.to_dict() for m in messages]
        await _run_in_pool(
            self._append_trace_sync,
            request_id, user_id, agent_id, session_id,
            workspace_id, branch, source, payload,
        )

    async def get_recent_trace_messages(
        self,
        *,
        user_id: str,
        agent_id: str = "default_agent",
        max_user_turns: int = 5,
        source: str = "default",
    ) -> List[Any]:
        """Return the most recent K user turns from history; this write is not appended yet, so there is no self-reference."""
        from ..pipelines.base import ChatMessage
        rows = await _run_in_pool(
            self._get_recent_trace_payloads_sync,
            user_id, agent_id, source, max(1, max_user_turns) * 4,
        )
        messages: List[ChatMessage] = []
        for payload in rows:
            for item in payload:
                if isinstance(item, dict):
                    messages.append(ChatMessage.from_dict(item))
        if not messages:
            return []
        from .preproc import keep_last_k_user_turns, prepare_extractor_messages
        return keep_last_k_user_turns(
            prepare_extractor_messages(
                messages,
                tool_max_bytes=int(self.config.coding.tool_result_max_bytes or 2048),
            ),
            k=max(1, max_user_turns),
        )

    async def list_by_workspace(
        self,
        user_id: str,
        workspace_id: str,
        branch: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List["CodingMemory"]:
        """List this user's complete coding memories filtered only by workspace_id (plus optional branch).

        Scope rules live in _scope_where (the same set the search path uses). No semantic ranking; ordered by
        updated_at descending (most recently modified first).
        """
        return await _run_in_pool(
            self._list_by_workspace_sync, user_id, workspace_id, branch, limit, offset
        )

    def _list_by_workspace_sync(
        self, user_id: str, workspace_id: str, branch: Optional[str],
        limit: int, offset: int,
    ) -> List["CodingMemory"]:
        scope_sql, scope_params = _scope_where(workspace_id, branch)
        with self._lock:
            assert self._conn is not None
            sql = (f"SELECT * FROM coding_memory_meta WHERE user_id = ? AND {scope_sql} "
                   f"ORDER BY updated_at DESC LIMIT ? OFFSET ?")
            params: List[Any] = [user_id] + scope_params + [limit, offset]
            cur = self._conn.execute(sql, params)
            return [_row_to_memory(r) for r in cur.fetchall()]

    # ------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------

    async def search_by_query_embedding(
        self,
        query_embedding: List[float],
        user_id: str,
        *,
        workspace_id: Optional[str] = None,
        branch: Optional[str] = None,
        top: int = 40,
        include_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Vector retrieval. Returns a deduplicated hit list sorted by score descending:
            [
                {
                    "experience_id": "...",
                    "score": 0.83,
                    "matched_key_kind": "task" | "aspect",
                    "matched_key_text": "...",
                },
                ...
            ]
        workspace_id / branch are the scope filter, rules in _scope_where: without them this is pure semantic recall
        (exactly the behavior before the filter existed); with them only experiences inside the scope per SQLite are kept, and the
        vector fetch is scaled by the scope's selectivity so "take top-N first, then filter" does not eat recall.
        """
        if self._vec is None:
            return []

        allowed: Optional[Set[str]] = None
        if workspace_id:
            allowed, total = await _run_in_pool(
                self._scoped_experience_ids_sync, user_id, workspace_id, branch
            )
            if not allowed:
                logger.info(
                    f"[coding-store] no experience in scope: "
                    f"workspace_id={workspace_id!r} branch={branch!r} "
                    f"(user has {total}); skip vdb search"
                )
                return []
            top = _scope_overfetch(top, len(allowed), total)

        raw = await self._vec.search(
            query_embedding=query_embedding,
            user_id=user_id,
            limit=top,
            score_threshold=0.0,
            only_latest=True,
            include_sources=include_sources,
            exclude_sources=exclude_sources,
        )

        best: Dict[str, Dict[str, Any]] = {}
        for hit in raw:
            node: MemoryNode = hit.get("node")
            score: float = hit.get("score", 0.0)
            if node is None:
                continue
            custom = node.custom or {}
            eid = custom.get("coding_experience_id")
            if not eid:
                continue
            # Trust the SQLite copy rather than the coding_workspace_id copy in the vector row payload:
            # they are not written in one transaction, so after an experience changes workspace the payload may still hold the old value.
            if allowed is not None and eid not in allowed:
                continue
            # dedup by experience_id, keeping the highest score
            cur = best.get(eid)
            if cur is None or score > cur["score"]:
                best[eid] = {
                    "experience_id": eid,
                    "score": score,
                    "matched_key_kind": custom.get("coding_key_kind", "aspect"),
                    "matched_key_text": custom.get("coding_key_text") or node.content,
                }
        # Sort
        return sorted(best.values(), key=lambda h: -h["score"])

    # ------------------------------------------------------------
    # Fact layer
    # ------------------------------------------------------------

    async def insert_fact(self, fact: WorkingFact) -> None:
        """Insert one fact (one row in SQLite plus one in the fact collection)."""
        if fact.created_at is None:
            fact.created_at = datetime.utcnow()
        await _run_in_pool(self._upsert_fact_sql_sync, fact)
        await self._upsert_fact_vec(fact)
        logger.info(
            f"[coding-store] inserted fact_id={fact.fact_id} "
            f"statement={fact.statement!r}"
        )

    async def get_fact(
        self, fact_id: str, user_id: Optional[str] = None
    ) -> Optional[WorkingFact]:
        facts = await self.get_facts([fact_id], user_id=user_id)
        return facts[0] if facts else None

    async def get_facts(
        self, fact_ids: List[str], user_id: Optional[str] = None
    ) -> List[WorkingFact]:
        if not fact_ids:
            return []
        return await _run_in_pool(self._get_facts_sync, fact_ids, user_id)

    async def search_facts(
        self,
        query_embedding: List[float],
        user_id: str,
        *,
        workspace_id: Optional[str] = None,
        branch: Optional[str] = None,
        top: int = 5,
        query_text: Optional[str] = None,
        min_score: float = 0.0,
        include_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
    ) -> List[Tuple[WorkingFact, float]]:
        """Semantic recall over the fact layer; returns only active facts, score descending, at most top entries.

        SUPERSEDE already deleted the vector row, so superseded facts should not be recallable; the SQLite status is checked
        once more here so an inconsistency between the two sides cannot release a stale conclusion.

        workspace_id / branch are the scope filter, rules in _scope_where, with the same semantics as the experience layer:
        absent means no filtering, present means filtering by the SQLite values with the vector fetch scaled by selectivity.

        The VDB is oversampled 4x over top: vector rows and SQLite rows are not written in one transaction, so a crash in between
        leaves orphans where the vector is still there but SQLite is gone, and a failed vector delete during a soft delete also leaves superseded
        residue. All of it is filtered out below; without oversampling it would consume the candidate budget directly -- in the worst case
        the first top entries are all residue and recall comes back empty.

        When query_text is given, a BM25 lexical channel is added before the truncation to top and the results are fused and reranked (see
        coding/lexical.py). The returned score is always the dense cosine score, never the fused score, because the layer above
        (client.py:851) passes its gate on a 0.30 cosine threshold, and returning fused scores (a different scale) would silently empty the fact layer.
        min_score lets the caller move that gate ahead of reranking: cutting the candidates that would fail the gate first and then
        reranking keeps the returned count identical to the pure dense count, with lexical only changing who is chosen, never reducing recall.
        """
        if self._fact_vec is None:
            return []

        fetch = top * 4
        if workspace_id:
            scoped, total = await _run_in_pool(
                self._count_active_facts_sync, user_id, workspace_id, branch
            )
            if not scoped:
                logger.info(
                    f"[coding-store] no active fact in scope: "
                    f"workspace_id={workspace_id!r} branch={branch!r} "
                    f"(user has {total}); skip fact vdb search"
                )
                return []
            fetch = _scope_overfetch(fetch, scoped, total)

        raw = await self._fact_vec.search(
            query_embedding=query_embedding,
            user_id=user_id,
            limit=fetch,
            score_threshold=0.0,
            only_latest=True,
            include_sources=include_sources,
            exclude_sources=exclude_sources,
        )

        scores: Dict[str, float] = {}
        for hit in raw:
            node: MemoryNode = hit.get("node")
            if node is None:
                continue
            fid = (node.custom or {}).get("working_fact_id")
            if not fid:
                continue
            score = hit.get("score", 0.0)
            if fid not in scores or score > scores[fid]:
                scores[fid] = score

        if not scores:
            return []

        ordered_ids = sorted(scores, key=lambda f: -scores[f])
        facts = await self.get_facts(ordered_ids, user_id=user_id)
        alive = [
            (f, scores[f.fact_id]) for f in facts
            if f.status == "active"
            and _scope_matches(f.workspace_id, f.branch, workspace_id, branch)
        ]
        kept = [(f, s) for f, s in alive if s >= min_score]
        # get_facts does not guarantee returning rows in ordered_ids order, so they are explicitly re-sorted by score.
        kept.sort(key=lambda p: -p[1])
        if alive and not kept:
            # The gate emptied the fact layer right here. The caller (client.py) then only sees an empty list and cannot tell
            # "nothing recalled" from "recalled but all below threshold", and the latter means the threshold was raised too far, so it is
            # reported wherever the scores are available.
            logger.warning(
                f"[coding-store] fact layer emptied by min_score={min_score}: "
                f"{len(alive)} active hit(s), top={max(s for _, s in alive):.3f}"
            )
        if len(kept) <= top or not (query_text or "").strip():
            return kept[:top]

        # The lexical channel only has a choice when there are more candidates than slots. The text must match the embedding side (_upsert_fact_vec embeds
        # `subject: statement`), otherwise the two channels would not be ranking the same thing.
        from .lexical import rerank

        by_id = {f.fact_id: (f, s) for f, s in kept}
        order = rerank(
            query_text,
            [(f.fact_id, s, f"{f.subject}: {f.statement}") for f, s in kept],
        )
        if order is None:
            return kept[:top]
        fused = [by_id[fid] for fid in order if fid in by_id][:top]
        dense_top = {f.fact_id for f, _ in kept[:top]}
        swapped = [f.fact_id for f, _ in fused if f.fact_id not in dense_top]
        if swapped:
            logger.info(
                f"[coding-store] bm25 rerank changed the top-{top}: "
                f"{len(swapped)} fact(s) swapped in from {len(kept)} candidates"
            )
        # Return the dense scores (see docstring); the order is already fused, so do not sort by score again.
        return fused

    async def get_fact_ancestors(
        self, fact_id: str, *, user_id: Optional[str] = None, max_depth: int = 8
    ) -> List[WorkingFact]:
        """Walk fact_link backwards and return this fact's predecessors in chronological order, excluding itself.

        SUPERSEDE only soft deletes, so the replaced fact stays in the store untouched with fact_link pointing at its successor,
        which means "what did this conclusion say earlier" needs no extra copy; a reverse lookup is enough.

        At most max_depth generations; it stops at the end, at a broken link, or on a cycle. A cycle should never happen, but once it
        does it is an infinite loop, which is not a risk worth taking for a summary.
        """
        return await _run_in_pool(
            self._get_fact_ancestors_sync, fact_id, user_id, max_depth
        )

    async def supersede_fact(self, old_fact_id: str, new_fact_id: str) -> bool:
        """Soft delete the old fact: status=superseded plus fact_link pointing at the new fact, and withdrawn from the vector database.

        The SQLite row is kept so the fact_link chain stays traceable; the vector row is deleted so it no longer takes recall slots.
        """
        changed = await _run_in_pool(
            self._supersede_fact_sql_sync, old_fact_id, new_fact_id
        )
        if not changed:
            return False
        if self._fact_vec is not None:
            try:
                await self._fact_vec.delete(old_fact_id)
            except Exception as e:
                logger.debug(f"[coding-store] delete fact vec {old_fact_id} ignored: {e}")
        logger.info(
            f"[coding-store] superseded fact_id={old_fact_id} by={new_fact_id}"
        )
        return True

    async def set_fact_derived_experiences(
        self, fact_id: str, experience_ids: List[str]
    ) -> None:
        """Overwrite fact.derived_experience_ids (the caller computes the union)."""
        await _run_in_pool(
            self._set_json_column_sync,
            "working_fact", "fact_id", fact_id,
            "derived_experience_ids", experience_ids,
        )

    async def set_experience_source_facts(
        self, experience_id: str, fact_ids: List[str]
    ) -> None:
        """Overwrite experience.source_fact_ids (the caller computes the union)."""
        await _run_in_pool(
            self._set_json_column_sync,
            "coding_memory_meta", "experience_id", experience_id,
            "source_fact_ids", fact_ids,
        )

    # ------------------------------------------------------------
    # Synchronous SQLite implementation
    # ------------------------------------------------------------

    def _fill_timestamps(self, memory: CodingMemory) -> None:
        now = datetime.utcnow()
        if memory.created_at is None:
            memory.created_at = now
        memory.updated_at = now

    def _upsert_sql_sync(self, memory: CodingMemory) -> None:
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                """
                INSERT INTO coding_memory_meta (
                    experience_id, user_id, agent_id, task, search_keys, solution,
                    boundary_envs, workspace_id, branch, session_id,
                    files, source_fact_ids, confidence, source, extraction_source,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(experience_id) DO UPDATE SET
                    user_id         = excluded.user_id,
                    agent_id        = excluded.agent_id,
                    task            = excluded.task,
                    search_keys     = excluded.search_keys,
                    solution        = excluded.solution,
                    boundary_envs   = excluded.boundary_envs,
                    workspace_id    = excluded.workspace_id,
                    branch          = excluded.branch,
                    session_id      = excluded.session_id,
                    files           = excluded.files,
                    source_fact_ids = excluded.source_fact_ids,
                    confidence      = excluded.confidence,
                    source          = excluded.source,
                    extraction_source = excluded.extraction_source,
                    updated_at      = excluded.updated_at
                """,
                (
                    memory.experience_id, memory.user_id, memory.agent_id,
                    memory.task, json.dumps(memory.search_keys, ensure_ascii=False),
                    memory.solution, memory.boundary_envs,
                    memory.workspace_id, memory.branch, memory.session_id,
                    json.dumps(memory.files, ensure_ascii=False),
                    json.dumps(memory.source_fact_ids, ensure_ascii=False),
                    memory.confidence, memory.source, memory.extraction_source,
                    (memory.created_at or datetime.utcnow()).isoformat(),
                    (memory.updated_at or datetime.utcnow()).isoformat(),
                ),
            )
    def _delete_sql_sync(self, experience_id: str, user_id: str) -> None:
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                "DELETE FROM coding_memory_meta WHERE experience_id = ? AND user_id = ?",
                (experience_id, user_id),
            )

    def _get_by_id_sync(
        self, experience_id: str, user_id: Optional[str]
    ) -> Optional[CodingMemory]:
        with self._lock:
            assert self._conn is not None
            if user_id:
                cur = self._conn.execute(
                    "SELECT * FROM coding_memory_meta WHERE experience_id = ? AND user_id = ?",
                    (experience_id, user_id),
                )
            else:
                cur = self._conn.execute(
                    "SELECT * FROM coding_memory_meta WHERE experience_id = ?",
                    (experience_id,),
                )
            row = cur.fetchone()
            return _row_to_memory(row) if row else None

    def _get_many_sync(
        self, experience_ids: List[str], user_id: Optional[str],
        include_sources: Optional[List[str]], exclude_sources: Optional[List[str]],
    ) -> List[CodingMemory]:
        if not experience_ids:
            return []
        with self._lock:
            assert self._conn is not None
            placeholders = ",".join("?" * len(experience_ids))
            if user_id:
                sql = f"SELECT * FROM coding_memory_meta WHERE experience_id IN ({placeholders}) AND user_id = ?"
                params = list(experience_ids) + [user_id]
            else:
                sql = f"SELECT * FROM coding_memory_meta WHERE experience_id IN ({placeholders})"
                params = list(experience_ids)
            if include_sources:
                marks = ",".join("?" * len(include_sources))
                sql += f" AND source IN ({marks})"
                params.extend(include_sources)
            if exclude_sources:
                marks = ",".join("?" * len(exclude_sources))
                sql += f" AND source NOT IN ({marks})"
                params.extend(exclude_sources)
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
            # Preserve the input order
            by_id = {r["experience_id"]: _row_to_memory(r) for r in rows}
            return [by_id[eid] for eid in experience_ids if eid in by_id]

    def _scoped_experience_ids_sync(
        self, user_id: str, workspace_id: str, branch: Optional[str]
    ) -> Tuple[Set[str], int]:
        """Return (the experience_ids inside the scope, the total number of experiences for this user).

        One query yields both the filtered result and the selectivity denominator: the first intersects vector hits, the second sets the overfetch factor.
        """
        scope_sql, scope_params = _scope_where(workspace_id, branch)
        with self._lock:
            assert self._conn is not None
            cur = self._conn.execute(
                f"SELECT experience_id, ({scope_sql}) AS in_scope "
                f"FROM coding_memory_meta WHERE user_id = ?",
                scope_params + [user_id],
            )
            rows = cur.fetchall()
        return {r["experience_id"] for r in rows if r["in_scope"]}, len(rows)

    def _count_active_facts_sync(
        self, user_id: str, workspace_id: str, branch: Optional[str]
    ) -> Tuple[int, int]:
        """Return (the number of active facts inside the scope, the total active facts for this user).

        Fact-layer filtering happens with _scope_matches after get_facts returns the objects; only the selectivity is needed here.
        """
        scope_sql, scope_params = _scope_where(workspace_id, branch)
        with self._lock:
            assert self._conn is not None
            row = self._conn.execute(
                f"SELECT COUNT(*) AS total, "
                f"SUM(CASE WHEN {scope_sql} THEN 1 ELSE 0 END) AS scoped "
                f"FROM working_fact WHERE user_id = ? AND status = 'active'",
                scope_params + [user_id],
            ).fetchone()
        return int(row["scoped"] or 0), int(row["total"] or 0)

    def _upsert_fact_sql_sync(self, fact: WorkingFact) -> None:
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                """
                INSERT INTO working_fact (
                    fact_id, user_id, agent_id, session_id, workspace_id, branch,
                    source, statement, files, subject, details,
                    fact_link, derived_experience_ids,
                    status, fact_chain_summary, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fact_id) DO UPDATE SET
                    user_id                = excluded.user_id,
                    agent_id               = excluded.agent_id,
                    session_id             = excluded.session_id,
                    workspace_id           = excluded.workspace_id,
                    branch                 = excluded.branch,
                    source                 = excluded.source,
                    statement              = excluded.statement,
                    files                  = excluded.files,
                    subject                = excluded.subject,
                    details                = excluded.details,
                    fact_link              = excluded.fact_link,
                    derived_experience_ids = excluded.derived_experience_ids,
                    status                 = excluded.status,
                    fact_chain_summary     = excluded.fact_chain_summary
                """,
                (
                    fact.fact_id, fact.user_id, fact.agent_id,
                    fact.session_id, fact.workspace_id, fact.branch, fact.source,
                    fact.statement,
                    json.dumps(fact.files, ensure_ascii=False),
                    fact.subject or "",
                    fact.details or "",
                    fact.fact_link,
                    json.dumps(fact.derived_experience_ids, ensure_ascii=False),
                    fact.status,
                    fact.fact_chain_summary or "",
                    (fact.created_at or datetime.utcnow()).isoformat(),
                ),
            )

    def _get_facts_sync(
        self, fact_ids: List[str], user_id: Optional[str]
    ) -> List[WorkingFact]:
        with self._lock:
            assert self._conn is not None
            placeholders = ",".join("?" * len(fact_ids))
            sql = f"SELECT * FROM working_fact WHERE fact_id IN ({placeholders})"
            params: List[Any] = list(fact_ids)
            if user_id:
                sql += " AND user_id = ?"
                params.append(user_id)
            cur = self._conn.execute(sql, params)
            by_id = {r["fact_id"]: _row_to_fact(r) for r in cur.fetchall()}
            # Preserve the input order
            return [by_id[fid] for fid in fact_ids if fid in by_id]

    def _get_fact_ancestors_sync(
        self, fact_id: str, user_id: Optional[str], max_depth: int
    ) -> List[WorkingFact]:
        with self._lock:
            assert self._conn is not None
            sql = "SELECT * FROM working_fact WHERE fact_link = ?"
            if user_id:
                sql += " AND user_id = ?"
            # Being superseded twice should not happen (supersede only applies to active), but if it does, take the earliest
            sql += " ORDER BY created_at ASC LIMIT 1"

            out: List[WorkingFact] = []
            seen = {fact_id}
            cur_id = fact_id
            for _ in range(max_depth):
                params = [cur_id] + ([user_id] if user_id else [])
                row = self._conn.execute(sql, params).fetchone()
                if row is None:
                    break
                parent = _row_to_fact(row)
                if parent.fact_id in seen:
                    logger.warning(
                        f"[coding-store] fact_link cycle at {parent.fact_id}, stop walking"
                    )
                    break
                seen.add(parent.fact_id)
                out.append(parent)
                cur_id = parent.fact_id
            out.reverse()          # the reverse walk goes near to far; callers want chronological order
            return out

    def _supersede_fact_sql_sync(self, old_fact_id: str, new_fact_id: str) -> bool:
        with self._lock:
            assert self._conn is not None
            cur = self._conn.execute(
                "UPDATE working_fact SET status = 'superseded', fact_link = ? "
                "WHERE fact_id = ? AND status = 'active'",
                (new_fact_id, old_fact_id),
            )
            return cur.rowcount > 0

    def _set_json_column_sync(
        self, table: str, key_column: str, key: str, column: str, value: List[str]
    ) -> None:
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                f"UPDATE {table} SET {column} = ? WHERE {key_column} = ?",
                (json.dumps(value, ensure_ascii=False), key),
            )

    def _list_user_tasks_sync(self, user_id: str, limit: int) -> List[str]:
        with self._lock:
            assert self._conn is not None
            cur = self._conn.execute(
                "SELECT task FROM coding_memory_meta WHERE user_id = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (user_id, limit),
            )
            return [r["task"] for r in cur.fetchall()]

    def _append_trace_sync(
        self,
        request_id: str,
        user_id: str,
        agent_id: str,
        session_id: Optional[str],
        workspace_id: Optional[str],
        branch: Optional[str],
        source: str,
        payload: List[Dict[str, Any]],
    ) -> None:
        with self._trace_lock:
            assert self._trace_conn is not None
            self._trace_conn.execute(
                "INSERT OR IGNORE INTO coding_trace "
                "(request_id,user_id,agent_id,session_id,workspace_id,branch,source,messages_json,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    request_id, user_id, agent_id or "default_agent", session_id,
                    workspace_id, branch, source,
                    json.dumps(payload, ensure_ascii=False), _now_iso(),
                ),
            )
            # last-K only needs recent traces; keeping 100 writes covers an unusually large single-turn distribution
            # while preventing raw tool traces from growing unbounded in SQLite.
            self._trace_conn.execute(
                "DELETE FROM coding_trace WHERE rowid IN ("
                "SELECT rowid FROM coding_trace WHERE user_id=? AND agent_id=? "
                "ORDER BY created_at DESC LIMIT -1 OFFSET 100)",
                (user_id, agent_id or "default_agent"),
            )

    def _get_recent_trace_payloads_sync(
        self, user_id: str, agent_id: str, source: str, limit: int,
    ) -> List[List[Dict[str, Any]]]:
        with self._trace_lock:
            assert self._trace_conn is not None
            rows = self._trace_conn.execute(
                "SELECT messages_json FROM coding_trace "
                "WHERE user_id=? AND agent_id=? AND source=? "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, agent_id or "default_agent", source, max(1, limit)),
            ).fetchall()
        out: List[List[Dict[str, Any]]] = []
        for row in reversed(rows):
            try:
                payload = json.loads(row["messages_json"] or "[]")
            except Exception:
                payload = []
            if isinstance(payload, list):
                out.append(payload)
        return out

    # ------------------------------------------------------------
    # VDB implementation
    # ------------------------------------------------------------

    async def _upsert_vec_keys(self, memory: CodingMemory) -> None:
        """Write task + search_keys into the VDB (one MemoryNode per row)."""
        if self._vec is None:
            return

        all_keys: List[Tuple[str, str]] = [("task", memory.task)]
        for k in memory.search_keys:
            if k:
                all_keys.append(("aspect", k))
        if not all_keys:
            return

        # batch embed
        texts = [t for _, t in all_keys]
        try:
            embeddings = await self.embed_service.embed_batch(texts)
        except Exception as e:
            logger.warning(f"[coding-store] embed_batch failed: {e}")
            return

        nodes: List[MemoryNode] = []
        for i, ((kind, text), emb) in enumerate(zip(all_keys, embeddings)):
            node = MemoryNode(
                node_id=_vec_node_id(memory.experience_id, i),
                user_id=memory.user_id,
                agent_id=memory.agent_id,
                session_id=memory.session_id or "",
                source=memory.source,
                # layer takes no part in chat-path filtering (separate collection), so L1_RAW is used as a placeholder
                layer=MemoryLayer.L1_RAW,
                content=text,
                embedding=emb,
                status=MemoryStatus.ACTIVE,
                custom={
                    "coding_experience_id": memory.experience_id,
                    "coding_key_kind": kind,
                    "coding_key_text": text,
                    "coding_workspace_id": memory.workspace_id or "",
                    "coding_branch": memory.branch or "",
                    "coding_kind_marker": "coding_key",
                },
            )
            nodes.append(node)

        try:
            await self._vec.upsert_batch(nodes)
        except Exception as e:
            logger.warning(f"[coding-store] vdb upsert_batch failed: {e}")

    async def _upsert_fact_vec(self, fact: WorkingFact) -> None:
        """Persist a fact into the vector database: one row per fact, with node_id equal to fact_id.

        The retrieval text is the statement, prefixed with subject when there is one. The subject is usually already
        present in the statement, but a pinpoint question ("what did Dylan eat for lunch the next day") hinges entirely on the weight of that subject word,
        so prefixing it once promotes it from "incidentally mentioned" to "title". details is deliberately kept out of the vector:
        it is the source text, its length is uncontrolled, embedding it would dilute the summary's semantics, and the problem it solves is
        "being able to read the value after a hit", not "hitting more easily".
        """
        if self._fact_vec is None or not fact.statement:
            return
        vector_text = (
            f"{fact.subject}: {fact.statement}" if fact.subject else fact.statement
        )
        try:
            embedding = await self.embed_service.embed(vector_text)
        except Exception as e:
            logger.warning(f"[coding-store] fact embed failed: {e}")
            return

        node = MemoryNode(
            node_id=fact.fact_id,
            user_id=fact.user_id,
            agent_id=fact.agent_id,
            session_id=fact.session_id or "",
            source=fact.source,
            # layer takes no part in chat-path filtering (separate collection), so L1_RAW is used as a placeholder
            layer=MemoryLayer.L1_RAW,
            content=vector_text,
            embedding=embedding,
            status=MemoryStatus.ACTIVE,
            custom={
                "working_fact_id": fact.fact_id,
                "coding_workspace_id": fact.workspace_id or "",
                "coding_branch": fact.branch or "",
                "coding_kind_marker": "working_fact",
                "coding_fact_subject": fact.subject or "",
            },
        )
        try:
            await self._fact_vec.upsert_batch([node])
        except Exception as e:
            logger.warning(f"[coding-store] fact vdb upsert failed: {e}")

    async def _delete_vec_keys_by_count(self, experience_id: str, count: int) -> None:
        """Delete deterministic node_ids 0..count-1."""
        if self._vec is None or count <= 0:
            return
        for i in range(count):
            try:
                await self._vec.delete(_vec_node_id(experience_id, i))
            except Exception as e:
                logger.debug(
                    f"[coding-store] delete vdb {_vec_node_id(experience_id, i)} ignored: {e}"
                )
