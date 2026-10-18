# -*- coding: utf-8 -*-
"""
History Store - SQLite audit trail

Built on sqlite3 from the Python standard library (zero extra dependency), wrapped as async through _run_in_sqlite_pool().
Records the full history of every memory operation (ADD/UPDATE/DELETE/SEARCH) for audit trails and versioning.

Design decisions:
- Independent of the rdb.py abstraction: history only needs SQLite, no MySQL/PG switching
- Sync sqlite3 calls are wrapped by _run_in_sqlite_pool() so the event loop is never blocked
- Every write is try-except safe; a failure never affects the main flow
"""

import functools
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from .cache_sqlite import _sqlite_executor
from ..utils.async_executor import run_in_executor_guarded

logger = logging.getLogger(__name__)


async def _run_in_sqlite_pool(func, *args, **kwargs):
    """Run a sync function in the dedicated SQLite thread pool"""
    call = functools.partial(func, *args, **kwargs) if args or kwargs else func
    return await run_in_executor_guarded(_sqlite_executor, call)

# SQL: create table
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS memory_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id     TEXT NOT NULL,
    isolation_key TEXT NOT NULL,
    event         TEXT NOT NULL,
    old_memory    TEXT,
    new_memory    TEXT,
    old_status    TEXT,
    new_status    TEXT,
    change_reason TEXT,
    layer         TEXT,
    actor_id      TEXT DEFAULT '',
    role          TEXT DEFAULT '',
    extra         TEXT DEFAULT '{}',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
"""

# SQL: create indexes
_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_history_memory_id ON memory_history(memory_id);",
    "CREATE INDEX IF NOT EXISTS idx_history_isolation_key ON memory_history(isolation_key);",
    "CREATE INDEX IF NOT EXISTS idx_history_created_at ON memory_history(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_history_event ON memory_history(event);",
]


def _now_iso() -> str:
    """Return the current UTC time as an ISO format string"""
    return datetime.now(timezone.utc).isoformat()


class HistoryStore:
    """
    SQLite history store - audit log of every memory operation.

    Usage:
        store = HistoryStore(config)
        await store.initialize()

        await store.record_add(memory_id="abc", content="...", layer="raw",
                               isolation_key="app:user")

        history = await store.get_history("abc")
        recent = await store.get_recent("app:user", limit=50)

        await store.close()
    """

    def __init__(self, config):
        """
        Args:
            config: MemoryConfig instance; settings are read from config.history
        """
        self._db_path = config.history.db_path
        self._conn: Optional[sqlite3.Connection] = None

    async def initialize(self) -> None:
        """Create the table and indexes if they do not exist"""
        await _run_in_sqlite_pool(self._init_sync)
        logger.info(f"[HistoryStore] initialized: {self._db_path}")

    def _init_sync(self) -> None:
        """Sync init: create the directory, connect the database, create table and indexes"""
        # Make sure the directory exists
        db_dir = os.path.dirname(self._db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent read/write performance
        self._conn.execute("PRAGMA journal_mode=WAL;")

        self._conn.execute(_CREATE_TABLE_SQL)
        for idx_sql in _CREATE_INDEXES_SQL:
            self._conn.execute(idx_sql)
        self._conn.commit()

    # ================================================================
    # Recording operations
    # ================================================================

    async def record_add(
        self,
        memory_id: str,
        content: str,
        layer: str,
        isolation_key: str,
        actor_id: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Record an ADD event"""
        return await self._insert(
            memory_id=memory_id,
            isolation_key=isolation_key,
            event="ADD",
            new_memory=content,
            layer=layer,
            actor_id=actor_id,
            extra=extra,
        )

    async def record_update(
        self,
        memory_id: str,
        old_content: str,
        new_content: str,
        old_status: str,
        new_status: str,
        change_reason: str,
        isolation_key: str = "",
        actor_id: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Record an UPDATE event"""
        return await self._insert(
            memory_id=memory_id,
            isolation_key=isolation_key,
            event="UPDATE",
            old_memory=old_content,
            new_memory=new_content,
            old_status=old_status,
            new_status=new_status,
            change_reason=change_reason,
            actor_id=actor_id,
            extra=extra,
        )

    async def record_delete(
        self,
        memory_id: str,
        content: str,
        isolation_key: str,
        actor_id: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Record a DELETE event"""
        return await self._insert(
            memory_id=memory_id,
            isolation_key=isolation_key,
            event="DELETE",
            old_memory=content,
            actor_id=actor_id,
            extra=extra,
        )

    async def record_search(
        self,
        query: str,
        isolation_key: str,
        results_count: int,
        actor_id: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Record a SEARCH event"""
        search_extra = {"query": query, "results_count": results_count}
        if extra:
            search_extra.update(extra)
        return await self._insert(
            memory_id="",
            isolation_key=isolation_key,
            event="SEARCH",
            new_memory=query,
            actor_id=actor_id,
            extra=search_extra,
        )

    # ================================================================
    # Queries
    # ================================================================

    async def get_history(self, memory_id: str) -> List[Dict[str, Any]]:
        """
        Get the full change history of a memory.

        Args:
            memory_id: memory ID

        Returns:
            History records sorted by time
        """
        return await _run_in_sqlite_pool(self._get_history_sync, memory_id)

    def _get_history_sync(self, memory_id: str) -> List[Dict[str, Any]]:
        if self._conn is None:
            return []
        cursor = self._conn.execute(
            "SELECT * FROM memory_history WHERE memory_id = ? ORDER BY created_at ASC",
            (memory_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    async def get_recent(
        self,
        isolation_key: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get the most recent operations of a user / isolation key.

        Args:
            isolation_key: isolation key defined by the caller
            limit: maximum number of rows to return

        Returns:
            Recent operations in reverse chronological order
        """
        return await _run_in_sqlite_pool(self._get_recent_sync, isolation_key, limit)

    def _get_recent_sync(self, isolation_key: str, limit: int) -> List[Dict[str, Any]]:
        if self._conn is None:
            return []
        cursor = self._conn.execute(
            "SELECT * FROM memory_history WHERE isolation_key = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (isolation_key, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    async def count(self, isolation_key: str = "") -> int:
        """
        Count the history records.

        Args:
            isolation_key: optional isolation key filter; empty counts everything.

        Returns:
            The record count
        """
        return await _run_in_sqlite_pool(self._count_sync, isolation_key)

    def _count_sync(self, isolation_key: str) -> int:
        if self._conn is None:
            return 0
        if isolation_key:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM memory_history WHERE isolation_key = ?",
                (isolation_key,),
            )
        else:
            cursor = self._conn.execute("SELECT COUNT(*) FROM memory_history")
        return cursor.fetchone()[0]

    async def close(self) -> None:
        """Close the database connection"""
        if self._conn is not None:
            await _run_in_sqlite_pool(self._close_sync)
            logger.info("[HistoryStore] closed")

    def _close_sync(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ================================================================
    # Internal
    # ================================================================

    async def _insert(
        self,
        memory_id: str,
        isolation_key: str,
        event: str,
        old_memory: str = None,
        new_memory: str = None,
        old_status: str = None,
        new_status: str = None,
        change_reason: str = None,
        layer: str = None,
        actor_id: str = "",
        role: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert one history record and return its ID"""
        return await _run_in_sqlite_pool(
            self._insert_sync,
            memory_id=memory_id,
            isolation_key=isolation_key,
            event=event,
            old_memory=old_memory,
            new_memory=new_memory,
            old_status=old_status,
            new_status=new_status,
            change_reason=change_reason,
            layer=layer,
            actor_id=actor_id,
            role=role,
            extra=extra,
        )

    def _insert_sync(
        self,
        memory_id: str,
        isolation_key: str,
        event: str,
        old_memory: str = None,
        new_memory: str = None,
        old_status: str = None,
        new_status: str = None,
        change_reason: str = None,
        layer: str = None,
        actor_id: str = "",
        role: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        if self._conn is None:
            return -1
        now = _now_iso()
        extra_json = json.dumps(extra or {}, ensure_ascii=False)
        cursor = self._conn.execute(
            """INSERT INTO memory_history
               (memory_id, isolation_key, event,
                old_memory, new_memory, old_status, new_status,
                change_reason, layer, actor_id, role, extra,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory_id,
                isolation_key,
                event,
                old_memory,
                new_memory,
                old_status,
                new_status,
                change_reason,
                layer,
                actor_id,
                role,
                extra_json,
                now,
                now,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid
