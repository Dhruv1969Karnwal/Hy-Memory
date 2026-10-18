"""
Agent Memory V2 - SqliteCache

Zero-dependency audit/observability persistence backend built on the Python standard library sqlite3.
Works out of the box on a single machine.

Design pattern follows history_store.py:
- Standard library sqlite3 + _run_in_sqlite_pool()
- WAL mode + check_same_thread=False
- threading.Lock guards write operations

Three tables:
- memory_operations / pipeline_logs / system_metrics: auditing and observability
"""

import concurrent.futures
import functools
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any

from .cache_base import CacheBase
from ..config import MemoryConfig
from ..utils.async_executor import run_in_executor_guarded

logger = logging.getLogger(__name__)

# Dedicated SQLite thread pool (does not contend with the VDB/Graph pools)
_SQLITE_POOL_SIZE = 32
_sqlite_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=_SQLITE_POOL_SIZE, thread_name_prefix="sqlite"
)


async def _run_in_sqlite_pool(func, *args, **kwargs):
    """Run a synchronous function in the dedicated SQLite thread pool"""
    call = functools.partial(func, *args, **kwargs) if args or kwargs else func
    return await run_in_executor_guarded(_sqlite_executor, call)


# DDL
# Table names / columns align with the MySQL partitioned version (including created_date); SQLite has no partitions, so these are plain CREATE TABLE statements.
_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS memory_operations_v2 (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id   TEXT NOT NULL,
    user_id      TEXT NOT NULL,
    agent_id     TEXT NOT NULL DEFAULT '',
    op           TEXT NOT NULL,       -- ADD / UPDATE
    memory_id    TEXT NOT NULL,       -- Node ID after the operation (ADD: new node ID; UPDATE: new node ID)
    old_memory_id TEXT,               -- Old node ID on UPDATE
    content      TEXT NOT NULL,       -- Content finally written
    layer        TEXT NOT NULL DEFAULT '',
    reason       TEXT NOT NULL DEFAULT '',
    supersedes   TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL,
    created_date TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_memop_request ON memory_operations_v2(request_id);
CREATE INDEX IF NOT EXISTS idx_memop_memory  ON memory_operations_v2(memory_id);
CREATE INDEX IF NOT EXISTS idx_memop_user    ON memory_operations_v2(user_id);

CREATE TABLE IF NOT EXISTS pipeline_logs_v2 (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id   TEXT NOT NULL,
    namespace    TEXT NOT NULL DEFAULT '',
    user_id      TEXT NOT NULL,
    agent_id     TEXT NOT NULL DEFAULT '',
    session_id   TEXT NOT NULL DEFAULT '',
    step         TEXT NOT NULL,       -- EXTRACT / SEARCH_QUERY / RECONCILE / SUMMARY
    prompt       TEXT NOT NULL,       -- Raw LLM prompt
    response     TEXT NOT NULL,       -- Raw LLM response
    parsed       TEXT NOT NULL DEFAULT '',  -- Parsed structured result (JSON)
    memory_ids   TEXT NOT NULL DEFAULT '',  -- Related memory_id list (JSON array); populated during the reconcile step
    elapsed_ms   REAL NOT NULL DEFAULT 0,
    prompt_tokens    INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens     INTEGER NOT NULL DEFAULT 0,
    event_at_ms  INTEGER NOT NULL DEFAULT 0,  -- When the step happened (epoch ms, same source as the JSONL log; consumers format it themselves)
    created_at   TEXT NOT NULL,
    created_date TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_plog_request ON pipeline_logs_v2(request_id);
CREATE INDEX IF NOT EXISTS idx_plog_user    ON pipeline_logs_v2(user_id);

CREATE TABLE IF NOT EXISTS system_metrics (
    minute_ts  TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_ts ON system_metrics(minute_ts);


-- Episode (internal session) state: groups "one thing" under the same episode_id across write batches.
-- Only three values are stored; the segmentation rule (time gap) lives on the writer side, so no business logic here.
CREATE TABLE IF NOT EXISTS episode_state (
    episode_key    TEXT PRIMARY KEY,   -- user::agent::session
    user_id        TEXT NOT NULL DEFAULT '',
    agent_id       TEXT NOT NULL DEFAULT '',
    session_id     TEXT NOT NULL DEFAULT '',
    episode_id     TEXT NOT NULL DEFAULT '',
    last_write_ts  REAL NOT NULL DEFAULT 0,
    memory_count   INTEGER NOT NULL DEFAULT 0,
    updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episode_user ON episode_state(user_id);
"""


class SqliteCache(CacheBase):
    """
    SQLite audit/observability persistence backend (memory_operations / pipeline_logs / system_metrics).

    Zero-dependency local backend for single-machine / development environments.
    Uses the standard library sqlite3, wrapped as async through _run_in_sqlite_pool().
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

        # Read db_path
        cache_cfg = getattr(config, "cache", None)
        if cache_cfg and getattr(cache_cfg, "db_path", None):
            self._db_path = cache_cfg.db_path
        else:
            from ..config import _default_data_dir
            self._db_path = os.getenv(
                "MEMORY_CACHE_DB_PATH",
                os.path.join(_default_data_dir(), "data", "cache.db"),
            )

    # ================================================================
    # Lifecycle
    # ================================================================

    async def initialize(self) -> None:
        await _run_in_sqlite_pool(self._init_sync)
        logger.debug(f"SqliteCache initialized: {self._db_path}")

    def _init_sync(self) -> None:
        db_dir = os.path.dirname(self._db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_CREATE_TABLES_SQL)
        self._migrate_pipeline_logs_columns()
        self._conn.commit()

    def _migrate_pipeline_logs_columns(self) -> None:
        """Add the namespace/session/event time columns to pipeline_logs_v2 on older databases."""
        try:
            cols = {
                row["name"]
                for row in self._conn.execute("PRAGMA table_info(pipeline_logs_v2)")
            }
            _text_cols = ("namespace", "session_id")
            for col in _text_cols:
                if col not in cols:
                    self._conn.execute(
                        f"ALTER TABLE pipeline_logs_v2 ADD COLUMN {col} TEXT NOT NULL DEFAULT ''"
                    )
            if "event_at_ms" not in cols:
                self._conn.execute(
                    "ALTER TABLE pipeline_logs_v2 ADD COLUMN event_at_ms INTEGER NOT NULL DEFAULT 0"
                )
        except Exception as e:
            logger.debug(f"[sqlite] pipeline_logs_v2 column migration skipped: {e}")

    async def close(self) -> None:
        if self._conn is not None:
            await _run_in_sqlite_pool(self._close_sync)
        logger.info("SqliteCache closed")

    def _close_sync(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ================================================================
    # Statistics
    # ================================================================

    async def get_stats(self) -> Dict[str, Any]:
        return {
            "backend": "sqlite",
            "db_path": self._db_path,
        }

    # ================================================================
    # Memory Operations Log
    # ================================================================

    async def store_memory_operation(
        self,
        request_id: str,
        user_id: str,
        agent_id: str,
        op: str,
        memory_id: str,
        content: str,
        layer: str = "",
        old_memory_id: Optional[str] = None,
        reason: str = "",
        supersedes: Optional[List[str]] = None,
    ) -> bool:
        """Record one knowledge-base mutation (ADD / EVOLVE)"""
        from ..utils.pipeline_observability import is_memory_operations_enabled
        if not is_memory_operations_enabled():
            return True
        try:
            import json as _json
            from datetime import datetime as dt
            now = dt.now()
            created_at = now.isoformat()
            created_date = now.strftime("%Y-%m-%d")
            supersedes_str = _json.dumps(supersedes or [], ensure_ascii=False)

            def _insert():
                with self._lock:
                    self._conn.execute(
                        """INSERT INTO memory_operations_v2
                           (request_id, user_id, agent_id, op, memory_id, old_memory_id,
                            content, layer, reason, supersedes, created_at, created_date)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (request_id, user_id, agent_id, op, memory_id,
                         old_memory_id, content, layer, reason, supersedes_str,
                         created_at, created_date),
                    )
                    self._conn.commit()

            await _run_in_sqlite_pool(_insert)
            return True
        except Exception as e:
            logger.warning(f"store_memory_operation failed: {e}")
            return False

    async def get_memory_operations(
        self,
        request_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query knowledge-base mutation records, filterable by request_id / memory_id / user_id.
        At least one filter must be given.
        """
        try:
            def _query():
                conditions = []
                params = []
                if request_id:
                    conditions.append("request_id = ?")
                    params.append(request_id)
                if memory_id:
                    conditions.append("memory_id = ?")
                    params.append(memory_id)
                if user_id:
                    conditions.append("user_id = ?")
                    params.append(user_id)

                if not conditions:
                    return []

                where = " AND ".join(conditions)
                rows = self._conn.execute(
                    f"SELECT * FROM memory_operations_v2 WHERE {where} "
                    f"ORDER BY id DESC LIMIT ?",
                    params + [limit],
                ).fetchall()
                result = []
                for r in rows:
                    row = dict(r)
                    # Deserialize the supersedes JSON string → list
                    import json as _json
                    raw_sup = row.get("supersedes", "")
                    try:
                        row["supersedes"] = _json.loads(raw_sup) if raw_sup else []
                    except Exception:
                        row["supersedes"] = []
                    result.append(row)
                return result

            return await _run_in_sqlite_pool(_query)
        except Exception as e:
            logger.warning(f"get_memory_operations failed: {e}")
            return []

    # ================================================================
    # Pipeline Logs (intermediate results of the LLM call chain)
    # ================================================================

    async def store_pipeline_log(
        self,
        request_id: str,
        user_id: str,
        agent_id: str,
        step: str,
        prompt: str,
        response: str,
        parsed: str = "",
        memory_ids: Optional[List[str]] = None,
        elapsed_ms: float = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        session_id: str = "",
        event_at_ms: int = 0,
    ) -> bool:
        """
        Record one intermediate pipeline result (EXTRACT / SEARCH_QUERY / RECONCILE / SUMMARY).

        user_id is split on "__" so the namespace is persisted separately; session_id, like user/agent, is an isolation dimension.
        event_at_ms: when the step happened (epoch ms, same source as the JSONL log timestamp); falls back to the insert time when omitted.
        """
        try:
            from datetime import datetime as dt
            from ..utils.pipeline_observability import split_namespace_user
            now = dt.now()
            created_at = now.isoformat()
            created_date = now.strftime("%Y-%m-%d")
            event_at_ms = int(event_at_ms) if event_at_ms else int(now.timestamp() * 1000)
            mem_ids_json = json.dumps(memory_ids or [])
            namespace, uid = split_namespace_user(user_id)

            def _insert():
                with self._lock:
                    self._conn.execute(
                        """INSERT INTO pipeline_logs_v2
                           (request_id, namespace, user_id, agent_id, session_id,
                            step, prompt, response,
                            parsed, memory_ids, elapsed_ms,
                            prompt_tokens, completion_tokens, total_tokens,
                            event_at_ms, created_at, created_date)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (request_id, namespace, uid, agent_id, session_id or "",
                         step, prompt, response,
                         parsed, mem_ids_json, elapsed_ms,
                         prompt_tokens, completion_tokens, total_tokens,
                         event_at_ms, created_at, created_date),
                    )
                    self._conn.commit()

            await _run_in_sqlite_pool(_insert)
            return True
        except Exception as e:
            logger.warning(f"store_pipeline_log failed: {e}")
            return False

    async def get_pipeline_logs(
        self,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        step: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query intermediate pipeline result logs.

        Filterable by request_id / user_id / step; at least one must be given.
        """
        try:
            def _query():
                conditions = []
                params = []
                if request_id:
                    conditions.append("request_id = ?")
                    params.append(request_id)
                if user_id:
                    conditions.append("user_id = ?")
                    params.append(user_id)
                if step:
                    conditions.append("step = ?")
                    params.append(step)

                if not conditions:
                    return []

                where = " AND ".join(conditions)
                rows = self._conn.execute(
                    f"SELECT * FROM pipeline_logs_v2 WHERE {where} "
                    f"ORDER BY id ASC LIMIT ?",
                    params + [limit],
                ).fetchall()
                out = [dict(r) for r in rows]
                # Derive a human-readable event_at (ISO ms) for the query API; left empty when event_at_ms is 0
                from datetime import datetime as _dt
                for _r in out:
                    _ms = _r.get("event_at_ms") or 0
                    _r["event_at"] = (
                        _dt.fromtimestamp(_ms / 1000.0).isoformat(timespec="milliseconds")
                        if _ms else ""
                    )
                return out

            return await _run_in_sqlite_pool(_query)
        except Exception as e:
            logger.warning(f"get_pipeline_logs failed: {e}")
            return []

    # ================================================================
    # System Metrics (persisted at minute granularity)
    # ================================================================

    async def store_metrics_minute(self, minute_ts: str, data: dict) -> None:
        """Store the incremental metrics for one minute bucket (UPSERT: repeated writes in the same minute are merged)"""
        try:
            def _store():
                now = datetime.now().isoformat()
                # UPSERT: merge the data when this minute already exists
                existing = self._conn.execute(
                    "SELECT data FROM system_metrics WHERE minute_ts = ?",
                    (minute_ts,),
                ).fetchone()
                if existing:
                    import json as _json
                    old_data = _json.loads(existing["data"])
                    merged = self._merge_metric_buckets(old_data, data)
                    self._conn.execute(
                        "UPDATE system_metrics SET data = ?, created_at = ? WHERE minute_ts = ?",
                        (_json.dumps(merged, default=str), now, minute_ts),
                    )
                else:
                    self._conn.execute(
                        "INSERT INTO system_metrics (minute_ts, data, created_at) VALUES (?, ?, ?)",
                        (minute_ts, json.dumps(data, default=str), now),
                    )
                self._conn.commit()
            await _run_in_sqlite_pool(_store)
        except Exception as e:
            logger.warning(f"store_metrics_minute failed: {e}")

    async def load_metrics_range(self, start_ts: str, end_ts: str) -> list:
        """Read minute metric data within the given time range"""
        try:
            def _load():
                rows = self._conn.execute(
                    "SELECT data FROM system_metrics WHERE minute_ts >= ? AND minute_ts <= ? ORDER BY minute_ts",
                    (start_ts, end_ts),
                ).fetchall()
                results = []
                for row in rows:
                    try:
                        results.append(json.loads(row["data"]))
                    except (json.JSONDecodeError, TypeError):
                        pass
                return results
            return await _run_in_sqlite_pool(_load)
        except Exception as e:
            logger.warning(f"load_metrics_range failed: {e}")
            return []

    async def cleanup_old_metrics(self, before_ts: str) -> None:
        """Delete metric data before the given timestamp"""
        try:
            def _cleanup():
                self._conn.execute(
                    "DELETE FROM system_metrics WHERE minute_ts < ?",
                    (before_ts,),
                )
                self._conn.commit()
            await _run_in_sqlite_pool(_cleanup)
        except Exception as e:
            logger.warning(f"cleanup_old_metrics failed: {e}")

    # ================================================================
    # Summary Buffer
    # ================================================================




    async def get_episode_state(self, episode_key: str) -> Optional[Dict[str, Any]]:
        try:
            def _query():
                with self._lock:
                    row = self._conn.execute(
                        "SELECT episode_id, last_write_ts, memory_count "
                        "FROM episode_state WHERE episode_key = ?",
                        (episode_key,),
                    ).fetchone()
                return dict(row) if row else None

            row = await _run_in_sqlite_pool(_query)
            if not row:
                return None
            return {
                "episode_id": row.get("episode_id") or "",
                "last_write_ts": float(row.get("last_write_ts") or 0),
                "memory_count": int(row.get("memory_count") or 0),
            }
        except Exception as e:
            logger.warning(f"get_episode_state failed: {e}")
            return None

    async def upsert_episode_state(
        self,
        episode_key: str,
        *,
        user_id: str,
        agent_id: str,
        session_id: str,
        episode_id: str,
        last_write_ts: float,
        memory_count: int,
    ) -> bool:
        try:
            from datetime import datetime as _dt

            def _write():
                with self._lock:
                    self._conn.execute(
                        """
                        INSERT INTO episode_state (
                            episode_key, user_id, agent_id, session_id,
                            episode_id, last_write_ts, memory_count, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(episode_key) DO UPDATE SET
                            episode_id = excluded.episode_id,
                            last_write_ts = excluded.last_write_ts,
                            memory_count = excluded.memory_count,
                            updated_at = excluded.updated_at
                        """,
                        (episode_key, user_id, agent_id, session_id, episode_id,
                         float(last_write_ts), int(memory_count),
                         _dt.now().isoformat()),
                    )
                    self._conn.commit()
                return True

            return await _run_in_sqlite_pool(_write)
        except Exception as e:
            logger.warning(f"upsert_episode_state failed: {e}")
            return False

    @staticmethod
    def _merge_metric_buckets(old: dict, new: dict) -> dict:
        """Merge two metric buckets (multiple flushes within the same minute)"""
        merged = dict(old)
        for key in ("sys1_started", "sys1_completed", "sys1_failed",
                    "sys2_started", "sys2_completed", "sys2_failed",
                    "vdb_ops", "graph_ops"):
            merged[key] = merged.get(key, 0) + new.get(key, 0)
        for key in ("vdb_ops_sum_ms", "graph_ops_sum_ms"):
            merged[key] = merged.get(key, 0) + new.get(key, 0)
        # timing sums
        for bucket_key in ("sys1_timing_sums", "sys2_timing_sums"):
            old_sums = merged.get(bucket_key, {})
            new_sums = new.get(bucket_key, {})
            for k, v in new_sums.items():
                old_sums[k] = old_sums.get(k, 0) + v
            merged[bucket_key] = old_sums
        return merged
