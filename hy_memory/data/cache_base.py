"""
Agent Memory V2 - CacheBase

Abstract base class of the audit / observability persistence layer.
Defines the public interface every backend (SQLite / MySQL) must implement:
memory_operations (change log), pipeline_logs (LLM call chain), system_metrics (metrics).
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class CacheBase(ABC):
    """
    Abstract base class for audit / observability persistence.

    Every backend (SqliteCache / MysqlCache) inherits from this class,
    and upper layers depend only on the CacheBase type.
    """

    # ================================================================
    # Lifecycle
    # ================================================================

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend connection / database."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close connections and release resources."""
        ...

    # ================================================================
    # Statistics
    # ================================================================

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        ...

    # ================================================================
    # Memory Operations Log
    # ================================================================

    @abstractmethod
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
        """Record one knowledge-base change operation (ADD / EVOLVE)"""
        ...

    @abstractmethod
    async def get_memory_operations(
        self,
        request_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query knowledge-base change records, filterable by request_id / memory_id / user_id"""
        ...

    # ================================================================
    # Pipeline Logs (intermediate results of the LLM call chain)
    # ================================================================

    @abstractmethod
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
        """Record one pipeline intermediate result (EXTRACT / SEARCH_QUERY / RECONCILE / SUMMARY)

        session_id is an isolation dimension like user_id/agent_id and is passed in explicitly by the caller.
        Implementations split user_id on "__" into a namespace that is stored separately.

        event_at_ms: when the step happened (epoch milliseconds; consumers convert it for display), captured uniformly by the upper layer,
        sharing its source with the timestamp of the JSONL log. When unset (0), implementations fall back to the insertion time; distinct from created_at (insertion time)."""
        ...

    @abstractmethod
    async def get_pipeline_logs(
        self,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        step: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query pipeline intermediate result logs"""
        ...

    # ================================================================
    # Summary Buffer (unconsumed turns staged for rolling summary)
    #
    # No-op by default, so backends without this capability can inherit directly.
    # Persistent backends (sqlite/mysql) override these with real implementations.
    # ================================================================




    async def get_episode_state(self, episode_key: str) -> Optional[Dict[str, Any]]:
        """Read the current episode state for an episode_key. Returns None by default (no state → start a new one)."""
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
        """Write / update episode state. No-op by default."""
        return False
