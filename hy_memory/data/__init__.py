"""
Agent Memory V2 - Data Layer

Three-engine storage architecture:
- VectorStore (Chroma/Qdrant/FAISS):  vector storage, semantic retrieval for L1 Raw / L2 Fact / Schema / Intention
- GraphStore  (Kuzu/Neo4j): graph storage, entity relations / graph traversal / timeline / knowledge gaps
- Cache       (SQLite/MySQL): audit/observability persistence (memory_operations / pipeline_logs / system_metrics)

Cache backends:
- SqliteCache: dependency-free local (default)
- MysqlCache:  standard MySQL persistence (requires the aiomysql package)
- create_cache(): factory function that selects the backend from configuration
"""

from .vector_store import VectorStore, create_vector_store
from .vector_store_base import VectorStoreBase
from .graph_store import GraphStore, create_graph_store
from .graph_store_base import GraphStoreBase
from .cache_base import CacheBase
from .cache_sqlite import SqliteCache
from .cache_mysql import MysqlCache
from .cache import create_cache
from .history_store import HistoryStore

# Legacy interface compatibility kept
from .kv_store import KVStore, KVStoreConfig, KVBackend

__all__ = [
    # V2 core
    "VectorStore",
    "VectorStoreBase",
    "create_vector_store",
    "GraphStore",
    "GraphStoreBase",
    "create_graph_store",
    "CacheBase",
    "SqliteCache",
    "MysqlCache",
    "create_cache",
    "HistoryStore",

    # V1 compat
    "KVStore",
    "KVStoreConfig",
    "KVBackend",
]
