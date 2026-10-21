"""
Agent Memory V2 - GraphStore provider factory

Selects the graph database backend (Kuzu / Neo4j) from config.

Usage:
    from .graph_store import create_graph_store, GraphStore

    # Factory function (recommended)
    graph = create_graph_store(config)

    # Backward-compatible alias
    graph = GraphStore(config)
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import MemoryConfig
from .graph_store_base import GraphStoreBase

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def graph_data_may_exist(config: MemoryConfig) -> bool:
    """
    Decide whether graph data exists that needs purging (independent of MEMORY_MODE).

    - Kuzu: return True only when a database file/directory already exists at db_path, so pro mode does not create an empty store.
    - Neo4j/Memgraph: external services, always try to connect and purge.
    """
    provider_raw = getattr(config.graph_store, "provider", None)
    if not provider_raw or not str(provider_raw).strip():
        raise ValueError(
            "config.graph_store.provider is required "
            "(one of: kuzu / neo4j / memgraph)"
        )
    provider = str(provider_raw).lower().strip()
    if provider in ("neo4j", "memgraph"):
        return True
    if provider != "kuzu":
        raise ValueError(
            f"Unknown graph_store provider {provider!r}; "
            f"must be one of: kuzu / neo4j / memgraph"
        )

    db_path = Path(config.graph_store.db_path)
    if not db_path.exists():
        return False
    if db_path.is_file():
        return True
    if db_path.is_dir():
        try:
            return any(db_path.iterdir())
        except OSError:
            return False
    return False


def create_graph_store(config: MemoryConfig) -> GraphStoreBase:
    """
    Factory function: create the matching GraphStore implementation from config.graph_store.provider.

    - "kuzu"     → KuzuGraphStore     (embedded)
    - "neo4j"    → Neo4jGraphStore    (client-server)
    - "memgraph" → Neo4jGraphStore    (bolt protocol compatible, reuses the Neo4j driver)

    A missing or unknown provider raises immediately; there is no silent fallback to kuzu.
    """
    provider = getattr(config.graph_store, 'provider', None)
    if not provider or not str(provider).strip():
        raise ValueError(
            "config.graph_store.provider is required "
            "(one of: kuzu / neo4j / memgraph)"
        )
    provider = str(provider).lower().strip()

    if provider in ("neo4j", "memgraph"):
        from .graph_store_neo4j import Neo4jGraphStore
        logger.debug(f"GraphStore provider: {provider}")
        return Neo4jGraphStore(config)
    elif provider == "kuzu":
        from .graph_store_kuzu import KuzuGraphStore
        logger.debug("GraphStore provider: kuzu")
        return KuzuGraphStore(config)
    else:
        raise ValueError(
            f"Unknown graph_store provider {provider!r}; "
            f"must be one of: kuzu / neo4j / memgraph"
        )


# Backward compatibility: `from .graph_store import GraphStore` still works
# GraphStore is now an alias for the factory function; GraphStore(config) is equivalent to create_graph_store(config)
def GraphStore(config: MemoryConfig) -> GraphStoreBase:
    """Backward-compatible alias -- equivalent to create_graph_store(config)"""
    return create_graph_store(config)
