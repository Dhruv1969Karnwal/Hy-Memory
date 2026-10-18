"""
Agent Memory V2 - Cache factory

Selects the cache backend from the configuration:
- sqlite (default): zero dependencies, works out of the box on a single machine
- mysql: standard MySQL persistence, requires the aiomysql package

Usage:
    from hy_memory.data.cache import create_cache
    cache = create_cache(config)
    await cache.initialize()
"""

import logging
from .cache_base import CacheBase
from ..config import MemoryConfig

logger = logging.getLogger(__name__)

_SUPPORTED_BACKENDS = ("sqlite", "mysql")


def create_cache(config: MemoryConfig) -> CacheBase:
    """
    Create a cache backend instance from the configuration.

    Supported backends:
    - "sqlite": SqliteCache (default, zero-dependency local persistence)
    - "mysql":  MysqlCache (standard MySQL, requires the aiomysql package)

    Args:
        config: MemoryConfig instance

    Returns:
        A CacheBase subclass instance

    Raises:
        ValueError: backend is missing or not one of sqlite/mysql
    """
    backend_raw = getattr(getattr(config, "cache", None), "backend", None)
    if not backend_raw or not str(backend_raw).strip():
        raise ValueError(
            "config.cache.backend is required (one of: sqlite / mysql)"
        )
    backend = str(backend_raw).lower().strip()

    if backend == "mysql":
        from .cache_mysql import MysqlCache
        logger.info("Cache backend: mysql")
        return MysqlCache(config)
    elif backend == "sqlite":
        from .cache_sqlite import SqliteCache
        return SqliteCache(config)
    else:
        raise ValueError(
            f"Unknown cache backend {backend!r}; must be one of: sqlite / mysql"
        )


__all__ = ["CacheBase", "create_cache"]
