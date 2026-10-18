"""
Agent Memory - KVStore key-value store

Provides fast key-value access.

Features:
- Key-value get/set/delete
- Batch operations
- TTL support
- Multiple backends (Redis, RocksDB, Memory)

Use cases:
- Storing user profiles
- Storing task state
- Caching hot data

Example:
    kv = KVStore(config)
    
    # Store
    kv.set("profile:user_123", {"name": "Alice", "age": 30})
    
    # Get
    profile = kv.get("profile:user_123")
    
    # Set a TTL
    kv.set("session:xxx", {"token": "..."}, ttl=3600)
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class KVBackend(Enum):
    """KV storage backend"""
    REDIS = "redis"
    ROCKSDB = "rocksdb"
    MEMORY = "memory"       # In-memory storage


@dataclass
class KVStoreConfig:
    """
    KV store configuration
    """
    backend: str = "memory"
    
    # Redis configuration
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    
    # Common configuration
    prefix: str = "agentmem:"
    default_ttl: int = 0             # 0 means it never expires
    max_size: int = 100000           # Maximum number of keys


@dataclass
class KVEntry:
    """KV entry, including its expiry time"""
    value: Any
    expires_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class KVStore:
    """
    Key-value store
    
    Provides simple key-value access.
    """
    
    def __init__(self, config: Optional[KVStoreConfig] = None):
        """
        Initialize the KV store
        
        Args:
            config: storage configuration
        """
        self.config = config or KVStoreConfig()
        self._client = None
        
        self._init_client()
        logger.info(f"KVStore initialized, backend={self.config.backend}")
    
    def _init_client(self) -> None:
        """Initialize the storage client"""
        if self.config.backend == "redis":
            self._init_redis()
        elif self.config.backend == "memory":
            self._init_memory()
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")
    
    def _init_redis(self) -> None:
        """Initialize the Redis client"""
        try:
            import redis
            
            self._client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password or None,
                decode_responses=True,
            )
            # Test the connection
            self._client.ping()
            
        except ImportError:
            raise ImportError("redis is required. Install with: pip install redis")
    
    def _init_memory(self) -> None:
        """Initialize the in-memory store"""
        self._memory_store: Dict[str, KVEntry] = {}
    
    def _make_key(self, key: str) -> str:
        """Build the prefixed key"""
        return f"{self.config.prefix}{key}"
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value
        
        Args:
            key: key
            default: default value
        
        Returns:
            The value, or the default
        """
        full_key = self._make_key(key)
        
        if self.config.backend == "memory":
            entry = self._memory_store.get(full_key)
            if entry is None:
                return default
            if entry.is_expired():
                del self._memory_store[full_key]
                return default
            return entry.value
        
        elif self.config.backend == "redis":
            value = self._client.get(full_key)
            if value is None:
                return default
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        
        return default
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: int = None,
    ) -> bool:
        """
        Set a value
        
        Args:
            key: key
            value: value
            ttl: expiry time in seconds, None uses the default
        
        Returns:
            Whether it succeeded
        """
        full_key = self._make_key(key)
        ttl = ttl if ttl is not None else self.config.default_ttl
        
        if self.config.backend == "memory":
            # Enforce the size limit
            if len(self._memory_store) >= self.config.max_size:
                self._evict_expired()
            
            expires_at = None
            if ttl > 0:
                expires_at = datetime.now() + timedelta(seconds=ttl)
            
            self._memory_store[full_key] = KVEntry(
                value=value,
                expires_at=expires_at,
            )
            return True
        
        elif self.config.backend == "redis":
            value_str = json.dumps(value) if not isinstance(value, str) else value
            if ttl > 0:
                self._client.setex(full_key, ttl, value_str)
            else:
                self._client.set(full_key, value_str)
            return True
        
        return False
    
    def delete(self, key: str) -> bool:
        """
        Delete a key
        
        Args:
            key: key
        
        Returns:
            Whether it succeeded
        """
        full_key = self._make_key(key)
        
        if self.config.backend == "memory":
            if full_key in self._memory_store:
                del self._memory_store[full_key]
                return True
        
        elif self.config.backend == "redis":
            return self._client.delete(full_key) > 0
        
        return False
    
    def exists(self, key: str) -> bool:
        """
        Check whether a key exists
        
        Args:
            key: key
        
        Returns:
            Whether it exists
        """
        full_key = self._make_key(key)
        
        if self.config.backend == "memory":
            entry = self._memory_store.get(full_key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._memory_store[full_key]
                return False
            return True
        
        elif self.config.backend == "redis":
            return self._client.exists(full_key) > 0
        
        return False
    
    def keys(self, pattern: str = "*") -> List[str]:
        """
        Get the list of matching keys
        
        Args:
            pattern: match pattern
        
        Returns:
            The list of keys
        """
        full_pattern = self._make_key(pattern)
        prefix_len = len(self.config.prefix)
        
        if self.config.backend == "memory":
            import fnmatch
            result = []
            for key in self._memory_store.keys():
                if fnmatch.fnmatch(key, full_pattern):
                    result.append(key[prefix_len:])
            return result
        
        elif self.config.backend == "redis":
            keys = self._client.keys(full_pattern)
            return [k[prefix_len:] for k in keys]
        
        return []
    
    def mget(self, keys: List[str]) -> Dict[str, Any]:
        """
        Batch get
        
        Args:
            keys: list of keys
        
        Returns:
            Dict of key-value pairs
        """
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result
    
    def mset(self, items: Dict[str, Any], ttl: int = None) -> bool:
        """
        Batch set
        
        Args:
            items: dict of key-value pairs
            ttl: expiry time
        
        Returns:
            Whether it succeeded
        """
        for key, value in items.items():
            self.set(key, value, ttl)
        return True
    
    def _evict_expired(self) -> int:
        """Evict expired keys"""
        if self.config.backend != "memory":
            return 0
        
        expired_keys = [
            key for key, entry in self._memory_store.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self._memory_store[key]
        
        return len(expired_keys)
    
    def clear(self) -> None:
        """Clear all data"""
        if self.config.backend == "memory":
            self._memory_store.clear()
        elif self.config.backend == "redis":
            keys = self._client.keys(f"{self.config.prefix}*")
            if keys:
                self._client.delete(*keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        if self.config.backend == "memory":
            return {
                "backend": "memory",
                "total_keys": len(self._memory_store),
                "max_size": self.config.max_size,
            }
        elif self.config.backend == "redis":
            info = self._client.info()
            return {
                "backend": "redis",
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", ""),
            }
        return {}
