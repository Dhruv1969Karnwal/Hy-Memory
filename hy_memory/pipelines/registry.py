"""
HY Memory - Component Factory

ComponentFactory is responsible for:
1. Managing shared resources (EmbedService, VectorStore, GraphStore, Cache)
2. Creating MemoryWriter / MemoryReader / System2Writer instances (singleton cache)
3. Managing component lifecycles

Usage:
    factory = ComponentFactory(config=config)
    writer = await factory.get_writer()
    reader = await factory.get_reader()
    system2 = await factory.get_system2_writer()
"""

from typing import Optional, Dict
import logging

from .base import WritePipeline, ReadPipeline
from ..config import MemoryConfig
from ..core.embed_service import EmbedService
from ..data.vector_store import create_vector_store
from ..data.vector_store_base import VectorStoreBase

logger = logging.getLogger(__name__)


# ================================================================
# Backward-compatible aliases — for legacy code / external imports
# ================================================================

class PipelineConfig:
    """Backward compat: deprecated, kept so that old imports do not raise ImportError."""
    def __init__(self, **kwargs):
        pass


class ComponentFactory:
    """
    Component factory

    Manages Writer / Reader / System2Writer instances and the shared resources.
    client.py creates and retrieves components through this factory.

    Usage:
        factory = ComponentFactory(config=config)
        writer = await factory.get_writer()       # MemoryWriter
        reader = await factory.get_reader()        # MemoryReader (default)
        reader = await factory.get_reader(reader_name="hybrid")     # named reader
        s2 = await factory.get_system2_writer()    # System2Writer (ultra mode)
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig.from_env()

        # Instance cache (singletons)
        self._writer: Optional[WritePipeline] = None
        self._reader: Optional[ReadPipeline] = None
        self._system2_writer: Optional[WritePipeline] = None

        # Multi-reader cache (indexed by reader_name)
        self._readers: Dict[str, ReadPipeline] = {}

        # Shared resources
        self._shared_embed_service: Optional[EmbedService] = None
        self._shared_vector_store: Optional[VectorStoreBase] = None
        self._shared_vector_store_initialized: bool = False
        self._shared_graph_store = None
        self._shared_cache = None

    # ================================================================
    # Shared resources
    # ================================================================

    def _get_shared_embed_service(self) -> EmbedService:
        if self._shared_embed_service is None:
            self._shared_embed_service = EmbedService(self.config)
        return self._shared_embed_service

    def _get_shared_vector_store(self) -> VectorStoreBase:
        if self._shared_vector_store is None:
            self._shared_vector_store = create_vector_store(self.config)
            logger.debug("ComponentFactory: VectorStore created")
        return self._shared_vector_store

    # ================================================================
    # Acquire components (auto-initialize + cache)
    # ================================================================

    async def get_writer(self, **kwargs) -> WritePipeline:
        """
        Get the MemoryWriter instance (System 1 writer).

        All modes (lite/pro/ultra) use the same MemoryWriter;
        the only difference is the agent_mode argument the client passes in.
        """
        if self._writer is not None:
            return self._writer

        from .writer import MemoryWriter
        writer = MemoryWriter(
            config=self.config,
            embed_service=self._get_shared_embed_service(),
            vector_store=self._get_shared_vector_store(),
            cache=self._shared_cache,
            # Memory relation graph (pro mode): the write side projects L2_FACT into the graph and builds RELATED_TO / TAGGED_WITH.
            # When None, the whole graph projection inside the writer is skipped.
            graph_store=self._shared_graph_store,
        )
        await writer.initialize()
        self._writer = writer
        logger.debug("MemoryWriter initialized")
        return writer

    async def get_reader(self, reader_name: str = "", **kwargs) -> ReadPipeline:
        """
        Get the MemoryReader instance.

        Args:
            reader_name: reader type to use (legacy / hybrid).
                         Empty string = the HY_MEMORY_READER env var, otherwise the default legacy.

        Readers can be switched per request; one instance is cached per reader type.
        """
        # Resolve the effective reader name
        from ._retrieval.config import resolve_reader_name
        effective_name = resolve_reader_name(reader_name)

        # Check the cache
        if effective_name in self._readers:
            return self._readers[effective_name]

        # Backward compat: with no name given, use the legacy _reader singleton
        if not reader_name and self._reader is not None:
            return self._reader

        reader_kwargs = {
            "config": self.config,
            "embed_service": self._get_shared_embed_service(),
            "vector_store": self._get_shared_vector_store(),
            "cache": self._shared_cache,
        }
        if self._shared_graph_store is not None:
            reader_kwargs["graph_store"] = self._shared_graph_store

        # Build the requested reader directly with _build_impl (bypassing the MemoryReader dispatcher)
        import os
        old_env = os.environ.get("HY_MEMORY_READER", "")
        os.environ["HY_MEMORY_READER"] = effective_name
        try:
            from .reader import MemoryReader
            reader = MemoryReader(**reader_kwargs)
            await reader.initialize()
        finally:
            if old_env:
                os.environ["HY_MEMORY_READER"] = old_env
            else:
                os.environ.pop("HY_MEMORY_READER", None)

        # Cache
        self._readers[effective_name] = reader
        if not reader_name:
            self._reader = reader
        logger.debug(f"MemoryReader initialized: {effective_name}")
        return reader

    async def get_system2_writer(self, **kwargs) -> WritePipeline:
        """
        Get the System2Writer instance (ultra mode only).

        System2Writer internally wraps a MemoryWriter (System 1) plus System 2 cognitive processing.
        """
        if self._system2_writer is not None:
            return self._system2_writer

        from .system2_writer import System2Writer
        s2_kwargs = {
            "config": self.config,
            "embed_service": self._get_shared_embed_service(),
            "vector_store": self._get_shared_vector_store(),
            "cache": self._shared_cache,
        }
        if self._shared_graph_store is not None:
            s2_kwargs["graph_store"] = self._shared_graph_store
        s2 = System2Writer(**s2_kwargs)
        await s2.initialize()
        self._system2_writer = s2
        logger.debug("System2Writer initialized")
        return s2

    # ================================================================
    # Lifecycle
    # ================================================================

    async def close(self) -> None:
        """Close all components and shared resources"""
        for name, inst in [("writer", self._writer), ("reader", self._reader),
                           ("system2_writer", self._system2_writer)]:
            if inst is not None:
                try:
                    await inst.close()
                except Exception as e:
                    logger.error(f"Failed to close {name}: {e}")

        if self._shared_vector_store:
            try:
                await self._shared_vector_store.close()
            except Exception as e:
                logger.error(f"Failed to close VectorStore: {e}")

        self._writer = None
        self._reader = None
        self._system2_writer = None
        self._shared_vector_store = None
        self._shared_vector_store_initialized = False
        self._shared_embed_service = None

        logger.debug("ComponentFactory closed")


# ================================================================
# Backward-compatible aliases
# ================================================================

PipelineRegistry = ComponentFactory
