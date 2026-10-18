"""
Agent Memory V2 - VectorStore provider factory

Selects the vector database backend (Chroma / Qdrant / FAISS) from config.

Usage:
    from .vector_store import create_vector_store, VectorStore

    # Factory function (recommended)
    store = create_vector_store(config)

    # Backward-compatible alias (creates the instance of the matching provider)
    store = VectorStore(config)
"""

import logging
from typing import TYPE_CHECKING

from ..config import MemoryConfig
from .vector_store_base import VectorStoreBase

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def create_vector_store(config: MemoryConfig) -> VectorStoreBase:
    """
    Factory function: create the matching VectorStore implementation from config.vector_store.provider.

    - "chroma"  → ChromaVectorStore  (embedded, no external service dependency)
    - "qdrant"  → QdrantVectorStore  (embedded / remote)
    - "faiss"   → FaissVectorStore   (embedded, CPU)
    A missing or unknown provider raises immediately; there is no silent fallback to chroma.
    """
    provider = getattr(config.vector_store, 'provider', None)
    if not provider or not str(provider).strip():
        raise ValueError(
            "config.vector_store.provider is required "
            "(one of: chroma / qdrant / faiss)"
        )
    provider = str(provider).lower().strip()

    if provider == "qdrant":
        from .vector_store_qdrant import QdrantVectorStore
        logger.debug("VectorStore provider: qdrant")
        return QdrantVectorStore(config)
    elif provider == "faiss":
        from .vector_store_faiss import FaissVectorStore
        logger.debug("VectorStore provider: faiss")
        return FaissVectorStore(config)
    elif provider == "chroma":
        from .vector_store_chroma import ChromaVectorStore
        logger.debug("VectorStore provider: chroma")
        return ChromaVectorStore(config)
    else:
        raise ValueError(
            f"Unknown vector_store provider {provider!r}; "
            f"must be one of: chroma / qdrant / faiss"
        )


# Backward compatibility: `from .vector_store import VectorStore` still works
# VectorStore is now an alias for the factory function; VectorStore(config) is equivalent to create_vector_store(config)
def VectorStore(config: MemoryConfig) -> VectorStoreBase:
    """Backward-compatible alias - equivalent to create_vector_store(config)"""
    return create_vector_store(config)
