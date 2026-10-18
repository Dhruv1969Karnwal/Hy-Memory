"""
Agent Memory - Core layer (Core Layer)

Core components:
- Scorer: scorer, computes the composite score of a memory
- Merger: merger, detects duplicate memories and merges them
- EmbedService: embedding service, text to vector
"""

from .scorer import MemoryScorer as Scorer
from .merger import Merger, MergeResult, MergerConfig
from .embed_service import EmbedService

__all__ = [
    "Scorer",
    "Merger",
    "MergeResult",
    "MergerConfig",
    "EmbedService",
]
