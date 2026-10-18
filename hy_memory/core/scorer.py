"""
Agent Memory - Memory scoring

Computes a composite memory score from semantic similarity, time decay, importance and other factors
"""

import math
from typing import List, Optional
from datetime import datetime

from ..models import MemoryEntry, MemoryScore
from ..config import RecallConfig


class MemoryScorer:
    """
    Memory scorer
    
    Computes a composite memory score, covering:
    - Semantic similarity score
    - Time decay score
    - Importance score
    - Access frequency score
    """
    
    def __init__(self, config: Optional[RecallConfig] = None):
        """
        Initialize the scorer
        
        Args:
            config: recall configuration
        """
        self.config = config or RecallConfig()
    
    def score(
        self,
        entry: MemoryEntry,
        semantic_score: float = 0.0,
        reference_time: Optional[datetime] = None
    ) -> MemoryScore:
        """
        Compute the composite score of a single memory
        
        Args:
            entry: memory entry
            semantic_score: semantic similarity score (0-1)
            reference_time: reference time used to compute time decay
        
        Returns:
            The scoring result
        """
        if reference_time is None:
            reference_time = datetime.now()
        
        # 1. Semantic similarity score (already provided by vector retrieval)
        semantic = max(0.0, min(1.0, semantic_score))
        
        # 2. Time decay score
        recency = self._calc_recency_score(entry, reference_time)
        
        # 3. Importance score
        importance = self._calc_importance_score(entry)
        
        # 4. Access frequency score
        access = self._calc_access_score(entry)
        
        return MemoryScore(
            semantic_score=semantic,
            recency_score=recency,
            importance_score=importance,
            access_score=access,
            semantic_weight=self.config.semantic_weight,
            recency_weight=self.config.recency_weight,
            importance_weight=self.config.importance_weight,
            access_weight=self.config.access_weight,
        )
    
    def _get_event_time(self, entry: MemoryEntry) -> Optional[datetime]:
        """Get the event time of entry, compatible with V2 MemoryNode and V1 metadata"""
        # V2: MemoryNode has created_at / valid_from directly
        if hasattr(entry, 'created_at') and entry.created_at is not None:
            return entry.created_at
        # V1: entry.metadata.event_time
        if hasattr(entry, 'metadata') and entry.metadata and hasattr(entry.metadata, 'event_time'):
            return entry.metadata.event_time
        return None

    def _get_importance(self, entry: MemoryEntry) -> float:
        """Get the importance of entry, compatible with V2 MemoryNode and V1 metadata"""
        # V2: MemoryNode exposes the confidence field as the importance indicator
        if hasattr(entry, 'confidence') and entry.confidence is not None:
            return entry.confidence
        # V1: entry.metadata.importance
        if hasattr(entry, 'metadata') and entry.metadata and hasattr(entry.metadata, 'importance'):
            return entry.metadata.importance
        return 0.5

    def _get_access_count(self, entry: MemoryEntry) -> int:
        """Get the access count of entry, compatible with V2 MemoryNode and V1 metadata"""
        # V2: MemoryNode has access_count directly
        if hasattr(entry, 'access_count') and entry.access_count is not None:
            return entry.access_count
        # V1: entry.metadata.access_count
        if hasattr(entry, 'metadata') and entry.metadata and hasattr(entry.metadata, 'access_count'):
            return entry.metadata.access_count
        return 0

    def _get_semantic_score(self, entry: MemoryEntry) -> float:
        """Get the semantic score of entry, compatible with the score object"""
        if hasattr(entry, 'score') and entry.score and hasattr(entry.score, 'semantic_score'):
            return entry.score.semantic_score
        return 0.0

    def _get_final_score(self, entry: MemoryEntry) -> float:
        """Get the final score of entry, compatible with the score object"""
        if hasattr(entry, 'score') and entry.score and hasattr(entry.score, 'final_score'):
            return entry.score.final_score
        return 0.0

    def _calc_recency_score(
        self,
        entry: MemoryEntry,
        reference_time: datetime
    ) -> float:
        """
        Compute the time decay score
        
        Uses exponential decay: score = decay_factor ^ (days / decay_days)
        """
        event_time = self._get_event_time(entry)
        if event_time is None:
            return 0.5  # Default medium score
        days_passed = (reference_time - event_time).days
        
        if days_passed <= 0:
            return 1.0
        
        # Exponential decay
        decay_days = self.config.recency_decay_days
        decay_factor = self.config.recency_decay_factor
        
        score = math.pow(decay_factor, days_passed / decay_days)
        return max(0.0, min(1.0, score))
    
    def _calc_importance_score(self, entry: MemoryEntry) -> float:
        """Compute the importance score"""
        return max(0.0, min(1.0, self._get_importance(entry)))
    
    def _calc_access_score(self, entry: MemoryEntry) -> float:
        """
        Compute the access frequency score
        
        Uses a logarithm: score = log(access_count + 1) / log(max_count + 1)
        """
        access_count = self._get_access_count(entry)
        
        # Assume the maximum access count is 100
        max_count = 100
        
        if access_count <= 0:
            return 0.0
        
        score = math.log(access_count + 1) / math.log(max_count + 1)
        return max(0.0, min(1.0, score))
    
    def score_batch(
        self,
        entries: List[MemoryEntry],
        reference_time: Optional[datetime] = None
    ) -> List[MemoryEntry]:
        """
        Compute scores in batch
        
        Args:
            entries: list of memories
            reference_time: reference time
        
        Returns:
            The memory list with scores
        """
        if reference_time is None:
            reference_time = datetime.now()
        
        for entry in entries:
            # Reuse the existing semantic score
            semantic_score = self._get_semantic_score(entry)
            entry.score = self.score(entry, semantic_score, reference_time)
        
        return entries
    
    def rank(
        self,
        entries: List[MemoryEntry],
        reference_time: Optional[datetime] = None
    ) -> List[MemoryEntry]:
        """
        Score and sort
        
        Args:
            entries: list of memories
            reference_time: reference time
        
        Returns:
            The memory list sorted by composite score
        """
        scored = self.score_batch(entries, reference_time)
        return sorted(scored, key=lambda e: self._get_final_score(e), reverse=True)
    
    def filter_by_threshold(
        self,
        entries: List[MemoryEntry],
        min_score: float = 0.0
    ) -> List[MemoryEntry]:
        """
        Filter by score threshold
        
        Args:
            entries: list of memories
            min_score: minimum score threshold
        
        Returns:
            The filtered memory list
        """
        return [e for e in entries if self._get_final_score(e) >= min_score]
    
    def adjust_weights(
        self,
        semantic_weight: Optional[float] = None,
        recency_weight: Optional[float] = None,
        importance_weight: Optional[float] = None,
        access_weight: Optional[float] = None
    ) -> None:
        """
        Adjust the scoring weights
        
        The weights are normalized automatically
        """
        weights = [
            semantic_weight or self.config.semantic_weight,
            recency_weight or self.config.recency_weight,
            importance_weight or self.config.importance_weight,
            access_weight or self.config.access_weight,
        ]
        
        # Normalization
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        
        self.config.semantic_weight = weights[0]
        self.config.recency_weight = weights[1]
        self.config.importance_weight = weights[2]
        self.config.access_weight = weights[3]
