"""
Agent Memory - Merger

Detects and merges duplicate memories to avoid information redundancy.

Capabilities:
- Similarity detection (based on semantic similarity)
- Merge decision (whether to merge and into which memory)
- Content merge (produce the merged content)
- Deduplication

Example:
    merger = Merger()
    
    # Check whether a merge is needed
    result = merger.check_merge(
        new_content="The user likes Sichuan food, especially mapo tofu",
        existing_memories=[
            {"id": "mem_1", "content": "The user likes Sichuan food", "score": 0.95},
            {"id": "mem_2", "content": "The user lives in Beijing", "score": 0.3},
        ]
    )
    
    if result.should_merge:
        print(f"Merging into {result.target_memory_id}")
        print(f"Merged content: {result.merged_content}")
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

from ..config import LLMConfig as GlobalLLMConfig

logger = logging.getLogger(__name__)


class MergeStrategy(Enum):
    """Merge strategies."""
    APPEND = "append"           # Append content
    REPLACE = "replace"         # Replace content
    SMART = "smart"             # Smart merge (requires an LLM)
    KEEP_BOTH = "keep_both"     # Keep both


@dataclass
class MergeResult:
    """
    Merge result
    """
    should_merge: bool = False
    target_memory_id: Optional[str] = None
    merged_content: Optional[str] = None
    strategy: MergeStrategy = MergeStrategy.KEEP_BOTH
    reason: str = ""
    similarity_score: float = 0.0


@dataclass
class MergerConfig:
    """
    Merger configuration
    """
    # Similarity thresholds
    merge_threshold: float = 0.85          # Above this value, a merge triggers
    duplicate_threshold: float = 0.95      # Above this value, treated as a duplicate
    
    # Merge strategies
    default_strategy: MergeStrategy = MergeStrategy.SMART
    
    # Other options
    max_content_length: int = 2000         # Max length after merging
    enable_smart_merge: bool = True        # Whether smart merging is enabled


class Merger:
    """
    Memory merger
    
    Detects and handles duplicate / similar memories.
    """
    
    def __init__(
        self,
        config: Optional[MergerConfig] = None,
        llm_config: Optional[GlobalLLMConfig] = None,
    ):
        """
        Initialize the merger
        
        Args:
            config: Merger configuration
            llm_config: LLM config (used to read settings such as max_tokens)
        """
        self.config = config or MergerConfig()
        self._llm_config = llm_config or GlobalLLMConfig()
        
        # Counters
        self._check_count = 0
        self._merge_count = 0
        
        logger.info("Merger initialized")
    
    def check_merge(
        self,
        new_content: str,
        existing_memories: List[Any],
        context: Dict[str, Any] = None,
    ) -> MergeResult:
        """
        Check whether a merge is needed
        
        Args:
            new_content: New memory content
            existing_memories: Existing memories (must have a score attribute or field)
            context: Context information
        
        Returns:
            The merge result
        """
        self._check_count += 1
        
        if not existing_memories:
            return MergeResult(should_merge=False, reason="No existing memories")
        
        # Find the most similar memory
        best_match = None
        best_score = 0.0
        
        for mem in existing_memories:
            # Fetch the similarity score
            score = self._get_similarity_score(mem)
            
            if score > best_score:
                best_score = score
                best_match = mem
        
        # Decide whether a merge is needed
        if best_score >= self.config.duplicate_threshold:
            # Exact duplicate; skip
            return MergeResult(
                should_merge=True,
                target_memory_id=self._get_memory_id(best_match),
                strategy=MergeStrategy.KEEP_BOTH,  # Keep the old one
                similarity_score=best_score,
                reason="Duplicate content detected",
            )
        
        if best_score >= self.config.merge_threshold:
            # Merge required
            merged_content = self._merge_content(
                new_content,
                self._get_content(best_match),
            )
            
            self._merge_count += 1
            
            return MergeResult(
                should_merge=True,
                target_memory_id=self._get_memory_id(best_match),
                merged_content=merged_content,
                strategy=MergeStrategy.SMART,
                similarity_score=best_score,
                reason="Similar content, merged",
            )
        
        # No merge needed
        return MergeResult(
            should_merge=False,
            similarity_score=best_score,
            reason="No similar memory found",
        )
    
    def _get_similarity_score(self, memory: Any) -> float:
        """Fetch the similarity score."""
        if isinstance(memory, dict):
            return memory.get("score", 0.0)
        elif hasattr(memory, "score"):
            score = memory.score
            if hasattr(score, "semantic_score"):
                return score.semantic_score
            return float(score) if score else 0.0
        return 0.0
    
    def _get_memory_id(self, memory: Any) -> str:
        """Fetch the memory ID."""
        if isinstance(memory, dict):
            return memory.get("id", memory.get("memory_id", ""))
        elif hasattr(memory, "memory_id"):
            return memory.memory_id
        return ""
    
    def _get_content(self, memory: Any) -> str:
        """Fetch the memory content."""
        if isinstance(memory, dict):
            return memory.get("content", "")
        elif hasattr(memory, "content"):
            return memory.content
        return ""
    
    def _merge_content(
        self,
        new_content: str,
        old_content: str,
    ) -> str:
        """
        Merge content
        
        Simple implementation: if the new content is longer, replace with it;
        otherwise append the new information.
        
        Args:
            new_content: New content
            old_content: Old content
        
        Returns:
            The merged content
        """
        # Simple strategy: use the longer content
        if len(new_content) > len(old_content) * 1.2:
            return new_content
        
        # Check whether the new content carries extra information
        # Simple approach: check if the new content is fully contained in the old content
        if new_content.strip() in old_content:
            return old_content
        
        # Append the new information
        # In production this should use an LLM for smart merging
        if len(old_content) + len(new_content) < self.config.max_content_length:
            # Avoid naive concatenation; this is only an example
            # Real behavior should extract the new information and update
            return new_content  # Use the new content
        
        return new_content
    
    async def smart_merge(
        self,
        new_content: str,
        old_content: str,
        llm_provider: Any = None,
    ) -> str:
        """
        Smart merge (uses an LLM)
        
        Args:
            new_content: New content
            old_content: Old content
            llm_provider: The LLM provider
        
        Returns:
            The merged content
        """
        if llm_provider is None:
            return self._merge_content(new_content, old_content)
        
        prompt = f"""Merge the following two memories into one, keeping all useful information:

Old memory:
{old_content}

New memory:
{new_content}

Requirements:
1. Keep all important information
2. On conflict, prefer the new memory
3. Keep the wording concise and clear
4. Do not add information absent from the originals

Merged memory:"""

        response = await llm_provider.complete(
            prompt=prompt,
            max_tokens=self._llm_config.agent_max_tokens,
        )
        
        return response.content.strip()
    
    def deduplicate(
        self,
        memories: List[Any],
        threshold: float = None,
    ) -> List[Any]:
        """
        Batch deduplication
        
        Args:
            memories: The memory list
            threshold: Similarity threshold
        
        Returns:
            The deduplicated list
        """
        if threshold is None:
            threshold = self.config.duplicate_threshold
        
        # Simple implementation: deduplicate by content hash
        seen = set()
        result = []
        
        for mem in memories:
            content = self._get_content(mem)
            content_hash = hash(content.strip().lower())
            
            if content_hash not in seen:
                seen.add(content_hash)
                result.append(mem)
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Return the statistics."""
        return {
            "check_count": self._check_count,
            "merge_count": self._merge_count,
            "merge_rate": (
                self._merge_count / self._check_count
                if self._check_count > 0 else 0
            ),
        }
