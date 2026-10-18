"""
Agent Memory V2 - Reflector agent (enhanced edition)

V2 design document:

  The reflector agent is responsible for:
  1. UpdateType classification: how the new information relates to what is already known (OVERRIDE/SUPPLEMENT/TEMPORAL/NEGATE/CONFLICT)
  2. Conflict detection:       identify contradiction, temporal change, granularity mismatch and other conflict types
  3. Implicit inference:       infer user attributes from behaviour/attitude/choice signals

  V1 interfaces retained (check_conflicts, suggest_merge)
  New V2 interfaces (classify_update_type, detect_conflicts_v2, extract_implicit_signals)
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import json
import re
import logging

from .llm_provider import LLMProvider
from ..config import LLMConfig as GlobalLLMConfig

logger = logging.getLogger(__name__)


# ================================================================
# Enums and data structures
# ================================================================

class ConflictType(Enum):
    """Conflict types (V1 compatible)"""
    CONTRADICTION = "contradiction"
    UPDATE = "update"
    DUPLICATE = "duplicate"
    RELATED = "related"


class ConflictAction(Enum):
    """Conflict handling suggestions (V1 compatible)"""
    KEEP_NEW = "keep_new"
    KEEP_OLD = "keep_old"
    MERGE = "merge"
    BOTH = "both"


class V2ConflictType(str, Enum):
    """V2 conflict types"""
    TEMPORAL_CHANGE = "temporal_change"
    CONTEXT_DEPENDENT = "context_dependent"
    CONTRADICTION = "contradiction"
    GRANULARITY_MISMATCH = "granularity_mismatch"


@dataclass
class ReflectResult:
    """Reflection result (V1 compatible)"""
    success: bool
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    should_merge: bool = False
    merge_target_id: Optional[str] = None
    merged_content: Optional[str] = None
    tokens_used: int = 0
    error: Optional[str] = None


@dataclass
class UpdateTypeResult:
    """UpdateType classification result (V2)"""
    success: bool
    update_type: str = ""  # OVERRIDE/SUPPLEMENT/TEMPORAL/NEGATE/CONFLICT
    target_fact_id: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""
    temporal_scope: Optional[str] = None
    tokens_used: int = 0
    error: Optional[str] = None


@dataclass
class ConflictDetectionResult:
    """Conflict detection result (V2)"""
    success: bool
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    tokens_used: int = 0
    error: Optional[str] = None


@dataclass
class ImplicitSignal:
    """Implicit signal"""
    signal_type: str = ""   # behavior / attitude / choice
    content: str = ""
    possible_traits: List[str] = field(default_factory=list)
    strength: str = "weak"  # weak / moderate / strong


@dataclass
class ImplicitInferenceResult:
    """Implicit inference result (V2)"""
    success: bool
    signals: List[ImplicitSignal] = field(default_factory=list)
    tokens_used: int = 0
    error: Optional[str] = None


# ================================================================
# Prompt templates
# ================================================================

UPDATE_TYPE_PROMPT = """Analyze how the new information relates to the existing information and determine the update type.

New information:
{new_fact}

Related information already known:
{existing_facts}

Determine the update type:
- OVERRIDE: explicit replacement (age change, moved house, changed job)
- SUPPLEMENT: supplement (a new hobby, a new experience)
- TEMPORAL: temporary change ("today I feel like eating ...", "lately I have been ...")
- NEGATE: negation of the old information ("no longer likes ...", "has already left ...")
- CONFLICT: a contradiction that cannot be decided (needs human/LLM follow-up confirmation)

Output JSON:
{{
  "update_type": "type",
  "target_fact_id": "id of the affected old information (or null)",
  "confidence": 0.0-1.0,
  "reasoning": "reason for the decision",
  "temporal_scope": "valid range when TEMPORAL (or null)"
}}"""

CONFLICT_DETECT_PROMPT = """Detect conflicts between the new information below and the existing memories.

New information:
{new_content}

Existing memories:
{existing_memories}

Conflict type legend:
- TEMPORAL_CHANGE: the user state genuinely changed over time (moved house, changed job)
- CONTEXT_DEPENDENT: different behaviour in different contexts (serious at work, lively among friends)
- CONTRADICTION: logical contradiction (conflicting statements about the same time)
- GRANULARITY_MISMATCH: different granularity that can coexist ("likes fruit" vs "likes apples")

Output JSON:
{{
  "conflicts": [
    {{
      "existing_memory_id": "id of the conflicting memory",
      "conflict_type": "TEMPORAL_CHANGE/CONTEXT_DEPENDENT/CONTRADICTION/GRANULARITY_MISMATCH",
      "description": "conflict description",
      "severity": "high/medium/low",
      "suggested_action": "override/contextualize/both_valid/flag_for_review"
    }}
  ]
}}

If there is no conflict return: {{"conflicts": []}}"""

IMPLICIT_INFERENCE_PROMPT = """Analyze the implicit signals in the conversation below and infer the user's likely attributes or preferences.

Conversation:
---
{content}
---

Implicit signal types:
- behavior: behavioural signal (e.g. "runs every day" → self-disciplined / health-conscious)
- attitude: attitude signal (e.g. "that is too expensive" → price-sensitive)
- choice: choice signal (e.g. "I'll still go for the minimal style" → minimalism)

Output JSON:
{{
  "signals": [
    {{
      "signal_type": "behavior/attitude/choice",
      "content": "original signal content",
      "possible_traits": ["inferable trait 1", "trait 2"],
      "strength": "weak/moderate/strong"
    }}
  ]
}}

Only output inferences backed by evidence, do not guess. If there is no signal return: {{"signals": []}}"""


# ================================================================
# Core implementation
# ================================================================

class Reflector:
    """
    Reflector agent (V2)

    Retains the V1 conflict detection and merge suggestion capabilities,
    and adds V2 UpdateType classification, V2 conflict detection and implicit inference.
    """

    # V1 Prompt
    CONFLICT_CHECK_PROMPT = """Analyze whether the new memory conflicts with the existing memories.

New memory:
{new_content}

Existing memories:
{existing_memories}

Output the analysis in JSON format:
{{
    "conflicts": [
        {{
            "memory_id": "id of the conflicting memory",
            "conflict_type": "contradiction/update/duplicate/related",
            "description": "conflict description",
            "suggestion": "keep_new/keep_old/merge/both"
        }}
    ],
    "should_merge": true/false,
    "merge_target_id": "id of the memory to merge into (if a merge is needed)",
    "merged_content": "the merged content (if a merge is needed)"
}}

Output JSON only, with no other explanation:"""

    def __init__(
        self,
        llm_provider: LLMProvider,
        llm_config: Optional[GlobalLLMConfig] = None,
    ):
        self.llm = llm_provider
        self._llm_config = llm_config or GlobalLLMConfig()
        self._call_count = 0
        self._total_tokens = 0
        logger.info("Reflector V2 initialized")

    # ================================================================
    # V1 compatible interfaces
    # ================================================================

    async def check_conflicts(
        self,
        new_content: str,
        existing_memories: List[Dict[str, Any]],
        context: Dict[str, Any] = None,
    ) -> ReflectResult:
        """V1 compatible: detect conflicts"""
        try:
            if not existing_memories:
                return ReflectResult(success=True)

            memories_text = "\n".join([
                f"[{m.get('id', 'unknown')}] {m.get('content', '')}"
                for m in existing_memories
            ])

            prompt = self.CONFLICT_CHECK_PROMPT.format(
                new_content=new_content, existing_memories=memories_text,
            )
            response = await self.llm.complete(
                prompt=prompt,
                max_tokens=self._llm_config.agent_max_tokens,
            )
            self._call_count += 1
            self._total_tokens += response.tokens_used

            data = self._parse_json(response.content)

            return ReflectResult(
                success=True,
                conflicts=data.get("conflicts", []),
                should_merge=data.get("should_merge", False),
                merge_target_id=data.get("merge_target_id"),
                merged_content=data.get("merged_content"),
                tokens_used=response.tokens_used,
            )
        except Exception as e:
            logger.error(f"Reflector.check_conflicts failed: {e}")
            return ReflectResult(success=False, error=str(e))

    async def suggest_merge(
        self, memory1: Dict[str, Any], memory2: Dict[str, Any],
    ) -> Optional[str]:
        """V1 compatible: produce a merge suggestion"""
        prompt = f"""Merge the following two memories into one, keeping all useful information and resolving the conflict.

Memory 1:
{memory1.get('content', '')}

Memory 2:
{memory2.get('content', '')}

Output the merged content directly:"""

        try:
            response = await self.llm.complete(
                prompt=prompt,
                max_tokens=self._llm_config.agent_max_tokens,
            )
            self._call_count += 1
            self._total_tokens += response.tokens_used
            return response.content.strip()
        except Exception as e:
            logger.error(f"suggest_merge failed: {e}")
            return None

    # ================================================================
    # V2 UpdateType classification
    # ================================================================

    async def classify_update_type(
        self,
        new_fact: str,
        existing_facts: List[Dict[str, Any]],
    ) -> UpdateTypeResult:
        """
        Determine how the new fact updates the existing facts.
        Called by System 2 Deep Extraction / Write Path.
        """
        try:
            facts_text = "\n".join([
                f"[{f.get('id', '?')}] {f.get('content', '')} "
                f"(time: {f.get('created_at', '?')}, confidence: {f.get('confidence', '?')})"
                for f in existing_facts
            ])

            prompt = UPDATE_TYPE_PROMPT.format(
                new_fact=new_fact,
                existing_facts=facts_text or "(no existing information)",
            )
            response = await self.llm.complete(
                prompt=prompt,
                max_tokens=self._llm_config.agent_max_tokens,
            )
            self._call_count += 1
            self._total_tokens += response.tokens_used

            data = self._parse_json(response.content)

            return UpdateTypeResult(
                success=True,
                update_type=data.get("update_type", "SUPPLEMENT"),
                target_fact_id=data.get("target_fact_id"),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                temporal_scope=data.get("temporal_scope"),
                tokens_used=response.tokens_used,
            )
        except Exception as e:
            logger.error(f"classify_update_type failed: {e}")
            return UpdateTypeResult(success=False, error=str(e))

    # ================================================================
    # V2 conflict detection
    # ================================================================

    async def detect_conflicts_v2(
        self,
        new_content: str,
        existing_memories: List[Dict[str, Any]],
    ) -> ConflictDetectionResult:
        """
        V2 conflict detection: distinguishes the four conflict types.
        """
        try:
            if not existing_memories:
                return ConflictDetectionResult(success=True)

            memories_text = "\n".join([
                f"[{m.get('id', '?')}] {m.get('content', '')} "
                f"(time: {m.get('created_at', '?')})"
                for m in existing_memories
            ])

            prompt = CONFLICT_DETECT_PROMPT.format(
                new_content=new_content,
                existing_memories=memories_text,
            )
            response = await self.llm.complete(
                prompt=prompt,
                max_tokens=self._llm_config.agent_max_tokens,
            )
            self._call_count += 1
            self._total_tokens += response.tokens_used

            data = self._parse_json(response.content)

            return ConflictDetectionResult(
                success=True,
                conflicts=data.get("conflicts", []),
                tokens_used=response.tokens_used,
            )
        except Exception as e:
            logger.error(f"detect_conflicts_v2 failed: {e}")
            return ConflictDetectionResult(success=False, error=str(e))

    # ================================================================
    # V2 implicit inference
    # ================================================================

    async def extract_implicit_signals(
        self,
        content: str,
    ) -> ImplicitInferenceResult:
        """
        Extract implicit signals from the conversation content.
        Called by the System 2 Trait Inferencer.
        """
        try:
            prompt = IMPLICIT_INFERENCE_PROMPT.format(content=content)

            response = await self.llm.complete(
                prompt=prompt,
                max_tokens=self._llm_config.agent_max_tokens,
            )
            self._call_count += 1
            self._total_tokens += response.tokens_used

            data = self._parse_json(response.content)

            signals = []
            for s in data.get("signals", []):
                signals.append(ImplicitSignal(
                    signal_type=s.get("signal_type", ""),
                    content=s.get("content", ""),
                    possible_traits=s.get("possible_traits", []),
                    strength=s.get("strength", "weak"),
                ))

            return ImplicitInferenceResult(
                success=True,
                signals=signals,
                tokens_used=response.tokens_used,
            )
        except Exception as e:
            logger.error(f"extract_implicit_signals failed: {e}")
            return ImplicitInferenceResult(success=False, error=str(e))

    # ================================================================
    # Utility methods
    # ================================================================

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """Parse JSON out of the LLM output"""
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {"conflicts": [], "should_merge": False}

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return {
            "call_count": self._call_count,
            "total_tokens": self._total_tokens,
            "avg_tokens_per_call": (
                self._total_tokens / self._call_count
                if self._call_count > 0 else 0
            ),
        }
