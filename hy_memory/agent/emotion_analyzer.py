"""
Agent Memory V2 - Emotion Analyzer (emotion analysis agent)

V2 design doc Section 3.1:

  Emotional Tagging: each memory node carries emotional valence and arousal.
  - valence: [-1, 1] negative to positive
  - arousal: [0, 1] calm to excited

  Emotion tagging is used for:
  1. Write Path: tag newly written L2 facts with emotions
  2. Read Path: emotion weighting (high-arousal memories resist decay, sensitive content flagged EMOTIONALLY_SENSITIVE)
  3. System 2: Sleep Replay prioritizes replaying high-emotional-arousal memories
  4. Reconsolidation: detect shifts in emotional tone

  Amygdala analogy: emotion tagging makes important, emotional memories easier to recall.
"""

import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging

from .llm_provider import LLMProvider

logger = logging.getLogger(__name__)


# ================================================================
# Prompt templates
# ================================================================

EMOTION_ANALYSIS_PROMPT = """Analyze the emotional characteristics of the following conversation content.

Conversation content:
---
{content}
---

Please analyze along two dimensions:
1. Valence: range [-1.0, 1.0]
   - -1.0: very negative (sadness, anger, fear)
   - 0.0:  neutral
   - 1.0:  very positive (happiness, excitement, satisfaction)

2. Emotional arousal: range [0.0, 1.0]
   - 0.0: very calm (relaxed, bored)
   - 0.5: moderate
   - 1.0: very agitated (excitement, anger, fear)

Output JSON:
{{
  "valence": 0.0,
  "arousal": 0.0,
  "dominant_emotion": "primary emotion label",
  "is_sensitive": false,
  "sensitivity_reason": "if sensitive, explain why (otherwise null)",
  "emotional_keywords": ["key emotion words"]
}}

Common emotion-dimension mapping reference:
  happy/content: valence=0.6~0.9, arousal=0.3~0.6
  excited/agitated: valence=0.7~1.0, arousal=0.7~1.0
  calm/relaxed: valence=0.2~0.5, arousal=0.0~0.2
  anxious/worried: valence=-0.3~-0.6, arousal=0.5~0.8
  sad/dejected: valence=-0.5~-0.9, arousal=0.1~0.4
  angry/discontent: valence=-0.6~-1.0, arousal=0.6~1.0
  neutral/narrative: valence=0.0, arousal=0.1~0.3"""

BATCH_EMOTION_PROMPT = """Analyze the emotional characteristics of the following content segments.

{items}

For each content segment output:
{{
  "results": [
    {{
      "index": 0,
      "valence": 0.0,
      "arousal": 0.0,
      "dominant_emotion": "emotion label"
    }}
  ]
}}"""


# ================================================================
# Data structures
# ================================================================

@dataclass
class EmotionAnalysisConfig:
    """Emotion analysis configuration"""
    max_tokens: int = 400
    # None = follow config.llm.temperature (hardcoding it would bypass user config,
    # and return a direct 400 on gateways that only accept specific temperatures)
    temperature: Optional[float] = None

    # Sensitivity threshold: mark as sensitive when arousal > this value and valence < 0
    sensitivity_arousal_threshold: float = 0.6
    sensitivity_valence_threshold: float = -0.3

    # Per-batch size for batch analysis
    batch_size: int = 10


@dataclass
class EmotionResult:
    """Emotion analysis result"""
    success: bool
    valence: float = 0.0
    arousal: float = 0.0
    dominant_emotion: str = "neutral"
    is_sensitive: bool = False
    sensitivity_reason: Optional[str] = None
    emotional_keywords: List[str] = field(default_factory=list)
    tokens_used: int = 0
    error: Optional[str] = None


# ================================================================
# Core implementation
# ================================================================

class EmotionAnalyzer:
    """
    Emotion analysis agent.

    Annotates text content with the two dimensions of emotional valence + arousal.
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: Optional[EmotionAnalysisConfig] = None,
    ):
        self.llm = llm
        self.config = config or EmotionAnalysisConfig()
        self._call_count = 0
        self._total_tokens = 0
        logger.info("EmotionAnalyzer initialized")

    async def analyze(self, content: str) -> EmotionResult:
        """
        Analyze the emotion of a single content segment.

        Args:
            content: conversation text

        Returns:
            EmotionResult containing valence, arousal, dominant_emotion, etc.
        """
        try:
            prompt = EMOTION_ANALYSIS_PROMPT.format(content=content)
            response = await self.llm.complete(
                prompt=prompt,
                max_tokens=self.config.max_tokens,
            )
            self._call_count += 1
            self._total_tokens += response.tokens_used

            data = self._parse_json(response.content)

            valence = self._clamp(float(data.get("valence", 0.0)), -1.0, 1.0)
            arousal = self._clamp(float(data.get("arousal", 0.0)), 0.0, 1.0)

            # Determine sensitivity
            is_sensitive = data.get("is_sensitive", False)
            if not is_sensitive:
                is_sensitive = (
                    arousal > self.config.sensitivity_arousal_threshold
                    and valence < self.config.sensitivity_valence_threshold
                )

            return EmotionResult(
                success=True,
                valence=valence,
                arousal=arousal,
                dominant_emotion=data.get("dominant_emotion", "neutral"),
                is_sensitive=is_sensitive,
                sensitivity_reason=data.get("sensitivity_reason"),
                emotional_keywords=data.get("emotional_keywords", []),
                tokens_used=response.tokens_used,
            )

        except Exception as e:
            logger.error(f"EmotionAnalyzer.analyze failed: {e}")
            return EmotionResult(success=False, error=str(e))

    async def analyze_batch(
        self, contents: List[str],
    ) -> List[EmotionResult]:
        """
        Batch emotion analysis.

        For large volumes of content, process in batches to reduce LLM call count.
        """
        results = []
        batch_size = self.config.batch_size

        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]

            if len(batch) == 1:
                result = await self.analyze(batch[0])
                results.append(result)
                continue

            # Batch prompt
            items_text = "\n".join(
                f"[{j}] {c}" for j, c in enumerate(batch)
            )
            prompt = BATCH_EMOTION_PROMPT.format(items=items_text)

            try:
                response = await self.llm.complete(
                    prompt=prompt,
                    max_tokens=self.config.max_tokens * len(batch),
                )
                self._call_count += 1
                self._total_tokens += response.tokens_used

                data = self._parse_json(response.content)
                batch_results = data.get("results", [])

                for j, c in enumerate(batch):
                    if j < len(batch_results):
                        r = batch_results[j]
                        valence = self._clamp(float(r.get("valence", 0.0)), -1.0, 1.0)
                        arousal = self._clamp(float(r.get("arousal", 0.0)), 0.0, 1.0)
                        results.append(EmotionResult(
                            success=True,
                            valence=valence,
                            arousal=arousal,
                            dominant_emotion=r.get("dominant_emotion", "neutral"),
                            tokens_used=response.tokens_used // len(batch),
                        ))
                    else:
                        results.append(EmotionResult(
                            success=True, valence=0.0, arousal=0.0,
                        ))

            except Exception as e:
                logger.warning(f"Batch emotion analysis failed, falling back: {e}")
                for c in batch:
                    result = await self.analyze(c)
                    results.append(result)

        return results

    # ================================================================
    # Utility methods
    # ================================================================

    @staticmethod
    def _clamp(value: float, min_v: float, max_v: float) -> float:
        return max(min_v, min(max_v, value))

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """Parse JSON from LLM output"""
        import re
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
        return {}

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return {
            "call_count": self._call_count,
            "total_tokens": self._total_tokens,
        }
