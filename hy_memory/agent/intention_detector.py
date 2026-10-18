"""
Agent Memory V2 - Intention Detector (intention detection agent)

V2 design doc §3.4 + §4.1:

  Detects prospective memory in user conversations (Prospective Memory).
  L6 layer: remembers "what to do later".

  Two trigger types:
  1. TIME_BASED: time trigger (e.g. "remind me next Monday")
  2. EVENT_BASED: event trigger (e.g. "next time travel comes up")

  Intention detection is used for:
  1. Write Path (System 1): lightweight detection, fast write into the L6 Intention Queue
  2. Read Path: check whether the current conversation triggers an existing intention
  3. System 2: deep analysis of intention details, setting up trigger conditions

  Prefrontal cortex analogy: prospective memory -- memory that fires automatically at some later moment.
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

INTENTION_DETECT_PROMPT = """Analyze the conversation below and detect whether the user expressed any future plan, intention or need.

Conversation:
---
{content}
---

Current time: {current_time}

Detect the following intention types:
1. Time-triggered: plans with an explicit point in time ("there is a meeting next Monday", "travelling in March")
2. Event-triggered: needs that should be surfaced in a specific context ("remind me next time XX comes up", "let me know if there is any information about XX")
3. Proactive care: signals hinting that attention will be needed later ("I have been under a lot of pressure lately", "my exam is coming up")

Output JSON:
{{
  "intentions": [
    {{
      "content": "description of the intention",
      "trigger_type": "time_based/event_based",
      "trigger_condition": "natural language description of the trigger condition",
      "trigger_time": "time in ISO format (for time_based, otherwise null)",
      "trigger_event_pattern": "pattern description of the trigger event (for event_based, otherwise null)",
      "priority": "high/medium/low",
      "expiry": "expiry time in ISO format (if any, otherwise null)",
      "is_proactive_care": false
    }}
  ]
}}

About is_proactive_care:
  Set it to true when the intention is not something the user asked for explicitly, but a signal the system should proactively care about.
  Example: the user says "I am so nervous, my exam is coming up" → the system should follow up on the result after the exam.

If no intention is detected, return: {{"intentions": []}}"""

INTENTION_TRIGGER_CHECK_PROMPT = """Decide whether the current conversation triggers any of the existing intentions below.

Current conversation:
---
{current_content}
---

Current time: {current_time}

Existing intentions:
{existing_intentions}

For each intention, decide whether the current conversation triggers it.

Output JSON:
{{
  "triggered": [
    {{
      "intention_id": "id of the triggered intention",
      "trigger_reason": "reason for the trigger",
      "confidence": 0.0-1.0
    }}
  ]
}}

If no intention is triggered: {{"triggered": []}}"""


# ================================================================
# Data structures
# ================================================================

@dataclass
class IntentionDetectorConfig:
    """Intention detector configuration."""
    detect_max_tokens: int = 600
    # None = follow config.llm.temperature
    detect_temperature: Optional[float] = None
    trigger_check_max_tokens: int = 500
    trigger_check_temperature: Optional[float] = None

    # Proactive care detection
    enable_proactive_care: bool = True


@dataclass
class DetectedIntention:
    """A detected intention."""
    content: str = ""
    trigger_type: str = "event_based"   # time_based / event_based
    trigger_condition: str = ""
    trigger_time: Optional[str] = None  # ISO format
    trigger_event_pattern: Optional[str] = None
    priority: str = "medium"            # high / medium / low
    expiry: Optional[str] = None
    is_proactive_care: bool = False


@dataclass
class IntentionDetectResult:
    """Result of intention detection."""
    success: bool
    intentions: List[DetectedIntention] = field(default_factory=list)
    tokens_used: int = 0
    error: Optional[str] = None


@dataclass
class TriggeredIntentionItem:
    """A triggered intention."""
    intention_id: str = ""
    trigger_reason: str = ""
    confidence: float = 0.0


@dataclass
class IntentionTriggerResult:
    """Result of the intention trigger check."""
    success: bool
    triggered: List[TriggeredIntentionItem] = field(default_factory=list)
    tokens_used: int = 0
    error: Optional[str] = None


# ================================================================
# Core implementation
# ================================================================

class IntentionDetector:
    """
    Intention detection agent.

    Capabilities:
    1. detect(): detect new intentions from a conversation (Write Path)
    2. check_triggers(): check whether the current conversation triggers existing intentions (Read Path)
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: Optional[IntentionDetectorConfig] = None,
    ):
        self.llm = llm
        self.config = config or IntentionDetectorConfig()
        self._call_count = 0
        self._total_tokens = 0
        logger.info("IntentionDetector initialized")

    async def detect(
        self,
        content: str,
        current_time: str = "",
    ) -> IntentionDetectResult:
        """
        Detect intentions from the conversation content.

        Args:
            content:      conversation text
            current_time: current time (ISO format)

        Returns:
            IntentionDetectResult holding the list of detected intentions
        """
        try:
            from datetime import datetime as dt
            prompt = INTENTION_DETECT_PROMPT.format(
                content=content,
                current_time=current_time or dt.now().isoformat(timespec="seconds"),
            )

            response = await self.llm.complete(
                prompt=prompt,
                max_tokens=self.config.detect_max_tokens,
            )
            self._call_count += 1
            self._total_tokens += response.tokens_used

            data = self._parse_json(response.content)

            intentions = []
            for item in data.get("intentions", []):
                # Skip proactive_care intentions when proactive care is disabled
                if not self.config.enable_proactive_care and item.get("is_proactive_care"):
                    continue

                intentions.append(DetectedIntention(
                    content=item.get("content", ""),
                    trigger_type=item.get("trigger_type", "event_based"),
                    trigger_condition=item.get("trigger_condition", ""),
                    trigger_time=item.get("trigger_time"),
                    trigger_event_pattern=item.get("trigger_event_pattern"),
                    priority=item.get("priority", "medium"),
                    expiry=item.get("expiry"),
                    is_proactive_care=item.get("is_proactive_care", False),
                ))

            return IntentionDetectResult(
                success=True,
                intentions=intentions,
                tokens_used=response.tokens_used,
            )

        except Exception as e:
            logger.error(f"IntentionDetector.detect failed: {e}")
            return IntentionDetectResult(success=False, error=str(e))

    async def check_triggers(
        self,
        current_content: str,
        existing_intentions: List[Dict[str, Any]],
        current_time: str = "",
    ) -> IntentionTriggerResult:
        """
        Check whether the current conversation triggers any existing intention.

        Args:
            current_content:     current conversation content
            existing_intentions: existing intentions [{id, content, trigger_condition, ...}]
            current_time:        current time

        Returns:
            IntentionTriggerResult holding the list of triggered intentions
        """
        try:
            if not existing_intentions:
                return IntentionTriggerResult(success=True)

            from datetime import datetime as dt
            intentions_text = "\n".join(
                f"[{i.get('id', '?')}] {i.get('content', '')} "
                f"(trigger condition: {i.get('trigger_condition', '?')})"
                for i in existing_intentions
            )

            prompt = INTENTION_TRIGGER_CHECK_PROMPT.format(
                current_content=current_content,
                current_time=current_time or dt.now().isoformat(timespec="seconds"),
                existing_intentions=intentions_text,
            )

            response = await self.llm.complete(
                prompt=prompt,
                max_tokens=self.config.trigger_check_max_tokens,
            )
            self._call_count += 1
            self._total_tokens += response.tokens_used

            data = self._parse_json(response.content)

            triggered = []
            for item in data.get("triggered", []):
                triggered.append(TriggeredIntentionItem(
                    intention_id=item.get("intention_id", ""),
                    trigger_reason=item.get("trigger_reason", ""),
                    confidence=float(item.get("confidence", 0.5)),
                ))

            return IntentionTriggerResult(
                success=True,
                triggered=triggered,
                tokens_used=response.tokens_used,
            )

        except Exception as e:
            logger.error(f"IntentionDetector.check_triggers failed: {e}")
            return IntentionTriggerResult(success=False, error=str(e))

    # ================================================================
    # Helpers
    # ================================================================

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """Parse JSON out of the LLM output."""
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
        """Get statistics."""
        return {
            "call_count": self._call_count,
            "total_tokens": self._total_tokens,
        }
