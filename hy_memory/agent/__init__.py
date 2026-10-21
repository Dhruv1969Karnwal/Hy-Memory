"""
Agent Memory - Intelligence layer (Agent Layer)

LLM-based intelligent processing: semantic understanding, information extraction, reasoning and judgement.
This layer is optional; the system keeps working when it is turned off.

Modules:
- MemAgent: agent coordinator, the unified entry point
- Abstractor: System 2 full-summary agent (session summary / schema induction / profile summary)
- Extractor: extraction agent, extracts structured information (V2: lightweight / deep extraction dual modes)
- Reflector: reflection agent, detects conflicts and updates (V2: UpdateType classification / conflict detection / implicit inference)
- EmotionAnalyzer: emotion analysis agent, dual-dimension valence + arousal labeling
- IntentionDetector: intention detection agent, prospective memory + proactive care
- LLMProvider: LLM provider, unified model invocation interface
"""

# Lite+Agent pipeline
from .mem_agent import MemAgent, AgentResult, ProcessMode
from .extractor import Extractor, ExtractResult
from .reflector import Reflector, ReflectResult, ConflictType, ConflictAction
from .llm_provider import LLMProvider, LLMResponse, LLMConfig, LLMBackend

# V2 Extractor
from .extractor import ExtractMode, V2ExtractResult

# System 2 / Pro Pipeline Abstractor
from .abstractor import Abstractor, SessionSummaryResult, SchemaAbstractResult, ProfileSummaryResult

# V2 Reflector
from .reflector import (
    V2ConflictType,
    UpdateTypeResult,
    ConflictDetectionResult,
    ImplicitSignal,
    ImplicitInferenceResult,
)

# V2 Emotion Analyzer
from .emotion_analyzer import EmotionAnalyzer, EmotionAnalysisConfig, EmotionResult

# V2 Intention Detector
from .intention_detector import (
    IntentionDetector,
    IntentionDetectorConfig,
    DetectedIntention,
    IntentionDetectResult,
    TriggeredIntentionItem,
    IntentionTriggerResult,
)

__all__ = [
    # MemAgent
    "MemAgent",
    "AgentResult",
    "ProcessMode",
    # Abstractor (System 2)
    "Abstractor",
    "SessionSummaryResult",
    "SchemaAbstractResult",
    "ProfileSummaryResult",
    # Extractor (V1)
    "Extractor",
    "ExtractResult",
    # Extractor (V2)
    "ExtractMode",
    "V2ExtractResult",
    # Reflector (V1)
    "Reflector",
    "ReflectResult",
    "ConflictType",
    "ConflictAction",
    # Reflector (V2)
    "V2ConflictType",
    "UpdateTypeResult",
    "ConflictDetectionResult",
    "ImplicitSignal",
    "ImplicitInferenceResult",
    # Emotion Analyzer (V2)
    "EmotionAnalyzer",
    "EmotionAnalysisConfig",
    "EmotionResult",
    # Intention Detector (V2)
    "IntentionDetector",
    "IntentionDetectorConfig",
    "DetectedIntention",
    "IntentionDetectResult",
    "TriggeredIntentionItem",
    "IntentionTriggerResult",
    # LLMProvider
    "LLMProvider",
    "LLMResponse",
    "LLMConfig",
    "LLMBackend",
]
