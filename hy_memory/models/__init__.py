"""
Agent Memory V2 - Data models

Core data structures, covering the new V2 models and the V1 compatibility types.
"""

# === V2 core models ===
from .memory import (
    # Enum types
    MemoryLayer,
    ContentType,
    MemoryStatus,
    SourceType,
    UpdateType,
    SchemaStatus,
    TriggerType,
    IntentionPriority,
    MetaCognitionTag,
    GapType,
    # Core node
    MemoryNode,
    VersionedFact,
    SchemaNode,
    IntentionNode,
    # Auxiliary models
    KnowledgeGap,
    LifeStage,
    TemporalEvent,
    UserTimeline,
    # Output protocol
    ProfileSummary,
    MemoryIndexEntry,
    MemorySummaryEntry,
    MetaCognitionReport,
    TriggeredIntention,
    ActivatedSchema,
    TemporalContext,
    MemoryContextPackage,
    # V1 compatibility
    MemoryEntry,
    MemoryMetadata,
    MemoryScore,
)

# === Request/response models ===
from .requests import (
    # Enums
    DeleteScope,
    MemoryInputType,
    TaskStatus,
    AgentProcessMode,
    # QA pairs
    QAPair,
    # Add memory
    AddRequest,
    AddResponse,
    AsyncAddResponse,
    AsyncTask,
    TaskStatusRequest,
    TaskStatusResponse,
    # Recall memory
    RecallRequest,
    RecallResponse,
    # Update memory
    UpdateRequest,
    UpdateResponse,
    # Delete memory
    DeleteRequest,
    DeleteResponse,
    BatchDeleteRequest,
    BatchDeleteResponse,
    # Get/list memory
    GetRequest,
    GetResponse,
    ListRequest,
    ListResponse,
    # User profile
    UserProfile,
    GetProfileRequest,
    GetProfileResponse,
    UpdateProfileRequest,
    UpdateProfileResponse,
    RebuildProfileRequest,
    RebuildProfileResponse,
    # Async task management
    SubmitTaskRequest,
    SubmitTaskResponse,
    GetTaskRequest,
    GetTaskResponse,
    CancelTaskRequest,
    CancelTaskResponse,
    ListTasksRequest,
    ListTasksResponse,
)

__all__ = [
    # === V2 enums ===
    "MemoryLayer",
    "ContentType",
    "MemoryStatus",
    "SourceType",
    "UpdateType",
    "SchemaStatus",
    "TriggerType",
    "IntentionPriority",
    "MetaCognitionTag",
    "GapType",
    # === V2 core nodes ===
    "MemoryNode",
    "VersionedFact",
    "SchemaNode",
    "IntentionNode",
    # === V2 auxiliary models ===
    "KnowledgeGap",
    "LifeStage",
    "TemporalEvent",
    "UserTimeline",
    # === V2 output protocol ===
    "ProfileSummary",
    "MemoryIndexEntry",
    "MemorySummaryEntry",
    "MetaCognitionReport",
    "TriggeredIntention",
    "ActivatedSchema",
    "TemporalContext",
    "MemoryContextPackage",
    # === V1 compatibility ===
    "MemoryEntry",
    "MemoryMetadata",
    "MemoryScore",
    # === Request/response enums ===
    "DeleteScope",
    "MemoryInputType",
    "TaskStatus",
    "AgentProcessMode",
    # === QA ===
    "QAPair",
    # === Add ===
    "AddRequest",
    "AddResponse",
    "AsyncAddResponse",
    "AsyncTask",
    "TaskStatusRequest",
    "TaskStatusResponse",
    # === Recall ===
    "RecallRequest",
    "RecallResponse",
    # === Update ===
    "UpdateRequest",
    "UpdateResponse",
    # === Delete ===
    "DeleteRequest",
    "DeleteResponse",
    "BatchDeleteRequest",
    "BatchDeleteResponse",
    # === Get/list ===
    "GetRequest",
    "GetResponse",
    "ListRequest",
    "ListResponse",
    # === User profile ===
    "UserProfile",
    "GetProfileRequest",
    "GetProfileResponse",
    "UpdateProfileRequest",
    "UpdateProfileResponse",
    "RebuildProfileRequest",
    "RebuildProfileResponse",
    # === Async tasks ===
    "SubmitTaskRequest",
    "SubmitTaskResponse",
    "GetTaskRequest",
    "GetTaskResponse",
    "CancelTaskRequest",
    "CancelTaskResponse",
    "ListTasksRequest",
    "ListTasksResponse",
]
