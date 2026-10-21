"""
Agent Memory V2 - Request/response models

Defines the request and response data structures of the API.
Aligned with the V2 data model (MemoryNode, MemoryContextPackage, etc.).
"""

from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from .memory import (
    MemoryLayer,
    MemoryNode,
    MemoryContextPackage,
)


# ============================================================
# Enums
# ============================================================

class DeleteScope(str, Enum):
    """Deletion scope."""
    MEMORY = "memory"
    SCENE = "scene"
    USER = "user"
    APP = "app"


class MemoryInputType(str, Enum):
    """Memory input type."""
    TEXT = "text"
    QA = "qa"
    DIALOGUE = "dialogue"


class TaskStatus(str, Enum):
    """Async task status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentProcessMode(str, Enum):
    """Agent processing mode."""
    DISABLED = "disabled"    # Agent processing disabled
    FULL = "full"            # Full processing (extraction + summary + conflict detection)


# ============================================================
# Add memory
# ============================================================

@dataclass
class QAPair:
    """Question/answer pair."""
    question: str = ""
    answer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"question": self.question, "answer": self.answer}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QAPair":
        return cls(question=data.get("question", ""), answer=data.get("answer", ""))

    def to_content(self) -> str:
        return f"Q: {self.question}\nA: {self.answer}"


@dataclass
class AddRequest:
    """
    Add memory request

    Separation of parameter responsibilities:
    - agent_mode: controls MemAgent processing depth (disabled / full)
    """
    # Required - identity
    uid: str = ""
    agent_id: str = ""

    # Session info (new in V2)
    session_id: str = ""

    # Input content
    input_type: MemoryInputType = MemoryInputType.TEXT
    content: Optional[str] = None
    qa_pair: Optional[QAPair] = None

    # Memory layer
    layer: Optional[Union[MemoryLayer, str]] = None

    # Time
    event_time: Optional[datetime] = None
    ttl_seconds: Optional[int] = None

    # Weight
    importance: float = 1.0

    # Extensions
    tags: List[str] = field(default_factory=list)
    custom: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None

    # Agent processing options
    agent_mode: AgentProcessMode = AgentProcessMode.FULL
    async_process: bool = False

    # V1 compatibility
    auto_extract: bool = False
    store_raw: bool = True
    infer: bool = False

    def __post_init__(self):
        if isinstance(self.layer, str):
            self.layer = MemoryLayer.from_string(self.layer)
        if isinstance(self.input_type, str):
            self.input_type = MemoryInputType(self.input_type)
        if isinstance(self.agent_mode, str):
            self.agent_mode = AgentProcessMode(self.agent_mode)
        if self.auto_extract and self.agent_mode == AgentProcessMode.DISABLED:
            self.agent_mode = AgentProcessMode.FULL
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
        self._validate()

    def _validate(self):
        if self.input_type == MemoryInputType.TEXT:
            if not self.content:
                raise ValueError("content is required when input_type is 'text'")
        elif self.input_type == MemoryInputType.QA:
            if not self.qa_pair:
                raise ValueError("qa_pair is required when input_type is 'qa'")

    def get_content(self) -> str:
        if self.input_type == MemoryInputType.TEXT:
            return self.content or ""
        elif self.input_type == MemoryInputType.QA:
            return self.qa_pair.to_content() if self.qa_pair else ""
        return ""

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "uid": self.uid,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "input_type": self.input_type.value,
            "layer": self.layer.value if self.layer else None,
            "importance": self.importance,
            "tags": self.tags,
            "agent_mode": self.agent_mode.value,
            "async_process": self.async_process,
        }
        if self.content:
            result["content"] = self.content
        if self.qa_pair:
            result["qa_pair"] = self.qa_pair.to_dict()
        return result


@dataclass
class AddResponse:
    """Add memory response (synchronous mode)."""
    success: bool = False
    memory_id: str = ""
    node_ids: List[str] = field(default_factory=list)
    layer: Optional[MemoryLayer] = None
    message: str = ""

    # Extra nodes extracted by System 1
    extracted_fact_ids: List[str] = field(default_factory=list)
    extracted_intention_ids: List[str] = field(default_factory=list)

    # V1 compatibility
    extracted_ids: List[str] = field(default_factory=list)
    abstract_key_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "memory_id": self.memory_id,
            "node_ids": self.node_ids,
            "layer": self.layer.value if self.layer else None,
            "message": self.message,
            "extracted_fact_ids": self.extracted_fact_ids,
            "extracted_intention_ids": self.extracted_intention_ids,
        }


@dataclass
class AsyncAddResponse:
    """Async add memory response."""
    success: bool = False
    task_id: str = ""
    status: TaskStatus = TaskStatus.PENDING
    message: str = ""
    estimated_time: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "task_id": self.task_id,
            "status": self.status.value,
            "message": self.message,
            "estimated_time": self.estimated_time,
        }


@dataclass
class TaskStatusRequest:
    """Query task status request."""
    task_id: str = ""


@dataclass
class TaskStatusResponse:
    """Task status response."""
    success: bool = False
    task_id: str = ""
    status: TaskStatus = TaskStatus.PENDING
    message: str = ""
    progress: int = 0
    result: Optional[AddResponse] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "task_id": self.task_id,
            "status": self.status.value,
            "message": self.message,
            "progress": self.progress,
            "result": self.result.to_dict() if self.result else None,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class AsyncTask:
    """Async task."""
    task_id: str = ""
    uid: str = ""
    agent_id: str = ""
    request: Optional[AddRequest] = None
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0
    result: Optional[AddResponse] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @classmethod
    def create(cls, request: AddRequest) -> "AsyncTask":
        return cls(
            task_id=str(uuid.uuid4()),
            uid=request.uid,
            agent_id=request.agent_id,
            request=request,
        )


# ============================================================
# Retrieve memory
# ============================================================

@dataclass
class RecallRequest:
    """
    Retrieve memory request

    V2 adaptations:
    - adds session_id (used for reconsolidation)
    - adds include_context_package (default True, returns a MemoryContextPackage)
    - adds token_budget (token budget)
    """
    query: str = ""
    uid: str = ""

    # Session info
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    agent_ids: List[str] = field(default_factory=list)
    layers: List[MemoryLayer] = field(default_factory=list)

    # Result limits
    limit: int = 10
    limit_per_layer: Optional[int] = None

    # Cross-scene
    cross_scene: bool = False

    # Filters
    tags: List[str] = field(default_factory=list)
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None
    min_importance: float = 0.0
    min_confidence: float = 0.0

    # V2: Context Package options
    include_context_package: bool = True
    token_budget: int = 2000
    include_meta_cognition: bool = True
    include_intentions: bool = True
    include_schemas: bool = True
    include_temporal: bool = True

    # V1 compatibility: scoring weights
    semantic_weight: float = 0.5
    recency_weight: float = 0.3
    importance_weight: float = 0.15
    access_weight: float = 0.05

    # Return options
    include_score: bool = True
    include_metadata: bool = True


@dataclass
class RecallResponse:
    """
    Retrieve memory response

    V2: the core return value is MemoryContextPackage; memories is kept for V1 compatibility.
    """
    success: bool = False
    context_package: Optional[MemoryContextPackage] = None
    memories: List[MemoryNode] = field(default_factory=list)
    total_count: int = 0
    message: str = ""
    by_category: Dict[str, List[MemoryNode]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "success": self.success,
            "total_count": self.total_count,
            "message": self.message,
            "memories": [m.to_dict() for m in self.memories],
            "by_category": {
                k: [m.to_dict() for m in v]
                for k, v in self.by_category.items()
            },
        }
        if self.context_package:
            result["context_package"] = self.context_package.to_dict()
        return result


# ============================================================
# Get memory
# ============================================================

@dataclass
class GetRequest:
    """Get a single memory request."""
    memory_id: str = ""


@dataclass
class GetResponse:
    """Get a single memory response."""
    success: bool = False
    memory: Optional[MemoryNode] = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "memory": self.memory.to_dict() if self.memory else None,
            "message": self.message,
        }


# ============================================================
# List memories
# ============================================================

@dataclass
class ListRequest:
    """List memories request."""
    uid: str = ""
    agent_id: Optional[str] = None
    layers: List[MemoryLayer] = field(default_factory=list)
    offset: int = 0
    limit: int = 100
    order_by: str = "created_at"
    order_desc: bool = True


@dataclass
class ListResponse:
    """List memories response."""
    success: bool = False
    memories: List[MemoryNode] = field(default_factory=list)
    total_count: int = 0
    offset: int = 0
    limit: int = 100
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "memories": [m.to_dict() for m in self.memories],
            "total_count": self.total_count,
            "offset": self.offset,
            "limit": self.limit,
            "message": self.message,
        }


# ============================================================
# Update memory
# ============================================================

@dataclass
class UpdateRequest:
    """Update memory request."""
    memory_id: str = ""
    content: Optional[str] = None
    importance: Optional[float] = None
    tags: Optional[List[str]] = None
    custom: Optional[Dict[str, Any]] = None
    conflict_strategy: str = "update"


@dataclass
class UpdateResponse:
    """Update memory response."""
    success: bool = False
    memory_id: str = ""
    message: str = ""
    conflict_resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "memory_id": self.memory_id,
            "message": self.message,
            "conflict_resolved": self.conflict_resolved,
        }


# ============================================================
# Delete memory
# ============================================================

@dataclass
class DeleteRequest:
    """Delete a single memory request."""
    memory_id: str = ""


@dataclass
class DeleteResponse:
    """Delete memory response."""
    success: bool = False
    deleted_count: int = 0
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "deleted_count": self.deleted_count,
            "message": self.message,
        }


@dataclass
class BatchDeleteRequest:
    """Batch delete memories request."""
    scope: DeleteScope = DeleteScope.MEMORY
    memory_ids: List[str] = field(default_factory=list)
    uid: Optional[str] = None
    agent_id: Optional[str] = None
    layers: List[MemoryLayer] = field(default_factory=list)
    before_time: Optional[datetime] = None
    confirm: bool = False


@dataclass
class BatchDeleteResponse:
    """Batch delete response."""
    success: bool = False
    deleted_count: int = 0
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "deleted_count": self.deleted_count,
            "message": self.message,
        }


# ============================================================
# User profile
# ============================================================

@dataclass
class UserProfile:
    """User profile."""
    uid: str = ""
    basic_info: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1

    # New in V2
    schemas: List[str] = field(default_factory=list)
    gotchas: List[str] = field(default_factory=list)
    personality: str = ""
    hypotheses: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "basic_info": self.basic_info,
            "preferences": self.preferences,
            "relationships": self.relationships,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "version": self.version,
            "schemas": self.schemas,
            "gotchas": self.gotchas,
            "personality": self.personality,
            "hypotheses": self.hypotheses,
        }


@dataclass
class GetProfileRequest:
    """Get user profile request."""
    uid: str = ""
    include_history: bool = False


@dataclass
class GetProfileResponse:
    """Get user profile response."""
    success: bool = False
    profile: Optional[UserProfile] = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "profile": self.profile.to_dict() if self.profile else None,
            "message": self.message,
        }


@dataclass
class UpdateProfileRequest:
    """Update user profile request."""
    uid: str = ""
    updates: Dict[str, Any] = field(default_factory=dict)
    merge: bool = True


@dataclass
class UpdateProfileResponse:
    """Update user profile response."""
    success: bool = False
    profile: Optional[UserProfile] = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "profile": self.profile.to_dict() if self.profile else None,
            "message": self.message,
        }


@dataclass
class RebuildProfileRequest:
    """Rebuild user profile request."""
    uid: str = ""
    force: bool = False


@dataclass
class RebuildProfileResponse:
    """Rebuild user profile response."""
    success: bool = False
    profile: Optional[UserProfile] = None
    memories_processed: int = 0
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "profile": self.profile.to_dict() if self.profile else None,
            "memories_processed": self.memories_processed,
            "message": self.message,
        }


# ============================================================
# Async task management
# ============================================================

@dataclass
class SubmitTaskRequest:
    """Submit async task request."""
    task_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    callback_url: Optional[str] = None


@dataclass
class SubmitTaskResponse:
    """Submit async task response."""
    success: bool = False
    task_id: str = ""
    status: str = "pending"
    message: str = ""
    estimated_time: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "task_id": self.task_id,
            "status": self.status,
            "message": self.message,
            "estimated_time": self.estimated_time,
        }


@dataclass
class GetTaskRequest:
    """Query task status request."""
    task_id: str = ""


@dataclass
class GetTaskResponse:
    """Query task status response."""
    success: bool = False
    task_id: str = ""
    task_type: str = ""
    status: str = "pending"
    progress: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    message: str = ""
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def is_completed(self) -> bool:
        return self.status == "completed"

    @property
    def is_failed(self) -> bool:
        return self.status == "failed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "message": self.message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class CancelTaskRequest:
    """Cancel task request."""
    task_id: str = ""


@dataclass
class CancelTaskResponse:
    """Cancel task response."""
    success: bool = False
    task_id: str = ""
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "task_id": self.task_id,
            "message": self.message,
        }


@dataclass
class ListTasksRequest:
    """List tasks request."""
    uid: Optional[str] = None
    task_type: Optional[str] = None
    status: Optional[str] = None
    offset: int = 0
    limit: int = 20


@dataclass
class ListTasksResponse:
    """List tasks response."""
    success: bool = False
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    total_count: int = 0
    offset: int = 0
    limit: int = 20
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "tasks": self.tasks,
            "total_count": self.total_count,
            "offset": self.offset,
            "limit": self.limit,
            "message": self.message,
        }
