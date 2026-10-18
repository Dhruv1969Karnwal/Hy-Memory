"""
HY Memory - abstract base classes + unified protocol

Defines the WritePipeline / ReadPipeline abstract interfaces plus the unified request/response protocol.

Design principles:
1. Unified protocol -- WriteRequest/Response and ReadRequest/Response are version-agnostic
2. Extensible -- each version passes its version-specific data in extra_data
3. Lifecycle management -- resources are managed through initialize() / close()
"""

import re
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

# Type forward declarations (avoids circular imports)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..utils.tracer import PipelineTracer


# ================================================================
# Unified request/response protocol
# ================================================================

@dataclass
class PipelineContext:
    """
    Pipeline execution context -- shared state spanning the whole write/read flow.
    
    Shared modules can pass intermediate results through this context for use inside the pipeline.
    """
    # Request identity
    request_id: str = ""
    
    # User isolation (two levels: user_id + agent_id)
    user_id: str = ""
    agent_id: str = ""

    # Session
    session_id: str = ""

    # Intermediate result storage (used inside the pipeline)
    intermediate: Dict[str, Any] = field(default_factory=dict)

    # Performance tracking
    start_time: Optional[datetime] = None
    timings: Dict[str, float] = field(default_factory=dict)

    def elapsed_ms(self) -> float:
        """Milliseconds elapsed from start_time until now"""
        if self.start_time is None:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds() * 1000

    @property
    def isolation_key(self) -> str:
        """Build the isolation key: {user_id}:{agent_id}"""
        parts = [self.user_id or "default", self.agent_id or "default"]
        return ":".join(parts)


@dataclass
class ToolCall:
    """
    A tool call initiated by the assistant (normalized representation for OpenAI / Anthropic).

    Every entry of the OpenAI-style `tool_calls` array and every Anthropic-style content
    block (`type=="tool_use"`) is normalized into this structure.
    """
    id: str = ""                         # call id (OpenAI: tc.id; Anthropic: tool_use block id)
    name: str = ""                       # tool name
    arguments: Dict[str, Any] = field(default_factory=dict)
    # OpenAI encodes arguments as a JSON string; the normalization layer json.loads them before passing them in
    # Anthropic already gives a dict; the normalization layer copies it as is

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            arguments=data.get("arguments", {}) or {},
        )


# ── user name line (multi-real-person dialogues only) ──
# A dialogue with names renders as:
#   The user's name is Tim
#   [user]: I like coffee
#   [John]: I like tea too
# The name occupies its own line and appears exactly once per block; the [user] label is **not replaced by the name**, so the perspective stays in the label.
# extract uses this line to take name for L0 basic_info (the user need not state their name in the dialogue).
_USER_NAME_LINE_ZH = "\u7528\u6237\u540d\u53eb{name}"
_USER_NAME_LINE_EN = "The user's name is {name}"

_USER_NAME_LINE_RE = re.compile(
    r"^(?:\u7528\u6237\u540d\u53eb|The user's name is)\s*(.+?)\s*$"
)


def render_user_name_line(name: str, is_zh: bool) -> str:
    """Render the leading user name line; returns an empty string when name is empty."""
    name = (name or "").strip()
    if not name:
        return ""
    tpl = _USER_NAME_LINE_ZH if is_zh else _USER_NAME_LINE_EN
    return tpl.format(name=name)


def parse_user_name_line(line: str) -> Optional[str]:
    """Return the name when the line is a user name line, otherwise None.

    L1_RAW stores exactly what get_flat_content() produces, so reading history back relies on this to recognize the name line,
    otherwise it would be treated as a bare message with no prefix (see MemoryWriter._get_recent_history).
    """
    m = _USER_NAME_LINE_RE.match((line or "").strip())
    return m.group(1) if m else None


@dataclass
class ChatMessage:
    """A single chat message, compatible with the OpenAI / Anthropic messages format (including tool calls)"""
    role: str = "user"       # user / assistant / system / tool
    content: str = ""

    name: Optional[str] = None
    # Speaker display name (the `name` field of OpenAI messages). Multi-real-person dialogues use it to tell the parties apart:
    #   role=user     → the current memory owner. A name only states "what the user is called"; the label still renders [user]
    #   role=assistant + name → **the other real person** (not an AI), rendered as [<name>]
    #   role=assistant without name → AI assistant, rendered as [assistant]
    # Perspective is decided by role; name is only a display name, and both sides may have one.
    # With role=tool this field is not used here (the tool name lives in tool_name).

    # ── tool extensions (used only on the productivity / coding paths; the chat path ignores them) ──
    tool_calls: List[ToolCall] = field(default_factory=list)
    # With role=assistant: the list of tool calls that were initiated
    # Always empty for other roles

    tool_call_id: Optional[str] = None
    # With role=tool: the call id this message corresponds to
    # None for other roles

    tool_name: Optional[str] = None
    # With role=tool: the tool name this message corresponds to (OpenAI provides it directly; Anthropic needs a lookup in the preceding tool_use, may be empty)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_name is not None:
            d["tool_name"] = self.tool_name
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", "") or "",
            name=data.get("name") or None,
            tool_calls=[ToolCall.from_dict(tc) for tc in (data.get("tool_calls") or [])],
            tool_call_id=data.get("tool_call_id"),
            tool_name=data.get("tool_name"),
        )

    def speaker_label(self) -> str:
        """Speaker label used for rendering (without brackets).

        role=user is always "user" (even when a name is present), so the perspective stays in the label;
        the other real person (role=assistant + name) uses their name; everything else falls back to the role.
        """
        if self.role == "user":
            return "user"
        if self.role == "assistant" and self.name:
            return self.name
        return self.role

    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def is_tool_message(self) -> bool:
        return self.role == "tool"


@dataclass
class WriteRequest:
    """
    Unified write request protocol.
    
    Every version of WritePipeline accepts requests in this format.
    Version-specific parameters are passed via extra.
    
    Input modes (choose one):
    1. content: str -- a single text (backward compatible)
    2. messages: List[ChatMessage] -- a multi-turn (user-assistant)*N dialogue
       messages takes precedence; content is used as a fallback when messages is empty.
    """
    # Required (choose one)
    content: str = ""
    messages: List[ChatMessage] = field(default_factory=list)
    
    # User isolation
    user_id: str = ""
    agent_id: str = ""
    # Call source (openclaw / cursor / codex, etc.), orthogonal to agent_id.
    source: str = "default"

    # Request trace id. Injected explicitly at the client entry point (contextvar-immune) and used across the whole write pipeline
    # as the single source of truth for persisting memory_ops / WRITE_SUMMARY / pipeline_logs. When empty, the writer
    # falls back to the contextvar.
    request_id: str = ""

    # Optional metadata
    content_type: str = "text"
    session_id: str = ""
    turn_index: Optional[int] = None
    role: str = "user"
    assistant_content: Optional[str] = None
    
    # Version-specific parameters (V1: agent_mode etc.; V2: gate_config etc.)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    # Existing memories (used by V1 merge detection)
    existing_memories: Optional[List[Dict[str, Any]]] = None

    # Memory timestamp (the current time datetime.now() is used when omitted)
    memory_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        from ..utils.source_filter import normalize_source
        self.source = normalize_source(self.source)


    # Caller-defined business meta, stored together with the memories produced by this call and returned verbatim on query.
    # Pure payload only: not embedded, not part of retrieval, never used as a filter condition.
    #
    # Deliberately separate from extra: extra mixes SDK-internal control keys (agent_mode / mode / layer),
    # and placing caller-defined keys next to them would let the two overwrite each other.
    #
    # The client_ prefix keeps this apart from the writer-internal "meta" (metadata of the extraction result,
    # shaped like {"content","layer","tags"}); the two are entirely unrelated.
    client_meta: Optional[Dict[str, Any]] = None

    # ── Coding memory path only (ignored on the chat path) ──
    workspace_id: Optional[str] = None
    # Repo identifier. Normalizing the git remote URL is recommended (e.g. "github.com/org/repo").
    # Persisted with experiences and facts, surfaced on retrieval, and available as an optional search / list filter;
    # a missing value never causes the write to be rejected.
    branch: Optional[str] = None
    # Branch name. Same rules as workspace_id.

    # ---- Convenience methods ----

    def has_messages(self) -> bool:
        """Whether multi-turn dialogue input is present"""
        return bool(self.messages)

    def get_user_queries(self) -> List[str]:
        """Extract the content of every user-role message (used to embed and check novelty)"""
        return [m.content for m in self.messages if m.role == "user" and m.content.strip()]

    def user_name(self) -> Optional[str]:
        """Display name of the current user (the role=user side); None when absent."""
        for m in self.messages:
            if m.role == "user" and m.name:
                return m.name.strip() or None
        return None

    def get_flat_content(self) -> str:
        """
        Return the full text content:
        - with messages → joined into dialogue format
        - otherwise → return content

        Rendered format (multi-real-person dialogue):
            The user's name is Tim
            [user]: I like coffee
            [John]: I like tea too

        Without a name the behaviour is identical to the previous one (`[user]: ...` / `[assistant]: ...`).
        The name line appears only when role=user carries a name, and exactly once per block; the language is decided from the message content
        (the same criterion the extractor uses to pick the zh/en prompt).
        """
        if not self.messages:
            return self.content or ""

        lines = [f"[{m.speaker_label()}]: {m.content}" for m in self.messages]

        name = self.user_name()
        if name:
            from ..utils.lang_detect import is_chinese

            name_line = render_user_name_line(
                name, is_chinese(messages=self.messages)
            )
            if name_line:
                lines.insert(0, name_line)

        return "\n".join(lines)

    def get_dialogue_text(self) -> str:
        """
        Return a dialogue format suited to LLM reading:
        - with messages → full dialogue format
        - with content + assistant_content → single-turn QA format
        - otherwise → the raw content
        """
        if self.messages:
            parts = []
            for m in self.messages:
                label = {"user": "User", "assistant": "Assistant", "system": "System"}.get(m.role, m.role)
                parts.append(f"{label}: {m.content}")
            return "\n".join(parts)
        if self.assistant_content:
            return f"User: {self.content}\nAssistant: {self.assistant_content}"
        return self.content or ""


@dataclass
class WriteResponse:
    """
    Unified write response protocol.
    
    Every version of WritePipeline returns responses in this format.
    """
    success: bool = False
    
    # Write result
    memory_id: str = ""
    layer: str = ""
    message: str = ""
    
    # Extracted entities (generic)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance
    elapsed_ms: float = 0.0
    tokens_used: int = 0
    
    # Error information
    error_code: int = 0
    error_message: str = ""
    
    # Version-specific data (V1: routing_confidence, should_merge, etc.; V2: gate_passed, l2_fact_ids, etc.)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "memory_id": self.memory_id,
            "layer": self.layer,
            "message": self.message,
            "entities": self.entities,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "tokens_used": self.tokens_used,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "extra": self.extra,
        }


@dataclass
class ReadRequest:
    """
    Unified read request protocol.
    """
    # Required
    query: str = ""

    # User isolation
    user_id: str = ""              # single user (backward compatible)
    user_ids: List[str] = field(default_factory=list)  # multi-user search
    agent_id: str = ""

    # Flexible search filters (list form)
    agent_ids: List[str] = field(default_factory=list)
    session_ids: List[str] = field(default_factory=list)
    # Call source filter. When both are given the semantics are IN(include) AND NOT IN(exclude),
    # with exclude taking precedence. The underlying store must apply this before top-k/limit.
    include_sources: List[str] = field(default_factory=list)
    exclude_sources: List[str] = field(default_factory=list)

    # Retrieval parameters
    limit: int = 10
    layers: Optional[List[str]] = None
    min_score: float = 0.4             # generic minimum score threshold (used as the recall bar by some readers)

    # Schema (L6) independent retrieval parameters
    schema_min_score: float = 0.4      # minimum score threshold for the Schema path
    schema_limit: int = 10             # maximum number of results for the Schema path

    # Proactive (Intention) independent retrieval parameters
    intention_limit: int = 0           # topk for the Intention path, 0 (default) disables it


    # Time filter
    created_after: Optional[float] = None  # Unix timestamp (float); returns only memories with gmt_created >= this value

    # Precomputed query vector (optional, avoids re-embedding)
    query_embedding: Optional[List[float]] = None

    # Session
    session_id: str = ""

    # Trace / observability
    # If the caller already generated a request_id at the async_search layer (recommended), it is passed through.
    # The reader uses it to associate pipeline_logs with this search so the inspector can query them.
    request_id: str = ""

    # Version-specific parameters (V2: token_budget, use_llm_understanding, etc.)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from ..utils.source_filter import normalize_source_filters
        self.include_sources, self.exclude_sources = normalize_source_filters(
            self.include_sources, self.exclude_sources,
        )


@dataclass
class ReadResponse:
    """
    Unified read response protocol.
    """
    success: bool = False
    
    # Retrieval results (generic format)
    memories: List[Dict[str, Any]] = field(default_factory=list)
    total_found: int = 0
    
    # Performance
    elapsed_ms: float = 0.0
    
    # Error information
    error_code: int = 0
    error_message: str = ""
    
    # Version-specific data (V2: context_package, query_understanding, recalled_node_ids, etc.)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "memories": self.memories,
            "total_found": self.total_found,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "error_code": self.error_code,
            "error_message": self.error_message,
            "extra": self.extra,
        }


# ================================================================
# Pipeline abstract base classes
# ================================================================

class WritePipeline(ABC):
    """
    Write abstract base class.

    Shared modules (embed_service, vector_store, etc.) are injected through the constructor.

    Example:
        class MemoryWriter(WritePipeline):
            async def write(self, request, ctx):
                ...
    """
    
    # Pipeline version identifier (subclasses must override)
    VERSION: str = "base"
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the resources the pipeline needs.
        
        For example: open storage connections, load models, and so on.
        Call it once before first use.
        """
        pass
    
    @abstractmethod
    async def write(
        self,
        request: WriteRequest,
        ctx: Optional[PipelineContext] = None,
        tracer: Optional["PipelineTracer"] = None,
    ) -> WriteResponse:
        """
        Execute the write flow.
        
        Args:
            request: unified write request
            ctx: pipeline context (optional, used to pass intermediate state)
            
        Returns:
            Unified write response
        """
        pass
    
    async def close(self) -> None:
        """Release resources (default no-op)"""
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} version={self.VERSION}>"


class ReadPipeline(ABC):
    """
    Read abstract base class.
    """
    
    # Pipeline version identifier (subclasses must override)
    VERSION: str = "base"
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the resources the pipeline needs."""
        pass
    
    @abstractmethod
    async def read(
        self,
        request: ReadRequest,
        ctx: Optional[PipelineContext] = None,
        tracer: Optional["PipelineTracer"] = None,
    ) -> ReadResponse:
        """
        Execute the read flow.
        
        Args:
            request: unified read request
            ctx: pipeline context
            
        Returns:
            Unified read response
        """
        pass
    
    async def close(self) -> None:
        """Release resources (default no-op)"""
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} version={self.VERSION}>"
