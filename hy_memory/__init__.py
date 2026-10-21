"""
HY Memory - Core SDK

Industrial-grade core framework for an agent memory system.

Quickstart:
    export OPENAI_API_KEY="sk-your-key-here"

    from hy_memory import HyMemoryClient

    # Simplest usage -- just set the OPENAI_API_KEY environment variable
    client = HyMemoryClient(mode="pro")
    client.add("The user likes sci-fi movies", user_id="test_user")
    results = client.search("What does the user like?", user_ids=["test_user"])
    client.close()

Default configuration:
    - embedder: OpenAI text-embedding-3-small (1536 dimensions)
    - llm: OpenAI gpt-4.1-nano
    - vector_store: embedded Chroma local (zero external dependencies)

    Just set the OPENAI_API_KEY environment variable and it runs.

    # Custom configuration
    client = HyMemoryClient.from_config({
        "vector_store": {"provider": "qdrant"},
        "graph_store": {
            "provider": "neo4j",
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password",
        },
        "enable_graph": True,
    })

Data isolation (three levels):
    - user_id:    level-1 key -- a memory library unique to each user
    - agent_id:   level-2 key -- isolates Agent scenes under the same user
    - session_id: level-3 key -- isolates sessions under the same Agent

Installation:
    pip install hy-memory          # core dependencies (includes Chroma, works out of the box)
    pip install hy-memory[qdrant]  # + Qdrant vector database
    pip install hy-memory[faiss]   # + FAISS vector database
    pip install hy-memory[all]     # includes all optional dependencies
"""

try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("hy-memory")
except Exception:
    __version__ = "0.1.0"

# ====== User-level API (recommended) ======
from .client import HyMemoryClient
from .inspector import MemoryInspector
from .runtime import SharedRuntime

# ====== Configuration ======
from .config import MemoryConfig

# ====== Embedding endpoint override (per-call; keeps model/dims, swaps only the access endpoint) ======
from .core.embed_service import embed_endpoint_override

# ====== Advanced API (as needed) ======
from .pipelines import (
    ComponentFactory,
    PipelineConfig,
    WritePipeline,
    ReadPipeline,
    ChatMessage,
    WriteRequest,
    WriteResponse,
    ReadRequest,
    ReadResponse,
)

# ====== Data models ======
from .models import (
    MemoryLayer,
    MemoryNode,
    MemoryEntry,
    MemoryMetadata,
    MemoryScore,
    MemoryInputType,
    AgentProcessMode,
    TaskStatus,
    DeleteScope,
    QAPair,
    AddRequest,
    AddResponse,
    AsyncAddResponse,
    RecallRequest,
    RecallResponse,
    UpdateRequest,
    UpdateResponse,
    DeleteRequest,
    DeleteResponse,
    BatchDeleteRequest,
    BatchDeleteResponse,
    GetRequest,
    GetResponse,
    ListRequest,
    ListResponse,
)


__all__ = [
    "__version__",
    # User-level API
    "HyMemoryClient",
    "MemoryInspector",
    "SharedRuntime",
    # Configuration
    "MemoryConfig",
    # Embedding endpoint override
    "embed_endpoint_override",
    # Advanced API
    "ComponentFactory",
    "PipelineConfig",
    "WritePipeline",
    "ReadPipeline",
    "ChatMessage",
    "WriteRequest",
    "WriteResponse",
    "ReadRequest",
    "ReadResponse",
    # Data models
    "MemoryLayer",
    "MemoryNode",
    "MemoryEntry",
    "MemoryMetadata",
    "MemoryScore",
    "MemoryInputType",
    "AgentProcessMode",
    "TaskStatus",
    "DeleteScope",
    "QAPair",
    "AddRequest",
    "AddResponse",
    "AsyncAddResponse",
    "RecallRequest",
    "RecallResponse",
    "UpdateRequest",
    "UpdateResponse",
    "DeleteRequest",
    "DeleteResponse",
    "BatchDeleteRequest",
    "BatchDeleteResponse",
    "GetRequest",
    "GetResponse",
    "ListRequest",
    "ListResponse",
]
