"""
HY Memory - Pipelines

pipelines/
├── base.py                # WritePipeline / ReadPipeline abstract bases + unified protocol
├── registry.py            # ComponentFactory -- component factory
├── writer.py              # MemoryWriter (System 1 writes)
├── reader.py              # MemoryReader (dispatcher → legacy/hybrid)
├── reader_base.py         # BaseReadPipeline (shared post-processing)
├── reader_legacy.py       # LegacyReadPipeline（vector recall）
├── reader_hybrid.py       # HybridReadPipeline（vector + BM25 recall）
├── system2_writer.py      # System2Writer (ultra mode: System 1 + System 2 scheduling)
├── system2_agent.py       # System 2 Agent (LLM + Graph tools)
├── system2_tools.py       # System 2 Tool definitions
├── cross_domain_sweeper.py # Cross-domain schema induction
└── _retrieval/            # Retrieval strategy submodules (BM25, RRF, tag_index, evolution)
"""

from .base import (
    WritePipeline,
    ReadPipeline,
    ChatMessage,
    ToolCall,
    WriteRequest,
    WriteResponse,
    ReadRequest,
    ReadResponse,
    PipelineContext,
)
from .registry import ComponentFactory, PipelineConfig

# Backward-compatible aliases
PipelineRegistry = ComponentFactory

__all__ = [
    # Abstract base classes
    "WritePipeline",
    "ReadPipeline",
    # Unified protocol
    "ChatMessage",
    "ToolCall",
    "WriteRequest",
    "WriteResponse",
    "ReadRequest",
    "ReadResponse",
    "PipelineContext",
    # Component factory
    "ComponentFactory",
    "PipelineRegistry",  # Backward compatibility
    "PipelineConfig",    # Backward compatibility
]
