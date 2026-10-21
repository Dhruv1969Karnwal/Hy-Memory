"""
HY Memory - Read (Dispatcher)

The unified public entry `MemoryReader` selects and proxies the real implementation
based on the environment variable `HY_MEMORY_READER`:

  - "legacy" → LegacyReadPipeline  (default, vector candidate retrieval)
  - "hybrid" → HybridReadPipeline  (vector + BM25 candidate retrieval)

Both share the same post-processing: profile/intention, episode, tag/entity, graph,
local rerank, evolution chains, quotas, and result formatting.

The cache parameter is injected from the Registry and used for trace writes on the read side; when None,
trace writes are silently skipped.
"""

from typing import Any, Optional
import logging

from .base import ReadPipeline, ReadRequest, ReadResponse, PipelineContext
from ..config import MemoryConfig
from ..core.embed_service import EmbedService
from ..data.vector_store_base import VectorStoreBase
from ..data.graph_store_base import GraphStoreBase
from ..utils.tracer import PipelineTracer
from ._retrieval import config as _retrieval_config

logger = logging.getLogger(__name__)


def _build_impl(
    config: MemoryConfig,
    embed_service: Optional[EmbedService],
    vector_store: Optional[VectorStoreBase],
    graph_store: Optional[GraphStoreBase] = None,
    cache: Any = None,
) -> ReadPipeline:
    """Select the internal implementation class via the environment variable and construct the instance."""
    name = _retrieval_config.resolve_reader_name()
    if name == _retrieval_config.READER_HYBRID:
        try:
            from .reader_hybrid import HybridReadPipeline
            return HybridReadPipeline(
                config, embed_service, vector_store,
                graph_store=graph_store, cache=cache,
            )
        except ImportError as e:
            logger.warning(f"[reader-dispatch] hybrid import failed: {e}; fallback to legacy")

    # Default / fallback
    from .reader_legacy import LegacyReadPipeline
    return LegacyReadPipeline(
        config, embed_service, vector_store,
        graph_store=graph_store, cache=cache,
    )


class MemoryReader(ReadPipeline):
    """
    Reader (public entry point).

    Dispatches to different implementations via the `HY_MEMORY_READER` environment variable:
      - legacy / hybrid
    Defaults to legacy; behavior matches pre-post13.
    """

    VERSION = "reader"

    def __init__(
        self,
        config: MemoryConfig,
        embed_service: Optional[EmbedService] = None,
        vector_store: Optional[VectorStoreBase] = None,
        graph_store: Optional[GraphStoreBase] = None,
        cache: Any = None,
    ):
        self.config = config
        self._impl: ReadPipeline = _build_impl(
            config, embed_service, vector_store, graph_store=graph_store, cache=cache,
        )
        logger.info(
            f"[reader-dispatch] active impl = {self._impl.__class__.__name__} "
            f"(HY_MEMORY_READER={_retrieval_config.resolve_reader_name()}) "
            f"graph_store={'yes' if graph_store else 'no'}"
        )

    async def initialize(self) -> None:
        await self._impl.initialize()

    async def read(
        self,
        request: ReadRequest,
        ctx: Optional[PipelineContext] = None,
        tracer: Optional[PipelineTracer] = None,
    ) -> ReadResponse:
        return await self._impl.read(request, ctx=ctx, tracer=tracer)

    async def close(self) -> None:
        await self._impl.close()

    # ---- Convenience properties: expose the internal implementation to diagnostic scripts ----
    @property
    def impl(self) -> ReadPipeline:
        return self._impl

    @property
    def embed_service(self) -> EmbedService:
        return getattr(self._impl, "embed_service", None)
