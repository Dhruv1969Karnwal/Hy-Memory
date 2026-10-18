# -*- coding: utf-8 -*-
"""
Qdrant BM25 sparse-vector encoder wrapper (fastembed Qdrant/bm25), process-wide singleton.

Qdrant-native BM25 uses sparse vectors:
  - Write: document text → fastembed .embed() → sparse(indices, values), with IDF/TF weights
  - Search: query → .query_embed() → sparse, query_points(using="bm25")

BM25 is asymmetric: the document side uses embed() (with IDF×TF weights), the query side uses query_embed()
(weights all 1). The two must not be used interchangeably.

The first SparseTextEmbedding('Qdrant/bm25') call downloads the ONNX model (hundreds of MB) and takes
~4.4s to load, so it must be cached as a process-wide singleton. Graceful degradation when fastembed is missing: return None so
callers disable qdrant sparse capability accordingly.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# (indices, values): usable directly as qdrant SparseVector(indices=, values=)
SparseTuple = Tuple[List[int], List[float]]

_encoder = None            # SparseTextEmbedding | None | False(=unavailable)
_lock = threading.Lock()

_MODEL_NAME = "Qdrant/bm25"


def _get_encoder():
    """Lazily load the fastembed BM25 singleton. Returns False when unavailable."""
    global _encoder
    if _encoder is not None:
        return _encoder
    with _lock:
        if _encoder is not None:
            return _encoder
        try:
            from fastembed import SparseTextEmbedding
            enc = SparseTextEmbedding(_MODEL_NAME)
            logger.info(f"[bm25-fastembed] SparseTextEmbedding('{_MODEL_NAME}') loaded")
            _encoder = enc
        except Exception as e:
            logger.warning(
                f"[bm25-fastembed] fastembed unavailable ({e}); "
                f"qdrant sparse BM25 disabled, keyword channel will be empty. "
                f"pip install fastembed"
            )
            _encoder = False
    return _encoder


def is_available() -> bool:
    """Whether the fastembed BM25 encoder is available."""
    return bool(_get_encoder())


def _to_tuple(sparse_embedding) -> Optional[SparseTuple]:
    """fastembed SparseEmbedding → (indices:List[int], values:List[float])。"""
    try:
        indices = [int(i) for i in sparse_embedding.indices]
        values = [float(v) for v in sparse_embedding.values]
        if not indices:
            return None
        return indices, values
    except Exception as e:
        logger.debug(f"[bm25-fastembed] sparse convert failed: {e}")
        return None


def encode_doc(text: str) -> Optional[SparseTuple]:
    """Document-side encoding (with IDF×TF weights); returns None when unavailable or for empty text."""
    enc = _get_encoder()
    if not enc or not text:
        return None
    try:
        out = list(enc.embed([text]))
        return _to_tuple(out[0]) if out else None
    except Exception as e:
        logger.debug(f"[bm25-fastembed] encode_doc failed: {e}")
        return None


def encode_docs(texts: List[str]) -> List[Optional[SparseTuple]]:
    """Batch document encoding; same length as input, empty texts map to None."""
    enc = _get_encoder()
    if not enc:
        return [None] * len(texts)
    idxs = [i for i, t in enumerate(texts) if t]
    result: List[Optional[SparseTuple]] = [None] * len(texts)
    if not idxs:
        return result
    try:
        encoded = list(enc.embed([texts[i] for i in idxs]))
        for j, i in enumerate(idxs):
            if j < len(encoded):
                result[i] = _to_tuple(encoded[j])
    except Exception as e:
        logger.debug(f"[bm25-fastembed] encode_docs failed: {e}")
    return result


def encode_query(text: str) -> Optional[SparseTuple]:
    """Query-side encoding (query_embed, weights all 1); returns None when unavailable or for empty text."""
    enc = _get_encoder()
    if not enc or not text:
        return None
    try:
        out = list(enc.query_embed(text))
        return _to_tuple(out[0]) if out else None
    except Exception as e:
        logger.debug(f"[bm25-fastembed] encode_query failed: {e}")
        return None
