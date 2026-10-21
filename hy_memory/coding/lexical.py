"""Lexical channel for the fact layer: BM25 + score fusion, reranking store.search_facts.

Why this channel exists: statement is the only field that enters the retrieval vector (store._upsert_fact_vec embeds only
`subject: statement`), while much of what it records is given quantities -- symbol names, numbers, units, labels. Dense
vectors are blunt about such tokens: the cosine between "step 3 input: n=64" and "step 4 input: n=32" is nearly identical,
yet the query wants exactly one of them. The lexical channel is sensitive to exact tokens and fills that hole.

Reranking only changes results when there are more candidates than top-k; with fewer candidates the channel is naturally a no-op.
The lexical weight should suffice to break up close-scoring semantic neighbours, but never outweigh clearly more relevant semantic candidates.

Two design constraints, both forced by the caller:

  1. Never return the fused score upward. client.py:851 gates on `s >= fact_min_score`
     (default 0.30), which is on the cosine scale; the fused score is on another scale, and returning it would
     silently empty the whole fact layer. Fusion therefore only determines order; returned scores stay dense cosine scores.

  2. Reranking only runs over candidates that are "already past the dense floor". Otherwise BM25 could push a fact with cosine 0.2
     into the top, only for the client gate to drop it, and the hybrid would recall less than dense-only. Cutting the pool by
     min_score first and reranking afterwards keeps the returned count equal to dense-only, just better chosen.
"""
from __future__ import annotations

import logging
import os
import re
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

ENV_DISABLE = "HY_FACT_LEXICAL_OFF"   # Set to 1 to turn the lexical channel off and fall back to dense-only
ENV_WEIGHT = "HY_FACT_LEXICAL_WEIGHT"

# Lexical weight relative to dense, meaning "how large a cosine gap lexical may at most overturn".
_LEX_WEIGHT = 0.05

# Numbers must tokenize as a whole, scientific notation included: the identity of a given quantity lives entirely in the full value, so "1e-6"
# cut into 1/e/6 can no longer be told apart from "6e-1". The numeric branch comes before the alphabetic one, otherwise the
# e in "1e-6" would be eaten by the alphabetic branch first.
_LATIN = re.compile(r"\d+(?:\.\d+)?(?:e[-+]?\d+)?|[a-z]+")
_CJK = re.compile(r"[\u4e00-\u9fff]+")  # CJK Unified Ideographs basic block

_warned = False


def _warn_once(msg: str) -> None:
    global _warned
    if not _warned:
        _warned = True
        logger.warning(f"[coding-lexical] {msg}")


def tokenize(text: str) -> List[str]:
    """Tokenizer for mixed Chinese/English text.

    Latin words and numbers come from the regex first (numbers stay whole, see the comment on _LATIN); CJK runs go to jieba.
    The first jieba call loads a dictionary (~0.4s), hence the lazy import here: stores without CJK never pay for it.
    """
    low = text.lower()
    tokens = _LATIN.findall(low)
    cjk = _CJK.findall(low)
    if cjk:
        try:
            import jieba

            for run in cjk:
                tokens.extend(t for t in jieba.lcut(run) if t.strip())
        except Exception as e:  # noqa: BLE001
            _warn_once(f"jieba unavailable ({e}); CJK falls back to bigrams")
            for run in cjk:
                tokens.extend(run[i:i + 2] for i in range(max(len(run) - 1, 1)))
    return tokens


def _weight() -> float:
    """Reads the env var on every call, so behaviour can be switched without changing code."""
    raw = os.getenv(ENV_WEIGHT)
    if not raw:
        return _LEX_WEIGHT
    try:
        return float(raw)
    except ValueError:
        _warn_once(f"{ENV_WEIGHT}={raw!r} is not a number; using {_LEX_WEIGHT}")
        return _LEX_WEIGHT


def rerank(
    query_text: str,
    candidates: Sequence[Tuple[str, float, str]],
    *,
    lex_weight: Optional[float] = None,
) -> Optional[List[str]]:
    """Returns the fused id order; None when unavailable or meaningless (the caller then stays dense-only).

    candidates are (id, dense cosine score, text used to build the lexical index). The text should match the embedding side
    (`subject: statement`); otherwise the two channels are not ranking the same thing.
    """
    if os.getenv(ENV_DISABLE, "").strip() in ("1", "true", "True"):
        return None
    if len(candidates) < 2:
        return None
    if lex_weight is None:
        lex_weight = _weight()
    try:
        # BM25Plus rather than BM25Okapi. Okapi's idf turns negative for terms "present in most documents" and is
        # then flattened by the epsilon floor: with a two-item candidate pool every term appears in >=50% of documents, so
        # both items score the same and cannot be separated at all. Small candidate pools are exactly this channel's
        # working range; Okapi would switch lexical off in the most common case. BM25Plus's delta term keeps scores positive.
        from rank_bm25 import BM25Plus
    except Exception as e:  # noqa: BLE001
        _warn_once(f"rank_bm25 unavailable ({e}); fact search stays dense-only")
        return None

    q = tokenize(query_text)
    if not q:
        return None
    corpus = [tokenize(text) for _, _, text in candidates]
    if not any(corpus):
        return None
    try:
        scores = BM25Plus(corpus).get_scores(q)
    except Exception as e:  # noqa: BLE001
        _warn_once(f"bm25 scoring failed ({e}); fact search stays dense-only")
        return None

    ids = [fid for fid, _, _ in candidates]
    # All candidates score the same => lexical has no discriminative power (no query token matched, or all matched equally).
    # Its "ranking" would then be arbitrary, and fusing it in would perturb the dense order with noise, so skip the rerank.
    lo, hi = min(scores), max(scores)
    if hi - lo <= 1e-9:
        return None

    # Fusion uses the raw cosine score plus the normalized lexical score, not the ranks of the two channels.
    #
    # The trouble with ranks is that they erase margins: with a single-digit candidate pool, normalizing ranks to [0,1]
    # amplifies a 0.0004 cosine difference and a 0.05 one equally into one rank apart, so "dense barely cares" and "dense
    # strongly objects" become indistinguishable. Cosine is already a comparable scale, so use it directly.
    #
    # This gives lex_weight an interpretable meaning: lexical may overturn at most a cosine margin of lex_weight.
    # No more than that.
    lex = {ids[i]: (scores[i] - lo) / (hi - lo) for i in range(len(ids))}
    dense = {fid: s for fid, s, _ in candidates}
    dense_pos = {fid: i for i, fid in enumerate(
        sorted(ids, key=lambda f: -dense[f]))}
    fused = {fid: dense[fid] + lex_weight * lex[fid] for fid in ids}
    # Ties resolve stably in dense order: dense is the primary index, lexical is a patch.
    return sorted(ids, key=lambda f: (-fused[f], dense_pos[f]))
