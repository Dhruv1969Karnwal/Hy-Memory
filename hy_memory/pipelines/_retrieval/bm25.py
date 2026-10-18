"""
BM25-lite: run the BM25 formula directly over the bounded in-memory candidate pool (20~100 items).

Design trade-offs:
  - No full-corpus inverted index is built. The retrieval pool is already fixed by the vector/tag
    channels, and BM25 only re-ranks inside the pool; at size <= 100 the pure Python computation takes microseconds to milliseconds, with zero external dependencies
  - The statistical base for IDF / avgdl is the "current candidate pool", not the whole corpus. This
    biases things slightly (a rare term has a different IDF corpus-wide than inside the pool), but the
    pool size keeps the relative ordering sound; besides, BM25 scores only feed the RRF rank, which is insensitive to absolute scores
  - Mixed Chinese/English is supported: English is tokenized on whitespace plus punctuation and
    lowercased, while a run of CJK characters becomes a single whole token (OMEGA originally did not handle Chinese; this is a minimal extension)

This module is a pure-function implementation with no side effects, which keeps it easy to unit test.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Tuple

from . import config


# Reuse the token regex from the intent module
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [tok.lower() if tok.isascii() else tok for tok in _TOKEN_RE.findall(text)]


def compute_bm25_scores(
    query_terms: List[str],
    candidate_contents: List[str],
    k1: float = None,
    b: float = None,
) -> List[float]:
    """
    Compute the BM25 score for every entry in candidate_contents.

    Args:
        query_terms: tokenized query token list (same shape as _tokenize produces)
        candidate_contents: list of candidate content strings
        k1, b: BM25 parameters; the config defaults are used when None

    Returns:
        Raw BM25 scores, the same length as candidate_contents.
        All zeros when query_terms is empty or the pool is empty.
    """
    if not query_terms or not candidate_contents:
        return [0.0] * len(candidate_contents)

    k1 = k1 if k1 is not None else config.BM25_K1
    b = b if b is not None else config.BM25_B

    N = len(candidate_contents)
    tokenized: List[List[str]] = [_tokenize(c) for c in candidate_contents]
    doc_lens: List[int] = [len(toks) for toks in tokenized]
    avgdl = (sum(doc_lens) / N) if N > 0 else 1.0
    avgdl = avgdl or 1.0  # Avoid division by zero

    # Precompute df for each query term
    q_terms = list({t for t in query_terms if t})
    df: Dict[str, int] = {}
    for t in q_terms:
        df[t] = sum(1 for toks in tokenized if t in toks)

    # Precompute term freq per doc (project each doc onto q_terms instead of building a full Counter)
    tf_per_doc: List[Dict[str, int]] = []
    for toks in tokenized:
        c = Counter(toks)
        tf_per_doc.append({t: c.get(t, 0) for t in q_terms})

    scores: List[float] = []
    for i, toks in enumerate(tokenized):
        dl = doc_lens[i] or 1
        score = 0.0
        tf_map = tf_per_doc[i]
        for t in q_terms:
            tf = tf_map.get(t, 0)
            if tf == 0:
                continue
            # IDF with +1 smoothing (BM25+, avoids negative IDF)
            idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            score += idf * (tf * (k1 + 1)) / denom
        scores.append(score)

    return scores


def score_and_rank(
    query_terms: List[str],
    candidates: List[Tuple[str, str]],
) -> List[Tuple[str, float]]:
    """
    Convenience function: take [(id, content), ...] and return [(id, score), ...] sorted by BM25 descending.
    Scores are divided by the max and normalized to [0, 1] (does not affect rank, easier to observe).
    """
    contents = [c for _, c in candidates]
    raw = compute_bm25_scores(query_terms, contents)
    max_s = max(raw) if raw else 0.0
    if max_s <= 0:
        norm = [0.0] * len(raw)
    else:
        norm = [s / max_s for s in raw]
    ranked = [(cid, norm[i]) for i, (cid, _) in enumerate(candidates)]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# Exposed so the intent module can reuse it
tokenize = _tokenize
