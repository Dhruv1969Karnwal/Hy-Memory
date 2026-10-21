"""
Intent classification plus query keyword extraction.

Chooses a weighting strategy among the three retrieval intents from the literal anchors and the conceptual phrasing of a query.

Three intents:
- NAVIGATIONAL: the query contains an exact identifier (CamelCase/snake_case/quoted phrase/path/ID/URL
  and the like); the vector channel is switched off and BM25 is weighted up
- CONCEPTUAL: the query asks about tendencies/patterns/attitudes (how/why/explain/tend to, and their equivalents),
  so the tag channel and the vector channel are weighted up
- FACTUAL: the default otherwise, vector-led with tag second

The regexes only catch high-confidence literal signals for NAVIGATIONAL; everything else falls back to the
CONCEPTUAL / FACTUAL keyword trigger tables. Getting FACTUAL and CONCEPTUAL wrong shifts the weights very little
and barely affects the outcome; getting NAVIGATIONAL wrong is costlier (literal match diluted by vectors), so NAV detection is written strictly.
"""

import re
from typing import List

from . import config


# ================================================================
# NAVIGATIONAL patterns (copied from OMEGA plus a light extension)
# ================================================================

_NAV_PATTERNS = [
    r"`[^`]+`",                          # wrapped in backticks
    r'"[^"]{2,}"',                       # exact phrase in double quotes
    r"'[^']{2,}'",                       # exact phrase in single quotes
    r"[/\\][\w.\-]+[/\\]",               # path (/src/foo/)
    r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b",    # snake_case
    # CamelCase / camelCase / PascalCase: the token must switch from "lower to upper" internally;
    # covers "SoundCloud", "camelCase", "PascalCase" and the like. All-lowercase or all-uppercase words do not match.
    r"\b[A-Za-z][a-z0-9]+[A-Z][A-Za-z0-9]*\b",
    r"\bmem-[a-f0-9]{8,}\b",             # hy_memory node id
    r"\b[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}\b",  # full UUID
    r"\bv?\d+\.\d+(?:\.\d+)?\b",         # version number 1.2 / v0.3.45
    r"\b[a-f0-9]{16,}\b",                # long hex string
    r"https?://\S+",                     # URL
]

_NAV_REGEX = re.compile("|".join(_NAV_PATTERNS))


# ================================================================
# CONCEPTUAL trigger words
# ================================================================

_CONCEPTUAL_TRIGGERS = {
    # English
    "how", "why", "explain", "approach", "strategy", "tend", "overall",
    "architecture", "design", "philosophy", "pattern", "style",
    "in general", "generally",
    # Chinese
    "\u600e\u4e48", "\u4e3a\u4ec0\u4e48", "\u5982\u4f55", "\u503e\u5411", "\u98ce\u683c", "\u6574\u4f53",
    "\u4e00\u822c", "\u901a\u5e38", "\u603b\u4f53", "\u6a21\u5f0f",
}


# ================================================================
# Keyword extraction
# ================================================================

# Same stopwords as OMEGA, English only; Chinese relies on the weak tokenizer that turns each run of Han characters into one token
_STOPWORDS_EN = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "having",
    "my", "mine", "your", "yours", "our", "ours", "their", "theirs",
    "i", "you", "we", "they", "he", "she", "it", "me", "him", "her", "us", "them",
    "this", "that", "these", "those",
    "what", "when", "where", "who", "whom", "how", "why", "which",
    "and", "or", "but", "not", "no", "nor",
    "for", "to", "of", "in", "on", "at", "by", "from", "with", "about",
    "as", "if", "then", "than", "so", "too", "very",
    "can", "could", "would", "should", "may", "might", "will", "shall",
    "am",
}


# ================================================================
# API
# ================================================================

def is_navigational(query: str) -> bool:
    """Whether the query carries an explicit literal identifier signal."""
    return bool(_NAV_REGEX.search(query or ""))


def is_conceptual(query: str) -> bool:
    """Whether the query is a concept/tendency question."""
    if not query:
        return False
    q_lower = query.lower()
    for trig in _CONCEPTUAL_TRIGGERS:
        if trig in q_lower:
            return True
    return False


def classify_intent(query: str) -> str:
    """
    Returns one of the three intents: NAVIGATIONAL / FACTUAL / CONCEPTUAL.

    Priority: NAV > CONCEPTUAL > FACTUAL (the default fallback).
    If `HY_MEMORY_READER_INTENT_OVERRIDE` is configured, the override value is returned directly (for debugging).
    """
    if config.INTENT_OVERRIDE in ("NAVIGATIONAL", "FACTUAL", "CONCEPTUAL"):
        return config.INTENT_OVERRIDE
    if is_navigational(query):
        return "NAVIGATIONAL"
    if is_conceptual(query):
        return "CONCEPTUAL"
    return "FACTUAL"


# Mixed Chinese/English tokenisation:
#   - a run of latin letters/digits is one token
#   - each run of Han characters is one token too (coarse segmentation, no fine splitting)
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+")


def extract_keywords(query: str) -> List[str]:
    """
    Extracts the keyword sequence from the query (for batch embed → tag_index matching).

    Rules:
      - latin/digit tokens must be at least 3 characters and not in the stopword list
      - Han tokens keep the original run (embedded as a whole to get the vector)
      - deduplicated while preserving order, at most `KEYWORD_MAX_COUNT` entries
    """
    if not query:
        return []
    raw_tokens = _TOKEN_RE.findall(query)
    seen = set()
    result: List[str] = []
    for tok in raw_tokens:
        t = tok.lower() if tok.isascii() else tok
        # stopword and too-short filtering for latin/digit tokens
        if tok.isascii():
            if len(t) < 3 or t in _STOPWORDS_EN:
                continue
        else:
            # Han token: a single character carries too little meaning
            if len(t) < 2:
                continue
        if t in seen:
            continue
        seen.add(t)
        result.append(t)
        if len(result) >= config.KEYWORD_MAX_COUNT:
            break
    return result
