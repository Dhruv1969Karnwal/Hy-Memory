"""
Input token counting via tiktoken.

Only meant to estimate the token count of user input (the raw content / messages); it does not aim to line up exactly
with any specific LLM tokenizer -- cl100k_base is far closer than len() for both Chinese and English, which is enough for usage
statistics and approximate billing. The encoder is cached process-wide (loading the BPE vocabulary takes ~3s the first time, later encode calls are sub-millisecond).
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_ENCODING = "cl100k_base"
_encoder = None  # Lazy loading plus in-process cache


def _get_encoder():
    """Return the cached tiktoken encoder; returns None when tiktoken is unavailable."""
    global _encoder
    if _encoder is not None:
        return _encoder
    try:
        import tiktoken
        _encoder = tiktoken.get_encoding(_DEFAULT_ENCODING)
    except Exception as e:
        logger.debug(f"[token_count] tiktoken unavailable, fallback to char heuristic: {e}")
        _encoder = False  # Mark it unavailable so we do not retry over and over
    return _encoder


def count_tokens(text: str) -> int:
    """Count the tokens of a text; falls back to the len(text)//4 rough estimate when tiktoken is unavailable."""
    if not text:
        return 0
    enc = _get_encoder()
    if enc:
        try:
            return len(enc.encode(text))
        except Exception as e:
            logger.debug(f"[token_count] encode failed, char fallback: {e}")
    return max(1, len(text) // 4)


def count_messages_tokens(messages: List["object"]) -> int:
    """
    Count the input tokens of a messages list.

    Counts only the plain input text (role + content joined), without the fixed chat template overhead.
    Elements may be objects with .role/.content attributes, or {"role","content"} dicts.
    """
    if not messages:
        return 0
    parts: List[str] = []
    for m in messages:
        role = getattr(m, "role", None) if not isinstance(m, dict) else m.get("role")
        content = getattr(m, "content", None) if not isinstance(m, dict) else m.get("content")
        if content:
            parts.append(f"{role or ''}: {content}")
    return count_tokens("\n".join(parts))


def count_input_tokens(content: Optional[str], messages: Optional[List["object"]]) -> int:
    """Single entry point for the add interface: counts by messages when present, otherwise by content."""
    if messages:
        return count_messages_tokens(messages)
    return count_tokens(content or "")
