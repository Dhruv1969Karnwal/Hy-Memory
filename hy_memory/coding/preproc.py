# -*- coding: utf-8 -*-
"""
Coding memory preprocessing utilities (rule based, zero LLM)

- has_any_tool_message  : detect whether messages contain tool messages (O(1) gate for the coding chain)
- strip_tool_messages   : strip tool messages plus the assistant tool_calls field, leaving them for the chat chain
- truncate_tool_message : keep head/tail of a long tool_result and cut the middle (no schema-aware compression)
- extract_files         : pull path / file_path and similar keys out of tool_calls.arguments
- extract_tool_summary  : simplified view handed to the LLM judge

See docs/scene_coding_usage.md for usage and scene routing.
"""

from typing import Any, Dict, List, Optional, Tuple

from ..pipelines.base import ChatMessage


# ================================================================
# Path key names (shared across OpenAI / Anthropic / Claude Code)
# ================================================================

PATH_KEYS: Tuple[str, ...] = (
    "path", "file_path", "filename", "file", "filepath",
    "target_path", "notebook_path", "src_path", "dst_path",
)


# ================================================================
# Detection / filtering
# ================================================================

def has_any_tool_message(messages: List[ChatMessage]) -> bool:
    """Whether messages contain a tool message or assistant tool_calls."""
    return any(m.is_tool_message() or m.has_tool_calls() for m in messages)


def strip_tool_messages(messages: List[ChatMessage]) -> List[ChatMessage]:
    """
    For the existing chat chain: drops role=tool messages;
    an assistant message with tool_calls keeps its content (the tool_calls field is dropped);
    a message whose content is empty after stripping and that has no tool_calls is discarded.
    """
    out: List[ChatMessage] = []
    for m in messages:
        if m.is_tool_message():
            continue
        if m.has_tool_calls():
            # Keep the text, drop tool_calls; if content is empty the whole message is discarded
            if m.content.strip():
                out.append(ChatMessage(role=m.role, content=m.content))
            # else: discard the whole message
            continue
        out.append(m)
    return out


# ================================================================
# tool_result truncation (no schema-aware compression)
# ================================================================

DEFAULT_MAX_TOOL_RESULT_BYTES = 2048
DEFAULT_HEAD_BYTES = 1024
DEFAULT_TAIL_BYTES = 512


def truncate_tool_result_text(
    text: str,
    max_bytes: int = DEFAULT_MAX_TOOL_RESULT_BYTES,
    head_bytes: int = DEFAULT_HEAD_BYTES,
    tail_bytes: int = DEFAULT_TAIL_BYTES,
) -> str:
    """
    Keeps the head and tail of a long string and cuts out the middle.

    Tool output schemas are not consistent across platforms, so rule-based compression ("line count for Read / exit code for Bash") is hopeless;
    the safest option is plain byte truncation, leaving the signal for the LLM to digest.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    # A custom max_bytes can be smaller than the default head+tail (1024+512). Scale the content window
    # down proportionally and reserve room for the omission marker; never return the oversized source text as is.
    marker_budget = min(96, max(24, max_bytes // 4))
    content_budget = max(2, max_bytes - marker_budget)
    requested = max(1, head_bytes + tail_bytes)
    if head_bytes + tail_bytes > content_budget:
        head_bytes = max(1, int(content_budget * head_bytes / requested))
        tail_bytes = max(1, content_budget - head_bytes)
    head = raw[:head_bytes].decode("utf-8", errors="ignore")
    tail = raw[-tail_bytes:].decode("utf-8", errors="ignore")
    omitted = len(raw) - head_bytes - tail_bytes
    return (
        head
        + f"\n\n[...{omitted} bytes omitted by truncate...]\n\n"
        + tail
    )


def truncate_tool_message(
    msg: ChatMessage,
    max_bytes: int = DEFAULT_MAX_TOOL_RESULT_BYTES,
) -> ChatMessage:
    """Truncate the content of role=tool messages. Other roles are returned unchanged."""
    if not msg.is_tool_message():
        return msg
    if len(msg.content.encode("utf-8")) <= max_bytes:
        return msg
    new_content = truncate_tool_result_text(msg.content, max_bytes=max_bytes)
    # Do not mutate the original object (the caller may reuse messages)
    return ChatMessage(
        role=msg.role,
        content=new_content,
        name=msg.name,
        tool_calls=list(msg.tool_calls),
        tool_call_id=msg.tool_call_id,
        tool_name=msg.tool_name,
    )


def truncate_messages(
    messages: List[ChatMessage],
    max_bytes: int = DEFAULT_MAX_TOOL_RESULT_BYTES,
) -> List[ChatMessage]:
    """Truncate every tool message in the messages list."""
    return [truncate_tool_message(m, max_bytes=max_bytes) for m in messages]


# ================================================================
# Long-session window compression: last-K user turns + assistant head/tail truncation
# ================================================================

DEFAULT_LAST_K_USER_TURNS = 5
# The original values were head 200 + tail 200, which assumed assistant text is prose where
# recovering the intention is enough. In practice the last assistant block often **is the deliverable itself**
# (an itinerary, a config block, an answer), and truncating it tears out the middle of the answer sheet.
#
# The threshold is 6000 (3000 each for head and tail): most normal deliverables fit and stay intact;
# anything above it is usually a long trial-and-error trajectory that **still must be cut** -- keeping it only dilutes attention.
DEFAULT_ASSISTANT_HEAD_CHARS = 3000
DEFAULT_ASSISTANT_TAIL_CHARS = 3000


def truncate_assistant_text(
    text: str,
    head_chars: int = DEFAULT_ASSISTANT_HEAD_CHARS,
    tail_chars: int = DEFAULT_ASSISTANT_TAIL_CHARS,
) -> str:
    """
    Character-level head/tail truncation of assistant.content.

    Unlike tool_result, assistant text is rarely huge, but a long trajectory still accumulates
    and bloats the prompt. Default head 3000 + tail 3000: the closing part of an assistant message
    is often the deliverable itself; see the notes above DEFAULT_ASSISTANT_HEAD_CHARS in this module.

    Note: the tool_calls list is not truncated here; the caller decides how to render it in _format_conversation.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    if len(text) <= head_chars + tail_chars:
        return text
    omitted = len(text) - head_chars - tail_chars
    return (
        text[:head_chars]
        + f" [...{omitted} chars omitted...] "
        + text[-tail_chars:]
    )


def truncate_assistant_messages(
    messages: List[ChatMessage],
    head_chars: int = DEFAULT_ASSISTANT_HEAD_CHARS,
    tail_chars: int = DEFAULT_ASSISTANT_TAIL_CHARS,
) -> List[ChatMessage]:
    """Truncate all assistant text in the list without dropping any message or user turn.

    ``New Messages`` must cover every turn passed to this write in full; last-K only belongs to
    historical context. This helper therefore only bounds the size of a single assistant message and does no window selection.
    """
    out: List[ChatMessage] = []
    for m in messages:
        if m.role != "assistant" or not m.content:
            out.append(m)
            continue
        new_content = truncate_assistant_text(
            m.content,
            head_chars=head_chars,
            tail_chars=tail_chars,
        )
        if new_content == m.content:
            out.append(m)
            continue
        out.append(ChatMessage(
            role=m.role,
            content=new_content,
            name=m.name,
            tool_calls=list(m.tool_calls),
            tool_call_id=m.tool_call_id,
            tool_name=m.tool_name,
        ))
    return out


def prepare_extractor_messages(
    messages: List[ChatMessage],
    *,
    tool_max_bytes: int = DEFAULT_MAX_TOOL_RESULT_BYTES,
    assistant_head_chars: int = DEFAULT_ASSISTANT_HEAD_CHARS,
    assistant_tail_chars: int = DEFAULT_ASSISTANT_TAIL_CHARS,
) -> List[ChatMessage]:
    """Bound the size of individual messages while keeping every input turn intact.

    Used for ``New Messages``. The order must be tool results first, then assistant text;
    neither step changes the message count or their relative order.
    """
    return truncate_assistant_messages(
        truncate_messages(messages, max_bytes=tool_max_bytes),
        head_chars=assistant_head_chars,
        tail_chars=assistant_tail_chars,
    )


def keep_last_k_user_turns(
    messages: List[ChatMessage],
    k: int = DEFAULT_LAST_K_USER_TURNS,
    assistant_head_chars: int = DEFAULT_ASSISTANT_HEAD_CHARS,
    assistant_tail_chars: int = DEFAULT_ASSISTANT_TAIL_CHARS,
) -> List[ChatMessage]:
    """
    Compresses a long trajectory into a window of the last K user turns and applies
    head/tail truncation to the assistant messages inside that window.

    A "user-turn" is one real user message (not tool_result-only) plus every following
    assistant / tool / tool_result message up to the next real user message.

    Motivation:
    - the extractor renders conversation_block in full, so a long batch dilutes LLM attention and causes
      under-extract
    - tool_result already has byte truncation (head 1024 + tail 512), but neither the long assistant text nor
      the message count itself was bounded
    - this function only compresses the **message count** and the assistant text length; tool_result is still
      handled by truncate_messages and stays unchanged

    Args:
        messages: the complete ChatMessage list (ordered)
        k: how many recent user turns to keep (default 5)
        assistant_head_chars / assistant_tail_chars: characters kept at the head and tail of assistant text

    Returns:
        The truncated messages list (order preserved, metadata such as tool_call_id unchanged).
        Returned as is when there are no messages or the user-turn count is <= k (assistant truncation still runs so behaviour stays consistent).
    """
    if not messages:
        return messages

    # 1) Find the start of every real user turn (real = role=user and not tool_result-only)
    user_turn_starts: List[int] = []
    for idx, m in enumerate(messages):
        if m.role == "user" and not m.is_tool_message():
            user_turn_starts.append(idx)

    # 2) Decide the window start: keeping the last k user turns means the k-th start from the end
    if len(user_turn_starts) > k:
        window_start = user_turn_starts[-k]
    else:
        window_start = 0

    windowed = messages[window_start:]

    # 3) Apply head/tail truncation to assistant messages (tool_calls preserved)
    out: List[ChatMessage] = []
    for m in windowed:
        if m.role == "assistant" and m.content:
            new_content = truncate_assistant_text(
                m.content,
                head_chars=assistant_head_chars,
                tail_chars=assistant_tail_chars,
            )
            if new_content == m.content:
                out.append(m)
            else:
                out.append(ChatMessage(
                    role=m.role,
                    content=new_content,
                    name=m.name,
                    tool_calls=list(m.tool_calls),
                    tool_call_id=m.tool_call_id,
                    tool_name=m.tool_name,
                ))
        else:
            out.append(m)
    return out


# ================================================================
# File path extraction
# ================================================================

def extract_files(messages: List[ChatMessage]) -> List[str]:
    """
    Pulls path / file_path / filename and similar keys from every tool_calls.arguments in messages.

    Returns the deduplicated, sorted list of paths. Reliable across OpenAI / Anthropic / Claude Code,
    one of the few tool signals that can be handled with rules.
    """
    paths: set = set()
    for m in messages:
        for tc in m.tool_calls:
            if not isinstance(tc.arguments, dict):
                continue
            for k in PATH_KEYS:
                v = tc.arguments.get(k)
                if isinstance(v, str) and v:
                    paths.add(v)
    return sorted(paths)


# ================================================================
# Simplified view of the LLM input on the write / search side
# ================================================================

def extract_tool_summary(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """
    Turn-level simplified view for the LLM judge / extractor.

    Each turn = one user message plus the tool_names of all assistant/tool messages up to the next user message.
    The chat path never calls this function (no tool message means the legacy chain).

    Returns:
        [
            {"turn": 0, "user": "...", "tools": ["Read", "Edit"], "has_tool_result": True},
            ...
        ]
    """
    out: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None

    def _flush():
        if cur is not None:
            # Deduplicate while preserving order
            seen = set()
            cur["tools"] = [t for t in cur["tools"] if not (t in seen or seen.add(t))]
            out.append(cur)

    for m in messages:
        if m.role == "user" and not m.is_tool_message():
            _flush()
            cur = {
                "turn": len(out),
                "user": (m.content or "").strip(),
                "tools": [],
                "has_tool_result": False,
            }
        else:
            if cur is None:
                # No leading user message (possibly a history assistant message): open an anonymous turn
                cur = {"turn": len(out), "user": "", "tools": [], "has_tool_result": False}
            if m.has_tool_calls():
                for tc in m.tool_calls:
                    if tc.name:
                        cur["tools"].append(tc.name)
            if m.is_tool_message():
                cur["has_tool_result"] = True
                if m.tool_name:
                    cur["tools"].append(m.tool_name)

    _flush()
    return out
