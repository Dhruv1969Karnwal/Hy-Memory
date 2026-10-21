"""Topic-scoped summary: slice summaries by **topic**, not by time window.

Fixed-window summaries accumulate a buffer over time, so the output can only ever be
"what this conversation was about" -- of limited retrieval value, since the content of one time window usually spans several unrelated topics.

With the Topic graph we can slice it differently: **summarize once the memories under a Topic cross a threshold**,
producing "here is what I know about your work". That is the summary granularity worth retrieving.

State lives in the summary node's own `custom` field, so no extra table is needed:
    custom = {
        "summary_kind": "topic",      # Distinguishes it from other summary types
        "topic_id": ..., "topic_name": ...,
        "covered_count": N,           # How many memories under this Topic this run covered
        "source_memory_ids": [...],
    }
When deciding whether to summarize again, read back the `covered_count` of the newest summary of that Topic
and compare it with the current degree of the Topic.

Old summaries are set to SHADOW and do not enter the supersedes evolution chain.
"""

import asyncio
import json
import logging
import threading
import weakref
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SUMMARY_KIND_TOPIC = "topic"


_TOPIC_LOCKS_GUARD = threading.Lock()
_TOPIC_LOCKS_BY_LOOP = weakref.WeakKeyDictionary()


def _graph_store_scope(graph_store) -> tuple:
    """Build a process-local stable scope for graph-store instances."""
    if hasattr(graph_store, "_db_path"):
        return (type(graph_store).__qualname__, str(graph_store._db_path))
    if hasattr(graph_store, "_url"):
        return (
            type(graph_store).__qualname__,
            str(graph_store._url),
            str(getattr(graph_store, "_database", "")),
        )
    return (type(graph_store).__qualname__, id(graph_store))


def _topic_lock(graph_store, isolation_key: str, topic_id: str) -> asyncio.Lock:
    """Return an event-loop-local lock for one graph scope and Topic."""
    loop = asyncio.get_running_loop()
    key = (_graph_store_scope(graph_store), isolation_key, topic_id)
    with _TOPIC_LOCKS_GUARD:
        locks = _TOPIC_LOCKS_BY_LOOP.get(loop)
        if locks is None:
            locks = weakref.WeakValueDictionary()
            _TOPIC_LOCKS_BY_LOOP[loop] = locks
        lock = locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            locks[key] = lock
        return lock


TOPIC_SUMMARY_PROMPT = """You are summarizing everything known about ONE topic.

## Topic

{topic_name}

## Memories under this topic (oldest → newest)

{memories}

---

Write a single dense summary of what is known about the user with respect to this
topic.

Rules:

* 2–4 sentences, third person, starting with "The user ..." (or the named person
  if the memories are about someone else).
* Cover the topic as a whole — state the current situation, and mention how it
  developed only when the change matters.
* Preserve concrete details: names, places, numbers, dates.
* Do NOT invent anything that is not in the memories.
* Do NOT enumerate the memories one by one; synthesize them.
* Output language must match the language of the memories.

Return ONLY the summary text, no preamble, no JSON, no quotes."""


TOPIC_SUMMARY_PROMPT_ZH = """You are summarizing everything known about **one topic**.

## Topic

{topic_name}

## Memories under this topic (oldest first)

{memories}

---

Write one high-density summary of the user on this topic.

Requirements:

* 2-4 sentences, third person, starting with "The user..." (if the memories are about someone else, start with that person's name).
* Cover the whole topic -- state the current situation; mention how it evolved only when the change itself matters.
* Keep concrete details: names of people, places, numbers, dates.
* Never invent anything that is not in the memories.
* Do not enumerate memories one by one; synthesize them into a single paragraph.
* Write the output in the same language as the memories.

Return only the summary body, no preamble, no JSON, no quotes."""


async def find_latest_topic_summary(
    *, vector_store, user_id: str, agent_id: Optional[str], topic_id: str,
) -> Optional[Any]:
    """Find the newest topic summary node for this Topic (None if there is none).

    The VDB filter cannot express fields inside `custom`, so we pull this user's L3 and filter on the Python side.
    The L3 volume is small (at most one active summary per Topic), so this cost is acceptable.
    """
    from ...models.memory import MemoryLayer, MemoryStatus

    try:
        nodes = await vector_store.list_by_user(
            user_id=user_id,
            agent_id=agent_id,
            layers=[MemoryLayer.L3_SUMMARY],
            status_filter=[MemoryStatus.ACTIVE],
            limit=500,
            with_vectors=False,
        )
    except Exception as e:
        logger.debug(f"[topic-summary] list L3 failed: {e}")
        return None

    matched = []
    for n in (nodes or []):
        custom = getattr(n, "custom", None) or {}
        if isinstance(custom, str):
            try:
                custom = json.loads(custom)
            except Exception:
                continue
        if custom.get("summary_kind") != SUMMARY_KIND_TOPIC:
            continue
        if custom.get("topic_id") != topic_id:
            continue
        matched.append(n)

    if not matched:
        return None
    from datetime import datetime as _dt
    matched.sort(key=lambda n: n.gmt_created or _dt.min)
    return matched[-1]


def covered_count_of(node: Any) -> int:
    custom = getattr(node, "custom", None) or {}
    if isinstance(custom, str):
        try:
            custom = json.loads(custom)
        except Exception:
            return 0
    try:
        return int(custom.get("covered_count") or 0)
    except (TypeError, ValueError):
        return 0


async def maybe_summarize_topics(
    *,
    graph_store,
    vector_store,
    embed_service,
    llm_provider,
    user_id: str,
    agent_id: Optional[str],
    session_id: str,
    isolation_key: str,
    touched_topic_ids: List[str],
    min_memories: int = 5,
    step: int = 5,
    max_memories: int = 60,
    max_concurrency: int = 3,
    temperature: Optional[float] = None,
    trace=None,
) -> Dict[str, Any]:
    """For the Topics touched by this write, generate / refresh summaries with bounded concurrency as needed.

    Trigger: the number of active memories under the Topic is ≥ `min_memories`, and it grew by ≥ `step` items since the last summary.
    This way we neither recompute on every single write (expensive) nor keep serving stale summaries.

    Returns:
        {"checked", "summarized", "skipped", "errors", "topics": [...]}
    """
    stats: Dict[str, Any] = {
        "checked": 0, "summarized": 0, "skipped": 0, "errors": 0, "topics": [],
    }
    if graph_store is None or llm_provider is None or not touched_topic_ids:
        return stats

    topic_names = {}
    try:
        from ...models.memory import MemoryNode

        # Topics are built on user::agent (without session); queries must use the same key
        for t in await graph_store.get_all_topics(
            MemoryNode.topic_scope_of(isolation_key)
        ):
            topic_names[t.get("topic_id")] = t.get("name") or ""
    except Exception as e:
        logger.debug(f"[topic-summary] get_all_topics failed: {e}")

    topic_ids = [topic_id for topic_id in dict.fromkeys(touched_topic_ids) if topic_id]
    stats["checked"] = len(topic_ids)
    semaphore = asyncio.Semaphore(max(1, min(16, int(max_concurrency or 1))))

    async def summarize_one(topic_id: str):
        # Lock before the semaphore: duplicate work on one Topic waits without
        # consuming capacity intended for independent Topics.
        async with _topic_lock(graph_store, isolation_key, topic_id):
            async with semaphore:
                try:
                    # Re-read after acquiring the Topic lock. Another graph task
                    # may have refreshed this Topic while this task was waiting.
                    mems = await graph_store.get_topic_memories(
                        topic_id, limit=max_memories
                    )
                    if len(mems) < min_memories:
                        return "skipped", None

                    prev = await find_latest_topic_summary(
                        vector_store=vector_store, user_id=user_id,
                        agent_id=agent_id, topic_id=topic_id,
                    )
                    prev_covered = covered_count_of(prev) if prev is not None else 0
                    if len(mems) - prev_covered < step:
                        return "skipped", None

                    from datetime import datetime as _dt

                    _t0 = _dt.now()
                    summary, _prompt, _tokens = await _generate(
                        llm_provider=llm_provider,
                        topic_name=topic_names.get(topic_id) or topic_id,
                        memories=[m.get("content") or "" for m in mems],
                        temperature=temperature,
                    )
                    if trace is not None:
                        await trace(
                            prompt=_prompt,
                            response=summary,
                            parsed={
                                "topic_id": topic_id,
                                "topic_name": topic_names.get(topic_id) or topic_id,
                                "covered_count": len(mems),
                                "skipped": not summary,
                            },
                            memory_ids=[
                                m.get("node_id") for m in mems if m.get("node_id")
                            ],
                            elapsed_ms=(_dt.now() - _t0).total_seconds() * 1000,
                            tokens=_tokens,
                        )
                    if not summary:
                        return "error", None

                    await _store(
                        vector_store=vector_store,
                        embed_service=embed_service,
                        summary=summary,
                        topic_id=topic_id,
                        topic_name=topic_names.get(topic_id) or "",
                        source_memory_ids=[m["node_id"] for m in mems],
                        covered_count=len(mems),
                        user_id=user_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        prev_node=prev,
                    )
                    return "summarized", {
                        "topic_id": topic_id,
                        "topic_name": topic_names.get(topic_id) or "",
                        "covered": len(mems),
                    }
                except Exception as e:
                    logger.debug(f"[topic-summary] topic {topic_id} failed: {e}")
                    return "error", None

    results = await asyncio.gather(*(summarize_one(topic_id) for topic_id in topic_ids))
    for status, topic in results:
        if status == "summarized":
            stats["summarized"] += 1
            stats["topics"].append(topic)
        elif status == "skipped":
            stats["skipped"] += 1
        else:
            stats["errors"] += 1

    if stats["summarized"]:
        logger.info(
            f"[topic-summary] {stats['summarized']} topic summaries written "
            f"(checked={stats['checked']} skipped={stats['skipped']})"
        )
    return stats


async def _generate(
    *, llm_provider, topic_name: str, memories: List[str],
    temperature: Optional[float] = None,
) -> tuple:
    """Return (summary text, actual prompt, token usage) -- the last two feed instrumentation, so production runs can be reviewed."""
    from ...utils.lang_detect import is_chinese

    body = "\n".join(f"- {c}" for c in memories if c)
    if not body:
        return "", "", {}
    tpl = TOPIC_SUMMARY_PROMPT_ZH if is_chinese(body) else TOPIC_SUMMARY_PROMPT
    prompt = tpl.format(topic_name=topic_name, memories=body)
    resp = await llm_provider.complete(
        prompt=prompt,
        max_tokens=400,
        temperature=temperature,
    )
    tokens = {
        "prompt": getattr(resp, "prompt_tokens", 0) or 0,
        "completion": getattr(resp, "completion_tokens", 0) or 0,
        "total": getattr(resp, "total_tokens", 0) or 0,
    }
    text = (getattr(resp, "content", "") or "").strip()
    # The LLM occasionally wraps the output in quotes or markdown fences
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        text = text.strip()
    return text.strip().strip('"').strip("'").strip(), prompt, tokens


async def _store(
    *,
    vector_store,
    embed_service,
    summary: str,
    topic_id: str,
    topic_name: str,
    source_memory_ids: List[str],
    covered_count: int,
    user_id: str,
    agent_id: Optional[str],
    session_id: str,
    prev_node: Optional[Any],
) -> str:
    from ...models.memory import (
        MemoryLayer, MemoryNode, MemoryStatus, SourceType,
    )
    from datetime import datetime

    embedding = await embed_service.embed_queued(summary)
    from ...utils.source_filter import merge_sources
    source_nodes = await vector_store.get_by_ids(source_memory_ids)
    node = MemoryNode(
        user_id=user_id,
        agent_id=agent_id or "default_agent",
        session_id=session_id or "default_session",
        source=merge_sources(
            getattr(source_node, "source", None) for source_node in source_nodes
        ),
        layer=MemoryLayer.L3_SUMMARY,
        content=summary,
        embedding=embedding,
        memory_at=datetime.now(),
        source_type=SourceType.INFERRED,
        status=MemoryStatus.ACTIVE,
        is_latest=True,
        tags=[topic_name] if topic_name else [],
        custom={
            "summary_kind": SUMMARY_KIND_TOPIC,
            "topic_id": topic_id,
            "topic_name": topic_name,
            "covered_count": covered_count,
            "source_memory_ids": list(source_memory_ids),
            # Audit only: topic summaries have no chain by design and do not enter the supersedes chain
            "supersedes_summary_id": getattr(prev_node, "node_id", "") or "",
        },
    )
    nid = await vector_store.upsert(node)

    # Old summaries are set to SHADOW so several summaries of one Topic are not recalled repeatedly
    if prev_node is not None and getattr(prev_node, "node_id", ""):
        try:
            await vector_store.update_payload(
                prev_node.node_id,
                {"status": MemoryStatus.SHADOW.value, "is_latest": False},
            )
        except Exception as e:
            logger.debug(f"[topic-summary] shadow prev summary failed: {e}")
    return nid
