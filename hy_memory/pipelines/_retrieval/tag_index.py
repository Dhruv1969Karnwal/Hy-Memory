"""
Per-user tag embedding index -- write-side maintained tag vector index.

Core design:
  - Storage: the same VDB as memories (abstracted via the VectorStoreBase interface), separate collection
    (named `{memories_collection}_tag_index`)
  - Write: lazily maintained on the writer side. After each successful memory write, call
    `ensure_tag_embedding` for every tag of the node; skip if it exists (uuid5 guarantees idempotence), otherwise embed + upsert
  - Delete: after client.delete removes a single memory, scan the main collection for each tag to
    verify its reference count; delete from tag_index only when it reaches 0
  - Query: the shared post-processing of legacy / hybrid can read this index behind a flag to nominate tag candidates
  - Failure policy: silent degradation with try-except across the whole chain; tag_index failures never affect the main flow

Limitations:
  - Non-Qdrant backends (Chroma / Faiss) must first implement or raise NotImplementedError
    in the tag_index methods of VectorStoreBase.
"""

import asyncio
from typing import List, Optional
import logging

from . import config

logger = logging.getLogger(__name__)


async def ensure_tag_embedding(
    vector_store,
    embed_service,
    user_id: str,
    tag: str,
) -> bool:
    """
    Lazy upsert: if the embedding for this (user_id, tag) does not exist, compute and write it;
    return True directly if it already exists.

    Returning True means "the tag is usable in tag_index"; False means failure.
    Exceptions are always swallowed and logged as warnings, never blocking the main flow.
    """
    if not config.TAG_INDEX_WRITE_ENABLED:
        return False
    if not user_id or not tag:
        return False
    try:
        exists = await vector_store.has_tag_embedding(user_id, tag)
        if exists:
            return True
    except NotImplementedError:
        return False
    except Exception as e:
        logger.debug(f"[tag-index] has_tag_embedding failed ({user_id}, {tag}): {e}")
        # Try a direct upsert (a failed exists check does not mean the write cannot proceed)

    try:
        # Reuse the embed interface from the search path (search favors low latency, bypassing the write batching queue)
        vec = await embed_service.embed_queued(tag)
    except Exception as e:
        logger.warning(f"[tag-index] embed tag failed ({user_id}, {tag}): {e}")
        return False

    try:
        await vector_store.upsert_tag_embedding(user_id=user_id, tag=tag, embedding=vec)
        return True
    except NotImplementedError:
        return False
    except Exception as e:
        logger.warning(f"[tag-index] upsert tag failed ({user_id}, {tag}): {e}")
        return False


async def ensure_tag_embeddings_for_node(
    vector_store,
    embed_service,
    user_id: str,
    tags: List[str],
) -> int:
    """
    Ensure a batch of tags have their embeddings in place in tag_index. Returns the success count.
    Checks/writes each tag concurrently. Each tag still has its own try-except without affecting others; a node commonly
    has 2-3 tags, and serializing would add embedding-gateway tail latencies directly onto the write critical path.
    """
    if not tags:
        return 0
    unique_tags = list(dict.fromkeys(tag for tag in tags if tag))
    # Qdrant can check all deterministic tag ids and upsert all missing points in
    # two VDB requests.  The old generic path did one existence read plus one
    # upsert per tag; a PMV2 chunk with 30 distinct tags could therefore add 60
    # foreground VDB round trips after the memories themselves were stored.
    batch_existing = getattr(vector_store, "existing_tag_embeddings", None)
    batch_upsert = getattr(vector_store, "upsert_tag_embeddings_batch", None)
    if callable(batch_existing) and callable(batch_upsert):
        try:
            existing = set(await batch_existing(user_id, unique_tags) or [])
            missing = [tag for tag in unique_tags if tag not in existing]
            if not missing:
                return len(existing)

            async def _embed_tag(tag: str):
                try:
                    return tag, await embed_service.embed_queued(tag)
                except Exception as exc:
                    logger.warning(
                        "[tag-index] embed tag failed (%s, %s): %s",
                        user_id, tag, exc,
                    )
                    return tag, None

            embedded = await asyncio.gather(*[_embed_tag(tag) for tag in missing])
            items = {tag: vec for tag, vec in embedded if vec}
            if items:
                stored = await batch_upsert(user_id, items)
                return len(existing) + int(stored or 0)
            return len(existing)
        except NotImplementedError:
            pass
        except Exception as exc:
            logger.warning(
                "[tag-index] batch maintain failed, falling back to per-tag path: %s",
                exc,
            )

    # Each coroutine enters embed_queued; the tags of one write get merged into a single batch by the queue, and concurrent writes
    # can share batches too, unlike calling embed_batch directly where every write sends its own HTTP request.
    results = await asyncio.gather(
        *[
            ensure_tag_embedding(
                vector_store=vector_store,
                embed_service=embed_service,
                user_id=user_id,
                tag=tag,
            )
            for tag in unique_tags
        ],
        return_exceptions=True,
    )
    return sum(result is True for result in results)


async def cleanup_tags_on_delete(
    vector_store,
    user_id: str,
    tags: List[str],
    isolation_key: str = "",
) -> int:
    """
    After deleting a memory, check the reference count of each of its tags; delete from tag_index when unreferenced.
    Returns the number of tag entries actually deleted.
    """
    if not config.TAG_INDEX_WRITE_ENABLED or not user_id or not tags:
        return 0
    removed = 0
    for tag in tags:
        if not tag:
            continue
        try:
            remaining = await vector_store.count_memories_with_tag(
                user_id=user_id, tag=tag, isolation_key=isolation_key,
            )
        except NotImplementedError:
            return 0  # backend unsupported, give up entirely
        except Exception as e:
            logger.debug(f"[tag-index] count_memories_with_tag failed ({user_id}, {tag}): {e}")
            continue

        if remaining > 0:
            continue

        try:
            await vector_store.delete_tag_embedding(user_id=user_id, tag=tag)
            removed += 1
        except NotImplementedError:
            return removed
        except Exception as e:
            logger.debug(f"[tag-index] delete_tag_embedding failed ({user_id}, {tag}): {e}")
    return removed


async def search_matching_tags(
    vector_store,
    user_id: str,
    keyword_embeddings: List[List[float]],
    topk: Optional[int] = None,
    min_score: Optional[float] = None,
) -> List[str]:
    """
    For each keyword embedding, retrieve the topk nearest tags in the per-user tag_index,
    then merge and deduplicate before returning.

    The returned tag list is deduplicated but unordered (the reader only cares about the MatchAny set).
    Any failure silently returns an empty list (the reader treats path B as empty).
    """
    if not keyword_embeddings or not user_id:
        return []
    topk = topk if topk is not None else config.TAG_MATCH_TOPK
    min_score = min_score if min_score is not None else config.TAG_MATCH_MIN_SCORE

    seen = set()
    result: List[str] = []
    for vec in keyword_embeddings:
        try:
            hits = await vector_store.search_tag_embeddings(
                user_id=user_id,
                query_embedding=vec,
                topk=topk,
                min_score=min_score,
            )
        except NotImplementedError:
            return []
        except Exception as e:
            logger.debug(f"[tag-index] search_tag_embeddings failed: {e}")
            continue
        for hit in hits or []:
            tag = hit.get("tag") if isinstance(hit, dict) else None
            if tag and tag not in seen:
                seen.add(tag)
                result.append(tag)
    return result


def backend_supports_tag_index(vector_store) -> bool:
    """
    Probe whether the vector_store implements the tag_index interface family.
    For internal tools to check whether the backend supports the tag index.
    """
    required = [
        "upsert_tag_embedding",
        "search_tag_embeddings",
        "delete_tag_embedding",
        "has_tag_embedding",
        "count_memories_with_tag",
    ]
    for name in required:
        fn = getattr(vector_store, name, None)
        if fn is None:
            return False
    # Fast opt-in via the private `_supports_tag_index` boolean, so subclasses do not need a real probe call
    flag = getattr(vector_store, "_supports_tag_index", None)
    return bool(flag) if flag is not None else True
