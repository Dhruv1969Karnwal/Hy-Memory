"""
Entity store write helper (shared by writer persistence and client batch migration).

Responsibility: given a memory's (memory_id, content), extract entities → embed → upsert
into the vector_store entity store (separate {collection}_entities collection).

All best-effort: any exception is swallowed and logged at debug, never affecting the main write/migration flow.
Silently skips when spaCy is unavailable or no entity is extracted (consistent with mem0 behavior).
"""

from typing import Any
import logging

logger = logging.getLogger(__name__)

# Max entities flushed per memory (aligned with the mem0 query-side limit)
MAX_ENTITIES_PER_MEMORY = 8


async def index_memory_entities(
    *,
    vector_store: Any,
    embed_service: Any,
    memory_id: str,
    content: str,
    user_id: str,
    agent_id: str = "",
    merge_threshold: float = 0.95,
) -> int:
    """Extract entities from one memory and flush them into the entity store. Returns the number of entities written/merged.

    best-effort: returns 0 on failure and never raises.
    """
    if not content or not memory_id or not user_id:
        return 0
    try:
        from .entities import extract_entities
    except Exception as e:
        logger.debug(f"[entity-index] entities module unavailable: {e}")
        return 0

    try:
        entities = extract_entities(content)
    except Exception as e:
        logger.debug(f"[entity-index] extract failed: {e}")
        return 0

    if not entities:
        return 0
    entities = entities[:MAX_ENTITIES_PER_MEMORY]

    texts = [t for (_etype, t) in entities if t and t.strip()]
    if not texts:
        return 0

    try:
        embeddings = await embed_service.embed_batch(texts)
    except Exception as e:
        logger.debug(f"[entity-index] embed_batch failed: {e}")
        return 0
    if not embeddings or len(embeddings) != len(texts):
        return 0

    done = 0
    for (etype, etext), emb in zip(entities, embeddings):
        try:
            await vector_store.upsert_entity(
                entity_text=etext,
                entity_type=etype,
                embedding=emb,
                memory_id=memory_id,
                user_id=user_id,
                agent_id=agent_id or "",
                merge_threshold=merge_threshold,
            )
            done += 1
        except NotImplementedError:
            # backend has no entity store support (non-chroma) → skip entirely
            logger.debug("[entity-index] backend has no entity store; skipping")
            return done
        except Exception as e:
            logger.debug(f"[entity-index] upsert_entity '{etext}' failed: {e}")
    return done
