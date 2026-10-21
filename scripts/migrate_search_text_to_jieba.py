#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migration: move payload.search_text of Qdrant collections from raw Chinese text (relying on
server-side MULTILINGUAL tokenization) to jieba pre-tokenized form (paired with the WHITESPACE
tokenizer, the server does no tokenization at all).

Background:
  Previously the in-process BM25 query side of reconcile/reader_hybrid used jieba, while the candidate
  side used whole Chinese runs (_TOKEN_RE matches a CJK-only character class), so Chinese BM25 scored zero; on the
  hybrid path the jieba-prefetched query was re-tokenized server-side by MULTILINGUAL, so the two sides
  disagreed on token boundaries. The new design uses jieba on both sides:
    1. _node_to_payload runs search_text through lemmatize_for_bm25 on write
    2. the Qdrant text index tokenizer switches to WHITESPACE
    3. existing data needs payload.search_text rewritten, which is exactly what this script does

Flow:
  For each collection:
    1) if search_text already has a text index and tokenizer != whitespace, drop it first
       (otherwise every update during set_payload rebuilds the old index, which is wasted work)
    2) scroll all points (with_payload=True, with_vectors=False; skipping vectors saves bandwidth)
    3) for each point: take content + tags from the payload, recompute search_text with lemmatize_for_bm25,
       and use set_payload to update that one field only, leaving every other payload field untouched
    4) once all updates are done, create_payload_index(WHITESPACE); Qdrant rebuilds the inverted index from the new payload

Idempotency:
  - collections already on the whitespace tokenizer are skipped (unless --force-reindex)
  - whether a single point's search_text was already jieba-tokenized cannot be detected directly (the
    source text may simply contain spaces), so the script always recomputes and overwrites; the result is idempotent

Usage:
    # dry run (only counts how many rows would change, writes nothing)
    python scripts/migrate_search_text_to_jieba.py --dry-run

    # real run over every collection
    python scripts/migrate_search_text_to_jieba.py

    # migrate only the given collection
    python scripts/migrate_search_text_to_jieba.py --collection hy_memory_test

    # tune the batch size
    python scripts/migrate_search_text_to_jieba.py --batch-size 200

    # Qdrant connection (falls back to env: MEMORY_VECTOR_HOST/PORT/PATH)
    python scripts/migrate_search_text_to_jieba.py --qdrant-host localhost --qdrant-port 6333

Notes:
  - Only the Qdrant backend is affected. On the Chroma/FAISS backends search_text is a dead field and
    needs no migration.
  - Upgrade the SDK to a version carrying the new _node_to_payload (whitespace tokenizer) before running,
    otherwise data written during the migration would still use the old logic.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import List, Optional

# Keep the script runnable standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hy_memory.pipelines._retrieval.lemmatize import lemmatize_for_bm25  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("migrate")


def _build_qdrant_client(args):
    from qdrant_client import QdrantClient

    if args.qdrant_path:
        logger.info(f"using local Qdrant at path={args.qdrant_path}")
        return QdrantClient(path=args.qdrant_path)

    host = args.qdrant_host or os.getenv("MEMORY_VECTOR_HOST") or "localhost"
    port = int(args.qdrant_port or os.getenv("MEMORY_VECTOR_PORT") or 6333)
    api_key = args.qdrant_api_key or os.getenv("MEMORY_VECTOR_API_KEY")
    https = bool(args.https or os.getenv("MEMORY_VECTOR_HTTPS"))

    logger.info(f"using remote Qdrant {host}:{port} https={https}")
    kwargs = {"host": host, "port": port, "https": https}
    if api_key:
        kwargs["api_key"] = api_key
    return QdrantClient(**kwargs)


def _list_collections(client) -> List[str]:
    cols = client.get_collections().collections
    return [c.name for c in cols]


def _detect_search_text_tokenizer(client, collection: str) -> Optional[str]:
    """Return the tokenizer of the current search_text field (lowercase string), or None when there is no index"""
    try:
        info = client.get_collection(collection)
    except Exception as e:
        logger.warning(f"[{collection}] get_collection failed: {e}")
        return None

    payload_schema = getattr(info, "payload_schema", None) or {}
    if not isinstance(payload_schema, dict):
        return None
    schema_entry = payload_schema.get("search_text")
    if schema_entry is None:
        return None
    params = getattr(schema_entry, "params", None)
    tok = getattr(params, "tokenizer", None) if params else None
    if tok is None:
        return None
    return str(getattr(tok, "value", tok)).lower()


def _drop_search_text_index(client, collection: str) -> bool:
    try:
        client.delete_payload_index(
            collection_name=collection,
            field_name="search_text",
        )
        logger.info(f"[{collection}] dropped existing search_text index")
        return True
    except Exception as e:
        logger.warning(f"[{collection}] drop_payload_index failed (maybe none): {e}")
        return False


def _create_whitespace_index(client, collection: str) -> None:
    from qdrant_client.models import TextIndexParams, TokenizerType

    client.create_payload_index(
        collection_name=collection,
        field_name="search_text",
        field_schema=TextIndexParams(
            type="text",
            tokenizer=TokenizerType.WHITESPACE,
            min_token_len=2,
            max_token_len=20,
        ),
    )
    logger.info(f"[{collection}] created search_text whitespace index")


def _compute_new_search_text(payload: dict) -> str:
    """Kept in sync with the logic in _node_to_payload"""
    content = payload.get("content") or ""
    tags = payload.get("tags") or []
    if isinstance(tags, str):
        # tags may also be stored as a string in the payload
        tags = [t for t in tags.split(",") if t]
    tags_text = " ".join(tags) if tags else ""
    raw = f"{content} {tags_text}".strip() if tags_text else content
    return lemmatize_for_bm25(raw) if raw else ""


def _migrate_collection(
    client,
    collection: str,
    *,
    dry_run: bool,
    batch_size: int,
    force_reindex: bool,
) -> dict:
    stats = {
        "collection": collection,
        "scanned": 0,
        "updated": 0,
        "skipped_unchanged": 0,
        "errors": 0,
        "tokenizer_before": None,
        "tokenizer_after": None,
        "elapsed_s": 0.0,
    }

    t0 = time.time()
    tok_before = _detect_search_text_tokenizer(client, collection)
    stats["tokenizer_before"] = tok_before
    logger.info(f"[{collection}] current tokenizer={tok_before!r}")

    if (
        tok_before is not None
        and tok_before.endswith("whitespace")
        and not force_reindex
    ):
        logger.info(f"[{collection}] already on whitespace tokenizer, skipping (use --force-reindex to override)")
        stats["tokenizer_after"] = tok_before
        stats["elapsed_s"] = round(time.time() - t0, 2)
        return stats

    # Step 1: drop old index (only if not dry-run)
    if not dry_run and tok_before is not None:
        _drop_search_text_index(client, collection)

    # Step 2: scroll + set_payload
    offset = None
    while True:
        scroll_result = client.scroll(
            collection_name=collection,
            scroll_filter=None,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if isinstance(scroll_result, tuple):
            points, next_offset = scroll_result
        else:
            points = scroll_result.points
            next_offset = scroll_result.next_page_offset

        if not points:
            break

        for p in points:
            stats["scanned"] += 1
            try:
                payload = p.payload or {}
                old = payload.get("search_text") or ""
                new = _compute_new_search_text(payload)
                if new == old:
                    stats["skipped_unchanged"] += 1
                    continue
                if dry_run:
                    stats["updated"] += 1
                    if stats["updated"] <= 3:
                        # Log a sample of the first few rows
                        logger.info(
                            f"[{collection}] sample update: id={p.id} "
                            f"old={old[:80]!r}... new={new[:80]!r}..."
                        )
                else:
                    client.set_payload(
                        collection_name=collection,
                        payload={"search_text": new},
                        points=[p.id],
                        wait=False,
                    )
                    stats["updated"] += 1
            except Exception as e:
                stats["errors"] += 1
                logger.warning(f"[{collection}] update failed for id={p.id}: {e}")

        if stats["scanned"] % 1000 == 0 or next_offset is None:
            logger.info(
                f"[{collection}] progress: scanned={stats['scanned']} "
                f"updated={stats['updated']} skipped={stats['skipped_unchanged']} "
                f"errors={stats['errors']}"
            )

        if next_offset is None:
            break
        offset = next_offset

    # Step 3: create new whitespace index
    if not dry_run:
        try:
            _create_whitespace_index(client, collection)
        except Exception as e:
            logger.error(f"[{collection}] create whitespace index failed: {e}")
            stats["errors"] += 1

        stats["tokenizer_after"] = _detect_search_text_tokenizer(client, collection)

    stats["elapsed_s"] = round(time.time() - t0, 2)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", help="migrate only the given collection; all collections when omitted")
    ap.add_argument("--dry-run", action="store_true", help="count only, write nothing")
    ap.add_argument("--batch-size", type=int, default=256, help="page size of a single scroll batch")
    ap.add_argument("--force-reindex", action="store_true",
                    help="force a search_text rewrite + index rebuild even when the tokenizer is already whitespace")
    ap.add_argument("--qdrant-host")
    ap.add_argument("--qdrant-port", type=int)
    ap.add_argument("--qdrant-path", help="local PersistentClient path")
    ap.add_argument("--qdrant-api-key")
    ap.add_argument("--https", action="store_true")
    args = ap.parse_args()

    client = _build_qdrant_client(args)

    if args.collection:
        targets = [args.collection]
    else:
        targets = _list_collections(client)
        logger.info(f"discovered {len(targets)} collections: {targets}")

    all_stats = []
    for col in targets:
        logger.info(f"=== migrating collection: {col} ===")
        try:
            s = _migrate_collection(
                client,
                col,
                dry_run=args.dry_run,
                batch_size=args.batch_size,
                force_reindex=args.force_reindex,
            )
        except Exception as e:
            logger.exception(f"[{col}] failed: {e}")
            s = {"collection": col, "error": str(e)}
        all_stats.append(s)

    # summary
    print()
    print("=" * 60)
    print("Migration summary")
    print("=" * 60)
    total_scanned = sum(s.get("scanned", 0) for s in all_stats)
    total_updated = sum(s.get("updated", 0) for s in all_stats)
    total_errors = sum(s.get("errors", 0) for s in all_stats)
    for s in all_stats:
        print(f"  - {s.get('collection')}: "
              f"scanned={s.get('scanned', 0)} "
              f"updated={s.get('updated', 0)} "
              f"skipped={s.get('skipped_unchanged', 0)} "
              f"errors={s.get('errors', 0)} "
              f"tok={s.get('tokenizer_before')!r}→{s.get('tokenizer_after')!r} "
              f"({s.get('elapsed_s', 0)}s)")
    print(f"TOTAL: scanned={total_scanned} updated={total_updated} errors={total_errors}")
    print(f"DRY-RUN: {args.dry_run}")

    if total_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
