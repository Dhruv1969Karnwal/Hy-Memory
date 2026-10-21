#!/usr/bin/env python3
"""Clean duplicate `RELATED_TO` edges from the graph (per node pair + relation_type keep the highest weight).

## Why duplicates exist

The old `add_edge` used `CREATE` instead of `MERGE`, so the same pair of nodes could get edges
of the same type created **repeatedly**:

- `UPDATE` is an in-place update (`node_id` unchanged), so a memory merged several times takes part in graph projection several times.
  If two old nodes show up in the same batch again, `co_occurs` gets created a second time.
- The System 2 (ultra) agent loop may call `add_edge` repeatedly for the same pair of nodes.

Large long-running databases can accumulate a lot of parallel edges, especially the untyped `related` written by System 2.

## Impact

Duplicate edges **do not amplify scoring** -- the read side `build_adjacency` folds parallel edges between
the same pair of nodes into the strongest one. The real cost is that they eat the `max_edges` budget of `get_related_edges`: Cypher's `LIMIT`
truncates first and deduplication happens on the Python side, so duplicates crowd out the real edges that should have been returned.

The current `add_edge` uses `MERGE` (`ON MATCH` keeps the larger weight), so new data no longer duplicates.
This script cleans up the historical data.

## Usage

    # first see how many duplicates there are (no data changes)
    python scripts/dedupe_graph_edges.py --dry-run

    # actually clean up, 5000 pairs per batch
    python scripts/dedupe_graph_edges.py --batch 5000

    # clean only one relation_type
    python scripts/dedupe_graph_edges.py --relation-type related

Connection parameters come from env: `NEO4J_URL` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` / `NEO4J_DATABASE`.
Neo4j only -- Kuzu is an embedded single-machine database, where rebuilding is far cheaper than deduplicating in place.
"""

import argparse
import asyncio
import os
import sys


def _args():
    p = argparse.ArgumentParser(description="clean duplicate RELATED_TO edges")
    p.add_argument("--dry-run", action="store_true", help="count only, modify nothing")
    p.add_argument("--batch", type=int, default=5000, help="how many pairs per batch (default 5000)")
    p.add_argument("--relation-type", default=None, help="only process the given relation_type")
    p.add_argument("--max-batches", type=int, default=1000, help="safety cap")
    return p.parse_args()


async def main() -> int:
    a = _args()
    try:
        from neo4j import AsyncGraphDatabase
    except ImportError:
        print("neo4j driver required: pip install neo4j", file=sys.stderr)
        return 2

    url = os.getenv("NEO4J_URL") or "bolt://localhost:7687"
    user = os.getenv("NEO4J_USERNAME") or "neo4j"
    pwd = os.getenv("NEO4J_PASSWORD") or ""
    db = os.getenv("NEO4J_DATABASE") or "neo4j"

    rt_filter = "AND r.relation_type = $rt" if a.relation_type else ""
    params = {"rt": a.relation_type} if a.relation_type else {}

    driver = AsyncGraphDatabase.driver(url, auth=(user, pwd))
    try:
        async with driver.session(database=db) as s:
            # ── stats ──
            res = await s.run(
                f"""
                MATCH (x:Memory)-[r:RELATED_TO]->(y:Memory)
                WHERE true {rt_filter}
                WITH x.node_id AS s1, y.node_id AS t1,
                     r.relation_type AS rt, count(*) AS dup
                WHERE dup > 1
                RETURN count(*) AS dup_pairs, sum(dup - 1) AS removable, max(dup) AS worst
                """,
                params,
            )
            row = await res.single()
            dup_pairs = int((row or {}).get("dup_pairs") or 0)
            removable = int((row or {}).get("removable") or 0)
            worst = int((row or {}).get("worst") or 0)
            print(f"duplicate pairs: {dup_pairs}   removable edges: {removable}   worst pair: {worst} copies")

            if a.dry_run or not dup_pairs:
                if a.dry_run:
                    print("--dry-run: no data was modified")
                return 0

            # ── batch delete: keep only the highest-weight edge per pair ──
            total = 0
            for i in range(a.max_batches):
                res = await s.run(
                    f"""
                    MATCH (x:Memory)-[r:RELATED_TO]->(y:Memory)
                    WHERE true {rt_filter}
                    WITH x.node_id AS s1, y.node_id AS t1,
                         r.relation_type AS rt, collect(r) AS rels
                    WHERE size(rels) > 1
                    WITH rels LIMIT $batch
                    // weight descending, keep the first one
                    WITH [e IN rels | e] AS es
                    UNWIND es AS e
                    WITH es, e ORDER BY coalesce(e.weight, 0) DESC
                    WITH es, collect(e) AS sorted
                    WITH sorted[1..] AS extras
                    UNWIND extras AS victim
                    DELETE victim
                    RETURN count(*) AS removed
                    """,
                    {**params, "batch": a.batch},
                )
                row = await res.single()
                removed = int((row or {}).get("removed") or 0)
                total += removed
                print(f"  batch {i + 1}: deleted {removed} (running total {total})")
                if removed == 0:
                    break

            print(f"done: deleted {total} duplicate edges in total")
    finally:
        await driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
