"""
Agent Memory V2 - Neo4jGraphStore

Graph storage layer implemented on the Neo4j graph database.

Uses the **synchronous** neo4j Python driver wrapped as async with asyncio.to_thread,
which fully avoids the event loop binding problem of AsyncGraphDatabase.driver
(same strategy as the Qdrant storage layer).
"""

from typing import Optional, List, Dict, Any, Sequence, Tuple
from datetime import datetime
import asyncio
import concurrent.futures
import logging
import json

from ..models.memory import MemoryNode, MemoryLayer, MemoryStatus
from ..config import MemoryConfig
from ..structure import parse_hierarchy_path_payload
from .graph_store_base import (
    GraphStoreBase,
    embed_text_for_graph,
    make_entity_id,
    normalize_entity_name,
)

logger = logging.getLogger(__name__)

# Dedicated Graph thread pool (does not compete with VDB/SQLite)
_GRAPH_POOL_SIZE = 64
_graph_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=_GRAPH_POOL_SIZE, thread_name_prefix="graph"
)


def _run_in_graph_pool(func, *args, **kwargs):
    """Run a synchronous function in the dedicated Graph thread pool."""
    import functools
    loop = asyncio.get_event_loop()
    if args or kwargs:
        return loop.run_in_executor(_graph_executor, functools.partial(func, *args, **kwargs))
    return loop.run_in_executor(_graph_executor, func)

# Neo4j schema initialization Cypher (constraints + indexes)
_NEO4J_SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.isolation_key IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Memory) REQUIRE m.node_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.topic_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (t:HierarchyTopic) REQUIRE t.hierarchy_topic_id IS UNIQUE",
    "CREATE INDEX IF NOT EXISTS FOR (m:Memory) ON (m.isolation_key)",
    "CREATE INDEX IF NOT EXISTS FOR (m:Memory) ON (m.layer)",
    "CREATE INDEX IF NOT EXISTS FOR (m:Memory) ON (m.status)",
    "CREATE INDEX IF NOT EXISTS FOR (m:Memory) ON (m.source)",
    "CREATE INDEX IF NOT EXISTS FOR (m:Memory) ON (m.valid_from)",
    "CREATE INDEX IF NOT EXISTS FOR (t:Topic) ON (t.isolation_key)",
    "CREATE INDEX IF NOT EXISTS FOR (t:HierarchyTopic) ON (t.isolation_key)",
    "CREATE INDEX IF NOT EXISTS FOR (t:HierarchyTopic) ON (t.subject)",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (v:VdbRef) REQUIRE v.node_id IS UNIQUE",
]


def _make_vector_index_statements(dims: int) -> List[str]:
    """Generate vector index DDL (depends on embedding_dims)."""
    return [
        # V_con content vector index
        f"""CREATE VECTOR INDEX memory_content_idx IF NOT EXISTS
FOR (m:Memory) ON (m.embedding)
OPTIONS {{indexConfig: {{
    `vector.dimensions`: {dims},
    `vector.similarity_function`: 'cosine'
}}}}""",
        # V_beh behavior vector index
        f"""CREATE VECTOR INDEX memory_behavior_idx IF NOT EXISTS
FOR (m:Memory) ON (m.beh_embedding)
OPTIONS {{indexConfig: {{
    `vector.dimensions`: {dims},
    `vector.similarity_function`: 'cosine'
}}}}""",
    ]


class Neo4jGraphStore(GraphStoreBase):
    """
    Neo4j graph store (recommended backend)

    Uses the **synchronous** neo4j GraphDatabase.driver + asyncio.to_thread,
    fully sidestepping the AsyncGraphDatabase event loop binding problem.
    """

    _supports_batch_topic_tags = True
    _supports_batch_memory_nodes = True
    _supports_batch_entity_mentions = True
    _supports_batch_related_edges = True
    _supports_batch_temporal_edges = True

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self._driver = None  # Synchronous driver

        # Embedding dimensions
        vs_dims = getattr(getattr(config, 'vector_store', None), 'embedding_dims', None)
        emb_dims = getattr(getattr(config, 'embedder', None), 'embedding_dims', None)
        self._embedding_dims = vs_dims or emb_dims or 1536

        graph_config = getattr(config, 'graph_store', None)
        self._url = getattr(graph_config, 'url', None) or "bolt://localhost:7687"
        self._username = getattr(graph_config, 'username', None) or "neo4j"
        self._password = getattr(graph_config, 'password', None) or "neo4j"
        self._database = getattr(graph_config, 'database', None) or "neo4j"

    async def initialize(self) -> None:
        """Initialize the Neo4j connection and create the Schema plus both vector indexes."""
        try:
            import neo4j
        except ImportError:
            raise ImportError(
                "neo4j is required. Install with: pip install neo4j"
            )

        import os as _os
        _neo4j_pool_size = int(_os.environ.get("NEO4J_POOL_SIZE", "256"))
        self._driver = neo4j.GraphDatabase.driver(
            self._url,
            auth=(self._username, self._password),
            max_connection_pool_size=_neo4j_pool_size,
            connection_acquisition_timeout=120,
        )
        logger.info(f"[neo4j] pool_size={_neo4j_pool_size}")
        # Verify connectivity
        await _run_in_graph_pool(self._driver.verify_connectivity)

        # Create constraints and indexes
        def _init_schema():
            with self._driver.session(database=self._database) as session:
                for stmt in _NEO4J_SCHEMA_STATEMENTS:
                    try:
                        session.run(stmt).consume()
                    except Exception as e:
                        logger.debug(f"Neo4j schema DDL note: {e}")

                # Create both vector HNSW indexes
                dims = self._embedding_dims
                for stmt in _make_vector_index_statements(dims):
                    try:
                        session.run(stmt).consume()
                    except Exception as e:
                        logger.debug(f"Neo4j vector index DDL note: {e}")

        await _run_in_graph_pool(_init_schema)

        logger.info(
            f"Neo4jGraphStore initialized (sync driver), url={self._url}, "
            f"database={self._database}, embedding_dims={self._embedding_dims}"
        )

    async def _run(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run a Cypher query and return the result list (one dict per row)."""
        if self._driver is None:
            return []
        def _exec():
            with self._driver.session(database=self._database) as session:
                result = session.run(query, params or {})
                return result.data()
        return await _run_in_graph_pool(_exec)

    async def _run_single(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Run a query and return the first row (or None)."""
        rows = await self._run(query, params)
        return rows[0] if rows else None

    async def _run_write(self, query: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Run a write operation."""
        if self._driver is None:
            return
        def _exec():
            with self._driver.session(database=self._database) as session:
                session.run(query, params or {}).consume()
        await _run_in_graph_pool(_exec)

    # ================================================================
    # Lifecycle
    # ================================================================

    async def close(self) -> None:
        """Close the connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
        logger.info("Neo4jGraphStore closed")

    async def get_stats(self) -> Dict[str, Any]:
        """Return graph statistics."""
        try:
            mc = await self._run_single("MATCH (m:Memory) RETURN count(m) AS cnt")
            uc = await self._run_single("MATCH (u:User) RETURN count(u) AS cnt")
            return {
                "backend": "neo4j",
                "url": self._url,
                "database": self._database,
                "memory_nodes": mc["cnt"] if mc else 0,
                "user_nodes": uc["cnt"] if uc else 0,
            }
        except Exception as e:
            return {"backend": "neo4j", "error": str(e)}

    # ================================================================
    # User node management
    # ================================================================

    async def ensure_user_node(self, isolation_key: str, user_id: str = "",
                                agent_id: str = "") -> None:
        """Ensure the User node exists."""
        try:
            await self._run_write(
                "MERGE (u:User {isolation_key: $ik}) "
                "ON CREATE SET u.user_id = $uid, "
                "u.agent_id = $sid, u.created_at = datetime()",
                {"ik": isolation_key, "uid": user_id, "sid": agent_id},
            )
        except Exception as e:
            logger.warning(f"ensure_user_node failed: {e}")

    # ================================================================
    # Memory node CRUD
    # ================================================================

    @staticmethod
    def _memory_node_params(node: MemoryNode) -> Dict[str, Any]:
        """Serialize one node identically for single and batch projection."""
        now = datetime.now().isoformat()
        embedding = (
            getattr(node, '_graph_embedding', None)
            or getattr(node, 'embedding', None)
        )
        behavior_embedding = getattr(node, '_graph_beh_embedding', None)
        return {
            "nid": node.node_id,
            "ik": node.get_isolation_key(),
            "uid": node.user_id,
            "sid": node.agent_id,
            "source": node.source,
            "layer": node.layer.value,
            "content": node.content,
            "status": node.status.value,
            "ver": getattr(node, 'version', 1) or 1,
            "conf": node.confidence,
            "ac": getattr(node, 'access_count', 0) or 0,
            "cat": (getattr(node, 'created_at', None) or datetime.now()).isoformat(),
            "vf": (node.valid_from or datetime.now()).isoformat(),
            "vu": node.valid_until.isoformat() if node.valid_until else None,
            "laat": getattr(node, 'last_accessed_at', None),
            "sbid": getattr(node, 'superseded_by_id', '') or '',
            "ec": json.dumps(getattr(node, 'evidence_chain', []) or []),
            "cust": json.dumps(getattr(node, 'custom', {}) or {}),
            "tags": json.dumps(getattr(node, 'tags', []) or []),
            "extra": "{}",
            "now": now,
            # The single-node query omits its SET clause for false-y vectors.
            "emb": embedding if embedding else None,
            "beh_emb": behavior_embedding if behavior_embedding else None,
        }

    async def upsert_memory_node(self, node: MemoryNode) -> str:
        """Create or update a Memory node plus the User→Memory edge."""
        isolation_key = node.get_isolation_key()
        await self.ensure_user_node(
            isolation_key, node.user_id, node.agent_id
        )

        params = self._memory_node_params(node)
        now = params["now"]

        # embedding SET clause (written only when present, so existing embeddings are not overwritten with NULL)
        emb_set_create = ", m.embedding = $emb" if params["emb"] else ""
        emb_set_match = ", m.embedding = $emb" if params["emb"] else ""

        # MERGE keeps this idempotent
        await self._run_write(
            f"""
            MERGE (m:Memory {{node_id: $nid}})
            ON CREATE SET
                m.isolation_key = $ik, m.user_id = $uid,
                m.agent_id = $sid, m.layer = $layer, m.content = $content,
                m.source = $source,
                m.status = $status, m.version = $ver,
                m.confidence = $conf,
                m.access_count = $ac,
                m.created_at = $cat, m.valid_from = $vf, m.valid_until = $vu,
                m.last_accessed_at = $laat,
                m.superseded_by_id = $sbid,
                m.evidence_chain = $ec,
                m.custom_json = $cust, m.tags = $tags, m.extra_json = $extra
                {emb_set_create}
            ON MATCH SET
                m.content = $content, m.status = $status, m.version = $ver,
                m.source = $source,
                m.confidence = $conf,
                m.access_count = $ac, m.last_accessed_at = $laat,
                m.valid_until = $vu, m.superseded_by_id = $sbid,
                m.custom_json = $cust, m.tags = $tags
                {emb_set_match}
            """,
            params,
        )

        # Ensure the HAS_MEMORY edge
        try:
            await self._run_write(
                """
                MATCH (u:User {isolation_key: $ik}), (m:Memory {node_id: $nid})
                MERGE (u)-[:HAS_MEMORY]->(m)
                ON CREATE SET u.created_at = $now
                """,
                {"ik": isolation_key, "nid": node.node_id, "now": now},
            )
        except Exception as e:
            logger.debug(f"HAS_MEMORY edge note: {e}")

        return node.node_id

    async def upsert_memory_nodes_batch(
        self, nodes: Sequence[MemoryNode],
    ) -> List[str]:
        """Upsert User + Memory + HAS_MEMORY for a projection in one round trip."""
        rows = [self._memory_node_params(node) for node in nodes if node.node_id]
        if not rows:
            return []
        await self._run_write(
            """
            UNWIND $rows AS row
            MERGE (u:User {isolation_key: row.ik})
            ON CREATE SET u.user_id = row.uid, u.agent_id = row.sid,
                          u.created_at = datetime()
            MERGE (m:Memory {node_id: row.nid})
            ON CREATE SET
                m.isolation_key = row.ik, m.user_id = row.uid,
                m.agent_id = row.sid, m.layer = row.layer,
                m.source = row.source,
                m.content = row.content, m.status = row.status,
                m.version = row.ver, m.confidence = row.conf,
                m.access_count = row.ac, m.created_at = row.cat,
                m.valid_from = row.vf, m.valid_until = row.vu,
                m.last_accessed_at = row.laat,
                m.superseded_by_id = row.sbid,
                m.evidence_chain = row.ec, m.custom_json = row.cust,
                m.tags = row.tags, m.extra_json = row.extra,
                m.embedding = row.emb, m.beh_embedding = row.beh_emb
            ON MATCH SET
                m.content = row.content, m.status = row.status,
                m.source = row.source, m.version = row.ver, m.confidence = row.conf,
                m.access_count = row.ac, m.last_accessed_at = row.laat,
                m.valid_until = row.vu, m.superseded_by_id = row.sbid,
                m.custom_json = row.cust, m.tags = row.tags,
                m.embedding = CASE
                    WHEN row.emb IS NULL THEN m.embedding ELSE row.emb END,
                m.beh_embedding = CASE
                    WHEN row.beh_emb IS NULL THEN m.beh_embedding ELSE row.beh_emb END
            MERGE (u)-[:HAS_MEMORY]->(m)
            ON CREATE SET u.created_at = row.now
            """,
            {"rows": rows},
        )
        return [row["nid"] for row in rows]

    async def export_clone_snapshot(self, user_id: str) -> Dict[str, Any]:
        """Read one user's complete Neo4j projection in one executor job.

        The session issues a small fixed set of set-oriented queries.  This
        replaces the historical ``O(memory_count)`` query loop used by clone.
        """
        if self._driver is None:
            return {}

        def _export():
            with self._driver.session(database=self._database) as session:
                params = {"uid": user_id}
                return {
                    "user_id": user_id,
                    "nodes": session.run(
                        "MATCH (m:Memory) WHERE m.user_id = $uid "
                        "RETURN m, m.embedding AS embedding, "
                        "m.beh_embedding AS beh_embedding",
                        params,
                    ).data(),
                    "evidence": session.run(
                        "MATCH (m:Memory)-[:DERIVED_FROM]->(v:VdbRef) "
                        "WHERE m.user_id = $uid "
                        "RETURN m.node_id AS source_id, v.node_id AS target_id, "
                        "v.layer AS layer",
                        params,
                    ).data(),
                    "related": session.run(
                        "MATCH (a:Memory)-[r:RELATED_TO]->(b:Memory) "
                        "WHERE a.user_id = $uid AND b.user_id = $uid "
                        "RETURN a.node_id AS source_id, b.node_id AS target_id, "
                        "r.relation_type AS relation_type, r.weight AS weight",
                        params,
                    ).data(),
                    "temporal": session.run(
                        "MATCH (a:Memory)-[r:TEMPORAL_NEXT]->(b:Memory) "
                        "WHERE a.user_id = $uid AND b.user_id = $uid "
                        "RETURN a.node_id AS source_id, b.node_id AS target_id, "
                        "r.topic_id AS topic_id",
                        params,
                    ).data(),
                    "cross": session.run(
                        "MATCH (a:Memory)-[:CROSS_ABSTRACTS_TO]->(b:Memory) "
                        "WHERE a.user_id = $uid AND b.user_id = $uid "
                        "RETURN a.node_id AS source_id, b.node_id AS target_id",
                        params,
                    ).data(),
                    "tags": session.run(
                        "MATCH (m:Memory)-[:TAGGED_WITH]->(t:Topic) "
                        "WHERE m.user_id = $uid "
                        "RETURN m.node_id AS source_id, t.topic_id AS topic_id, t.name AS name, "
                        "t.embedding AS embedding, t.isolation_key AS isolation_key",
                        params,
                    ).data(),
                    "entities": session.run(
                        "MATCH (m:Memory)-[:MENTIONS]->(e:Entity) "
                        "WHERE m.user_id = $uid "
                        "RETURN m.node_id AS source_id, e.name AS name, "
                        "e.label AS label, e.isolation_key AS isolation_key",
                        params,
                    ).data(),
                }

        return await _run_in_graph_pool(_export)

    @staticmethod
    def _clone_scope(scope: str, src_user_id: str, dst_user_id: str) -> str:
        value = str(scope or "")
        if value == src_user_id:
            return dst_user_id
        prefix = src_user_id + "::"
        return dst_user_id + value[len(src_user_id):] if value.startswith(prefix) else value

    async def import_clone_snapshot(
        self,
        snapshot: Dict[str, Any],
        *,
        dst_user_id: str,
        id_map: Dict[str, str],
        embed_service=None,
    ) -> int:
        """Import a remapped clone snapshot through set-oriented primitives."""
        from ..utils.clone_references import remap_memory_references

        src_user_id = str(snapshot.get("user_id") or "")
        source_nodes = []
        node_rows = list(snapshot.get("nodes") or [])
        import uuid as _uuid
        for row in node_rows:
            node = self._record_to_memory_node(row["m"])
            id_map.setdefault(node.node_id, str(_uuid.uuid4()))
        for row in node_rows:
            node = self._record_to_memory_node(row["m"])
            old_id = node.node_id
            data = remap_memory_references(node.to_dict(), id_map)
            data["node_id"] = id_map[old_id]
            data["user_id"] = dst_user_id
            cloned = MemoryNode.from_dict(data)
            if row.get("embedding") is not None:
                cloned._graph_embedding = list(row["embedding"])
            if row.get("beh_embedding") is not None:
                cloned._graph_beh_embedding = list(row["beh_embedding"])
            source_nodes.append((old_id, cloned))

        if source_nodes:
            written = await self.upsert_memory_nodes_batch(
                [node for _old_id, node in source_nodes]
            )
            if len(written) != len(source_nodes):
                raise RuntimeError("Neo4j clone node batch was incomplete")

        evidence_rows = []
        for row in snapshot.get("evidence") or []:
            source_id = id_map.get(str(row.get("source_id") or ""))
            old_target = str(row.get("target_id") or "")
            if source_id and old_target:
                evidence_rows.append({
                    "source_id": source_id,
                    "target_id": id_map.get(old_target, old_target),
                    "layer": str(row.get("layer") or ""),
                })
        if evidence_rows:
            await self._run_write(
                """
                UNWIND $rows AS row
                MERGE (v:VdbRef {node_id: row.target_id})
                ON CREATE SET v.layer = row.layer
                WITH row, v
                MATCH (m:Memory {node_id: row.source_id})
                MERGE (m)-[:DERIVED_FROM]->(v)
                """,
                {"rows": evidence_rows},
            )

        related = []
        seen_related = set()
        for row in snapshot.get("related") or []:
            old_src, old_tgt = str(row.get("source_id") or ""), str(row.get("target_id") or "")
            relation = str(row.get("relation_type") or "related")
            key = (frozenset((old_src, old_tgt)), relation)
            if not old_src or not old_tgt or key in seen_related:
                continue
            seen_related.add(key)
            if old_src in id_map and old_tgt in id_map:
                related.append((
                    id_map[old_src], id_map[old_tgt], "RELATED_TO",
                    {"relation_type": relation, "weight": float(row.get("weight") or 1.0)},
                ))
        if related:
            await self.add_edges_batch(related)

        cross_rows = [
            {"source_id": id_map[src], "target_id": id_map[tgt], "now": datetime.now().isoformat()}
            for row in snapshot.get("cross") or []
            for src, tgt in [(str(row.get("source_id") or ""), str(row.get("target_id") or ""))]
            if src in id_map and tgt in id_map
        ]
        if cross_rows:
            await self._run_write(
                """
                UNWIND $rows AS row
                MATCH (a:Memory {node_id: row.source_id}),
                      (b:Memory {node_id: row.target_id})
                MERGE (a)-[r:CROSS_ABSTRACTS_TO]->(b)
                ON CREATE SET r.created_at = row.now
                """,
                {"rows": cross_rows},
            )

        tag_rows = []
        topic_id_map: Dict[str, str] = {}
        for row in snapshot.get("tags") or []:
            old_id = str(row.get("source_id") or "")
            name = str(row.get("name") or "").strip()
            if old_id not in id_map or not name:
                continue
            scope = self._clone_scope(row.get("isolation_key"), src_user_id, dst_user_id)
            topic_id = f"topic_{_uuid.uuid5(_uuid.NAMESPACE_URL, f'{scope}:{name}').hex[:12]}"
            old_topic_id = str(row.get("topic_id") or "")
            if old_topic_id:
                topic_id_map[old_topic_id] = topic_id
            tag_rows.append({
                "mid": id_map[old_id], "tid": topic_id, "ik": scope,
                "name": name, "emb": row.get("embedding"), "now": datetime.now().isoformat(),
            })
        if tag_rows:
            await self._run_write(
                """
                UNWIND $rows AS row
                MERGE (t:Topic {topic_id: row.tid})
                ON CREATE SET t.isolation_key = row.ik, t.name = row.name,
                              t.embedding = row.emb, t.created_at = row.now
                WITH row, t
                MATCH (m:Memory {node_id: row.mid})
                MERGE (m)-[:TAGGED_WITH]->(t)
                """,
                {"rows": tag_rows},
            )

        temporal = [
            (
                id_map[src], id_map[tgt],
                topic_id_map.get(
                    str(row.get("topic_id") or ""),
                    str(row.get("topic_id") or ""),
                ),
            )
            for row in snapshot.get("temporal") or []
            for src, tgt in [(str(row.get("source_id") or ""), str(row.get("target_id") or ""))]
            if src in id_map and tgt in id_map
        ]
        if temporal:
            await self.add_temporal_edges_batch(temporal)

        entities_by_memory: Dict[str, List[Tuple[str, str]]] = {}
        scope_by_memory: Dict[str, str] = {}
        for row in snapshot.get("entities") or []:
            old_id = str(row.get("source_id") or "")
            if old_id not in id_map:
                continue
            new_id = id_map[old_id]
            entities_by_memory.setdefault(new_id, []).append((
                str(row.get("name") or ""), str(row.get("label") or ""),
            ))
            scope_by_memory[new_id] = self._clone_scope(
                row.get("isolation_key"), src_user_id, dst_user_id,
            )
        if entities_by_memory:
            await self.replace_entity_mentions_batch([
                (memory_id, entities, scope_by_memory.get(memory_id, ""))
                for memory_id, entities in entities_by_memory.items()
            ])

        for _old_id, node in source_nodes:
            custom = getattr(node, "custom", None) or {}
            from ..structure import topic_assignments_from_custom
            for assignment in topic_assignments_from_custom(custom):
                await self.add_topic_path(
                    node.node_id, assignment["path"], node.get_topic_scope_key(),
                    assignment["subject"],
                    node_types=assignment.get("node_types"),
                    role=assignment.get("role") or "participant",
                    embed_service=embed_service,
                )
        return len(source_nodes)

    async def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Fetch a single Memory node."""
        row = await self._run_single(
            "MATCH (m:Memory {node_id: $nid}) RETURN m",
            {"nid": node_id},
        )
        if not row:
            return None
        return self._record_to_memory_node(row["m"])

    async def get_nodes_by_ids(self, node_ids: List[str]) -> List[MemoryNode]:
        """Fetch nodes in batch."""
        if not node_ids:
            return []
        rows = await self._run(
            "MATCH (m:Memory) WHERE m.node_id IN $nids RETURN m",
            {"nids": node_ids},
        )
        return [self._record_to_memory_node(r["m"]) for r in rows]

    async def get_all_nodes(
        self,
        isolation_key: str,
        layer: Optional[MemoryLayer] = None,
        status: Optional[MemoryStatus] = None,
        limit: int = 100,
    ) -> List[MemoryNode]:
        """Fetch a node list matching the given filters."""
        conditions = ["m.isolation_key = $ik"]
        params: Dict[str, Any] = {"ik": isolation_key, "lim": limit}

        if layer:
            conditions.append("m.layer = $layer")
            params["layer"] = layer.value
        if status:
            conditions.append("m.status = $status")
            params["status"] = status.value

        where = " AND ".join(conditions)
        rows = await self._run(
            f"MATCH (m:Memory) WHERE {where} RETURN m LIMIT $lim",
            params,
        )
        return [self._record_to_memory_node(r["m"]) for r in rows]

    async def get_nodes_by_user(
        self,
        user_id: str,
        layer: Optional[MemoryLayer] = None,
        status: Optional[MemoryStatus] = None,
        limit: int = 100,
    ) -> List[MemoryNode]:
        """Return one user's nodes across all agent/session isolation keys."""
        if not user_id:
            return []
        conditions = ["m.user_id = $uid"]
        params: Dict[str, Any] = {"uid": user_id, "lim": int(limit)}
        if layer:
            conditions.append("m.layer = $layer")
            params["layer"] = layer.value
        if status:
            conditions.append("m.status = $status")
            params["status"] = status.value
        rows = await self._run(
            f"MATCH (m:Memory) WHERE {' AND '.join(conditions)} "
            "RETURN m LIMIT $lim",
            params,
        )
        return [self._record_to_memory_node(r["m"]) for r in rows]

    async def get_profile(self, isolation_key: str) -> Optional[MemoryNode]:
        """Fetch the L5 Identity Profile node."""
        row = await self._run_single(
            "MATCH (m:Memory) WHERE m.isolation_key = $ik AND m.layer = $layer "
            "AND m.status = 'active' RETURN m LIMIT 1",
            {"ik": isolation_key, "layer": MemoryLayer.L4_IDENTITY.value},
        )
        if not row:
            return None
        return self._record_to_memory_node(row["m"])

    async def update_node(self, node_id: str, updates: Dict[str, Any]) -> bool:
        """Update node properties."""
        if not updates:
            return True
        set_clauses = []
        params: Dict[str, Any] = {"nid": node_id}
        for key, value in updates.items():
            safe_key = key.replace("-", "_")
            set_clauses.append(f"m.{safe_key} = ${safe_key}")
            params[safe_key] = value

        set_str = ", ".join(set_clauses)
        try:
            await self._run_write(
                f"MATCH (m:Memory {{node_id: $nid}}) SET {set_str}",
                params,
            )
            return True
        except Exception as e:
            logger.warning(f"update_node failed for {node_id}: {e}")
            return False

    async def delete_node(self, node_id: str) -> bool:
        """Delete a Memory node together with its edges."""
        try:
            await self._run_write(
                "MATCH (m:Memory {node_id: $nid}) DETACH DELETE m",
                {"nid": node_id},
            )
            return True
        except Exception as e:
            logger.warning(f"delete_node failed for {node_id}: {e}")
            return False

    async def delete_all_nodes(self, isolation_key: str) -> int:
        """Delete all Memory nodes under one isolation key."""
        try:
            count_row = await self._run_single(
                "MATCH (m:Memory) WHERE m.isolation_key = $ik RETURN count(m) AS cnt",
                {"ik": isolation_key},
            )
            count = count_row["cnt"] if count_row else 0

            await self._run_write(
                "MATCH (m:Memory) WHERE m.isolation_key = $ik DETACH DELETE m",
                {"ik": isolation_key},
            )
            return count
        except Exception as e:
            logger.warning(f"delete_all_nodes failed for {isolation_key}: {e}")
            return 0

    async def clear_projection(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        *,
        layer: str = "l2_fact",
    ) -> Dict[str, int]:
        """See base class. Narrowed by layer so System 2's l6_schema is not deleted as well."""
        out = {"memories": 0, "topics": 0, "entities": 0}
        try:
            cond = "m.layer = $layer AND m.user_id = $uid"
            params: Dict[str, Any] = {"layer": layer, "uid": user_id}
            if agent_id:
                cond += " AND m.agent_id = $aid"
                params["aid"] = agent_id

            row = await self._run_single(
                f"MATCH (m:Memory) WHERE {cond} RETURN count(m) AS cnt", params
            )
            out["memories"] = int((row or {}).get("cnt") or 0)
            await self._run_write(
                f"MATCH (m:Memory) WHERE {cond} DETACH DELETE m", params
            )

            # Delete only genuinely orphan Topics inside the target user/agent scope. The old implementation had no scope
            # condition and wiped historical orphans from other namespaces across the whole database during a single user backfill.
            topic_scope = (
                f"{user_id}::{agent_id}" if agent_id else f"{user_id}::"
            )
            topic_condition = (
                "t.isolation_key = $scope OR t.isolation_key STARTS WITH $prefix"
            )
            # A Topic without Memory members must still be kept as long as it has SUBTOPIC_OF children.
            row = await self._run_single(
                f"""
                MATCH (t:Topic) WHERE ({topic_condition})
                  AND NOT (t)<-[:TAGGED_WITH]-(:Memory)
                WITH collect(t) AS orphans
                FOREACH (x IN orphans | DETACH DELETE x)
                RETURN size(orphans) AS cnt
                """,
                {"scope": topic_scope, "prefix": topic_scope + "::"},
            )
            out["topics"] = int((row or {}).get("cnt") or 0)
            hierarchy_scope = (
                f"{user_id}::{agent_id}" if agent_id else f"{user_id}::"
            )
            if agent_id:
                out["topics"] += await self.prune_hierarchy_topics(hierarchy_scope)
            else:
                scope_rows = await self._run(
                    "MATCH (t:HierarchyTopic) "
                    "WHERE t.isolation_key STARTS WITH $prefix "
                    "RETURN DISTINCT t.isolation_key AS isolation_key",
                    {"prefix": hierarchy_scope},
                )
                for scope_row in scope_rows:
                    out["topics"] += await self.prune_hierarchy_topics(
                        str(scope_row.get("isolation_key") or "")
                    )

            row = await self._run_single(
                f"""
                MATCH (e:Entity) WHERE ({topic_condition.replace('t.', 'e.')})
                  AND NOT (e)<-[:MENTIONS]-(:Memory)
                WITH collect(e) AS orphans
                FOREACH (x IN orphans | DETACH DELETE x)
                RETURN size(orphans) AS cnt
                """,
                {"scope": topic_scope, "prefix": topic_scope + "::"},
            )
            out["entities"] = int((row or {}).get("cnt") or 0)
        except Exception as e:
            logger.warning(f"clear_projection failed for {user_id}: {e}")
        return out

    async def delete_by_metadata(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Delete Memory nodes by a combination of metadata fields."""
        try:
            conditions = ["m.user_id = $uid"]
            params: Dict[str, Any] = {"uid": user_id}
            if agent_id is not None:
                conditions.append("m.agent_id = $aid")
                params["aid"] = agent_id
            if session_id is not None:
                # Memory nodes in the Graph store do not carry session_id directly,
                # but isolation_key = "user_id::agent_id::session_id",
                # so the exact isolation_key is built to filter on
                ik = f"{user_id}::{agent_id or 'default_agent'}::{session_id}"
                conditions.append("m.isolation_key = $ik")
                params["ik"] = ik

            where = " AND ".join(conditions)

            count_row = await self._run_single(
                f"MATCH (m:Memory) WHERE {where} RETURN count(m) AS cnt",
                params,
            )
            count = count_row["cnt"] if count_row else 0

            await self._run_write(
                f"MATCH (m:Memory) WHERE {where} DETACH DELETE m",
                params,
            )

            user_conds = ["u.user_id = $uid"]
            user_params: Dict[str, Any] = {"uid": user_id}
            if session_id is not None:
                ik = f"{user_id}::{agent_id or 'default_agent'}::{session_id}"
                user_conds.append("u.isolation_key = $ik")
                user_params["ik"] = ik
            elif agent_id is not None:
                user_conds.append("u.agent_id = $aid")
                user_params["aid"] = agent_id
            await self._run_write(
                f"MATCH (u:User) WHERE {' AND '.join(user_conds)} DETACH DELETE u",
                user_params,
            )

            topic_params: Dict[str, Any] = {"pfx": f"{user_id}::"}
            if agent_id is not None and session_id is None:
                topic_params["pfx"] = f"{user_id}::{agent_id}::"
            elif session_id is not None:
                topic_params["pfx"] = f"{user_id}::{agent_id or 'default_agent'}::{session_id}"
                await self._run_write(
                    "MATCH (t:Topic) WHERE t.isolation_key = $pfx DETACH DELETE t",
                    topic_params,
                )
            else:
                await self._run_write(
                    "MATCH (t:Topic) WHERE t.isolation_key STARTS WITH $pfx DETACH DELETE t",
                    topic_params,
                )

            # HierarchyTopic spans sessions. A session delete only prunes nodes
            # made empty by the removed L2s; broader deletes remove the scope.
            hierarchy_scope = f"{user_id}::{agent_id or 'default_agent'}"
            if session_id is not None:
                await self.prune_hierarchy_topics(hierarchy_scope)
            else:
                hierarchy_params: Dict[str, Any] = {"pfx": f"{user_id}::"}
                hierarchy_where = "t.isolation_key STARTS WITH $pfx"
                if agent_id is not None:
                    hierarchy_params["scope"] = hierarchy_scope
                    hierarchy_where = "t.isolation_key = $scope"
                await self._run_write(
                    f"MATCH (t:HierarchyTopic) WHERE {hierarchy_where} "
                    "DETACH DELETE t",
                    hierarchy_params,
                )

            await self._run_write(
                "MATCH (v:VdbRef) "
                "OPTIONAL MATCH ()-[r:DERIVED_FROM]->(v) "
                "WITH v, count(r) AS cnt WHERE cnt = 0 "
                "DETACH DELETE v",
                {},
            )

            logger.info(
                f"[graph-store] delete_by_metadata user={user_id} agent={agent_id} "
                f"session={session_id} memory_nodes={count}"
            )
            return count
        except Exception as e:
            logger.warning(f"delete_by_metadata failed (user_id={user_id}): {e}")
            return 0

    # ================================================================
    # Edge operations
    # ================================================================

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add a Memory→Memory relation edge (RELATED_TO), building both directions automatically."""
        props = properties or {}
        now = datetime.now().isoformat()

        edge_type_upper = edge_type.upper()
        if edge_type_upper != "RELATED_TO":
            logger.debug(f"Unsupported edge type: {edge_type}, only RELATED_TO is supported via add_edge")
            return False

        params = {
            "src": source_id, "tgt": target_id,
            "rtype": props.get("relation_type", "related"),
            "w": props.get("weight", 1.0),
            "now": now,
        }

        try:
            # Forward A→B plus reverse B→A, all in one Cypher statement.
            #
            # MERGE is mandatory, not CREATE: the same node pair gets the same edge type **repeatedly** --
            # UPDATE is in place (node_id unchanged), so a memory merged several times takes part in projection several times,
            # and if two old nodes land in the same batch again co_occurs would be created twice; the System 2 agent loop
            # may even be called repeatedly for one pair (measured in production: `related` had 200k duplicate pairs, up to 32 copies).
            #
            # Duplicate edges do not inflate scoring (build_adjacency folds them into the strongest one), but they consume
            # the max_edges budget of get_related_edges -- Cypher's LIMIT truncates first,
            # deduplication happens on the Python side, so duplicates crowd out the real edges that should have been returned.
            # ON MATCH keeps the larger weight, matching the read-side "fold into the strongest" semantics.
            await self._run_write(
                """
                MATCH (a:Memory {node_id: $src}), (b:Memory {node_id: $tgt})
                MERGE (a)-[r1:RELATED_TO {relation_type: $rtype}]->(b)
                    ON CREATE SET r1.weight = $w, r1.created_at = $now
                    ON MATCH SET r1.weight =
                        CASE WHEN $w > r1.weight THEN $w ELSE r1.weight END
                MERGE (b)-[r2:RELATED_TO {relation_type: $rtype}]->(a)
                    ON CREATE SET r2.weight = $w, r2.created_at = $now
                    ON MATCH SET r2.weight =
                        CASE WHEN $w > r2.weight THEN $w ELSE r2.weight END
                """,
                params,
            )
            return True
        except Exception as e:
            logger.warning(f"add_edge failed ({source_id})-[{edge_type}]->({target_id}): {e}")
            return False

    async def add_edges_batch(
        self,
        edges: Sequence[
            Tuple[str, str, str, Optional[Dict[str, Any]]]
        ],
    ) -> List[bool]:
        """Create bidirectional RELATED_TO pairs with one Neo4j request."""
        rows: List[Dict[str, Any]] = []
        valid: List[bool] = []
        now = datetime.now().isoformat()
        for source_id, target_id, edge_type, properties in edges:
            if str(edge_type).upper() != "RELATED_TO":
                valid.append(False)
                continue
            props = properties or {}
            valid.append(True)
            rows.append({
                "src": source_id,
                "tgt": target_id,
                "rtype": props.get("relation_type", "related"),
                "w": props.get("weight", 1.0),
                "now": now,
            })
        if rows:
            await self._run_write(
                """
                UNWIND $rows AS row
                MATCH (a:Memory {node_id: row.src}),
                      (b:Memory {node_id: row.tgt})
                MERGE (a)-[r1:RELATED_TO {relation_type: row.rtype}]->(b)
                    ON CREATE SET r1.weight = row.w,
                                  r1.created_at = row.now
                    ON MATCH SET r1.weight =
                        CASE WHEN row.w > r1.weight THEN row.w ELSE r1.weight END
                MERGE (b)-[r2:RELATED_TO {relation_type: row.rtype}]->(a)
                    ON CREATE SET r2.weight = row.w,
                                  r2.created_at = row.now
                    ON MATCH SET r2.weight =
                        CASE WHEN row.w > r2.weight THEN row.w ELSE r2.weight END
                """,
                {"rows": rows},
            )
        return valid

    async def add_topic_tag(
        self,
        memory_node_id: str,
        topic_name: str,
        isolation_key: str,
        embed_service=None,
        similarity_threshold: float = 0.85,
    ) -> str:
        """Attach a memory node to a topic (Memory→Topic), with tag normalization

        When embed_service is provided:
          1. embed the new tag
          2. cosine-compare it with existing Topics (same isolation_key)
          3. above similarity_threshold → reuse the existing Topic
          4. otherwise create a new Topic (with embedding)

        Without embed_service it degrades to the original behaviour (exact match on name).
        """
        import uuid as _uuid
        import numpy as np
        now = datetime.now().isoformat()

        # --- Tag normalization ---
        if embed_service is not None:
            try:
                tag_embedding = await embed_text_for_graph(embed_service, topic_name)

                # Fetch existing Topics
                existing = await self.get_all_topics(isolation_key)
                best_match = None
                best_sim = 0.0

                if existing and tag_embedding:
                    q_vec = np.array(tag_embedding, dtype=np.float32)
                    q_norm = np.linalg.norm(q_vec) + 1e-10
                    for t in existing:
                        t_emb = t.get("embedding")
                        if not t_emb:
                            continue
                        t_vec = np.array(t_emb, dtype=np.float32)
                        t_norm = np.linalg.norm(t_vec) + 1e-10
                        sim = float(np.dot(q_vec, t_vec) / (q_norm * t_norm))
                        if sim > best_sim:
                            best_sim = sim
                            best_match = t

                if best_match and best_sim >= similarity_threshold:
                    # Reuse an existing Topic
                    topic_id = best_match["topic_id"]
                    logger.debug(
                        f"[tag-normalize] '{topic_name}' → reuse '{best_match['name']}' "
                        f"(sim={best_sim:.3f})"
                    )
                else:
                    # Create a new Topic (with embedding)
                    topic_id = f"topic_{_uuid.uuid5(_uuid.NAMESPACE_URL, f'{isolation_key}:{topic_name}').hex[:12]}"
                    await self._run_write(
                        """
                        MERGE (t:Topic {topic_id: $tid})
                        ON CREATE SET t.isolation_key = $ik, t.name = $name,
                                      t.created_at = $now, t.embedding = $emb
                        """,
                        {"tid": topic_id, "ik": isolation_key, "name": topic_name,
                         "now": now, "emb": tag_embedding},
                    )
                    logger.debug(f"[tag-normalize] '{topic_name}' → new Topic (no match >= {similarity_threshold})")

                # Create the TAGGED_WITH edge (MERGE prevents duplicates)
                await self._run_write(
                    """
                    MATCH (m:Memory {node_id: $mid}), (t:Topic {topic_id: $tid})
                    MERGE (m)-[:TAGGED_WITH]->(t)
                    """,
                    {"mid": memory_node_id, "tid": topic_id},
                )
                return topic_id

            except Exception as e:
                logger.warning(f"[tag-normalize] failed, fallback to exact match: {e}")
                # Fallback to the original logic below

        # --- Original logic (when embed_service is absent) ---
        topic_id = f"topic_{_uuid.uuid5(_uuid.NAMESPACE_URL, f'{isolation_key}:{topic_name}').hex[:12]}"
        try:
            await self._run_write(
                """
                MERGE (t:Topic {topic_id: $tid})
                ON CREATE SET t.isolation_key = $ik, t.name = $name, t.created_at = $now
                """,
                {"tid": topic_id, "ik": isolation_key, "name": topic_name, "now": now},
            )
            # ⚠️ This once passed an extra params dict (`_run_write` accepts only one),
            # so the TypeError was silently swallowed by the except below → **TAGGED_WITH edges were never created**.
            # As a result every tag going through the fallback became an isolated Topic: get_topic_memories returned nothing,
            # Topic summaries never triggered, the temporal spine found no same-Topic members, and tag bridging found nothing either.
            await self._run_write(
                """
                MATCH (m:Memory {node_id: $mid}), (t:Topic {topic_id: $tid})
                MERGE (m)-[:TAGGED_WITH]->(t)
                """,
                {"mid": memory_node_id, "tid": topic_id},
            )
        except Exception as e:
            logger.warning(f"add_topic_tag failed for {memory_node_id}/{topic_name}: {e}")

        return topic_id

    @staticmethod
    def _match_existing_topic(
        topics: List[Dict[str, Any]],
        embedding: List[float],
        threshold: float,
    ) -> Optional[str]:
        """Return the closest existing topic above the unchanged cosine gate."""
        best_id, best_sim = None, threshold
        norm_q = sum(value * value for value in embedding) ** 0.5
        if norm_q == 0:
            return None
        for topic in topics:
            existing = topic.get("embedding")
            if not existing:
                continue
            norm_t = sum(value * value for value in existing) ** 0.5
            if norm_t == 0:
                continue
            similarity = sum(
                left * right for left, right in zip(embedding, existing)
            ) / (norm_q * norm_t)
            if similarity >= best_sim:
                best_id, best_sim = topic.get("topic_id"), similarity
        return best_id

    async def add_topic_tags_batch(
        self,
        assignments: List[Tuple[str, str, str]],
        *,
        embed_service=None,
        similarity_threshold: float = 0.85,
    ) -> List[Tuple[str, str]]:
        """Normalize and persist a projection's tags with bounded round trips."""
        if not assignments:
            return []
        import uuid as _uuid

        normalized = []
        seen = set()
        for memory_node_id, topic_name, isolation_key in assignments:
            item = (
                str(memory_node_id), str(topic_name).strip(), str(isolation_key),
            )
            if not all(item) or item in seen:
                continue
            seen.add(item)
            normalized.append(item)
        if not normalized:
            return []

        scopes = sorted({item[2] for item in normalized})
        topics_by_scope = {
            scope: list(await self.get_all_topics(scope) or [])
            for scope in scopes
        }
        existing_ids = {
            str(topic.get("topic_id"))
            for topics in topics_by_scope.values()
            for topic in topics
            if topic.get("topic_id")
        }
        missing_tags = list(dict.fromkeys(
            topic_name
            for _memory_id, topic_name, isolation_key in normalized
            if f"topic_{_uuid.uuid5(_uuid.NAMESPACE_URL, f'{isolation_key}:{topic_name}').hex[:12]}"
            not in existing_ids
        ))
        embeddings: Dict[str, Optional[List[float]]] = {
            tag: None for tag in missing_tags
        }
        if embed_service is not None and missing_tags:
            try:
                batch_fn = getattr(embed_service, "embed_batch", None)
                vectors = (
                    await batch_fn(missing_tags)
                    if callable(batch_fn)
                    else [
                        await embed_text_for_graph(embed_service, tag)
                        for tag in missing_tags
                    ]
                )
                for tag, vector in zip(missing_tags, vectors or []):
                    if vector:
                        embeddings[tag] = list(vector)
            except Exception as exc:
                logger.debug("Neo4j batch topic embedding skipped: %s", exc)

        now = datetime.now().isoformat()
        resolved: List[Tuple[str, str]] = []
        rows = []
        for memory_node_id, topic_name, isolation_key in normalized:
            topic_id = (
                f"topic_{_uuid.uuid5(_uuid.NAMESPACE_URL, f'{isolation_key}:{topic_name}').hex[:12]}"
            )
            embedding = embeddings.get(topic_name)
            if topic_id not in existing_ids and embedding:
                matched = self._match_existing_topic(
                    topics_by_scope.get(isolation_key, []),
                    embedding,
                    similarity_threshold,
                )
                if matched:
                    topic_id = matched
            if topic_id not in existing_ids:
                existing_ids.add(topic_id)
                topics_by_scope.setdefault(isolation_key, []).append({
                    "topic_id": topic_id,
                    "name": topic_name,
                    "embedding": embedding,
                })
            rows.append({
                "mid": memory_node_id,
                "tid": topic_id,
                "ik": isolation_key,
                "name": topic_name,
                "emb": embedding,
                "now": now,
            })
            resolved.append((memory_node_id, topic_id))

        await self._run_write(
            """
            UNWIND $rows AS row
            MERGE (t:Topic {topic_id: row.tid})
            ON CREATE SET t.isolation_key = row.ik, t.name = row.name,
                          t.created_at = row.now, t.embedding = row.emb
            WITH row, t
            MATCH (m:Memory {node_id: row.mid})
            MERGE (m)-[:TAGGED_WITH]->(t)
            """,
            {"rows": rows},
        )
        return resolved

    async def clear_hierarchy_memberships(self, memory_node_id: str) -> int:
        if not memory_node_id:
            return 0
        row = await self._run_single(
            "MATCH (:Memory {node_id: $mid})-[r:MEMBER_OF]->(:HierarchyTopic) "
            "WITH collect(r) AS rels FOREACH (x IN rels | DELETE x) "
            "RETURN size(rels) AS cnt",
            {"mid": memory_node_id},
        )
        return int((row or {}).get("cnt") or 0)

    async def prune_hierarchy_topics(self, isolation_key: str) -> int:
        if not isolation_key:
            return 0
        total = 0
        while True:
            row = await self._run_single(
                """
                MATCH (t:HierarchyTopic {isolation_key: $ik})
                WHERE NOT (t)<-[:MEMBER_OF]-(:Memory)
                  AND NOT (t)<-[:SUBTOPIC_OF]-(:HierarchyTopic)
                WITH collect(t) AS orphans
                FOREACH (x IN orphans | DETACH DELETE x)
                RETURN size(orphans) AS cnt
                """,
                {"ik": isolation_key},
            )
            count = int((row or {}).get("cnt") or 0)
            if not count:
                break
            total += count
        return total

    async def add_topic_path(
        self, memory_node_id: str, topic_path: List[str], isolation_key: str,
        subject: str, *, node_types: Optional[List[str]] = None,
        role: str = "subject", embed_service=None,
    ) -> List[str]:
        """Create a typed subject-scoped hierarchy and attach the L2 to its leaf."""
        if not memory_node_id or not subject:
            return []
        import json as _json
        from ..structure import (
            hierarchy_path_payload,
            hierarchy_topic_id,
            normalize_hierarchy_node_types,
            normalize_topic_paths,
        )

        paths = normalize_topic_paths(topic_path)
        if not paths:
            return []
        path = paths[0]
        typed_identity = node_types is not None
        types = normalize_hierarchy_node_types(node_types, len(path))
        ids: List[str] = []
        parent_id = ""
        now = datetime.now().isoformat()
        try:
            for depth, name in enumerate(path, start=1):
                prefix = path[:depth]
                prefix_types = types[:depth]
                topic_id = hierarchy_topic_id(
                    isolation_key, subject, prefix,
                    prefix_types if typed_identity else None,
                )
                if not topic_id:
                    return []
                ids.append(topic_id)
                exists = await self._run_single(
                    "MATCH (t:HierarchyTopic {hierarchy_topic_id: $tid}) "
                    "RETURN t.hierarchy_topic_id AS topic_id",
                    {"tid": topic_id},
                )
                if not exists:
                    label = (
                        f"{subject}: "
                        + " > ".join(
                            f"{kind}:{segment}"
                            for segment, kind in zip(prefix, prefix_types)
                        )
                    )
                    embedding = await embed_text_for_graph(embed_service, label)
                    await self._run_write(
                        """
                        MERGE (t:HierarchyTopic {hierarchy_topic_id: $tid})
                        ON CREATE SET t.isolation_key = $ik, t.subject = $subject,
                                      t.name = $name, t.path_json = $path,
                                      t.depth = $depth, t.embedding = $emb,
                                      t.created_at = $now
                        """,
                        {
                            "tid": topic_id, "ik": isolation_key,
                            "subject": subject, "name": name,
                            "path": hierarchy_path_payload(prefix, prefix_types),
                            "depth": depth, "emb": embedding, "now": now,
                        },
                    )
                if parent_id:
                    await self._run_write(
                        """
                        MATCH (c:HierarchyTopic {hierarchy_topic_id: $child}),
                              (p:HierarchyTopic {hierarchy_topic_id: $parent})
                        MERGE (c)-[:SUBTOPIC_OF]->(p)
                        """,
                        {"child": topic_id, "parent": parent_id},
                    )
                parent_id = topic_id
            await self._run_write(
                """
                MATCH (m:Memory {node_id: $mid}),
                      (t:HierarchyTopic {hierarchy_topic_id: $tid})
                MERGE (m)-[:MEMBER_OF]->(t)
                """,
                {"mid": memory_node_id, "tid": ids[-1]},
            )
            return ids
        except Exception as e:
            logger.warning(f"add_topic_path failed for {memory_node_id}/{path}: {e}")
            return []

    async def get_hierarchy_topics(
        self, isolation_key: str, *, subject: str = "",
    ) -> List[Dict[str, Any]]:
        condition = "t.isolation_key = $ik"
        params: Dict[str, Any] = {"ik": isolation_key}
        if subject:
            condition += " AND toLower(t.subject) = toLower($subject)"
            params["subject"] = subject
        rows = await self._run(
            f"""
            MATCH (t:HierarchyTopic) WHERE {condition}
            OPTIONAL MATCH (m:Memory)-[:MEMBER_OF]->(t)
            WHERE m.status = 'active'
            RETURN t.hierarchy_topic_id AS topic_id, t.name AS name,
                   t.subject AS subject, t.path_json AS path_json,
                   t.depth AS depth, t.embedding AS embedding,
                   count(DISTINCT m) AS direct_member_count
            """,
            params,
        )
        edge_rows = await self._run(
            """
            MATCH (c:HierarchyTopic)-[:SUBTOPIC_OF]->(p:HierarchyTopic)
            WHERE c.isolation_key = $ik AND p.isolation_key = $ik
            RETURN c.hierarchy_topic_id AS child_id,
                   p.hierarchy_topic_id AS parent_id
            """,
            {"ik": isolation_key},
        )
        parent_by_child = {row["child_id"]: row["parent_id"] for row in edge_rows}
        child_counts: Dict[str, int] = {}
        for edge in edge_rows:
            parent = edge["parent_id"]
            child_counts[parent] = child_counts.get(parent, 0) + 1
        for row in rows:
            parsed_path = parse_hierarchy_path_payload(row.pop("path_json") or "[]")
            row["path"] = parsed_path["path"]
            row["node_types"] = parsed_path["node_types"]
            row["node_type"] = (
                parsed_path["node_types"][-1]
                if parsed_path["node_types"] else "facet"
            )
            row["parent_id"] = parent_by_child.get(row["topic_id"], "")
            row["child_count"] = child_counts.get(row["topic_id"], 0)
        return rows

    async def get_hierarchy_topic_memories(
        self, hierarchy_topic_id: str, *, limit: int = 200,
    ) -> List[Dict[str, Any]]:
        return await self._run(
            """
            MATCH (m:Memory)-[:MEMBER_OF]->
                  (:HierarchyTopic {hierarchy_topic_id: $tid})
            WHERE m.status = 'active'
            RETURN m.node_id AS node_id, m.content AS content LIMIT $limit
            """,
            {"tid": hierarchy_topic_id, "limit": int(limit)},
        )

    async def get_hierarchy_memberships(
        self, isolation_key: str,
    ) -> List[Dict[str, str]]:
        return await self._run(
            """
            MATCH (m:Memory)-[:MEMBER_OF]->
                  (t:HierarchyTopic {isolation_key: $ik})
            WHERE m.status = 'active'
            RETURN m.node_id AS node_id,
                   t.hierarchy_topic_id AS topic_id
            """,
            {"ik": isolation_key},
        )

    async def add_entity_mention(
        self,
        memory_node_id: str,
        name: str,
        label: str,
        isolation_key: str,
    ) -> str:
        """Memory -[:MENTIONS]-> Entity。"""
        if not memory_node_id or not name:
            return ""
        name_disp = str(name).strip()
        name_norm = normalize_entity_name(name_disp)
        lab = (label or "").strip().upper()
        if not name_norm or not lab:
            return ""
        entity_id = make_entity_id(isolation_key, lab, name_disp)
        now = datetime.now().isoformat()
        try:
            await self._run_write(
                """
                MERGE (e:Entity {entity_id: $eid})
                ON CREATE SET e.isolation_key = $ik, e.name = $name,
                              e.name_norm = $nn, e.label = $lab, e.created_at = $now
                """,
                {
                    "eid": entity_id, "ik": isolation_key, "name": name_disp,
                    "nn": name_norm, "lab": lab, "now": now,
                },
            )
            await self._run_write(
                """
                MATCH (m:Memory {node_id: $mid}), (e:Entity {entity_id: $eid})
                MERGE (m)-[:MENTIONS]->(e)
                """,
                {"mid": memory_node_id, "eid": entity_id},
            )
            return entity_id
        except Exception as e:
            logger.warning(
                f"add_entity_mention failed for {memory_node_id}/{name_disp}: {e}"
            )
            return ""

    async def replace_entity_mentions_batch(
        self,
        assignments: Sequence[
            Tuple[str, Sequence[Tuple[str, str]], str]
        ],
    ) -> List[Tuple[str, str]]:
        """Clear and rebuild several memories' MENTIONS edges atomically."""
        now = datetime.now().isoformat()
        rows: List[Dict[str, Any]] = []
        resolved: List[Tuple[str, str]] = []
        for memory_node_id, entities, isolation_key in assignments:
            if not memory_node_id:
                continue
            entity_rows: List[Dict[str, str]] = []
            for name, label in entities:
                name_disp = str(name).strip()
                name_norm = normalize_entity_name(name_disp)
                lab = (label or "").strip().upper()
                if not name_norm or not lab:
                    continue
                entity_id = make_entity_id(isolation_key, lab, name_disp)
                entity_rows.append({
                    "eid": entity_id,
                    "ik": isolation_key,
                    "name": name_disp,
                    "nn": name_norm,
                    "lab": lab,
                })
                resolved.append((memory_node_id, entity_id))
            rows.append({"mid": memory_node_id, "entities": entity_rows})
        if not rows:
            return []
        await self._run_write(
            """
            UNWIND $rows AS row
            MATCH (m:Memory {node_id: row.mid})
            OPTIONAL MATCH (m)-[old:MENTIONS]->(:Entity)
            DELETE old
            WITH DISTINCT m, row
            FOREACH (ent IN row.entities |
                MERGE (e:Entity {entity_id: ent.eid})
                ON CREATE SET e.isolation_key = ent.ik, e.name = ent.name,
                              e.name_norm = ent.nn, e.label = ent.lab,
                              e.created_at = $now
                MERGE (m)-[:MENTIONS]->(e)
            )
            """,
            {"rows": rows, "now": now},
        )
        return resolved

    async def clear_entity_mentions(self, memory_node_id: str) -> int:
        if not memory_node_id:
            return 0
        try:
            row = await self._run_single(
                """
                MATCH (m:Memory {node_id: $mid})-[r:MENTIONS]->(:Entity)
                WITH collect(r) AS rs
                FOREACH (x IN rs | DELETE x)
                RETURN size(rs) AS cnt
                """,
                {"mid": memory_node_id},
            )
            return int((row or {}).get("cnt") or 0)
        except Exception as e:
            logger.debug(f"clear_entity_mentions note: {e}")
            return 0

    async def find_memories_by_entities(
        self,
        isolation_key: str,
        entities: Sequence[Tuple[str, str]],
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        if not isolation_key or not entities:
            return []
        norms = sorted({
            normalize_entity_name(n) for n, _ in entities if n and str(n).strip()
        })
        if not norms:
            return []
        try:
            rows = await self._run(
                """
                MATCH (m:Memory)-[:MENTIONS]->(e:Entity)
                WHERE e.isolation_key = $ik AND e.name_norm IN $norms
                      AND m.status = 'active'
                RETURN DISTINCT m.node_id AS node_id, e.name AS entity_name,
                       e.label AS entity_label
                LIMIT $lim
                """,
                {"ik": isolation_key, "norms": norms, "lim": int(limit)},
            )
            return [
                {
                    "node_id": r["node_id"],
                    "entity_name": r.get("entity_name"),
                    "entity_label": r.get("entity_label"),
                    "match_score": 1.0,
                }
                for r in rows
                if r.get("node_id")
            ]
        except Exception as e:
            logger.warning(f"find_memories_by_entities failed: {e}")
            return []

    # ================================================================
    # Topic retrieval (tag convergence + tag bridging recall)
    # ================================================================

    async def get_all_topics(self, isolation_key: str) -> List[Dict[str, Any]]:
        """Return every Topic node under an isolation_key (with degree).

        `degree` = number of active memories attached to the Topic, which profile aggregation needs
        (Topic degree + freshness ordering ≈ "what kind of person the user is", at zero LLM cost).
        Both backends must return the same shape, otherwise the feature fails silently on one of them.
        """
        try:
            rows = await self._run(
                """
                MATCH (t:Topic {isolation_key: $ik})
                OPTIONAL MATCH (m:Memory)-[:TAGGED_WITH]->(t)
                WHERE m.status = 'active'
                RETURN t.topic_id AS topic_id, t.name AS name,
                       t.embedding AS embedding, count(m) AS degree
                """,
                {"ik": isolation_key},
            )
            return [
                {
                    "topic_id": r["topic_id"],
                    "name": r["name"],
                    "embedding": r.get("embedding"),
                    "degree": int(r.get("degree") or 0),
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"get_all_topics failed: {e}")
            return []

    async def tag_bridge_search(
        self,
        topic_ids: List[str],
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Tag bridging recall: look up TAGGED_WITH from a Topic back to Memory."""
        if not topic_ids:
            return []
        try:
            rows = await self._run(
                """
                MATCH (t:Topic)<-[:TAGGED_WITH]-(m:Memory)
                WHERE t.topic_id IN $tids AND m.status = 'active'
                RETURN DISTINCT m.node_id AS node_id, m.content AS content,
                       m.layer AS layer, m.confidence AS confidence,
                       t.name AS topic_name, m.custom_json AS custom_json
                LIMIT $lim
                """,
                {"tids": topic_ids, "lim": limit},
            )
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning(f"tag_bridge_search failed: {e}")
            return []

    # ================================================================
    # V3: VdbRef (shadow node) + DERIVED_FROM edge
    # ================================================================

    async def ensure_vdbref(self, node_id: str, layer: str) -> None:
        """Ensure the VdbRef shadow node exists."""
        try:
            await self._run_write(
                "MERGE (v:VdbRef {node_id: $nid}) ON CREATE SET v.layer = $layer",
                {"nid": node_id, "layer": layer},
            )
        except Exception as e:
            logger.debug(f"ensure_vdbref note: {e}")

    async def add_derived_from(self, memory_node_id: str, vdbref_node_id: str) -> bool:
        """Add a DERIVED_FROM edge: Memory → VdbRef."""
        try:
            now = datetime.now().isoformat()
            await self._run_write(
                """
                MATCH (m:Memory {node_id: $mid}), (v:VdbRef {node_id: $vid})
                MERGE (m)-[r:DERIVED_FROM]->(v)
                ON CREATE SET r.created_at = $now
                """,
                {"mid": memory_node_id, "vid": vdbref_node_id, "now": now},
            )
            return True
        except Exception as e:
            logger.warning(f"add_derived_from failed ({memory_node_id})->({vdbref_node_id}): {e}")
            return False

    async def find_referencing_memories(
        self,
        vdb_node_ids: List[str],
        limit: int = 50,
        include_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Reverse lookup: VDB node_id → the Graph Memory nodes referencing it."""
        if not vdb_node_ids:
            return []
        results = []
        source_filter = ""
        if include_sources:
            source_filter += " AND m.source IN $include_sources"
        if exclude_sources:
            source_filter += " AND NOT m.source IN $exclude_sources"
        for vdb_id in vdb_node_ids:
            try:
                rows = await self._run(
                    f"""
                    MATCH (v:VdbRef {{node_id: $vid}})<-[:DERIVED_FROM]-(m:Memory)
                    WHERE m.status = 'active' {source_filter}
                      AND NOT m.node_id STARTS WITH 'catalog_'
                    RETURN m.node_id AS node_id, m.content AS content,
                           m.layer AS layer, m.confidence AS confidence,
                           m.custom_json AS custom_json, m.source AS memory_source
                    LIMIT $limit
                    """,
                    {"vid": vdb_id, "limit": limit,
                     "include_sources": list(include_sources or []),
                     "exclude_sources": list(exclude_sources or [])},
                )
                for row in rows:
                    results.append({
                        "node_id": row["node_id"],
                        "content": row["content"],
                        "layer": row["layer"],
                        "confidence": row["confidence"],
                        "custom_json": row.get("custom_json", ""),
                        "evidence_vdb_id": vdb_id,
                        "retrieval_source": "reverse_evidence",
                        "memory_source": row.get("memory_source") or "default",
                    })
            except Exception as e:
                logger.debug(f"find_referencing_memories for {vdb_id}: {e}")
        return results[:limit]

    async def get_evidence_vdbrefs(self, memory_node_id: str) -> List[Dict[str, Any]]:
        """Return every DERIVED_FROM evidence VdbRef of one Graph Memory."""
        try:
            rows = await self._run(
                """
                MATCH (m:Memory {node_id: $mid})-[:DERIVED_FROM]->(v:VdbRef)
                RETURN v.node_id AS node_id, v.layer AS layer
                """,
                {"mid": memory_node_id},
            )
            return [{"node_id": r["node_id"], "layer": r["layer"]} for r in rows]
        except Exception as e:
            logger.debug(f"get_evidence_vdbrefs for {memory_node_id}: {e}")
            return []

    # ================================================================
    # Graph retrieval
    # ================================================================

    async def get_topic_memories(
        self, topic_id: str, *, limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """List active memories under a Topic (ordered by created_at). Semantics in the base class.

        ⚠️ `ORDER BY` must use the **projected alias** (`created_at`), never `m.created_at`:
        after `RETURN DISTINCT` Neo4j only recognizes variables present in the projection, otherwise the whole query is a SyntaxError.
        The exception here is swallowed by `logger.debug` + `return []`, so it stays quiet --
        in practice `MEMORY_TOPIC_SUMMARY_ENABLED=true` never gets member memories and no summary is ever produced.
        """
        if not topic_id:
            return []
        try:
            rows = await self._run(
                """
                MATCH (m:Memory)-[:TAGGED_WITH]->(t:Topic {topic_id: $tid})
                WHERE m.status = 'active'
                RETURN DISTINCT m.node_id AS node_id, m.content AS content,
                       m.created_at AS created_at
                ORDER BY created_at
                LIMIT $lim
                """,
                {"tid": topic_id, "lim": int(limit)},
            )
            # created_at is projected only for ordering and stays out of the payload -- same shape as the Kuzu backend
            return [
                {"node_id": r.get("node_id"), "content": r.get("content")}
                for r in rows if r.get("node_id")
            ]
        except Exception as e:
            logger.debug(f"get_topic_memories({topic_id}): {e}")
            return []

    async def add_temporal_edge(
        self, prev_id: str, next_id: str, topic_id: str = "",
    ) -> bool:
        """Create intra-Topic temporal spine edges (TEMPORAL_NEXT, directed prev→next). Semantics in the base class."""
        if not prev_id or not next_id or prev_id == next_id:
            return False
        try:
            await self._run_write(
                """
                MATCH (a:Memory {node_id: $p}), (b:Memory {node_id: $n})
                MERGE (a)-[r:TEMPORAL_NEXT]->(b)
                ON CREATE SET r.topic_id = $tid, r.created_at = $now
                """,
                {"p": prev_id, "n": next_id, "tid": topic_id or "",
                 "now": datetime.now().isoformat()},
            )
            return True
        except Exception as e:
            logger.debug(f"add_temporal_edge({prev_id}->{next_id}): {e}")
            return False

    async def add_temporal_edges_batch(
        self, edges: Sequence[Tuple[str, str, str]],
    ) -> List[bool]:
        """Create directed TEMPORAL_NEXT edges with one Neo4j request."""
        rows = [
            {
                "p": prev_id,
                "n": next_id,
                "tid": topic_id or "",
                "now": datetime.now().isoformat(),
            }
            for prev_id, next_id, topic_id in edges
            if prev_id and next_id and prev_id != next_id
        ]
        if not rows:
            return [False for _edge in edges]
        await self._run_write(
            """
            UNWIND $rows AS row
            MATCH (a:Memory {node_id: row.p}), (b:Memory {node_id: row.n})
            MERGE (a)-[r:TEMPORAL_NEXT]->(b)
            ON CREATE SET r.topic_id = row.tid, r.created_at = row.now
            """,
            {"rows": rows},
        )
        valid_iter = iter([True] * len(rows))
        return [
            next(valid_iter) if prev_id and next_id and prev_id != next_id else False
            for prev_id, next_id, _topic_id in edges
        ]

    async def get_related_edges(
        self,
        node_ids: List[str],
        *,
        max_edges: int = 2000,
        relation_types: Optional[List[str]] = None,
        include_temporal: bool = True,
        include_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return RELATED_TO (plus optional TEMPORAL_NEXT) edges with one end inside node_ids. Semantics in the base class."""
        if not node_ids:
            return []

        # add_edge creates bidirectional edges, so querying the forward direction covers both; undirected dedup happens on the Python side
        rtype_filter = ""
        params: Dict[str, Any] = {"ids": list(node_ids), "lim": int(max_edges) * 2}
        if relation_types:
            rtype_filter = "AND r.relation_type IN $rtypes"
            params["rtypes"] = list(relation_types)
        source_filter = ""
        if include_sources:
            source_filter += " AND b.source IN $include_sources"
        if exclude_sources:
            source_filter += " AND NOT b.source IN $exclude_sources"
        params["include_sources"] = list(include_sources or [])
        params["exclude_sources"] = list(exclude_sources or [])

        out: List[Dict[str, Any]] = []
        seen_pairs: set = set()
        try:
            rows = await self._run(
                f"""
                MATCH (a:Memory)-[r:RELATED_TO]->(b:Memory)
                WHERE a.node_id IN $ids {rtype_filter} {source_filter}
                RETURN a.node_id AS source_id, b.node_id AS target_id,
                       r.relation_type AS relation_type, r.weight AS weight,
                       b.status AS target_status, b.layer AS target_layer
                LIMIT $lim
                """,
                params,
            )
            for row in rows:
                src, tgt = row.get("source_id"), row.get("target_id")
                if not src or not tgt or src == tgt:
                    continue
                # The dedup key must include relation_type: add_edge creates bidirectional edges, so
                # A→B / B→A must fold into one; but one node pair **can carry several relations**
                # (both co_occurs and causes), which are two distinct edges and must not be merged.
                pair = (frozenset((src, tgt)), row.get("relation_type") or "related")
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                w = row.get("weight")
                out.append({
                    "source_id": src,
                    "target_id": tgt,
                    "relation_type": row.get("relation_type") or "related",
                    "weight": float(w) if w is not None else 1.0,
                    "target_status": row.get("target_status"),
                    "target_layer": row.get("target_layer"),
                    "directed": False,
                })
                if len(out) >= max_edges:
                    break
        except Exception as e:
            logger.debug(f"get_related_edges: {e}")

        # ── Temporal spine: TEMPORAL_NEXT is directed (source = earlier, target = later) ──
        # Take the edge when either end lies in node_ids (both directions queried), otherwise "what happened before X" never works.
        if include_temporal and (
            not relation_types or "temporal_next" in relation_types
        ):
            try:
                # Split the two endpoint lookups so Neo4j can use the unique
                # Memory.node_id index for each branch.  The previous OR
                # predicate caused a relationship scan on larger graphs.
                rows = await self._run(
                    f"""
                    CALL {{
                        WITH $ids AS ids
                        MATCH (a:Memory)-[r:TEMPORAL_NEXT]->(b:Memory)
                        WHERE a.node_id IN ids
                          {source_filter}
                        RETURN a.node_id AS source_id, b.node_id AS target_id,
                               b.status AS target_status, b.layer AS target_layer,
                               a.status AS source_status, a.layer AS source_layer
                        UNION ALL
                        WITH $ids AS ids
                        MATCH (a:Memory)-[r:TEMPORAL_NEXT]->(b:Memory)
                        WHERE b.node_id IN ids
                          {source_filter.replace('b.source', 'a.source')}
                        RETURN a.node_id AS source_id, b.node_id AS target_id,
                               b.status AS target_status, b.layer AS target_layer,
                               a.status AS source_status, a.layer AS source_layer
                    }}
                    RETURN DISTINCT source_id, target_id,
                           target_status, target_layer, source_status, source_layer
                    LIMIT $lim
                    """,
                    {"ids": list(node_ids), "lim": int(max_edges),
                     "include_sources": list(include_sources or []),
                     "exclude_sources": list(exclude_sources or [])},
                )
                for row in rows:
                    src, tgt = row.get("source_id"), row.get("target_id")
                    if not src or not tgt or src == tgt:
                        continue
                    pair = (frozenset((src, tgt)), "temporal_next")
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    out.append({
                        "source_id": src,
                        "target_id": tgt,
                        "relation_type": "temporal_next",
                        "weight": 1.0,
                        "target_status": row.get("target_status"),
                        "target_layer": row.get("target_layer"),
                        "source_status": row.get("source_status"),
                        "source_layer": row.get("source_layer"),
                        "directed": True,
                    })
                    if len(out) >= max_edges:
                        break
            except Exception as e:
                logger.debug(f"get_related_edges(temporal): {e}")

        return out

    async def expand_from_anchors(
        self,
        anchor_ids: List[str],
        hop: int = 1,
        max_nodes: int = 50,
    ) -> List[Dict[str, Any]]:
        """Expand from anchor nodes along graph edges for N hops."""
        if not anchor_ids:
            return []

        expanded = []
        seen = set(anchor_ids)

        for anchor_id in anchor_ids:
            try:
                # Outgoing edges
                rows = await self._run(
                    """
                    MATCH (a:Memory {node_id: $aid})-[r]->(b:Memory)
                    WHERE b.status = 'active'
                    RETURN b.node_id AS node_id, b.content AS content,
                           b.layer AS layer, b.confidence AS confidence,
                           type(r) AS edge_type
                    LIMIT $lim
                    """,
                    {"aid": anchor_id, "lim": max_nodes},
                )
                for row in rows:
                    nid = row["node_id"]
                    if nid not in seen:
                        seen.add(nid)
                        expanded.append({
                            "node_id": nid,
                            "content": row["content"],
                            "layer": row["layer"],
                            "confidence": row["confidence"],
                            "edge_type": row["edge_type"],
                            "from_anchor": anchor_id,
                            "source": "graph_expand",
                        })

                # Incoming edges
                rows2 = await self._run(
                    """
                    MATCH (b:Memory)-[r]->(a:Memory {node_id: $aid})
                    WHERE b.status = 'active'
                    RETURN b.node_id AS node_id, b.content AS content,
                           b.layer AS layer, b.confidence AS confidence,
                           type(r) AS edge_type
                    LIMIT $lim
                    """,
                    {"aid": anchor_id, "lim": max_nodes},
                )
                for row in rows2:
                    nid = row["node_id"]
                    if nid not in seen:
                        seen.add(nid)
                        expanded.append({
                            "node_id": nid,
                            "content": row["content"],
                            "layer": row["layer"],
                            "confidence": row["confidence"],
                            "edge_type": row["edge_type"],
                            "from_anchor": anchor_id,
                            "source": "graph_expand",
                        })

            except Exception as e:
                logger.debug(f"expand_from_anchors for {anchor_id}: {e}")

            if len(expanded) >= max_nodes:
                break

        return expanded[:max_nodes]

    async def expand_with_tags(
        self,
        anchor_ids: List[str],
        hop: int = 2,
        max_nodes: int = 500,
        isolation_key: str = "",
    ) -> List[Dict[str, Any]]:
        """BFS expansion: RELATED_TO + CROSS_ABSTRACTS_TO + TAGGED_WITH, N hops.

        Every hop queries at the same time:
        1. Memory -[RELATED_TO]- Memory (both directions)
        2. Memory -[CROSS_ABSTRACTS_TO]- Memory (both directions, basic↔abstract)
        3. Memory -[TAGGED_WITH]-> Topic <-[TAGGED_WITH]- Memory (bridged through Topic)

        Every result must be a Memory node (no Topic/VdbRef or other intermediate nodes).
        When isolation_key is non-empty, only nodes with the same isolation_key are expanded.
        """
        if not anchor_ids:
            return []

        seen = set(anchor_ids)
        all_expanded = []
        frontier = list(anchor_ids)

        # isolation_key filter condition
        ik_filter = "AND m.isolation_key = $ik" if isolation_key else ""
        params_base = {"ik": isolation_key} if isolation_key else {}

        for current_hop in range(1, hop + 1):
            if not frontier or len(all_expanded) >= max_nodes:
                break

            next_frontier = []

            # --- RELATED_TO + CROSS_ABSTRACTS_TO neighbours (both directions, merged query) ---
            try:
                rows = await self._run(
                    f"""
                    UNWIND $ids AS aid
                    MATCH (a:Memory {{node_id: aid}})-[:RELATED_TO|CROSS_ABSTRACTS_TO]-(m:Memory)
                    WHERE m.status = 'active' {ik_filter}
                    RETURN DISTINCT m.node_id AS node_id, m.content AS content,
                           m.layer AS layer, m.confidence AS confidence,
                           m.custom_json AS custom_json
                    """,
                    {"ids": frontier, **params_base},
                )
                for row in rows:
                    nid = row["node_id"]
                    if nid not in seen:
                        seen.add(nid)
                        all_expanded.append({
                            "node_id": nid,
                            "content": row["content"],
                            "layer": row["layer"],
                            "confidence": row["confidence"],
                            "custom_json": row.get("custom_json", ""),
                            "hop": current_hop,
                            "source": "related_to",
                        })
                        next_frontier.append(nid)
            except Exception as e:
                logger.debug(f"expand_with_tags RELATED_TO|CROSS_ABSTRACTS_TO hop {current_hop}: {e}")

            # --- TAGGED_WITH bridging (Memory→Topic→Memory) ---
            try:
                rows = await self._run(
                    f"""
                    UNWIND $ids AS aid
                    MATCH (a:Memory {{node_id: aid}})-[:TAGGED_WITH]->(t:Topic)<-[:TAGGED_WITH]-(m:Memory)
                    WHERE m.status = 'active' AND m.node_id <> aid {ik_filter}
                    RETURN DISTINCT m.node_id AS node_id, m.content AS content,
                           m.layer AS layer, m.confidence AS confidence,
                           m.custom_json AS custom_json, t.name AS via_tag
                    """,
                    {"ids": frontier, **params_base},
                )
                for row in rows:
                    nid = row["node_id"]
                    if nid not in seen:
                        seen.add(nid)
                        all_expanded.append({
                            "node_id": nid,
                            "content": row["content"],
                            "layer": row["layer"],
                            "confidence": row["confidence"],
                            "custom_json": row.get("custom_json", ""),
                            "hop": current_hop,
                            "source": f"tag:{row['via_tag']}",
                        })
                        next_frontier.append(nid)
            except Exception as e:
                logger.debug(f"expand_with_tags TAGGED_WITH hop {current_hop}: {e}")

            frontier = next_frontier

        return all_expanded[:max_nodes]

    # ================================================================
    # Vector retrieval (dual HNSW indexes)
    # ================================================================

    async def vector_search(
        self,
        query_embedding: List[float],
        isolation_key: str,
        layers: Optional[List[str]] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        user_id: str = "",
        include_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """V_con vector retrieval over Graph Memory nodes — **Pre-Filter mode**

        Uses the Neo4j 5.18+ ``vector.similarity.cosine()`` function:
        MATCH filtering by isolation_key / layer / status first,
        then vector similarity is computed on the filtered subset.

        Compared with the old ``db.index.vector.queryNodes()`` approach:
        - queryNodes is a global HNSW top-K → post-filter, giving very low recall for sparse users
        - this approach is pre-filter → brute-force cosine, guaranteeing 100% recall
        - the performance gap is <10ms (per-user node counts are usually <100), entirely acceptable

        When isolation_key is empty but user_id is not, a cross-agent search runs
        (matching every node whose isolation_key starts with user_id + "::").
        """
        if self._driver is None or not query_embedding:
            return []

        try:
            # layer filter
            layer_filter = ""
            if layers:
                layer_list = ", ".join(f"'{l}'" for l in layers)
                layer_filter = f"AND m.layer IN [{layer_list}]"

            # Isolation filter: exact ik or user_id prefix
            if isolation_key:
                ik_filter = "m.isolation_key = $ik"
            elif user_id:
                ik_filter = "m.isolation_key STARTS WITH $ik"
                isolation_key = user_id + "::"
            else:
                return []

            source_filter = ""
            if include_sources:
                source_filter += " AND m.source IN $include_sources"
            if exclude_sources:
                source_filter += " AND NOT m.source IN $exclude_sources"

            rows = await self._run(
                f"""
                MATCH (m:Memory)
                WHERE {ik_filter} AND m.status = 'active' {source_filter}
                    {layer_filter}
                    AND m.embedding IS NOT NULL
                WITH m, vector.similarity.cosine(m.embedding, $vec) AS score
                WHERE score >= $threshold
                RETURN m.node_id AS node_id, m.content AS content,
                       m.layer AS layer, m.confidence AS confidence,
                       score, m.custom_json AS custom_json, m.source AS memory_source
                ORDER BY score DESC
                LIMIT $lim
                """,
                {
                    "vec": query_embedding,
                    "ik": isolation_key,
                    "threshold": (1.0 + score_threshold) / 2.0,  # raw cosine → Neo4j (1+cos)/2 scale
                    "lim": limit,
                    "include_sources": list(include_sources or []),
                    "exclude_sources": list(exclude_sources or []),
                },
            )

            # Neo4j vector.similarity.cosine() returns (1+cos)/2 ∈ [0,1],
            # converted back to raw cosine similarity [-1,1] (same scale as Qdrant):
            #   raw_cosine = 2 * neo4j_score - 1
            return [
                {
                    "node_id": r["node_id"],
                    "content": r["content"],
                    "layer": r["layer"],
                    "confidence": r["confidence"],
                    "score": 2.0 * r["score"] - 1.0,
                    "custom_json": r["custom_json"],
                    "retrieval_source": "graph_vector_search",
                    "memory_source": r.get("memory_source") or "default",
                }
                for r in rows
            ]

        except Exception as e:
            logger.warning(f"vector_search failed: {e}")
            return []

    async def beh_vector_search(
        self,
        query_beh_embedding: List[float],
        isolation_key: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """V_beh vector retrieval over L6 basic nodes — **Pre-Filter mode**

        Uses ``vector.similarity.cosine()`` on beh_embedding for an exact
        vector search after the isolation_key pre-filter.
        """
        if self._driver is None or not query_beh_embedding:
            return []

        try:
            rows = await self._run(
                """
                MATCH (m:Memory {isolation_key: $ik})
                WHERE m.status = 'active' AND m.layer = 'l6_schema'
                    AND m.beh_embedding IS NOT NULL
                WITH m, vector.similarity.cosine(m.beh_embedding, $vec) AS score
                RETURN m.node_id AS node_id, m.content AS content,
                       m.embedding AS embedding,
                       score AS beh_score
                ORDER BY score DESC
                LIMIT $lim
                """,
                {
                    "vec": query_beh_embedding,
                    "ik": isolation_key,
                    "lim": limit,
                },
            )

            # beh_score is Neo4j's (1+cos)/2 ∈ [0,1]
            # distance = 1 - similarity, keeping the 0~1 scale
            return [
                {
                    "node_id": r["node_id"],
                    "content": r["content"],
                    "embedding": r["embedding"],
                    "beh_distance": 1.0 - r["beh_score"],
                }
                for r in rows
            ]

        except Exception as e:
            logger.warning(f"beh_vector_search failed: {e}")
            return []

    async def update_embedding(
        self,
        node_id: str,
        embedding: Optional[List[float]] = None,
        beh_embedding: Optional[List[float]] = None,
    ) -> bool:
        """Update the embedding / beh_embedding vector properties of a node

        Neo4j allows SET on properties covered by a vector index, and the index updates itself.
        """
        if self._driver is None:
            return False
        try:
            sets = []
            params: Dict[str, Any] = {"nid": node_id}
            if embedding is not None:
                sets.append("m.embedding = $emb")
                params["emb"] = embedding
            if beh_embedding is not None:
                sets.append("m.beh_embedding = $beh")
                params["beh"] = beh_embedding
            if not sets:
                return True
            set_str = ", ".join(sets)
            await self._run_write(
                f"MATCH (m:Memory {{node_id: $nid}}) SET {set_str}",
                params,
            )
            return True
        except Exception as e:
            logger.warning(f"update_embedding failed for {node_id}: {e}")
            return False

    # ================================================================
    # Cross-domain induction (Cross-Domain)
    # ================================================================

    async def add_cross_abstracts_to(self, basic_id: str, core_id: str) -> bool:
        """Add a CROSS_ABSTRACTS_TO edge: L6 basic → L6 core (directed)."""
        if self._driver is None:
            return False
        try:
            now = datetime.now().isoformat()
            await self._run_write(
                """
                MATCH (a:Memory {node_id: $src}), (b:Memory {node_id: $tgt})
                CREATE (a)-[:CROSS_ABSTRACTS_TO {created_at: $now}]->(b)
                """,
                {"src": basic_id, "tgt": core_id, "now": now},
            )
            return True
        except Exception as e:
            logger.warning(f"add_cross_abstracts_to failed ({basic_id})->({core_id}): {e}")
            return False

    async def find_cores_from_basics(
        self,
        basic_ids: List[str],
        max_hops: int = 2,
    ) -> List[Dict[str, Any]]:
        """From an L6 node, follow CROSS_ABSTRACTS_TO to higher-level nodes."""
        if self._driver is None or not basic_ids:
            return []

        try:
            rows = await self._run(
                f"""
                MATCH (start:Memory)
                      -[:CROSS_ABSTRACTS_TO*1..{int(max_hops)}]->
                      (core:Memory)
                WHERE start.node_id IN $nids
                  AND core.status = 'active'
                RETURN DISTINCT core.node_id AS node_id,
                       core.content AS content,
                       core.confidence AS confidence
                """,
                {"nids": basic_ids},
            )
            return [
                {
                    "node_id": r["node_id"],
                    "content": r["content"],
                    "confidence": r["confidence"],
                }
                for r in rows
            ]
        except Exception as e:
            logger.debug(f"find_cores_from_basics failed: {e}")
            return []

    async def get_cross_abstracts_targets(self, basic_id: str) -> List[str]:
        """Return the CROSS_ABSTRACTS_TO → core node_ids already attached to one L6 basic node."""
        if self._driver is None:
            return []
        try:
            rows = await self._run(
                """
                MATCH (a:Memory {node_id: $nid})-[:CROSS_ABSTRACTS_TO]->(b:Memory)
                RETURN b.node_id AS node_id
                """,
                {"nid": basic_id},
            )
            return [r["node_id"] for r in rows]
        except Exception as e:
            logger.debug(f"get_cross_abstracts_targets for {basic_id}: {e}")
            return []

    # ================================================================
    # Internal helpers
    # ================================================================

    @staticmethod
    def _record_to_memory_node(node_data: Any) -> MemoryNode:
        """Convert a Neo4j Node into a MemoryNode."""
        # Node objects returned by the Neo4j driver support dict-like access
        if hasattr(node_data, 'items'):
            d = dict(node_data)
        elif isinstance(node_data, dict):
            d = node_data
        else:
            d = {}

        # Parse JSON fields
        evidence_chain = d.get("evidence_chain", "[]")
        if isinstance(evidence_chain, str):
            try:
                evidence_chain = json.loads(evidence_chain)
            except (json.JSONDecodeError, TypeError):
                evidence_chain = []

        custom = d.get("custom_json", "{}")
        if isinstance(custom, str):
            try:
                custom = json.loads(custom)
            except (json.JSONDecodeError, TypeError):
                custom = {}

        tags = d.get("tags", "[]")
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except (json.JSONDecodeError, TypeError):
                tags = []

        # Time fields - Neo4j may return an ISO string or a datetime
        def _parse_dt(val):
            if val is None:
                return None
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                try:
                    return datetime.fromisoformat(val)
                except (ValueError, TypeError):
                    return None
            # neo4j DateTime object
            if hasattr(val, 'to_native'):
                return val.to_native()
            return None

        return MemoryNode.from_dict({
            "node_id": d.get("node_id", ""),
            "user_id": d.get("user_id", ""),
            "agent_id": d.get("agent_id", ""),
            "source": d.get("source", "default") or "default",
            "layer": d.get("layer", "l1_raw"),
            "content": d.get("content", ""),
            "status": d.get("status", "active"),
            "version": d.get("version", 1),
            "confidence": d.get("confidence", 1.0),
            "access_count": d.get("access_count", 0),
            "created_at": _parse_dt(d.get("created_at")),
            "valid_from": _parse_dt(d.get("valid_from")),
            "valid_until": _parse_dt(d.get("valid_until")),
            "last_accessed_at": _parse_dt(d.get("last_accessed_at")),
            "superseded_by_id": d.get("superseded_by_id"),
            "evidence_chain": evidence_chain,
            "embedding": list(d.get("embedding") or []),
            "custom": custom,
            "tags": tags,
        })
