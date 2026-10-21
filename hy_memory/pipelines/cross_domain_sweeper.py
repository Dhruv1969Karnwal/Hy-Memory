"""
Cross-Domain Schema Induction Sweeper

Cross-domain breakthrough induction: discovers deep behavioural isomorphisms in the user's L6 basic Schemas and produces L6 core Schemas.

The algorithm in three steps:
  Step 1: behavioural dimension lift — call the LLM on L6 basic to produce a behavioural-psychology abstraction → embed → V_beh
  Step 2: collision scan — matrix cosine collisions + Union-Find clustering + Core merge
  Step 3: LLM breakthrough induction — call the LLM on each collision cluster to produce an L6 core

Trigger point: at the end of the evolve() flow, once the System2 Agent finishes
Trigger condition: the user has >= MIN_BASIC_FOR_SWEEP L6 basic nodes (default 5)

Trace design:
  - Shares one request_id with the System2 Agent, so a write request links the whole pipeline
  - Every trace carries memory_ids, so any memory_id can be traced back through all the steps it passed
  - pipeline_log steps: SWEEPER_SUMMARY / SWEEPER_BEH_EMBED / SWEEPER_COLLISION / SWEEPER_CORE_CREATE / SWEEPER_CORE_MERGE
  - memory_operation ops: SWEEPER_BEH_EMBED / SWEEPER_CORE_CREATE / SWEEPER_CROSS_EDGE / SWEEPER_CORE_MERGE
"""

import os
import json
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from ..models.memory import MemoryLayer, MemoryNode, MemoryStatus, SourceType
from ..data.graph_store_base import GraphStoreBase
from ..core.embed_service import EmbedService
from ..utils.source_filter import merge_sources

logger = logging.getLogger(__name__)

# ====================================================================
# Configuration
# ====================================================================

_CROSS_DOMAIN_ENABLE = os.getenv("MEMORY_CROSS_DOMAIN_ENABLE", "true").lower() == "true"
_MIN_BASIC_FOR_SWEEP = int(os.getenv("MEMORY_CROSS_DOMAIN_MIN_BASICS", "5"))
_BEH_THRESHOLD = float(os.getenv("MEMORY_CROSS_DOMAIN_BEH_THRESHOLD", "0.70"))
_CON_THRESHOLD = float(os.getenv("MEMORY_CROSS_DOMAIN_CON_THRESHOLD", "0.60"))
_ANN_TOPK = int(os.getenv("MEMORY_CROSS_DOMAIN_ANN_TOPK", "10"))

# ====================================================================
# LLM Prompts
# ====================================================================

_BEHAVIOR_ABSTRACTION_PROMPT = """Task: You are a behavioral psychologist. Abstract the following schema into a pure psychological/behavioral description. Strip ALL domain-specific nouns and scenarios — output ONLY the underlying behavioral style and psychological motivation.

CRITICAL LANGUAGE RULE: Your output MUST be in the SAME language as the input schema below. If the input is in English, you MUST output in English. If the input is in Chinese, you MUST output in Chinese. Violating this rule invalidates the output.

Input schema:
{content}

Output (strict JSON, same language as input):
{{"abstraction_for_embedding": "...pure behavioral description, same language as input..."}}"""

_CROSS_DOMAIN_INDUCTION_PROMPT = """Task: You are a deep pattern analyst. The system has detected structural resonance among the following schemas from different areas of the user's life. On the surface they appear unrelated, but their underlying behavioral logic is strikingly similar.

Your job: synthesize a HIGHER-ORDER pattern that explains WHY these behaviors co-occur — something the user themselves may not be consciously aware of. Think in terms of:
- Deep cognitive style (how they process information and make decisions)
- Core psychological need (what drives them at a fundamental level)
- Hidden mental model (the implicit belief system behind these behaviors)

Be insightful. Be precise. Quality over quantity — only output a synthesis if the connection is genuinely compelling and logically airtight. If the evidence is weak or the connection is superficial, output null.

LANGUAGE RULE: Your output (core_pattern, reasoning) MUST be in the SAME language as the input schemas below. If the schemas are in Chinese, output in Chinese. If in English, output in English. Do NOT translate or switch language.

Input schemas:
{patterns}

Output (strict JSON, or null if no compelling synthesis):
{{"core_pattern": "...one sentence describing the higher-order pattern, same language as input...", "reasoning": "...why these schemas are connected at a deep level, same language as input...", "confidence": 0.85}}"""


# ====================================================================
# Union-Find
# ====================================================================

class UnionFind:
    """Union-Find (Disjoint Set Union) for merging collision pairs."""

    def __init__(self):
        self._parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self._parent[ry] = rx

    def clusters(self) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {}
        for x in self._parent:
            root = self.find(x)
            groups.setdefault(root, []).append(x)
        return groups


# ====================================================================
# CrossDomainSweeper
# ====================================================================

class CrossDomainSweeper:
    """Cross-domain breakthrough induction scanner."""

    def __init__(
        self,
        graph_store: GraphStoreBase,
        embed_service: EmbedService,
        llm_call,  # async callable(prompt: str) -> str
        user_id: str,
        agent_id: str,
        request_id: str = "",
        cache: Any = None,
    ):
        self.graph_store = graph_store
        self.embed_service = embed_service
        self.llm_call = llm_call
        self.user_id = user_id
        self.agent_id = agent_id
        self.request_id = request_id
        self._cache = cache
        self._isolation_key = MemoryNode.build_isolation_key(user_id, agent_id)

    # ================================================================
    # Trace helpers
    # ================================================================

    async def _log_pipeline(
        self, step: str, prompt: str = "", response: str = "",
        parsed: Optional[Dict] = None, memory_ids: Optional[List[str]] = None,
        elapsed_ms: float = 0,
    ) -> None:
        """Write pipeline_log (best-effort)."""
        if not self._cache or not self.request_id:
            return
        try:
            await self._cache.store_pipeline_log(
                request_id=self.request_id,
                user_id=self.user_id,
                agent_id=self.agent_id,
                step=step,
                prompt=prompt,
                response=response,
                parsed=json.dumps(parsed or {}, ensure_ascii=False, default=str),
                memory_ids=memory_ids,
                elapsed_ms=elapsed_ms,
            )
        except Exception as e:
            logger.debug(f"[sweeper] store_pipeline_log({step}) failed: {e}")

    async def _log_mem_op(
        self, op: str, memory_id: str, content: str = "",
        reason: str = "",
    ) -> None:
        """Write memory_operation (best-effort)."""
        if not self._cache or not self.request_id:
            return
        try:
            await self._cache.store_memory_operation(
                request_id=self.request_id,
                user_id=self.user_id,
                agent_id=self.agent_id,
                op=op,
                memory_id=memory_id,
                content=content[:500],
                layer="graph",
                reason=reason,
            )
        except Exception as e:
            logger.debug(f"[sweeper] store_memory_operation({op}) failed: {e}")

    # ================================================================
    # Main entry
    # ================================================================

    async def sweep(self) -> Dict[str, Any]:
        """Run one cross-domain scan and return statistics."""
        _t_start = datetime.now()

        if not _CROSS_DOMAIN_ENABLE:
            return {"skipped": True, "reason": "disabled"}

        # Fetch all L6 basic nodes
        all_basics = await self.graph_store.get_all_nodes(
            isolation_key=self._isolation_key,
            layer=MemoryLayer.L6_SCHEMA,
            status=MemoryStatus.ACTIVE,
            limit=500,
        )
        basics = [
            n for n in all_basics
            # Core is no longer excluded — every L6 node takes part in collisions (supports recursive lifting)
        ]

        if not basics:
            return {"skipped": True, "reason": "no L6 basics"}

        logger.info(f"[sweeper] Starting cross-domain sweep: {len(basics)} L6 basics")

        # Step 1: make sure every basic has a beh_embedding (done regardless of how many there are)
        new_beh_count = await self._ensure_beh_embeddings(basics)

        # Step 2: collision scan (only meaningful with enough nodes)
        if len(basics) < _MIN_BASIC_FOR_SWEEP:
            result = {
                "basics_count": len(basics),
                "new_beh_embeddings": new_beh_count,
                "collisions": 0,
                "cores_created": 0,
                "cores_merged": 0,
                "collision_skipped": f"only {len(basics)} basics (need {_MIN_BASIC_FOR_SWEEP} for collision)",
            }
            elapsed_ms = (datetime.now() - _t_start).total_seconds() * 1000
            await self._log_pipeline(
                step="SWEEPER_SUMMARY", parsed=result, elapsed_ms=elapsed_ms,
            )
            return result

        collision_clusters, collision_details = await self._scan_collisions(basics)

        # Step 2 trace: collision details
        if collision_details:
            await self._log_pipeline(
                step="SWEEPER_COLLISION",
                parsed={
                    "total_basics_with_beh": collision_details.get("total_with_beh", 0),
                    "collision_pairs": collision_details.get("pairs", []),
                    "clusters_count": len(collision_clusters),
                },
                memory_ids=[
                    nid for cluster in collision_clusters for nid in cluster
                ],
            )

        if not collision_clusters:
            result = {
                "basics_count": len(basics),
                "new_beh_embeddings": new_beh_count,
                "collisions": 0,
                "cores_created": 0,
                "cores_merged": 0,
            }
            elapsed_ms = (datetime.now() - _t_start).total_seconds() * 1000
            await self._log_pipeline(
                step="SWEEPER_SUMMARY", parsed=result, elapsed_ms=elapsed_ms,
            )
            return result

        # Step 3: LLM induction + graph writes
        cores_created = 0
        cores_merged = 0
        created_memory_ids = []
        for cluster_ids in collision_clusters:
            created, merged, new_ids = await self._process_cluster(cluster_ids, basics)
            cores_created += created
            cores_merged += merged
            created_memory_ids.extend(new_ids)

        elapsed_ms = (datetime.now() - _t_start).total_seconds() * 1000
        result = {
            "basics_count": len(basics),
            "new_beh_embeddings": new_beh_count,
            "collisions": len(collision_clusters),
            "cores_created": cores_created,
            "cores_merged": cores_merged,
            "created_memory_ids": created_memory_ids,
        }

        # SWEEPER_SUMMARY trace
        await self._log_pipeline(
            step="SWEEPER_SUMMARY", parsed=result, elapsed_ms=elapsed_ms,
        )

        logger.info(f"[sweeper] Sweep complete: {result}")
        return result

    # ================================================================
    # Step 1: behavioural dimension lift
    # ================================================================

    async def _ensure_beh_embeddings(self, basics: List[MemoryNode]) -> int:
        """Ensure every L6 basic has a beh_embedding, generating one where it is missing."""
        count = 0
        for node in basics:
            custom = getattr(node, "custom", None) or {}
            if custom.get("_has_beh"):
                continue

            try:
                # LLM behavioural lift
                prompt = _BEHAVIOR_ABSTRACTION_PROMPT.format(content=node.content)
                llm_response = await self.llm_call(prompt)
                # Tolerate LLM responses with JSON wrapped in a markdown code fence
                _resp_text = llm_response.strip()
                if _resp_text.startswith("```"):
                    _lines = _resp_text.split("\n")
                    _lines = _lines[1:]  # Drop ```json
                    if _lines and _lines[-1].strip() == "```":
                        _lines = _lines[:-1]
                    _resp_text = "\n".join(_lines).strip()
                parsed = json.loads(_resp_text)
                abstraction = parsed.get("abstraction_for_embedding", "")
                if not abstraction:
                    continue

                # Embed → V_beh
                beh_embedding, beh_embed_meta = await self.embed_service.embed_queued_with_meta(
                    abstraction
                )

                # Write the Graph node
                await self.graph_store.update_embedding(
                    node.node_id, beh_embedding=beh_embedding
                )

                # Mark as generated + store the behaviour abstraction text
                custom["_has_beh"] = True
                custom["beh_abstraction"] = abstraction
                await self.graph_store.update_node(
                    node.node_id, {"custom_json": json.dumps(custom, ensure_ascii=False)}
                )

                # Trace: one memory_operation per beh_embedding write
                await self._log_mem_op(
                    op="SWEEPER_BEH_EMBED",
                    memory_id=node.node_id,
                    content=json.dumps({
                        "abstraction": abstraction,
                        "original_content": node.content[:200],
                    }, ensure_ascii=False),
                    reason="behavior abstraction → beh_embedding",
                )

                # Trace: pipeline_log records the LLM prompt/response
                await self._log_pipeline(
                    step="SWEEPER_BEH_EMBED",
                    prompt=prompt,
                    response=llm_response,
                    parsed={
                        "abstraction": abstraction,
                        "node_id": node.node_id,
                        "provider_meta": beh_embed_meta,
                    },
                    memory_ids=[node.node_id],
                )

                count += 1
                logger.debug(f"[sweeper] beh_embedding set for {node.node_id[:12]}")
            except Exception as e:
                logger.warning(f"[sweeper] beh_embedding failed for {node.node_id[:12]}: {e}")

        return count

    async def _fetch_basics_with_beh(self) -> List[Dict[str, Any]]:
        """Return all L6 basic nodes that have a beh_embedding (uniform interface, supports Neo4j / Kuzu)."""
        gs = self.graph_store
        results = []

        # Neo4j backend
        if hasattr(gs, '_run'):
            rows = await gs._run(
                "MATCH (m:Memory) WHERE m.isolation_key = $ik "
                "AND m.layer = 'l6_schema' AND m.status = 'active' "
                "AND m.beh_embedding IS NOT NULL "
                "RETURN m.node_id AS node_id, m.content AS content, "
                "m.embedding AS embedding, m.beh_embedding AS beh_embedding",
                {"ik": self._isolation_key},
            )
            for r in rows:
                results.append({
                    "node_id": r["node_id"],
                    "content": r["content"],
                    "embedding": r["embedding"],
                    "beh_embedding": r["beh_embedding"],
                })
            return results

        # Kuzu backend
        if hasattr(gs, '_execute') and hasattr(gs, '_available') and gs._available:
            r = gs._execute(
                "MATCH (m:Memory) WHERE m.isolation_key = $ik "
                "AND m.layer = 'l6_schema' AND m.status = 'active' "
                "AND m.beh_embedding IS NOT NULL "
                "RETURN m.node_id, m.content, m.embedding, m.beh_embedding;",
                {"ik": self._isolation_key},
            )
            while r is not None and r.has_next():
                row = r.get_next()
                results.append({
                    "node_id": row[0],
                    "content": row[1],
                    "embedding": row[2],
                    "beh_embedding": row[3],
                })
            return results

        logger.warning("[sweeper] Unknown graph_store backend, cannot fetch beh_embeddings")
        return []

    # ================================================================
    # Step 2: collision scan
    # ================================================================

    async def _scan_collisions(
        self,
        basics: List[MemoryNode],
    ) -> Tuple[List[List[str]], Dict[str, Any]]:
        """Matrix collision scan + Union-Find clustering. Returns (clusters, details)."""
        import numpy as np

        # Take all L6 basic nodes with a beh_embedding from the Graph (always through the abstract interface)
        all_with_beh = []
        try:
            all_with_beh = await self._fetch_basics_with_beh()
        except Exception as e:
            logger.warning(f"[sweeper] _scan_collisions fetch failed: {e}")
            return [], {}

        N = len(all_with_beh)
        if N < 2:
            return [], {"total_with_beh": N}

        # Build the matrix
        beh_matrix = np.array(
            [np.array(d["beh_embedding"], dtype=np.float32) for d in all_with_beh]
        )
        con_matrix = np.array(
            [np.array(d["embedding"], dtype=np.float32) for d in all_with_beh]
        )

        # L2 normalization
        beh_norms = np.linalg.norm(beh_matrix, axis=1, keepdims=True) + 1e-10
        con_norms = np.linalg.norm(con_matrix, axis=1, keepdims=True) + 1e-10
        beh_normed = beh_matrix / beh_norms
        con_normed = con_matrix / con_norms

        # N×N cosine similarity
        beh_sim = beh_normed @ beh_normed.T
        con_sim = con_normed @ con_normed.T

        # Collision condition
        mask = (beh_sim > _BEH_THRESHOLD) & (con_sim < _CON_THRESHOLD) & ~np.eye(N, dtype=bool)
        hit_indices = np.argwhere(mask)

        if len(hit_indices) == 0:
            logger.debug(f"[sweeper] No collisions found (N={N})")
            return [], {"total_with_beh": N}

        # Preload the CROSS_ABSTRACTS_TO targets of all nodes (to filter pairs that already have a relation)
        node_ids = [d["node_id"] for d in all_with_beh]
        targets_map: Dict[str, set] = {}  # node_id → set of target_ids
        for nid in node_ids:
            try:
                targets = await self.graph_store.get_cross_abstracts_targets(nid)
                targets_map[nid] = set(targets)
            except Exception:
                targets_map[nid] = set()

        # Collect collision pair details (for tracing)
        # Filter out pairs already linked by a direct CROSS_ABSTRACTS_TO relation (A→B or B→A)
        pairs_detail = []
        uf = UnionFind()
        for i, j in hit_indices:
            if i < j:
                nid_i = all_with_beh[i]["node_id"]
                nid_j = all_with_beh[j]["node_id"]

                # Skip pairs with an existing direct edge (prevents parent-child re-collision)
                if nid_j in targets_map.get(nid_i, set()):
                    continue  # A→B already exists
                if nid_i in targets_map.get(nid_j, set()):
                    continue  # B→A already exists
                # Skip pairs sharing the same target (already synthesized)
                shared_targets = targets_map.get(nid_i, set()) & targets_map.get(nid_j, set())
                if shared_targets:
                    continue  # both A→C and B→C exist

                uf.union(nid_i, nid_j)
                pairs_detail.append({
                    "a": nid_i[:12], "b": nid_j[:12],
                    "beh_sim": round(float(beh_sim[i, j]), 4),
                    "con_sim": round(float(con_sim[i, j]), 4),
                })
                logger.debug(
                    f"[sweeper] Collision: {nid_i[:12]} ↔ {nid_j[:12]} "
                    f"(beh={beh_sim[i,j]:.3f}, con={con_sim[i,j]:.3f})"
                )

        clusters = uf.clusters()
        result_clusters = [ids for ids in clusters.values() if len(ids) >= 2]
        logger.info(f"[sweeper] Found {len(result_clusters)} collision clusters from {len(pairs_detail)} pairs")

        details = {
            "total_with_beh": N,
            "pairs": pairs_detail[:50],  # Truncated so the trace does not grow too large
            "beh_threshold": _BEH_THRESHOLD,
            "con_threshold": _CON_THRESHOLD,
        }
        return result_clusters, details

    # ================================================================
    # Step 3: LLM induction + graph writes
    # ================================================================

    async def _process_cluster(
        self,
        cluster_ids: List[str],
        all_basics: List[MemoryNode],
    ) -> Tuple[int, int, List[str]]:
        """Process one collision cluster; returns (cores_created, cores_merged, created_memory_ids)."""
        basic_map = {n.node_id: n for n in all_basics}
        # Look up existing core
        existing_core_ids = set()
        for nid in cluster_ids:
            targets = await self.graph_store.get_cross_abstracts_targets(nid)
            existing_core_ids.update(targets)

        if existing_core_ids:
            # Existing Core → merge (connect the unconnected cluster nodes to the existing core)
            core_id = list(existing_core_ids)[0]
            core_node = await self.graph_store.get_node(core_id)
            merged_ids = []
            for nid in cluster_ids:
                # Prevent self-loops: no edge when the node is itself the core
                if nid == core_id:
                    continue
                targets = await self.graph_store.get_cross_abstracts_targets(nid)
                if core_id not in targets:
                    await self.graph_store.add_cross_abstracts_to(nid, core_id)
                    merged_ids.append(nid)

                    # Trace: one memory_operation per edge (both basic_id and core_id stay traceable)
                    await self._log_mem_op(
                        op="SWEEPER_CORE_MERGE",
                        memory_id=nid,
                        content=json.dumps({"basic_id": nid, "core_id": core_id}, ensure_ascii=False),
                        reason=f"merged basic → existing core {core_id[:12]}",
                    )

            # Trace: pipeline_log records a merge overview
            await self._log_pipeline(
                step="SWEEPER_CORE_MERGE",
                parsed={
                    "core_id": core_id,
                    "merged_basic_ids": merged_ids,
                    "cluster_size": len(cluster_ids),
                },
                memory_ids=[core_id] + merged_ids,
            )
            merged_source = merge_sources([
                getattr(core_node, "source", None),
                *(
                    getattr(basic_map.get(node_id), "source", None)
                    for node_id in cluster_ids
                ),
            ])
            if core_node is not None and merged_source != core_node.source:
                await self.graph_store.update_node(core_id, {"source": merged_source})

            logger.debug(f"[sweeper] Merged {len(merged_ids)} basics → existing core {core_id[:12]}")
            return 0, 1, []

        # Create a new Core
        patterns_text = ""
        for i, nid in enumerate(cluster_ids):
            node = basic_map.get(nid)
            content = node.content if node else nid
            patterns_text += f"Schema {chr(65+i)}: {content}\n"

        try:
            prompt = _CROSS_DOMAIN_INDUCTION_PROMPT.format(patterns=patterns_text.strip())
            llm_response = await self.llm_call(prompt)

            # Handle null response (LLM found no compelling connection)
            if not llm_response or llm_response.strip().lower() == "null":
                logger.info("[sweeper] LLM returned null — no compelling cross-domain pattern")
                return 0, 0, []

            # Tolerate LLM responses with JSON wrapped in a markdown code fence
            _resp_text = llm_response.strip()
            if _resp_text.startswith("```"):
                _lines = _resp_text.split("\n")
                _lines = _lines[1:]
                if _lines and _lines[-1].strip() == "```":
                    _lines = _lines[:-1]
                _resp_text = "\n".join(_lines).strip()
            parsed = json.loads(_resp_text)
            core_content = parsed.get("core_pattern", "")
            reasoning = parsed.get("reasoning", "")
            confidence = parsed.get("confidence", 0.85)

            if not core_content:
                logger.warning("[sweeper] LLM returned empty core_pattern")
                return 0, 0, []

            # Embed core content
            core_embedding, core_embed_meta = await self.embed_service.embed_queued_with_meta(
                core_content
            )

            # Create the L6 core node
            core_id = str(uuid.uuid4())
            core_node = MemoryNode(
                node_id=core_id,
                user_id=self.user_id,
                agent_id=self.agent_id,
                source=merge_sources(
                    getattr(basic_map.get(node_id), "source", None)
                    for node_id in cluster_ids
                ),
                layer=MemoryLayer.L6_SCHEMA,
                content=core_content,
                confidence=confidence,
                status=MemoryStatus.ACTIVE,
                source_type=SourceType.INFERRED,
                memory_at=datetime.now(),
                custom={"schema_type": "abstract", "reasoning": reasoning},
            )
            if core_embedding:
                core_node._graph_embedding = core_embedding
            await self.graph_store.upsert_memory_node(core_node)

            # Trace: core creation recorded as memory_operation
            await self._log_mem_op(
                op="SWEEPER_CORE_CREATE",
                memory_id=core_id,
                content=json.dumps({
                    "core_content": core_content,
                    "confidence": confidence,
                    "source_basics": cluster_ids,
                }, ensure_ascii=False),
                reason=f"cross-domain induction from {len(cluster_ids)} basics",
            )

            # Create CROSS_ABSTRACTS_TO edges + inherit evidence
            inherited_evidence_ids: set = set()
            for nid in cluster_ids:
                await self.graph_store.add_cross_abstracts_to(nid, core_id)

                # Collect the source nodes' evidence VdbRefs (for inheritance)
                try:
                    evidence_refs = await self.graph_store.get_evidence_vdbrefs(nid)
                    for ref in evidence_refs:
                        vid = ref.get("node_id") or ref.get("vdb_id", "")
                        if vid:
                            inherited_evidence_ids.add(vid)
                except Exception as e:
                    logger.debug(f"[sweeper] get_evidence for {nid[:12]} failed: {e}")

                # Trace: one memory_operation per edge (both endpoint node_ids traceable)
                await self._log_mem_op(
                    op="SWEEPER_CROSS_EDGE",
                    memory_id=nid,
                    content=json.dumps({"basic_id": nid, "core_id": core_id}, ensure_ascii=False),
                    reason=f"basic → core {core_id[:12]}",
                )

            # Trace: pipeline_log records the LLM prompt/response plus the full context
            await self._log_pipeline(
                step="SWEEPER_CORE_CREATE",
                prompt=prompt,
                response=llm_response,
                parsed={
                    "core_id": core_id,
                    "core_content": core_content,
                    "confidence": confidence,
                    "source_basic_ids": cluster_ids,
                    "provider_meta": core_embed_meta,
                },
                memory_ids=[core_id] + cluster_ids,
            )

            # Inherit evidence from the source nodes: give the core DERIVED_FROM edges to every source VdbRef
            if inherited_evidence_ids:
                for vid in inherited_evidence_ids:
                    try:
                        await self.graph_store.add_derived_from(core_id, vid)
                    except Exception as e:
                        logger.debug(f"[sweeper] inherit evidence {vid[:12]} → core {core_id[:12]} failed: {e}")
                logger.debug(
                    f"[sweeper] Inherited {len(inherited_evidence_ids)} evidence VdbRefs → core {core_id[:12]}"
                )

            logger.info(
                f"[sweeper] Created L6 core {core_id[:12]}: '{core_content[:60]}' "
                f"from {len(cluster_ids)} basics, evidence_inherited={len(inherited_evidence_ids)}"
            )
            return 1, 0, [core_id]

        except Exception as e:
            logger.error(f"[sweeper] Core creation failed: {e}")
            return 0, 0, []
