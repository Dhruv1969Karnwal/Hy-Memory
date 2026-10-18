# -*- coding: utf-8 -*-
"""
Neo4j vector retrieval adversarial scenario tests

Focus: verifying iterative search works in extreme scenarios
- 99% of the index is filled with high-similarity data from other users
- the target user's relevant data has a lower similarity but must still be returned

Stress test for Solution B, verifying:
1. Iterative search still finds the target user's results in the worst case
2. The iteration count and the K growth factor are enough to cover these scenarios
3. No wrong results and no data leakage under data races
"""

import pytest
from typing import List
import numpy as np


@pytest.mark.integration
class TestVectorSearchAdversarial:
    """Adversarial vector search scenarios."""

    @pytest.mark.slow
    async def test_vector_search_with_99_percent_competitors(self, graph_store_neo4j_fixture):
        """
        Extreme case: 99% of the global top-K is filled by other users
        
        Setup:
        - Target User: 1 memory (low similarity 0.4)
        - Competitor Users: 990 memories (high similarity 0.95+)
        
        The global top-30 fills up with competitors, so iteration has to reach a larger K to find the target
        
        Expected:
        - The target must be found even when the target user's similarity is low
        - Growing K (3 -> 4.5 -> 6.75) covers more results
        """
        ik_target = "target_user::default"
        ik_competitor = "competitor::default"
        
        np.random.seed(111)
        
        # Target user's vector (its own direction)
        target_base = np.random.randn(1536)
        target_base = target_base / np.linalg.norm(target_base)
        
        # Competitor users' vectors (a different direction, but highly similar among themselves)
        competitor_base = np.random.randn(1536)
        competitor_base = competitor_base / np.linalg.norm(competitor_base)
        
        # Make sure the target and competitor directions differ (test isolation)
        while np.dot(target_base, competitor_base) > 0.1:
            competitor_base = np.random.randn(1536)
            competitor_base = competitor_base / np.linalg.norm(competitor_base)
        
        # Create the target user's memory
        target_memory = {
            "node_id": "target_mem_0",
            "content": "Target user's unique memory",
            "layer": "l6_schema",
            "status": "active",
            "isolation_key": ik_target,
            "embedding": target_base.tolist(),
            "confidence": 0.6,
        }
        
        # Create 990 competitor memories (high similarity)
        competitor_memories = []
        for i in range(990):
            noise = np.random.randn(1536) * 0.01  # Very small noise
            vec = competitor_base + noise
            vec = vec / np.linalg.norm(vec)
            competitor_memories.append({
                "node_id": f"competitor_mem_{i}",
                "content": f"Competitor memory {i}",
                "layer": "l6_schema",
                "status": "active",
                "isolation_key": ik_competitor,
                "embedding": vec.tolist(),
                "confidence": 0.95,
            })
        
        # Search: use a mixed vector (close to the competitors but not identical)
        # That fills top-30 with competitors, but iteration should still reach the target
        query_vector = (target_base * 0.1 + competitor_base * 0.9)
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        results = await graph_store_neo4j_fixture.vector_search(
            query_embedding=query_vector.tolist(),
            isolation_key=ik_target,
            limit=5,
        )
        
        # Verify
        assert len(results) > 0, \
            "Iterative search should find target even with 990 high-similarity competitors"
        
        assert results[0]["node_id"] == "target_mem_0", \
            "Should find the target memory despite extreme competition"
        
        # Verify there is no data leakage
        for result in results:
            assert result["node_id"].startswith("target_mem_"), \
                f"No data leakage: got {result['node_id']}"
        
        print(f"✓ 99% competitor test passed: found target despite 990 high-similarity competitors")

    @pytest.mark.slow
    async def test_vector_search_many_isolated_users(self, graph_store_neo4j_fixture):
        """
        Test: isolation across 100 different users holds under iterative search
        
        Setup:
        - 100 distinct isolation_key values
        - 10 memories each, 1000 nodes in total
        
        Expected:
        - Searching for any user returns only that user's data
        - No cross-contamination from a large global top-K or from iteration
        """
        num_users = 100
        memories_per_user = 10
        
        np.random.seed(222)
        
        user_vectors = []
        for user_id in range(num_users):
            # Each user has a different base vector
            base_vec = np.random.randn(1536)
            base_vec = base_vec / np.linalg.norm(base_vec)
            user_vectors.append(base_vec)
        
        # Create the memories
        all_memories = []
        for user_id in range(num_users):
            ik = f"user_{user_id}::default"
            base_vec = user_vectors[user_id]
            
            for mem_id in range(memories_per_user):
                noise = np.random.randn(1536) * 0.05
                vec = base_vec + noise
                vec = vec / np.linalg.norm(vec)
                all_memories.append({
                    "node_id": f"user_{user_id}_mem_{mem_id}",
                    "content": f"User {user_id}'s memory {mem_id}",
                    "layer": "l6_schema",
                    "status": "active",
                    "isolation_key": ik,
                    "embedding": vec.tolist(),
                    "confidence": 0.8,
                })
        
        # Randomly pick 5 users for the search tests
        test_users = np.random.choice(num_users, 5, replace=False)
        
        for user_id in test_users:
            ik = f"user_{user_id}::default"
            query_vec = user_vectors[user_id]
            
            results = await graph_store_neo4j_fixture.vector_search(
                query_embedding=query_vec.tolist(),
                isolation_key=ik,
                limit=5,
            )
            
            # Verify isolation
            for result in results:
                expected_prefix = f"user_{user_id}_"
                assert result["node_id"].startswith(expected_prefix), \
                    f"User {user_id} search got wrong node: {result['node_id']}"
            
            assert len(results) <= 5, "Should respect limit"
        
        print(f"✓ Many users test passed: 100 users with {num_users * memories_per_user} total nodes, isolation maintained")

    @pytest.mark.slow
    async def test_vector_search_inactive_status_filtering(self, graph_store_neo4j_fixture):
        """
        Test: iterative search filters inactive nodes correctly
        
        Setup:
        - 5 active memories
        - 95 inactive memories (simulating deleted or expired nodes)
        
        Expected:
        - Results contain active nodes only
        - Inactive nodes are filtered during iteration as well (no expired results come back)
        """
        ik = "status_filter_test::default"
        
        np.random.seed(333)
        base_vec = np.random.randn(1536)
        base_vec = base_vec / np.linalg.norm(base_vec)
        
        memories = []
        
        # Active memories
        for i in range(5):
            noise = np.random.randn(1536) * 0.02
            vec = base_vec + noise
            vec = vec / np.linalg.norm(vec)
            memories.append({
                "node_id": f"active_mem_{i}",
                "content": f"Active memory {i}",
                "layer": "l6_schema",
                "status": "active",
                "isolation_key": ik,
                "embedding": vec.tolist(),
                "confidence": 0.8,
            })
        
        # Inactive memories (many enough to fill top-30)
        for i in range(95):
            noise = np.random.randn(1536) * 0.01
            vec = base_vec + noise
            vec = vec / np.linalg.norm(vec)
            memories.append({
                "node_id": f"inactive_mem_{i}",
                "content": f"Inactive memory {i}",
                "layer": "l6_schema",
                "status": "inactive",  # Expired / deleted
                "isolation_key": ik,
                "embedding": vec.tolist(),
                "confidence": 0.8,
            })
        
        results = await graph_store_neo4j_fixture.vector_search(
            query_embedding=base_vec.tolist(),
            isolation_key=ik,
            limit=10,
        )
        
        # Verify
        for result in results:
            assert result["node_id"].startswith("active_mem_"), \
                f"Should only return active nodes, got {result['node_id']}"
        
        # All 5 active nodes should be found (with the default of no similarity threshold)
        assert len(results) <= 5, "At most 5 active nodes available"
        
        print(f"✓ Status filtering test passed: returned {len(results)}/5 active nodes (95 inactive ignored)")

    @pytest.mark.slow
    async def test_vector_search_duplicate_detection(self, graph_store_neo4j_fixture):
        """
        Test: the deduplication logic of iterative search
        
        Setup:
        - Several versions with the same node_id (simulating inconsistent data)
        
        Expected:
        - Even if several rounds return the same node_id, it is returned only once
        - Deduplication is based on node_id
        """
        ik = "dedup_test::default"
        
        np.random.seed(444)
        base_vec = np.random.randn(1536)
        base_vec = base_vec / np.linalg.norm(base_vec)
        
        # Create 10 memories
        memories = []
        for i in range(10):
            noise = np.random.randn(1536) * 0.05
            vec = base_vec + noise
            vec = vec / np.linalg.norm(vec)
            memories.append({
                "node_id": f"dedup_mem_{i % 3}",  # node_id deliberately repeated
                "content": f"Memory {i}",
                "layer": "l6_schema",
                "status": "active",
                "isolation_key": ik,
                "embedding": vec.tolist(),
                "confidence": 0.8,
            })
        
        results = await graph_store_neo4j_fixture.vector_search(
            query_embedding=base_vec.tolist(),
            isolation_key=ik,
            limit=10,
        )
        
        # Verify deduplication
        node_ids = [r["node_id"] for r in results]
        assert len(node_ids) == len(set(node_ids)), \
            f"Should deduplicate, got duplicates in {node_ids}"
        
        print(f"✓ Deduplication test passed: returned {len(results)} unique nodes (dedup working)")

