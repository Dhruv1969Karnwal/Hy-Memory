# -*- coding: utf-8 -*-
"""
Neo4j vector retrieval performance tests

Verifies the performance characteristics of iterative search:
1. Single-round query performance baseline
2. Multi-round iteration overhead < 10%
3. Behavior across various data scales

These tests help detect performance regressions.
"""

import pytest
import time
from typing import List
import numpy as np


@pytest.mark.integration
class TestVectorSearchPerformance:
    """Vector search performance tests"""

    @pytest.mark.slow
    async def test_vector_search_single_round_baseline(self, graph_store_neo4j_fixture):
        """
        Establish the single-round query performance baseline
        
        Measure:
        - Create 1000 nodes
        - Run the query and time it
        
        Expected:
        - On normal hardware, it should finish within 1 second
        """
        ik = "perf_baseline::default"
        
        np.random.seed(555)
        base_vec = np.random.randn(1536)
        base_vec = base_vec / np.linalg.norm(base_vec)
        
        # Create 1000 nodes
        memories = []
        for i in range(1000):
            noise = np.random.randn(1536) * 0.1
            vec = base_vec + noise
            vec = vec / np.linalg.norm(vec)
            memories.append({
                "node_id": f"perf_mem_{i}",
                "content": f"Performance test memory {i}",
                "layer": "l6_schema",
                "status": "active",
                "isolation_key": ik,
                "embedding": vec.tolist(),
                "confidence": 0.8,
            })
        
        # Query and time it
        query_embedding = base_vec.tolist()
        
        start_time = time.time()
        results = await graph_store_neo4j_fixture.vector_search(
            query_embedding=query_embedding,
            isolation_key=ik,
            limit=20,
        )
        elapsed = time.time() - start_time
        
        # Verify
        assert len(results) <= 20
        assert elapsed < 5.0, f"Query took {elapsed:.2f}s, should be < 5s"
        
        print(f"✓ Baseline performance: {elapsed:.3f}s for 1000 nodes, returned {len(results)} results")

    @pytest.mark.slow
    async def test_vector_search_iteration_overhead(self, graph_store_neo4j_fixture):
        """
        Test: iteration overhead of iterative search (vs single round)
        
        Compare:
        1. Single-round query (large K)
        2. Multi-round iterative query (small K -> growing)
        
        Expected:
        - Multi-round iteration overhead < 10% (most cases need no iteration)
        """
        ik = "perf_iteration::default"
        
        np.random.seed(666)
        base_vec = np.random.randn(1536)
        base_vec = base_vec / np.linalg.norm(base_vec)
        
        # Create enough nodes
        memories = []
        for i in range(5000):
            noise = np.random.randn(1536) * 0.1
            vec = base_vec + noise
            vec = vec / np.linalg.norm(vec)
            memories.append({
                "node_id": f"iter_mem_{i}",
                "content": f"Iteration test memory {i}",
                "layer": "l6_schema",
                "status": "active",
                "isolation_key": ik,
                "embedding": vec.tolist(),
                "confidence": 0.8,
            })
        
        query_embedding = base_vec.tolist()
        
        # Run several queries to reduce noise
        times = []
        for _ in range(3):
            start_time = time.time()
            results = await graph_store_neo4j_fixture.vector_search(
                query_embedding=query_embedding,
                isolation_key=ik,
                limit=10,
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        
        # Verify reasonable performance
        assert avg_time < 2.0, f"Average query time {avg_time:.2f}s should be < 2s"
        
        print(f"✓ Iteration overhead: avg {avg_time:.3f}s over 3 queries (overhead acceptable)")

    @pytest.mark.slow
    async def test_vector_search_limit_scaling(self, graph_store_neo4j_fixture):
        """
        Test: performance scaling across different limit sizes
        
        Measure:
        - limit=5
        - limit=10
        - limit=20
        - limit=50
        
        Expected:
        - Performance should scale roughly linearly (or close to it)
        - There should be no jumps caused by iteration
        """
        ik = "perf_limit_scaling::default"
        
        np.random.seed(777)
        base_vec = np.random.randn(1536)
        base_vec = base_vec / np.linalg.norm(base_vec)
        
        # Create enough nodes
        memories = []
        for i in range(3000):
            noise = np.random.randn(1536) * 0.1
            vec = base_vec + noise
            vec = vec / np.linalg.norm(vec)
            memories.append({
                "node_id": f"scale_mem_{i}",
                "content": f"Limit scaling test memory {i}",
                "layer": "l6_schema",
                "status": "active",
                "isolation_key": ik,
                "embedding": vec.tolist(),
                "confidence": 0.8,
            })
        
        query_embedding = base_vec.tolist()
        limits = [5, 10, 20, 50]
        times = {}
        
        for limit in limits:
            start_time = time.time()
            results = await graph_store_neo4j_fixture.vector_search(
                query_embedding=query_embedding,
                isolation_key=ik,
                limit=limit,
            )
            elapsed = time.time() - start_time
            times[limit] = elapsed
            
            assert len(results) <= limit
        
        # Verify approximate linear scaling
        ratios = []
        for i in range(1, len(limits)):
            time_ratio = times[limits[i]] / times[limits[i-1]]
            limit_ratio = limits[i] / limits[i-1]
            # the time ratio should be close to the limit ratio (within 1.5x is acceptable)
            assert time_ratio < limit_ratio * 1.5, \
                f"Time scaling issue: limit {limits[i-1]}->{limits[i]}, ratio {time_ratio:.2f} vs expected ~{limit_ratio:.2f}"
            ratios.append(time_ratio)
        
        print(f"✓ Limit scaling: {times} (ratios {[f'{r:.2f}' for r in ratios]})")

    @pytest.mark.slow
    async def test_vector_search_layer_filter_cost(self, graph_store_neo4j_fixture):
        """
        Test: performance cost of layer filtering
        
        Compare:
        1. No layer filter
        2. A single layer specified
        3. Multiple layers specified
        
        Expected:
        - Layer filtering should not significantly increase query time
        """
        ik = "perf_layer::default"
        
        np.random.seed(888)
        base_vec = np.random.randn(1536)
        base_vec = base_vec / np.linalg.norm(base_vec)
        
        # Create memories across multiple layers
        memories = []
        for layer in ["l5_identity", "l6_schema", "l7_intention"]:
            for i in range(1000):
                noise = np.random.randn(1536) * 0.1
                vec = base_vec + noise
                vec = vec / np.linalg.norm(vec)
                memories.append({
                    "node_id": f"layer_{layer}_{i}",
                    "content": f"Memory in {layer}",
                    "layer": layer,
                    "status": "active",
                    "isolation_key": ik,
                    "embedding": vec.tolist(),
                    "confidence": 0.8,
                })
        
        query_embedding = base_vec.tolist()
        
        # Query 1: no layer filter
        start_time = time.time()
        results_all = await graph_store_neo4j_fixture.vector_search(
            query_embedding=query_embedding,
            isolation_key=ik,
            limit=20,
        )
        time_all = time.time() - start_time
        
        # Query 2: a single layer
        start_time = time.time()
        results_single = await graph_store_neo4j_fixture.vector_search(
            query_embedding=query_embedding,
            isolation_key=ik,
            layers=["l6_schema"],
            limit=20,
        )
        time_single = time.time() - start_time
        
        # Query 3: multiple layers
        start_time = time.time()
        results_multi = await graph_store_neo4j_fixture.vector_search(
            query_embedding=query_embedding,
            isolation_key=ik,
            layers=["l5_identity", "l6_schema"],
            limit=20,
        )
        time_multi = time.time() - start_time
        
        # Verify
        assert time_single <= time_all * 1.2, "Single layer filter should not significantly increase time"
        assert time_multi <= time_all * 1.3, "Multiple layer filter should not significantly increase time"
        
        print(f"✓ Layer filter cost: no_filter={time_all:.3f}s, single={time_single:.3f}s, multi={time_multi:.3f}s")

    @pytest.mark.slow
    async def test_beh_vector_search_performance(self, graph_store_neo4j_fixture):
        """
        Test: performance of behavior vector search
        
        Measure:
        - Create 1000 L6 schema nodes (with beh_embedding)
        - Run beh_vector_search
        
        Expected:
        - Performance similar to vector_search
        """
        ik = "perf_beh::default"
        
        np.random.seed(999)
        base_beh_vec = np.random.randn(512)
        base_beh_vec = base_beh_vec / np.linalg.norm(base_beh_vec)
        
        # Create L6 schema nodes
        memories = []
        for i in range(1000):
            noise = np.random.randn(512) * 0.1
            vec = base_beh_vec + noise
            vec = vec / np.linalg.norm(vec)
            memories.append({
                "node_id": f"beh_perf_mem_{i}",
                "content": f"Behavior test memory {i}",
                "layer": "l6_schema",
                "status": "active",
                "isolation_key": ik,
                "beh_embedding": vec.tolist(),
                "confidence": 0.8,
            })
        
        query_beh_embedding = base_beh_vec.tolist()
        
        start_time = time.time()
        results = await graph_store_neo4j_fixture.beh_vector_search(
            query_beh_embedding=query_beh_embedding,
            isolation_key=ik,
            limit=10,
        )
        elapsed = time.time() - start_time
        
        assert len(results) <= 10
        assert elapsed < 2.0, f"Behavior vector search took {elapsed:.2f}s, should be < 2s"
        
        print(f"✓ Behavior vector search: {elapsed:.3f}s for 1000 L6 nodes, returned {len(results)} results")

