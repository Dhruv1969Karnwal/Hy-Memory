"""
HY Memory Lite — Retrieval helper modules.

This package centralizes the shared utilities for the lite pipeline's read side, used by both reader_legacy / reader_hybrid.
Any change to retrieval/ranking/evolution-chain synthesis should be implemented here first,
to avoid duplicated code scattered across reader implementation classes.

Module layout:
- config:       reads env vars; threshold/weight constants
- intent:       intention classification + keyword extraction
- bm25:         in-memory BM25-lite implementation
- rrf:          Reciprocal Rank Fusion + intention weights
- evolution:    evolution-chain backtracking + evolved memory synthesis
- tag_index:    per-user tag embedding index (maintained on the write side, used by the shared read flow behind a flag)
- reconcile_retrieval: hybrid candidate retrieval for Reconcile (vector pool + BM25)
"""
