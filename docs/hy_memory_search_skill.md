# Hy Memory Search and Reader Guide

This document describes the public parameters and return contract of the current `HyMemoryClient.search()` / `async_search()`.

## Reader Selection

```python
client.search("query", user_ids=["u1"], reader="hybrid")
client.search("query", user_ids=["u1"])  # HY_MEMORY_READER; defaults to legacy
```

Only these are supported today:

- `legacy`: candidate recall uses vector retrieval.
- `hybrid`: adds backend keyword/BM25 candidates on top of the vector candidates.

An unknown name logs a warning and falls back to `legacy`. Both Readers share one read pipeline and differ only
in whether keyword candidate recall runs; Schema, Intention, Episode, Tag/Entity, Graph, the evolution chain, local
Rerank, quotas and the public result format are identical.

The keyword channel depends on the Vector Store. Chroma explicitly returns empty keyword results, so on Chroma
`hybrid` degrades to pure vector candidate recall; only backends with keyword or sparse/BM25 support produce extra candidates.

## Chat Search Parameters

```python
result = client.search(
query="What does the user like?",
    user_ids=["u1"],
    agent_ids=None,
    session_ids=None,
    include_sources=["openclaw", "codex"],
    exclude_sources=["cursor"],
    limit=10,
    min_score=0.4,
    schema_limit=10,
    schema_min_score=0.4,
    intention_limit=0,
    created_after=None,
    layers=None,
    reader="",
)
```

| Parameter | Current semantics |
|---|---|
| `limit` | Maximum number of results on the `normal` channel, default `10` |
| `min_score` | Minimum score on the `normal` channel, default `0.4` |
| `schema_limit` | Independent cap for L6 Schema, default `10`; `0` disables it, a negative value means unlimited |
| `schema_min_score` | Threshold for L6 forward semantic recall, default `0.4`; Schema hit through reverse evidence is not filtered again by this value at the Client layer |
| `intention_limit` | Independent cap for L7 Proactive, default `0`, which disables that recall channel |
| `created_after` | Lower bound of the event time, Unix timestamp |
| `layers` | Recall only the given MemoryLayer; `l1_raw` is never returned to callers by Search |
| `reader` | Overrides the Reader for this call; an empty string uses `HY_MEMORY_READER` |
| `include_sources` | Call-source allowlist; an empty list means no restriction |
| `exclude_sources` | Call-source denylist; on conflict with the allowlist the denylist wins |

Provide either `query` or `queries`; `user_ids` must be a non-empty list.

L0 basic info has been split out of Search:

```python
profile = client.get_user_profile(user_id="u1")
```

This API reads the full profile by `user_id` only. Ultra catalog/supplement is attached only through this API as
well, and does not consume `schema_limit`.

## Search Mode and Productivity Routing

- `search_mode="default"` + a single `query`: fixed chat/normal, zero LLM.
- `search_mode="default"` + `queries=[...]`: one LLM call does scene classification and rewrite when Coding is on;
  only rewrite/time when Coding is off but Query Rewrite is on; zero LLM when both are off.
- `search_mode="agentic"`: single query and multiple queries both run one LLM judgment, rewrite and time parsing.
- Explicit `scene="chat" | "working" | "coding"`: skips the judge/rewrite LLM.

Under default, multiple queries over `MEMORY_QUERY_REWRITE_MAX_CHARS` are not rewritten: with Coding on it only
classifies, with Coding off the last query is used directly.

Once routed to `working` / `coding`, Search reads the separate productivity memory store and returns `facts` /
`experiences` instead of the three-channel chat structure. `coding` also returns `chain_block` / `chain_used`.

## Chat Success Response Structure

```json
{
  "request_id": "...",
  "memories": {
    "normal": [],
    "schema": [],
    "proactive": []
  },
  "rewrite": {
    "used": false,
    "original_query": "What does the user like?",
    "effective_query": "What does the user like?",
    "created_after": null,
    "scene": "normal"
  },
  "elapsed_ms": 123.4
}
```

When the Reader fails, the response still carries the three empty `memories` buckets plus `request_id` /
`elapsed_ms`, but the current failure fallback does not include `rewrite`.

The stable base fields of every public result are `memory_id`, `content`, `score`, `layer`, `user_id`,
`agent_id`, `source`, `retrieval_source`, `owner`, `tags`, `memory_at`, `gmt_created` and `meta`. `source` is the
writing application; `retrieval_source` is a diagnostic marker for the internal recall channel (for example
VDB / Schema / Episode). Specific capabilities may add `evolution_chain`, `episode_id`,
`capsule_memories`, `timeline` or `rerank_score`.

Private scores inside the Reader such as `_semantic` and `_bm25` are not exposed as per-memory fields by
`HyMemoryClient.search()`; detailed recall diagnostics still belong to the pipeline trace.

Source filtering is done by the underlying backend before `limit` / Top-K: Qdrant uses the same payload
Filter, Chroma uses a metadata where, FAISS uses an ID selector, and Neo4j/Kuzu Graph filters on properties
before computing similarity. Recall is never split into one channel per source, and nothing relies on filtering after over-fetch.
Memories derived from multiple sources use `source="mixed"`; with any source filter enabled they are excluded
fail-closed unless the allowlist explicitly contains `mixed`.

## Local Rerank

The local cross-encoder Rerank is post-processing shared by both Readers, off by default:

| Variable | Default | Description |
|---|---|---|
| `MEMORY_LOCAL_RERANK_ENABLED` | `false` | Whether local Rerank is enabled |
| `MEMORY_LOCAL_RERANK_MODEL` | `nreimers/mmarco-mMiniLMv2-L6-H384-v1` | Small multilingual Hugging Face model name or local path |
| `MEMORY_LOCAL_RERANK_DEVICE` | `auto` | Device such as `auto` / `cpu` / `cuda` |
| `MEMORY_LOCAL_RERANK_BATCH_SIZE` | `16` | Batch size of a single model forward |
| `MEMORY_LOCAL_RERANK_MAX_LENGTH` | `512` | Maximum tokenizer length |
| `MEMORY_LOCAL_RERANK_POOL_FACTOR` | `3` | Candidate pool multiple over the final `limit` |

It can also be configured through the Client:

```python
from hy_memory import HyMemoryClient

client = HyMemoryClient.from_config({
    "recall": {
        "local_rerank_enabled": True,
        "local_rerank_model": "nreimers/mmarco-mMiniLMv2-L6-H384-v1",
        "local_rerank_device": "auto",
        "local_rerank_batch_size": 16,
        "local_rerank_max_length": 512,
        "local_rerank_pool_factor": 3,
    },
})
```

Install the optional dependency:

```bash
pip install 'hy-memory[rerank]==2.0.1'
```

Once enabled, the `HyMemoryClient(...)` constructor downloads or loads the model, places it on the device and runs
one dummy forward; after construction Search never triggers model loading. A failed initialization raises
`RuntimeError` right away, instead of letting the service look ready and only failing on the first user request.
A single inference failure or an invalid score at runtime still fails open and keeps the pre-Rerank order.

The SDK sets no concurrency cap for Rerank Search and does not serialize model forwards; request concurrency is
controlled by the caller or the deployment layer. `MEMORY_LOCAL_RERANK_BATCH_SIZE` only sizes the batch of one forward, it is not a concurrency limit.
