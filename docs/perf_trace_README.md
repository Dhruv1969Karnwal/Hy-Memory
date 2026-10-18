# Performance and Pipeline Observability

Hy Memory provides three complementary kinds of observability data:

1. `TRACE_PERF`: INFO logs on the write path, quick to grep for a single request.
2. Pipeline logs: stage-level structured records, optionally written as JSONL, with persisted trace queryable through `MemoryInspector`.
3. System metrics: process load, latency and storage operation statistics aggregated over a time window.

All three correlate requests by `request_id`, but they differ in purpose and fields.

## 1. TRACE_PERF Logs

The write path emits INFO logs in this format:

```text
TRACE_PERF [<request_id>] <STAGE> <key=value ...>
```

System 1 stages that can appear today:

| Stage | Meaning | Key fields |
|---|---|---|
| `S1_START` | Actual write processing starts | `waiting`, `user` |
| `S1_L1_DONE` | L1_RAW embedding and persistence finished | `l1_ms` |
| `S1_EXTRACT_DONE` | Extractor finished | `extract_ms`, `summary_ms`, `memory` |
| `S1_RECONCILE_DONE` | Reconcile decisions finished | `ops`, `candidates`, plus the count per op |
| `S1_UPDATE_BATCH_DONE` | Batch UPDATE finished | `nodes` |
| `S1_ADD_BATCH_DONE` | Batch ADD finished | `nodes` |
| `S1_INTENTION_DONE` | L7 intention write finished | `intentions` |
| `S1_PERSIST_DONE` | Extraction results persisted | `ops` |
| `S1_WORKFLOW_DONE` | System 1 workflow finished | `workflow_ms` |

Not every request emits every stage. In lite mode, with empty extraction, without intentions or without batch ADD/UPDATE, the corresponding lines stay absent.

To follow a single request:

```bash
grep 'TRACE_PERF \[<request_id>\]' app.log
```

`hy_memory/core/workload.py` also emits queueing logs with the `TRACE_PERF` prefix, but not in the `[request_id] STAGE` form above; do not lump the two together when analyzing logs.

### Timing returned by writes

A chat write returns System 1 timing when it is available:

```python
result = client.add("User likes coffee", user_id="u1")
print(result.get("timing", {}))
```

The fields are:

```text
sys1_waiting_ms
sys1_l1_process_ms
sys1_workflow_ms
sys1_ops_avg_ms
```

These are stage durations, not fixed service-level thresholds. What counts as reasonable depends on the LLM, embedding, vector database, machine resources and concurrency.

## 2. Pipeline logs

Pipeline logs have two outputs with different field contracts.

Key fields of the JSONL file:

```text
request_id, namespace, user_id, agent_id, session_id,
step, prompt, content, parsed, memory_ids,
tokens, elapsed_ms, timestamp
```

Key fields of the persisted Pipeline Trace that the Inspector reads:

```text
request_id, namespace, user_id, agent_id, session_id,
step, prompt, response, parsed, memory_ids,
elapsed_ms, prompt_tokens, completion_tokens, total_tokens,
event_at_ms, created_at
```

### Two outputs

- The JSONL log is enabled by default, with directory precedence `MEMORY_LOG_DIR` → `MEMORY_TRACE_LOG_DIR` → `{MEMORY_DATA_DIR}/logs/pipeline`.
- The persisted trace is controlled by `MEMORY_PIPELINE_TRACE_ENABLED`, on by default, and written to the cache backend for the Inspector to query.

Both writes are best-effort; a failed observability write must not interrupt the main read or write flow.

`MEMORY_LOG_LLM_CONTENT=false` by default, so `prompt` and the LLM output (the JSONL `content`, the persisted
Trace `response`) keep only their length and a summary hash. Full input and output are recorded only when it is
explicitly set to `true`; assess the sensitive-data risk before enabling this in production.

### Common steps

Steps commonly seen on the write path:

```text
S1_EMBED
S1_L1_UPSERT
EXTRACT
SEARCH_QUERY
RECONCILE_CANDIDATES
RECONCILE
RECONCILE_APPLY
TOOL_CALLS
WRITE_SUMMARY
S1_AGENT_FAILED
S1_L1_ROLLBACK
```

Steps commonly seen on the working/coding path:

```text
CODING_JUDGE
CODING_EXTRACT
CODING_RECONCILE
READ_CODING_EMBED
READ_CODING_VDB
READ_CODING_RESULT
```

Read-side steps of a chat search all start with `READ_`. The current set is:

```text
READ_SCENE_JUDGE
READ_REQUEST
READ_EMBED_QUERY
READ_INTENT
READ_RECALL_VEC
READ_HIDDEN_L0_CONTEXT
READ_KEYWORD_EMBED
READ_TAG_MATCH
READ_RECALL_TAG
READ_BM25
READ_RRF
READ_MERGE_SCHEMA
READ_EVOLUTION
READ_INTENTION
READ_GRAPH_RERANK
READ_SUMMARY
```

The steps actually emitted depend on the reader, scene, configuration and recall results; no single search is guaranteed to contain the whole list.

### Using MemoryInspector

`MemoryInspector` is a public API:

```python
from hy_memory import HyMemoryClient, MemoryInspector

client = HyMemoryClient(mode="pro")
inspector = MemoryInspector(client)

write_result = client.add("User likes coffee", user_id="u1")
write_trace = inspector.get_full_trace(write_result["request_id"])

search_result = client.search("What does the user like?", user_ids=["u1"])
read_trace = inspector.get_read_trace(search_result["request_id"])

extract_logs = inspector.get_pipeline_logs(
    request_id=write_result["request_id"],
    step="EXTRACT",
)
```

The async counterparts are `async_get_full_trace()`, `async_get_read_trace()` and `async_get_pipeline_logs()`.

## 3. System metrics

Process-level metrics currently collect:

- System 1 started, completed, failed and active counts, plus the average duration of each stage.
- Completions over the last 60 seconds.
- VDB and Graph operation counts and average durations.

The returned schema also keeps the `active_sys2`, `queued_sys2`, `sys2_requests` and System 2 latency fields, but the current production path never calls `sys2_enqueue()`, `sys2_start()` or `sys2_end()`, so these values do not reflect Ultra background tasks. Do not use them to judge current System 2 load.

Public SDK:

```python
metrics = client.get_metrics(minutes=10)
metrics = inspector.get_system_metrics(minutes=10)
metrics = await client.async_get_metrics(minutes=10)
```

HTTP Server endpoint:

```text
GET /api/v1/metrics?minutes=10
```

Key response fields:

```text
uptime_seconds
window_minutes
requests
avg_latency_ms
throughput
storage_ops
sys2_requests
since
```

Average duration or storage fields may be empty when there are no samples. Metrics aggregate per minute; when the cache backend supports the relevant methods they are persisted, flushed every minute in the background, and data older than 7 days is cleaned up. active/queued and the last 60 seconds of throughput are live state of the current process.

## 4. Targeted Verification

```bash
python -m pytest -q \
  tests/test_pipeline_observability.py \
  tests/test_op_trace_logs.py \
  tests/coding/test_coding_pipeline_log.py \
  tests/test_shared_runtime.py
```
