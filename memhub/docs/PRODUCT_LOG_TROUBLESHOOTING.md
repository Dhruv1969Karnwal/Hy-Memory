# MemHub Product Log Troubleshooting Guide

This guide is for product, QA, and front-line support. The goal is not to explain the internals, but to answer three questions:

1. Did the data import successfully;
2. Is memory distillation still running, and how many memories did it produce;
3. If it stalled, did it stall at import, at the model call, or at the final write.

The structured diagnostic logs at the default `info` level contain no conversation bodies, no model input/output, and no API keys. `debug`
additionally enables the HY pipeline trace, which may contain full prompt/response and the authorized conversation bodies.
When troubleshooting, prefer the app's "Export logs" to produce the file, and never send the whole `~/.one-memory` data directory;
pipeline logs produced while debug is on must be reviewed and sanitized by a developer before they are shared.

## 1. Which log level to set

### Everyday observation: `info`

`info` is the default level and already covers import counts, task queueing, model distillation results, pending counts, and failure terminal states.
While just watching the product work normally, there is no need to turn on debug.

### Reproducing a specific issue: `debug`

Turn it on only when a developer explicitly asks to trace one click, one recall, or one specific model task:

> Settings → Diagnostics → Detailed logs (for debugging)

Switching this on is the same as `debug`. Once the issue is reproduced:

1. Click "Record issue time";
2. Click "Export logs (send to developers for debugging)";
3. Turn off "Detailed logs (for debugging)".

debug may write full model prompt/response into the local debug logs, so it must not stay on for long. Switching back to `info`
only stops further body capture; the debug files already produced are not deleted automatically.

Do not judge whether the system is healthy with the level set to `warning` or `error` only, because normal progress events are
`info`.

## 2. Where the logs are

Inside the app, use "Diagnostics → Export logs" directly. When a developer guides you remotely, prefer
`memhub logs`, which filters unsafe fields again. When you need to inspect the raw structured logs on the spot, Core logs live by default at:

```text
~/.one-memory/logs/diagnostic-core.jsonl
```

Every line is one JSON object. In a macOS terminal, set this first:

```bash
LOG="$HOME/.one-memory/logs/diagnostic-core.jsonl"
```

## 3. Which keys to filter on

Product troubleshooting only needs to know these fields:

- `timestamp`: when it happened;
- `level`: `info`, `warn`, `error`;
- `event`: the event name, which is the most important filter key;
- `outcome`: results such as `started`, `succeeded`, `completed`, `failed`;
- `error_code`: the stable error code, which can be sent straight to a developer;
- `job_id`: one import job;
- `task_id`: one memory distillation task;
- `request_id` / `trace_id`: chains one model or UI request together;
- `details.result_count`: how many memories this run finally linked;
- `details.pending_count`: how many items are still pending after this task;
- `details.stored_count`: how many raw messages this import added;
- `duration_ms`: how long it took.

Never search by user wording, conversation content, or API keys; by design none of that ever reaches the diagnostic logs.

## 4. What you should see when the system is healthy

### A. Core has started

```text
event = core.ready
outcome = succeeded
```

Without `core.ready`, neither the later import nor distillation will run.

### B. Raw messages are being imported

```text
event = import.finished
outcome = completed
details.stored_count >= 0
details.error_count = 0
```

`stored_count=0` may simply mean there were no new messages; as long as the status is not failed, 0 alone is not a failure.

### C. Memory tasks enter the queue

```text
event = memory_scheduler.tasks_queued
outcome = succeeded
details.prepared_count > 0
```

This means the imported data has entered the memory distillation queue.

### D. The SDK is distilling

```text
event = hy.ingest_started
outcome = started
```

It should then be followed by:

```text
event = hy.ingest_completed
outcome = succeeded
details.result_count = number of memories the SDK returned this call
```

If a reconcile runs, you also see:

```text
event = hy.reconcile.completed
details.created_count
details.updated_count
details.superseded_count
details.deleted_count
```

### E. MemHub finished this distillation

The best signal for product to watch is:

```text
event = memory_task.finished
outcome = completed
details.result_count = memories this task finally wrote or linked
details.pending_count = tasks still left to process
details.attempt = which attempt number this was
```

A healthy run looks like this:

- `memory_task.finished` keeps appearing;
- `pending_count` trends down overall, or keeps moving while imports keep arriving;
- `result_count` is above 0 intermittently;
- the same `error_code` does not keep repeating.

`result_count=0` is not necessarily a failure. When a stretch of conversation holds nothing worth keeping long term, 0 is the normal
result. Only when many tasks return 0 in a row and `hy.ingest_failed`, `memory_task.failed`, or a backlog that only grows is also
present should you escalate.

## 5. Copy-paste filter commands

Prefer these safe read commands:

```bash
memhub logs --since 2h --event memory_task.finished --json
memhub logs --since 2h --event hy.reconcile.completed --json
memhub logs --since 24h --errors-only --tail 500 --json
memhub status --json
```

The `jq` commands below are only for `diagnostic-core.jsonl`. Do not run them against `core.log`, `service.log`, or the debug
pipeline logs and then send the complete output to someone else.

### Recent failures

```bash
jq -r '
  select(.level == "error" or .outcome == "failed") |
  [.timestamp, .service, .event, (.error_code // "-"), (.task_id // .job_id // "-")] |
  @tsv
' "$LOG"
```

### Distillation counts and remaining backlog per run

```bash
jq -r '
  select(.event == "memory_task.finished") |
  [.timestamp, .outcome, (.details.result_count // 0),
   (.details.pending_count // 0), (.details.attempt // 0), (.error_code // "-")] |
  @tsv
' "$LOG"
```

The output columns are, in order:

```text
time  outcome  memories this run  pending left  attempt  error code
```

### Cumulative completions and cumulative memories in the current log

```bash
jq -s '
  [.[] | select(.event == "memory_task.finished" and .outcome == "completed")] |
  {
    completed_tasks: length,
    memory_results: (map(.details.result_count // 0) | add // 0),
    latest_pending: (last.details.pending_count // 0)
  }
' "$LOG"
```

The `memory_results` here sums the per-task result counts and tells you whether the pipeline keeps producing; it is not the
deduplicated row total of the current memory store.

### Import counts

```bash
jq -r '
  select(.event == "import.finished") |
  [.timestamp, .details.source_id, .outcome,
   (.details.stored_count // 0), (.details.revised_count // 0),
   (.details.skipped_count // 0), (.details.error_count // 0)] |
  @tsv
' "$LOG"
```

### Tracing one specific task

First copy a `task_id` from a failed or finished log line:

```bash
TASK_ID="paste-task_id"
jq --arg id "$TASK_ID" 'select(.task_id == $id)' "$LOG"
```

## 6. How to read the common stop points

- `core.ready` present but no `import.finished`: the source is not authorized, sync is off, or reading the source failed.
- `import.finished` is normal but there is no `memory_scheduler.tasks_queued`: the imported content has not formed complete conversations yet, or it is outside the processing window.
- `memory_task.started` present but no `hy.ingest_completed`: focus on model configuration, balance, network, timeouts, and `hy.ingest_failed`.
- `hy.ingest_completed` present but `memory_task.finished` failed: focus on the final governance write and the matching `error_code`.
- `pending_count` does not drop at all for a long time: check whether the same error code keeps appearing, or whether the tasks are all sitting in
  `waiting_for_model`.
- Import counts keep rising while no distillation task shows up at all: that is abnormal and must not be explained away as "this conversation had no memory value".

## 7. What product should provide when reporting to development

Please provide:

1. the approximate time the issue happened;
2. the app version and platform;
3. whether detailed logs were enabled;
4. what was shown after clicking "Record issue time";
5. the diagnostics file obtained through "Export logs";
6. the `error_code` values that repeated recently;
7. the `result_count` and `pending_count` in the last few `memory_task.finished` lines.

Do not provide API keys, and do not package and send the whole memory database.
