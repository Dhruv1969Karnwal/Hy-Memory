# One Memory diagnostic logging contract

Updated: 2026-09-24. Related to D07, D26, RT-D41.

## Purpose

Diagnostic logs exist so that, after something goes wrong for the user, the real execution chain
Desktop -> Runtime -> Core -> HY -> Host can be reconstructed by recency, run session, one operation, or an incident id.
The logs are not user behaviour analytics, call records, or a memory source: they never enter `call_events` or the memory database, and nothing is uploaded over the network.

## Event format

Every line is one `DiagnosticEventV1` JSON object. Required fields are `schema_version=1`, the UTC
`timestamp`, `level`, `service`, `component`, `event`, `outcome`, and `session_id`.
Optional correlation fields are `trace_id`, `incident_id`, `instance_id`, `operation_id`, `request_id`,
`job_id`, `task_id`, `duration_ms`, the status, the stable error code, the error type, and the sanitized repo-relative stack frames.

`details` only accepts the scalar whitelist declared in code. Never record conversation text, queries, answers, model input or output,
request bodies, response bodies, tokens, API keys, secret refs, full user paths, or an exception message/cause. Unknown fields and
non-conforming identifiers are refused. A failed log write must never block the main product path.

## Correlation rules

- Each Desktop, Runtime, and Core start has its own `session_id`.
- A user request generates a UUID `trace_id`, propagated between Web, Broker, and Core via `X-One-Memory-Trace-Id`;
  invalid inbound values are replaced and never recorded as-is.
- Existing operation/request/job/task identities follow the business contracts; no parallel identity is created.
- "Record issue time" creates an `incident_id` and records the current Runtime instance. A query by incident returns
  events from the same desktop session or the same Runtime instance from 15 minutes before to 5 minutes after the mark.

## Storage and reading

Logs live under One Memory Home as `logs/diagnostic-<service>.jsonl`, with services runtime, core,
host-bridge, desktop, and cli. A single file is 4 MiB and keeps 3 rotations; on top of that, files older than 14 days are cleaned up and the
total is capped at 128 MiB. On POSIX, directories are 0700 and files are 0600.

The reader only enumerates regular files with those fixed names, never follows symlinks, and accepts no arbitrary path. Damaged lines are skipped
and counted, so one broken line never hides the rest of the evidence. The diagnostic export is a single sanitized JSON holding the version, platform, architecture,
non-secret status, and at most 5000 verified events; it contains no database or configuration file.

## Coverage and throttling

Must record start/stop, attach/spawn, identity verification, import and memory task status, model/HY calls, host recall,
permission and installation operations, key IPC, failure classification, retries, duration, and terminal state. High-frequency polls that change nothing and per-item raw text are not recorded;
imports report phase and count summaries so that logging itself never becomes a performance problem.

Product-side progress decisions use these stable events:

- `import.finished`: `stored_count`, `revised_count`, `skipped_count`, `error_count`;
- `hy.ingest.completed`: the `result_count` the SDK returned for this call;
- `hy.reconcile.completed`: the created, updated, superseded, and deleted counts;
- `memory_task.finished`: the final `result_count`, the remaining `pending_count`, and `attempt`.

Of these, `memory_task.finished` is the final event the product uses to decide whether distillation keeps making progress; it cannot be
replaced by `hy.ingest.completed` alone, because the latter happens before the Governance write.

## Acceptance

A privacy test must first plant a detectable secret as a positive control, then prove that neither the real pipeline logs nor the export contain it.
Using isolated data, at minimum induce a Core start failure, an HY failure, and a host connection failure, and reconstruct from the logs alone the failing component, the time,
the correlation IDs, the stable error code, and the code location. Passing automated tests only earns a verified marker; the real experience of the
installed app still needs its own acceptance run.
