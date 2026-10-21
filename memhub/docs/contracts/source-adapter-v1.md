# SourceAdapter V1

Status: frozen on 9/2; 9/3 only allows backward-compatible hardening.

## Inputs and outputs

Every source implements:

- `descriptor`: source ID, display name, storage shape, and capability claims.
- `detect()`: only decides whether a readable source exists inside a root the user authorized;
  it must never scan the real Home directory to "guess" what the user installed.
- `scan(options)`: asynchronously streams out `ScannedEvidence | ScanFailure |
  ScanSkip | ScanProgressEvent`.

`ScanOptions.cursors` is the multi-file cursor set for one user and one source. Each cursor holds
an opaque `file_key`, the file identity, the committed checkpoint, the observed size, and the
mtime. The adapter may advance a file's cursor only after the matching Evidence is committed, or once
that record can be safely skipped. Later appends to older files must stay rediscoverable, so a single "last file" cursor is not enough.

User authorization is a precondition for reading. Environment variables only discover candidate roots
and never substitute for authorization. Every candidate file must be revalidated when opened for realpath ownership, symlinks, regular file type, and the size budget.

## Evidence

Required:

- user, source, session, raw role, true author, verbatim wording, and at least one origin.
- Use `message_id` when it exists; without a stable message ID the source must provide a deterministic
  `source_ordinal`. The ordinal only distinguishes position within one session, never cross-channel correlation.
- A hook / plugin that does not know `message_id` but holds the same stable identifier writes `correlation_id`;
  a later file scan supplying the same `message_id` merges them into one Evidence.
- The adapter must confirm that `correlation_id` and the source `message_id` share one namespace; a local
  sequence number or a self-generated UUID must never be written into this field.

Constraints:

- A raw role must never be used to derive the true author directly.
- Tool call arguments, tool result bodies, reasoning, file contents, and command output stay out of
  Evidence by default; when only the conversation structure needs them, emit minimal metadata without content or a `ScanSkip`.
- A missing source timestamp stays `null`; import time must not impersonate it.
- When the same source message arrives via file, hook, or plugin with the same `message_id` /
  `correlation_id`, merge the origins; without a reliable correlation key, never merge just because the text matches.
- Identical text across sources is never merged.
- Each origin keeps its own `raw_meta` and `source_version`; later hook metadata must never overwrite
  the file locator information. A single origin's `raw_meta` is capped at 64 KiB.
- Only text Evidence is accepted today; no attachment, image, or audio support is claimed.

## Source capabilities

- `can_append`
- `can_observe_edits`
- `can_observe_deletes`
- `can_reconcile_complete_set`

Only an explicit delete event from `can_observe_deletes`, or a healthy complete reconciliation from
`can_reconcile_complete_set`, may drive source deletion.
A source being temporarily unreadable, a locked file, or an unknown schema is never read as a deletion.

## Events, errors, and recovery

- `ScannedEvidence`: produces one text Evidence.
- `ScanSkip`: a legitimate exclusion, such as tool content, reasoning, no text, or a budget limit; not counted as an error.
- `ScanFailure`: a single corrupt record or a busy source; the runner records it and continues or resumes later.
- `ScanProgressEvent`: reports only the real phase and the increment of discovered targets; when the total is unknown, no percentage may be estimated.
- skip / failure allow enumerated machine codes only. Persisted errors must not contain body text, absolute paths, or free-text exceptions.
- Conversation error identifiers must be irreversibly hashed before entering the task database.
- A whole-adapter exception puts the job in `failed`; a normal Core exit puts running tasks in
  `interrupted`, which is not a failure.
- A `paused`, `interrupted`, or `failed` job resumes from the multi-file cursor set.
- Evidence must be committed atomically before the matching cursor advances; if the process crashes in
  between, a rerun may only merge and must not create new Evidence or body revisions.

## Version policy

V1 fields only gain backward-compatible additions; removals, renames, and changes to existing meaning require a new contract version.
