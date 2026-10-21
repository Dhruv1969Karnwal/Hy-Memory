# Privacy

MemHub is a local-first shared memory application that processes only the AI conversation sources
its users explicitly authorize. One Memory Home in this document refers to the legacy compatibility data directory, by default `~/.one-memory`.

## Local data

- Source originals are stored as Evidence in the user's local data directory.
- Source reads, model processing, and host reads are independent authorization boundaries.
- Pausing, revocation, correction, and deletion keep the same semantics regardless of the install entry point.
- The uninstaller does not delete user memory data by default; deleting data requires a separate user confirmation.

## Network access

When the user chooses a third-party model service, MemHub sends that service the authorized
content required to complete the selected task. The model provider, the data sent, and the purpose must be visible before enabling it. The product must not send API keys, arbitrary request headers,
local absolute paths, or raw content unrelated to the task to any model.

## What is not collected

The product contains no screen, Accessibility / AXTree, OCR, keyboard/mouse, clipboard, or background audio capture.

## Diagnostic and debug logs

The default diagnostic level is `info`, and structured diagnostic logs accept only sanitized
fields: they record no conversation bodies, model prompt/response, secrets, access tokens, or full
local paths. Structured logs live under One Memory Home in `logs/`, at 4 MiB per file with 3 rotated copies kept; files older than 14 days are cleaned up and the total is capped at 128 MiB.
A diagnostic export contains version, platform, architecture, non-secret status, and up to 5000
verified events, and no database or configuration files. The detailed contract is in
[`docs/contracts/diagnostic-logging.md`](docs/contracts/diagnostic-logging.md).

When the user explicitly switches the diagnostic level to `debug`, Core enables the HY pipeline
trace and sets `MEMORY_LOG_LLM_CONTENT` to `true`. That mode writes full model prompt/response into
local debug logs, which may contain authorized conversation bodies and model output; use it only for
deliberate troubleshooting, do not leave it on long-term, and do not share the resulting logs without
inspection and sanitization. Switching back to `info` stops further body logging but does not automatically delete debug files already written.

Actual behavior is governed by the current source, the network request implementation, and the release notes. If you find the implementation inconsistent with this statement, report it through
the private channel described in [`SECURITY.md`](SECURITY.md).
