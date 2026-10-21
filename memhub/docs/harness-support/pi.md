# Pi one-click insert support

Last updated: 2026-10-14

## Bottom line

Pi has no MCP config entry point, but the current production permissions layer supports the CLI channel, so Pi can use and revoke a Credential. It installs **a CLI Recall Skill + a Capture Extension**.

## Integration and product status

| Item | Pi current implementation |
|---|---|
| One-click insert status | Available in production code |
| How Pi queries MemHub | The Skill runs the Recall CLI; no MCP |
| How the model knows when to query | CLI version of the Skill |
| Automatic Prefetch before prompts | No automatic injection; the model runs the command itself as the Skill directs |
| New Pi sessions into MemHub in real time | `one-memory-capture.js` Extension |
| Credential | Supported; the CLI reads it from a private file, and it can be revoked |

## What each toggle does

### Turning on "One-click insert"

It installs the CLI Recall Skill and the Capture Extension, creates the Credential and grants it. Pi has to run the Recall CLI itself as the Skill directs; this is neither MCP nor automatic Prefetch.

### Turning off "One-click insert"

It revokes the Pi Credential first, then restores or deletes the Skill, the Capture Extension and the Journal; the Sync toggle is unchanged.

### Turning on the "Sync toggle"

Core backfills Pi Sessions and allows the Extension's `before_agent_start` and `turn_end` Capture events into Evidence.

### Turning off the "Sync toggle"

Core stops scanning and pending Job claims and refuses Extension Capture intake; running Jobs may finish and existing Memory is kept. As long as one-click insert stays on, Pi's Recall CLI can still query existing Memory.

### Combined effect

Pi's Recall exit is the CLI Skill and its live intake is the Extension; neither is MCP. Turning sync off only freezes intake; it does not revoke Pi's permission to query existing Memory.

## What gets installed

```text
~/.pi/agent/skills/memhub-recall/SKILL.md
~/.pi/agent/extensions/one-memory-capture.js
~/.pi/agent/.one-memory/recall-install.json
```

No MCP config is created, because Pi has nothing that reads `mcp.json` or `mcpServers`.

## Recall channel

The Skill runs `recall-cli.cjs` through a shell command and sets:

```text
ONE_MEMORY_BROKER_DISCOVERY
ONE_MEMORY_HOST_CREDENTIAL_FILE
```

The Recall CLI reads the credential from the private Credential file and then talks to the local Broker. The Broker checks the current Grant on every call, so the Credential can be revoked.

## Capture Extension

`one-memory-capture.js` listens for:

```text
before_agent_start
turn_end
```

and calls `one-memory capture pi`. Capture failures never block Pi.

## Permissions, re-opening and closing

- Default Tools: `om_recall`, `om_conversations`, `om_diagnostic_trace`.
- Default Scopes: `all-apps`, `source:pi`.
- Already installed and `approved`: no reinstall; only repair the command paths in the Skill and Probe.
- Permission revoked: reissue the Installation and Credential, then grant again.
- Closing: revoke first, then restore or delete the Skill, the Extension and the Journal.
- A Skill or Extension modified by the user is not force-deleted.

## Stale UI copy

[host-state.ts](../../apps/web/src/host-state.ts) still contains the old comment and the unsupported copy claiming Pi has no MCP and therefore cannot get a revocable Credential, and some tests still hand-build Pi as `unsupported_transport`.

That contradicts the current production implementation: `HostPermissions` supports `channel="cli"`, and the Pi Skill also carries the Credential file path. This copy and those tests are stale leftovers.

## Taking effect and restart

A Probe can only prove the Recall CLI runs; it cannot prove Pi has loaded the new Skill and Extension. Quit Pi fully, re-open and start a new session.
