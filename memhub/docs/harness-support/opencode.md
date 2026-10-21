# OpenCode one-click insert support

Last updated: 2026-10-14

## Bottom line

OpenCode ships with one-click insert. It installs **the OpenCode MCP config + a live Capture Plugin** and no Skill.

## Integration and product status

| Item | OpenCode current implementation |
|---|---|
| One-click insert status | Available |
| How OpenCode queries MemHub | MCP Server |
| How the model knows when to query | No Skill; relies on the MCP Tool Description |
| Automatic Prefetch before prompts | None |
| New OpenCode sessions into MemHub in real time | `one-memory-capture.js` Plugin |
| Traditional Plugin | Capture is a local JS Plugin; Recall itself is still MCP |

## What each toggle does

### Turning on "One-click insert"

It writes `mcp.one-memory`, installs the Capture Plugin, creates the Credential and grants it. The OpenCode model has to call MCP on its own to Recall anything; the Capture Plugin never injects memories into prompts.

### Turning off "One-click insert"

Revoke first, then restore or delete the MCP, the Capture Plugin and the Journal; the Sync toggle is left untouched.

### Turning on the "Sync toggle"

Core scans the OpenCode database, resumes history Jobs and allows live Capture Plugin events into Evidence.

### Turning off the "Sync toggle"

Core stops database scanning and pending Job claims, and later Capture Plugin calls are refused by Core; running Jobs may finish and existing Memory is kept. If OpenCode's MCP is still installed, it can keep querying existing Memory.

### Combined effect

With both toggles on, OpenCode has active MCP Recall plus two session intake paths (database scan and Capture Plugin). There is no automatic Prefetch today.

## What gets installed

```text
~/.config/opencode/opencode.json
  └── mcp.one-memory

~/.config/opencode/plugin/one-memory-capture.js

~/.config/opencode/.one-memory/recall-install.json
```

The MCP config shape:

```json
{
  "mcp": {
    "one-memory": {
      "type": "local",
      "command": ["/path/to/MemHub", "/path/to/mcp.cjs"],
      "environment": {
        "ONE_MEMORY_HOST_CREDENTIAL_FILE": "/path/to/credential"
      },
      "enabled": true
    }
  }
}
```

OpenCode has no Skill mechanism the installer accepts, so `skillPath=null`; usage guidance relies on the MCP Tool Description.

## Capture Plugin

`one-memory-capture.js` listens for message part updates and session idle, and passes the text through:

```text
one-memory capture opencode
```

to Core. Plugin timeouts and exceptions never block OpenCode.

This means OpenCode has two paths for sessions to reach MemHub:

1. the Core Adapter scanning the OpenCode database;
2. the Capture Plugin pushing in real time.

The Sync toggle is still meant to be the final gate for accepting and processing, but the two paths overlap in responsibility.

## Permissions, re-opening and closing

- Default Tools: `om_recall`, `om_conversations`, `om_diagnostic_trace`.
- Default Scopes: `all-apps`, `source:opencode`.
- Already installed and `approved`: no reinstall; only repair the command and Probe.
- Closing: revoke first, then restore or delete `mcp.one-memory`, the Capture Plugin and the Journal.
- Config or Plugin modified by the user is not force-deleted; the grant is revoked and manual handling is requested.

## Taking effect and restart

MemHub's Probe does not mean a running OpenCode has hot-loaded the config and the Plugin. Quit OpenCode fully, re-open and start a new session.
