# Codex one-click insert support

Last updated: 2026-10-14

## Bottom line

Codex ships with one-click insert. What it installs is **TOML MCP config + Skill + Capture Hooks + Credential Grant**.

## Integration and product status

| Item | Codex current implementation |
|---|---|
| One-click insert status | Available |
| How Codex queries MemHub | MCP Server |
| How the model knows when to query | Skill |
| Automatic Prefetch before prompts | None |
| New Codex sessions into MemHub in real time | `UserPromptSubmit`, `Stop` Capture Hooks |
| Plugin | None |

## What each toggle does

### Turning on "One-click insert"

It writes the TOML MCP table, installs the Skill and the Capture Hooks, and creates the Codex Credential. The Codex model has to call MCP on its own for any memory query to happen.

### Turning off "One-click insert"

It revokes the Credential and restores or deletes the TOML, Skill, Hooks and Journal; the Sync toggle is unaffected.

### Turning on the "Sync toggle"

Core backfills `.codex/sessions`, resumes frozen Jobs, and allows Hook Capture to enter Evidence.

### Turning off the "Sync toggle"

Scanning and pending Job claims stop and Hook Capture is refused by Core; running Jobs may finish and existing Memory stays queryable.

### Combined effect

The two toggles control the Recall exit and the session intake respectively; only both on connects both directions, and there is still no automatic Prefetch.

## What gets installed

```text
~/.codex/config.toml
  └── [mcp_servers.one-memory]
      └── [mcp_servers.one-memory.env]

~/.codex/skills/memhub-recall/SKILL.md

~/.codex/hooks.json
  ├── UserPromptSubmit → one-memory capture codex
  └── Stop             → one-memory capture codex

~/.codex/.one-memory/recall-install.json
```

The TOML editor maintains only the `mcp_servers.one-memory` table and its `.env` table, preserving other config, comments and ordering as much as possible. See [toml-mcp.ts](../../packages/host-bridge/src/toml-mcp.ts).

## Permissions

- Default Tools: `om_recall`, `om_conversations`, `om_diagnostic_trace`.
- Default Scopes: `all-apps`, `source:codex`.
- The Credential Secret lives in a MemHub private directory; `config.toml` only stores the file path.

## Re-opening and closing

- Already installed and `approved`: no reinstall; only repair Runtime/MCP paths and Probe.
- Incomplete permissions: reinstall, create a new Installation and Credential, then grant.
- Closing: revoke first, then restore the TOML, Skill, Hooks and Journal.
- MCP, Skill or Hook files modified by the user are not force-deleted; the grant is revoked and manual handling is requested.

## Taking effect and restart

Current code does not guarantee a running Codex hot-loads `config.toml`, the Skill and the Hooks. Quit Codex fully, re-open and start a new session.
