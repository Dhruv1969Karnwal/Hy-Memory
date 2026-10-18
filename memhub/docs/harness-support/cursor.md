# Cursor one-click insert support

Last updated: 2026-10-14

## Bottom line

Cursor ships with one-click insert. It is not a Cursor Extension; it is **MCP config + Skill + Capture Hooks + Credential Grant**.

## Integration and product status

| Item | Cursor current implementation |
|---|---|
| One-click insert status | Available |
| How Cursor queries MemHub | MCP Server |
| How the model knows when to query | Skill |
| Automatic Prefetch before prompts | None |
| New Cursor sessions into MemHub in real time | `hooks.json` Capture Hooks |
| Traditional Cursor Extension | None |

## What each toggle does

### Turning on "One-click insert"

It installs the MCP, the Skill, the `beforeSubmitPrompt`/`afterAgentResponse` Capture Hooks, and creates and grants the Cursor Credential. MCP is for Cursor to query on its own; the Hooks only forward visible session events to Core in real time and do no Recall injection.

### Turning off "One-click insert"

It revokes the Cursor Credential first, then removes or restores the MCP, Skill, Hooks and Journal. The Sync toggle state is unchanged.

### Turning on the "Sync toggle"

Core backfills Cursor history sources, resumes frozen Jobs and enables polling/Watcher; new events sent by Capture Hooks are also allowed into Evidence.

### Turning off the "Sync toggle"

Core stops scanning, stops claiming pending Jobs and also refuses Hook Capture intake; already-running Jobs may finish and existing Memory is kept. Cursor's MCP/Skill are still present and the Credential has not been revoked, so Cursor can still query previously stored Memory.

### Combined effect

- Insert on, sync off: Cursor can still Recall, but new conversations no longer enter MemHub.
- Insert off, sync on: MemHub keeps learning from Cursor, but the session side cannot Recall.
- Both on: both directions are connected; Recall happens because the model calls it, not as automatic Prefetch on prompts.

## What gets installed

```text
~/.cursor/mcp.json
  └── mcpServers.one-memory

~/.cursor/skills/memhub-recall/SKILL.md

~/.cursor/hooks.json
  ├── beforeSubmitPrompt → one-memory capture cursor
  └── afterAgentResponse → one-memory capture cursor

~/.cursor/.one-memory/recall-install.json
```

The config only stores the Credential file path, never the Secret. Pre- and post-install state and hashes are recorded in the Journal.

The Layout lives in [recall-installer.ts](../../packages/host-bridge/src/recall-installer.ts), and Capture merging lives in [capture-hooks.ts](../../packages/host-bridge/src/capture-hooks.ts).

## Capture and the Sync toggle

The Capture Hook uses a fail-open shell command: a failure never stops Cursor from submitting a prompt or finishing an answer. The Hook is only the live event entry; Core's capture path still checks the sync config and Root authorization. Therefore:

```text
one-click insert → installs the Capture capability into Cursor
the Sync toggle → decides whether events actually enter and get processed
```

## Permissions

- Default Tools: `om_recall`, `om_conversations`, `om_diagnostic_trace`.
- Default Scopes: `all-apps`, `source:cursor`.

## Re-opening and closing

- Already installed and `approved`: no reinstall; only repair command paths and Probe.
- Incomplete permissions: reinstall and issue a fresh Credential; nothing appends a second MCP or Hook entry.
- Closing: revoke first, then restore or delete the MCP, Skill, Hooks and Journal.
- Files with user edits are never force-deleted; after revocation, unpermitted leftovers may remain.

## Taking effect and restart

Cursor has a dedicated restart prompt for Credential updates. Quit Cursor fully, then re-open and start a new conversation; closing only the chat tab does not guarantee the background process reloads the MCP, Skill and Hooks.
