# Claude Code one-click insert support

Last updated: 2026-10-14

## Bottom line

Claude Code ships with one-click insert. What it installs is **MCP config + Skill + three Capture Hooks**, not a traditional plugin.

## Integration and product status

| Item | Claude Code current implementation |
|---|---|
| One-click insert status | Available |
| How Claude Code queries MemHub | MCP Server |
| How the model knows when to query | Skill |
| Automatic Prefetch before prompts | Not by one-click insert itself; a separate sync Hook mechanism exists |
| New Claude Code sessions into MemHub in real time | `UserPromptSubmit`, `Stop`, `SubagentStop` Capture Hooks |
| Traditional Plugin | None |

## What each toggle does

### Turning on "One-click insert"

It writes the MCP entry into `~/.claude.json`, installs the Skill, merges the three Capture Hooks into `settings.json`, and creates the granted Credential.

### Turning off "One-click insert"

It revokes the grant first, then precisely deletes or restores the MCP entry, the Skill and the Capture Hooks. It does not turn Core sync off.

### Turning on the "Sync toggle"

Core scans `.claude/projects` for unprocessed sessions, resumes Jobs, and allows Capture Hook events to enter Evidence.

### Turning off the "Sync toggle"

Core freezes: scanning and pending Job claims stop, later Capture Hook calls are refused by Core, running Jobs may finish, and existing Memory is kept.

### Combined effect

- Insert on, sync off: Claude Code can still query old Memory, but new sessions no longer enter MemHub.
- Insert off, sync on: MemHub keeps reading Claude Code, but Claude Code has no Recall tools.
- Both on: both directions are connected; the one-click insert MCP is active querying, not automatic Prefetch.

## What gets installed

```text
~/.claude.json
  └── mcpServers.one-memory

~/.claude/skills/memhub-recall/SKILL.md

~/.claude/settings.json
  ├── UserPromptSubmit → one-memory capture claude-code
  ├── Stop             → one-memory capture claude-code
  └── SubagentStop     → one-memory capture claude-code

~/.claude/.one-memory/recall-install.json
```

The Layout and the MCP/Skill writing live in [recall-installer.ts](../../packages/host-bridge/src/recall-installer.ts), and Hook generation lives in [capture-hooks.ts](../../packages/host-bridge/src/capture-hooks.ts).

## Difference from the standalone sync Hooks

The project also ships a separate set of Claude Code-specific "sync Hooks". They are not the three Capture Hooks installed by one-click insert. The current permissions layer checks for conflicts between the two mechanisms, and when the standalone sync Hooks are present, installing or uninstalling Recall may require dealing with them first.

## Permissions

- Default Tools: `om_recall`, `om_conversations`, `om_diagnostic_trace`.
- Default Scopes: `all-apps`, `source:claude-code`.

## Re-opening and closing

- Already installed and `approved`: only repair command paths and Probe.
- Incomplete permissions: run Preview/Install again, issue a fresh Credential, then grant.
- Closing: revoke first, then precisely remove MemHub's MCP, Skill and Capture Hooks, keeping other Hooks.
- Files modified by the user are not force-deleted; the grant is revoked and manual review is requested.

## Taking effect and restart

MemHub does not restart Claude Code and cannot prove an existing process has hot-loaded `~/.claude.json`, the Skill and the Hooks. Quit fully, then re-open and start a new session.
