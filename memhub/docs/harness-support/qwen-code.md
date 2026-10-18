# Qwen Code one-click insert support

Last updated: 2026-10-14

## Bottom line

Qwen Code ships with one-click insert. It installs **JSON MCP config + Skill + Credential Grant** and no Capture Hook.

## Integration and product status

| Item | Qwen Code current implementation |
|---|---|
| One-click insert status | Available |
| How Qwen Code queries MemHub | MCP Server |
| How the model knows when to query | Skill |
| Automatic Prefetch before prompts | None |
| New Qwen Code sessions into MemHub in real time | No Hook; relies on Core scanning |
| Plugin | None |

## What each toggle does

### Turning on "One-click insert"

It installs the MCP into `settings.json`, writes the Skill, creates the Credential and grants it. The model calls MCP on its own when it needs to; nothing is Prefetched automatically.

### Turning off "One-click insert"

Revoke first, then restore or delete the MCP, the Skill and the Journal; Core sync is unaffected.

### Turning on the "Sync toggle"

Core scans `$QWEN_HOME/projects` (default `~/.qwen/projects`) for sessions it has not processed yet, resumes frozen Jobs and keeps polling for new content.

### Turning off the "Sync toggle"

Core freezes: scanning and pending Job claims stop, running Jobs may finish and existing Memory is kept. If Qwen Code's MCP is still installed, it can keep querying existing Memory.

### Combined effect

One-click insert controls whether Qwen Code can Recall; sync controls whether MemHub keeps learning from Qwen Code. Only both on connects both directions, and it is still the model calling on its own, not automatic Prefetch.

## What gets installed

```text
~/.qwen/settings.json
  └── mcpServers.one-memory

~/.qwen/skills/memhub-recall/SKILL.md

~/.qwen/.one-memory/recall-install.json
```

The installer preserves `$version`, model settings and other MCP Servers, though the JSON may be reformatted. An existing `mcpServers.one-memory` not managed by MemHub is refused by default.

## Permissions, re-opening and closing

- Default Tools: `om_recall`, `om_conversations`, `om_diagnostic_trace`.
- Default Scopes: `all-apps`, `source:qwen-code`.
- Already installed and `approved`: only repair the command and Probe.
- Incomplete permissions: reinstall, issue a fresh Credential and grant.
- Closing: revoke first, then restore or delete the MCP, the Skill and the Journal.
- Config or Skill modified by the user is not force-deleted.

## Known path issue

Core import honors Qwen's own `QWEN_HOME`, but the Desktop Host Layout defaults to a fixed `~/.qwen` and only recognizes MemHub's `ONE_MEMORY_HOST_HOME` isolation variable.

A custom `QWEN_HOME` can produce:

```text
Core import: $QWEN_HOME
Recall install: ~/.qwen
```

In other words, sync and one-click insert point at different Qwen Homes.

## Taking effect and restart

Current code does not guarantee a running Qwen Code hot-loads `settings.json` and the Skill. Quit fully, re-open and start a new session.
