# Hermes one-click insert support

Last updated: 2026-10-14

## Bottom line

Hermes has "One-click insert" available, but it is not a plugin install. It silently edits YAML in the Electron background, writes the Skill, creates the Credential and grants it, so on macOS you usually never see an install window or progress bar.

## Integration and product status

| Item | Hermes current implementation |
|---|---|
| One-click insert status | Available |
| How Hermes queries MemHub | MCP Server |
| How the model knows when to query | `memhub-recall` Skill |
| MemHub automatically injecting memories into prompts | None |
| New Hermes sessions into MemHub in real time | No Hook/Plugin; relies on Core scanning |
| Traditional plugin | None |

Hermes uses MemHub by having the model call the MCP tool itself when it needs to. There is no pre-prompt Prefetch today, and no memory-provider Hook that pushes memories into the context automatically.

## What each toggle does

### Turning on "One-click insert"

It writes the MCP Server into `config.yaml`, installs the Skill, creates the Credential and grants the Recall permissions. Hermes can then call MemHub on its own.

### Turning off "One-click insert"

It revokes the Hermes Credential first, then restores or deletes the MCP config, the Skill and the Journal. It does not change whether MemHub keeps reading Hermes conversations.

### Turning on the "Sync toggle"

Core validates the `.hermes` Root, creates or resumes the history Job, scans the Hermes session database for records that have not been processed, and keeps discovering new records through Core's poller.

### Turning off the "Sync toggle"

Core stops scanning and stops claiming Jobs that are not running; pending Jobs freeze and running Jobs may finish. Hermes has no Capture Hook, so once it is off there is no other real-time intake path. Existing Memory stays queryable.

### Combined effect

| One-click insert | Sync | Result |
|---|---|---|
| Off | Off | Both directions are closed |
| On | Off | Hermes can query existing Memory, but MemHub no longer reads new Hermes sessions |
| Off | On | MemHub reads Hermes sessions, but Hermes cannot call MemHub |
| On | On | Both directions connected; Recall is still the Hermes model calling on its own, not automatic Prefetch |

## What actually happens when you turn it on

```text
~/.hermes/config.yaml
  └── mcp_servers.one-memory

~/.hermes/skills/memhub-recall/SKILL.md

~/.hermes/.one-memory/recall-install.json

<MemHub dataDir>/host-permissions/...
  └── Hermes Credential, Installation, Grant
```

The config looks roughly like:

```yaml
mcp_servers:
  one-memory:
    command: /Applications/MemHub.app/Contents/MacOS/MemHub
    args:
      - /Applications/MemHub.app/Contents/Resources/mcp.cjs
    env:
      ELECTRON_RUN_AS_NODE: "1"
      ONE_MEMORY_BROKER_DISCOVERY: "..."
      ONE_MEMORY_MANAGED_BY: "one-memory"
      ONE_MEMORY_HOST_CREDENTIAL_FILE: ".../credential"
```

The Hermes Layout is defined in [recall-installer.ts](../../packages/host-bridge/src/recall-installer.ts), and the YAML-local editing lives in [yaml-mcp.ts](../../packages/host-bridge/src/yaml-mcp.ts). The editor maintains only `mcp_servers.one-memory` and preserves models, Gateway settings and comments as much as possible.

## Skill, Capture and permissions

- Skill: `~/.hermes/skills/memhub-recall/SKILL.md`.
- Capture: not installed today. Hermes uses a single memory-provider slot, and occupying it automatically could overwrite a Provider the user already has.
- Default Tools: `om_recall`, `om_conversations`, `om_diagnostic_trace`.
- Default Scopes: `all-apps`, `source:hermes`.

## Re-opening

- Already installed and `approved`: no reinstall; only repair the MemHub command path and Probe.
- File exists but permissions are incomplete: run Preview and install again, issue a fresh Credential and grant.
- An existing `one-memory` Server not managed by MemHub: refuse to overwrite by default.

## Closing

A normal close revokes the Credential first, then:

- Delete or restore `mcp_servers.one-memory`;
- Delete or restore `memhub-recall/SKILL.md`;
- Delete the install Journal.

Once the config or Skill has been modified by the user, MemHub does not force-delete; it revokes what it can and asks for a manual check.

## Taking effect and restart

A successful Probe only proves the MCP command can connect to MemHub; it does not prove a running Hermes has hot-loaded it. Quit Hermes fully, re-open and start a new session.
