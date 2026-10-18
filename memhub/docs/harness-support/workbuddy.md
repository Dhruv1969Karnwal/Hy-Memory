# WorkBuddy one-click insert support

Last updated: 2026-10-14

## Bottom line

WorkBuddy is the only Harness today that really installs as a `.codebuddy-plugin` directory. It writes the Marketplace copy, the Runtime Cache copy, three Registries, the Skill, the Hooks, the Credential and the Grant all at once.

## Integration and product status

| Item | WorkBuddy current implementation |
|---|---|
| One-click insert status | Available |
| How WorkBuddy queries MemHub | MCP Server inside the Plugin |
| How the model knows when to query | Skill inside the Plugin |
| Automatic Prefetch/injection | Yes; the Plugin Hook runs Recall Injection |
| New WorkBuddy sessions into MemHub in real time | Plugin Capture Hooks |
| Install shape | A real `.codebuddy-plugin` |

What makes WorkBuddy most different from the other Harnesses is that it not only lets the model call MCP on its own; it also runs Recall Injection in the `SessionStart` and `UserPromptSubmit` Hooks and hands the context MemHub returns straight to the current session. The `Stop` Hook only handles Capture.

## What each toggle does

### Turning on "One-click insert"

It installs the full Plugin, MCP, Skill, the automatic Recall Injection Hook and the Capture Hook, and creates the WorkBuddy Credential. From then on WorkBuddy can both call MCP on its own and fetch context automatically when Hooks fire.

### Turning off "One-click insert"

It revokes the Credential first, then restores the Registries, deletes the Plugin tree, the Skill, the Injection/Capture Hooks and the Journal. It does not change Core's sync settings.

### Turning on the "Sync toggle"

Core backfills WorkBuddy history sessions, resumes frozen Jobs and allows new sessions sent by the Plugin Capture Hook into Evidence.

### Turning off the "Sync toggle"

Core stops scanning and pending Job claims and refuses Capture intake; running Jobs may finish and existing Memory is kept. Because the one-click insert Credential is still valid, WorkBuddy can still Recall on its own and the Injection Hook may still query existing Memory.

### Combined effect

- Insert on, sync off: WorkBuddy can still query existing Memory on its own or inject it automatically, but new conversations no longer enter MemHub.
- Insert off, sync on: MemHub keeps reading WorkBuddy, with no MCP, Skill or automatic injection on the session side.
- Both on: both directions connected, plus automatic Recall Injection.

## Plugin identity

```text
marketplaceName = one-memory
pluginName      = one-memory-recall
skillName       = memhub-recall
version         = 1.0.0
pluginId        = one-memory-recall@one-memory
```

## Actual install directories

Marketplace copy:

```text
~/.workbuddy/plugins/marketplaces/one-memory/
├── .codebuddy-plugin/marketplace.json
└── plugins/one-memory-recall/
    ├── .codebuddy-plugin/plugin.json
    ├── hooks/hooks.json
    └── skills/memhub-recall/SKILL.md
```

Runtime Cache copy:

```text
~/.workbuddy/plugins/cache/one-memory/one-memory-recall/1.0.0/
├── .codebuddy-plugin/plugin.json
├── hooks/hooks.json
└── skills/memhub-recall/SKILL.md
```

WorkBuddy loads the Skill from the Marketplace copy and the MCP from the Cache copy, so both locations are real runtime dependencies. See [codebuddy-plugin.ts](../../packages/host-bridge/src/codebuddy-plugin.ts).

## Registry changes

| File | What MemHub writes |
|---|---|
| `~/.workbuddy/plugins/known_marketplaces.json` | registers the `one-memory` Marketplace |
| `~/.workbuddy/plugins/installed_plugins.json` | registers `one-memory-recall@one-memory` in the Cache |
| `~/.workbuddy/settings.json` | `enabledPlugins["one-memory-recall@one-memory"] = true` |

Install Journal:

```text
~/.workbuddy/.one-memory/recall-plugin-install.json
```

## Plugin content and Capture

Inside the Plugin there is:

- the MCP Server config;
- the `memhub-recall` Skill;
- the `SessionStart`, `UserPromptSubmit` and `Stop` Hooks.

The exact responsibilities of the Hooks:

- `SessionStart`: runs Recall Injection and returns the MemHub context plus a cross-session hint;
- `UserPromptSubmit`: runs Recall Injection and also triggers Capture asynchronously;
- `Stop`: only runs Capture and never starts another Recall.

Failures are fail-open and never block WorkBuddy.

## Legacy install migration

The old scattered install lives at:

```text
~/.workbuddy/mcp.json
~/.workbuddy/skills/one-memory-recall/SKILL.md
~/.workbuddy/.one-memory/recall-install.json
```

The migration order installs the new Plugin fully first and only then deletes the old install, so WorkBuddy is never left with no Recall at all if the new install fails.

## Permissions, re-opening and closing

- Default Tools: `om_recall`, `om_conversations`, `om_diagnostic_trace`.
- Default Scopes: `all-apps`, `source:workbuddy`.
- Already installed and `approved`: no reinstall; only repair the command and Probe.
- Closing: revoke first, then restore the three Registries and delete the Marketplace/Cache plugin trees and the Journal.
- A modified Plugin or Registry is not force-deleted; revoke what can be revoked and ask for manual review.

The installer uses staging + rename for the plugin tree so WorkBuddy never sees a half-written plugin. Registry files use very conservative hash checking, so if WorkBuddy or another plugin modifies the same Registry, automatic uninstall may be refused.

## Taking effect and restart

A Probe does not prove a running WorkBuddy has re-read the Marketplace, Cache, Registries, Skill and Hooks. Restart WorkBuddy, or at least start a new session.
