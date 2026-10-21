# Harness one-click insert support index

Last updated: 2026-10-14

The "One-click insert" toggle controls the **MemHub → Harness** Recall path; the "Sync toggle" controls session reading and processing in the **Harness → MemHub** direction. The two are not the same feature.

Turning on one-click insert usually runs:

```text
Run Preview in the background and compute the fingerprint
→ Create the install-only Installation and a private Credential
→ Write Harness config, Skill, Hook or Plugin
→ Write the install Journal
→ Grant Recall Tools and Scopes
→ Run the connection Probe
```

Default grant on the detail page:

```text
Tools: om_recall, om_conversations, om_diagnostic_trace
Scopes: all-apps, source:<current source>
```

Preview is never shown to the user; clicking the toggle itself counts as confirmation. On close, the Credential is usually revoked first, then managed files are restored or removed. Files the user has modified are never force-deleted.

## Support matrix

| Harness | One-click insert status | How the Harness queries MemHub | How new sessions reach MemHub | Sync toggle data source |
|---|---|---|---|---|
| [Hermes](hermes.md) | Available | MCP + Skill | No automatic Hook; Core scans | `.hermes` SQLite/file scan |
| [Cursor](cursor.md) | Available | MCP + Skill | Capture Hooks | Cursor Adapter scan + live Hook entry |
| [Claude Code](claude-code.md) | Available | MCP + Skill | Capture Hooks | `.claude/projects` scan + live Hook entry |
| [Codex](codex.md) | Available | MCP + Skill | Capture Hooks | `.codex/sessions` scan + live Hook entry |
| [WorkBuddy](workbuddy.md) | Available | MCP + Skill inside a `.codebuddy-plugin` | Plugin Hooks | WorkBuddy Adapter scan + Plugin Hook |
| [OpenCode](opencode.md) | Available | MCP, no Skill | JS Capture Plugin | OpenCode DB scan + live Plugin entry |
| [Qwen Code](qwen-code.md) | Available | MCP + Skill | No automatic Hook | `$QWEN_HOME/projects` scan |
| [Pi](pi.md) | Available | CLI Skill, no MCP | JS Capture Extension | Pi session scan + live Extension entry |
| [OpenClaw](openclaw.md) | Frozen for the shipped product | None today; only the unexposed low-level MCP writer | No Hook, Plugin or Prefetch | Adapter frozen too; the toggle is unavailable |
| [DeepSeek Harness](deepseek-harness.md) | Excluded from the detail page; the Profile UI has a bug | Profile-level MCP Patch | No automatic Hook | `$DSH_HOME/sessions` scan |

## Exact shared semantics of the Sync toggle

### Turning the Sync toggle on

Core will:

1. Check that the source directory is authorized and the Root identity is valid;
2. Establish a live-read baseline so the tail of pre-existing files is not mistaken for new live data;
3. Create a history Job when no existing history Job is present;
4. Write `sync_enabled` to `true` for the source in `<data_dir>/config.json`;
5. Schedule pending Jobs and resume paused, interrupted and failed Jobs;
6. Add the source to the polling/file Watcher;
7. Backfill historical Sessions that have not been processed yet and keep processing new Sessions from then on;
8. Resume the corresponding Memory Task when automatic memory processing is enabled.

### Turning the Sync toggle off

Core will:

1. Remove the source from the set of sources eligible to run in the live Watcher/polling loop;
2. Write `sync_enabled` to `false` for the source in `config.json`;
3. Stop the Jobs in the claim Job table that have not started yet; those Jobs keep their original state and wait for the next time the toggle is turned on;
4. Stop scanning for new Sessions through the Adapter;
5. Let the small number of already-running Jobs run to completion; they are not killed;
6. If Harness Hooks/Plugins keep calling Capture afterwards, Core still refuses intake because sync is off;
7. Memory that was already generated is not deleted and stays usable for Search and Recall.

Turning the Sync toggle off does not uninstall MCP, Skills, Hooks or Plugins, and it does not revoke the Credential Harnesses use to query memories. Those belong to the "One-click insert" toggle.

## Shared semantics

- Already installed with `approved` permissions: opening again usually does not reinstall; it only repairs Runtime paths and runs a Probe.
- `install-only`, `revoked` or `relink-required` states usually trigger a reinstall, a freshly issued Credential and a grant.
- On close, revoke first, then uninstall; when files carry user edits, only revoke and ask for manual review.
- A Probe means MemHub itself ran the installed on-disk command to check it; it does not prove a running Harness has hot-loaded anything.
- The safest way to make changes take effect is to fully quit the target Harness, re-open it and start a new session.

Unified production entry points:

- [Web toggle](../../apps/web/src/ApplicationDetail.tsx)
- [Web install and grant orchestration](../../apps/web/src/host-state.ts)
- [Desktop IPC](../../apps/desktop/src/main.ts)
- [Permission and install coordination](../../packages/runtime-control/src/host-permissions.ts)
- [Generic file installer](../../packages/host-bridge/src/recall-installer.ts)
- [Capture installer](../../packages/host-bridge/src/capture-hooks.ts)
