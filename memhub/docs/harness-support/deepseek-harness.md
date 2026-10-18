# DeepSeek Harness one-click insert support

Last updated: 2026-10-14

## Bottom line

The Profile-level MCP installer for DeepSeek Harness is implemented, but today a desktop Profile install fails at the Preview stage because of a Host ID validation bug.

## Integration and product status

| Item | DeepSeek Harness current implementation |
|---|---|
| One-click insert on the app detail page | Deliberately excluded, because no Profile can be selected |
| Profile install on the source management page | Shown in the UI, but currently fails due to the Host ID bug |
| How the Harness queries MemHub | In theory through the Profile-level Cordis MCP Client |
| Skill | None |
| Automatic Prefetch before prompts | None |
| Capture Hook/Plugin | None |
| How new sessions reach MemHub | Core scans `$DSH_HOME/sessions` |

## What each toggle does

### "One-click insert"

It is unavailable on the app detail page. On the source management page it would in theory write `cordis.patch.yml` for each Profile, create a separate Credential and grant it; but today `deepseek-harness:<profile>` contains a colon and is rejected by the permissions layer, so no file is actually written.

### Turning on the "Sync toggle"

The Sync toggle applies to the whole `deepseek-harness` source, not to an individual Profile. Core scans `$DSH_HOME/sessions`, backfills historical sessions and keeps polling for new Sessions.

### Turning off the "Sync toggle"

Core stops scanning and pending Job claims, running Jobs may finish, and existing Memory is kept. It does not modify any Profile's `cordis.patch.yml` or Credential.

### Combined semantics after the one-click install is fixed

Once the Profile Host ID is fixed in the future:

- One-click insert for a Profile only decides whether that Profile can query MemHub;
- The global Sync toggle decides whether MemHub keeps reading Sessions across all of DeepSeek Harness;
- There is no automatic Prefetch and no Capture Hook.

## Profile granularity

Default `DSH_HOME=~/.dsh`; each Profile is independent:

```text
$DSH_HOME/profiles/<profile>/cordis.yml
$DSH_HOME/profiles/<profile>/cordis.patch.yml
$DSH_HOME/.one-memory/recall-install-<profile>.json
```

The Host IDs enumerated by the desktop app are:

```text
deepseek-harness:web
deepseek-harness:headless
...
```

In theory each Profile owns an independent Patch, Journal, Credential and Grant; Core session sync still scans `$DSH_HOME/sessions` as the single unified `deepseek-harness` source.

## MCP Patch

The installer maintains a managed text block inside `cordis.patch.yml`:

```yaml
# >>> one-memory recall — managed block, do not edit >>>
- insert:
    - id: one-memory-recall
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: onememory
        transport: stdio
        command: '/Applications/MemHub.app/Contents/MacOS/MemHub'
        args: ['/Applications/MemHub.app/Contents/Resources/mcp.cjs']
        env:
          ONE_MEMORY_HOST_CREDENTIAL_FILE: '.../credential'
# <<< one-memory recall — managed block <<<
```

See [dsh-patch.ts](../../packages/host-bridge/src/dsh-patch.ts). The installer does not parse the whole YAML, because a Cordis Patch can contain content such as `!!js` that ordinary YAML libraries cannot round-trip safely.

DeepSeek Harness gets no Skill and no Capture Hook; new sessions reaching MemHub relies on Core scanning.

## UI status

- The app detail page excludes DeepSeek Harness, because a single source page cannot select a Profile.
- The first-import bulk connect flow also excludes it.
- The source management page lists each Profile and shows a separate install button for it.

## The currently confirmed bug

An install from the source management page currently does fail:

```text
The system generates hostId = deepseek-harness:<profile>
HostPermissions only accepts [a-z][a-z0-9-]*
HostGrantStore also rejects the colon
```

So installation is rejected before anything is written to `cordis.patch.yml` or any Credential is created. This is not DeepSeek Harness failing to support MCP; it is a mismatch between the Profile Host ID format and the permission validation.

## Permission semantics risk after the fix

The default grant selection on the source management page may currently include:

- `om_recall`, `om_ask`, `om_conversations`, `om_diagnostic_trace`;
- `all-apps`;
- every authorized Source and Workspace.

That is broader than the generic grant on the app detail page. After the Host ID is fixed, it should be confirmed whether every Profile is really meant to get permissions this broad by default.

## Repeat install and uninstall

The underlying Patch install is idempotent: an existing managed block is replaced, and a second copy is never appended. When the same Row ID appears outside MemHub's sentinel block, it refuses to take over by default.

The normal uninstall design revokes the Profile Credential first, then removes the managed block while keeping the rest of the Patch; if no Row remains afterwards, it restores `[]`. A managed block modified by the user is not force-deleted.
