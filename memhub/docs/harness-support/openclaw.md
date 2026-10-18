# OpenClaw one-click insert support

Last updated: 2026-10-14

## Bottom line

OpenClaw today is "the low-level MCP installer is implemented, but the full chain in the shipped product is frozen". Users cannot officially enable one-click insert from the current App.

## Integration and product status

| Item | OpenClaw actual status today |
|---|---|
| One-click insert status | Not available |
| How OpenClaw queries MemHub | None today; only the unexposed MCP config writer |
| Skill | Not installed |
| Plugin | Not installed |
| Prompt Hook | Not installed |
| Automatic Prefetch/context injection | Not implemented |
| New OpenClaw sessions into MemHub in real time | No Hook; the Core Adapter is frozen too |
| Credential/Grant | The official flow creates neither |

Users cannot install any plugin, Hook or Prefetch Provider into OpenClaw through the MemHub App today.

## What would happen if only the low-level MCP were unblocked

The theoretical call chain is:

```text
OpenClaw reads openclaw.json
→ starts mcp.servers.one-memory
→ the OpenClaw model calls om_recall / om_context / om_conversations itself when needed
→ MemHub's mcp.cjs calls Core through the local Broker
```

That is only active tool calling. It is not:

```text
the user submits a prompt
→ MemHub automatically Prefetches
→ memories are automatically injected into the OpenClaw context
```

Nothing implemented today does the latter automatically.

## What the two product switches actually do today

### "One-click insert"

Unavailable today. Web, Desktop, Runtime, the Connection Controller and the permissions layer all reject OpenClaw, so no MCP is written, no Credential is created and no Skill/Hook/Plugin is installed.

### "Sync toggle"

Also unavailable today. The OpenClaw source is globally paused inside Core: no `stat` on directories, no probing, no Adapter created, no scanning and no executable Jobs created.

### So the combined result today

No matter what the user tries, neither direction is connected in the shipped product:

```text
OpenClaw → MemHub: closed
MemHub → OpenClaw: closed
Automatic Prefetch: does not exist
```

## What real OpenClaw support would require

At least two product options exist:

1. **Support active Recall over MCP only**: unblock the MCP install, Credential and grant; do not touch the memory-provider, and whether the model queries at all depends on the Tool Description.
2. **Support automatic Prefetch**: this also needs an OpenClaw memory-provider integration, with clear rules for handling an existing Provider, how to combine or refuse, how to roll back, how to fail open on timeout, and how to keep injected content from being ingested back into memory.

The second option is completely unimplemented today.

## Current product limits

- The Web one-click mapping has no OpenClaw.
- The Desktop Host list actively filters OpenClaw out.
- Runtime Host Control, the Connection Controller, the Connection Client and `HostPermissions` all reject OpenClaw.
- The Core import layer puts OpenClaw on the paused list: no probing, no starting and no reading.

## Theoretical behavior of the low-level installer

Calling the low-level installer directly would write:

```text
~/.openclaw/openclaw.json
  └── mcp.servers.one-memory

~/.openclaw/.one-memory/recall-install.json
```

It installs no Skill and no Capture Hook:

- OpenClaw personal Skills live in the shared `~/.agents/skills`, so writing there directly would affect other Agents;
- `skills.load.extraDirs` would need extra install and rollback state that is not done yet;
- real-time capture would have to take the single memory-provider slot, possibly overwriting a Provider the user already has.

## Why it is frozen

- Import has only been verified at fixture level, with no acceptance run on real OpenClaw session replays;
- User role attribution is unresolved, which can keep content from entering memory extraction;
- Skill, Capture and permission boundaries have not formed a complete product flow yet.

## Low-level CLI risk

The development installer CLI can still write OpenClaw MCP config directly, but it creates no official Credential or Grant, and rollback of the same CLI is disabled.

That can leave behind:

```text
one-memory present in openclaw.json
but no valid Credential/Grant
and no way to manage it through the official product flow
```

That is development tooling residue, not product support.

## Uninstall

The low-level library functions can restore or delete `mcp.servers.one-memory` according to the Journal, but the official `HostPermissions.uninstall()` is unavailable because OpenClaw is excluded. Callers that use the low-level installer directly must take responsibility for Credential and uninstall safety themselves.
