# CLI experience flow (a walkthrough users can follow end to end)

This file lists only steps **verified on real hardware**. Anything marked "rejects" is deliberately not wired up today: it fails with a clear error instead of pretending to succeed.

## 0. Preparation

The release directory (all nine items present): `dist-runtime6` (produced with `node scripts/build_runtime_distribution.cjs dist-runtime6 darwin arm64 0.0.0).

For the walkthrough, use an **isolated data directory** and leave your daily `~/.one-memory` alone:

```bash
export ONE_MEMORY_DATA_DIR=/tmp/my_try
```

The CLI is available in two ways:

```bash
# Option A: use the build output from the repository
node apps/cli/dist/one-memory.cjs <command>

# Option B: npm tarball install (verified in an isolated directory)
npm install /path/to/one-memory-cli-0.0.0.tgz
npx one-memory <command>
```

## 1. Startup and health check

```bash
node apps/cli/dist/one-memory.cjs status
node apps/cli/dist/one-memory.cjs doctor
```

`status` shows whether the Runtime is running; `doctor` runs read-only checks per the diagnostic contract (no repairs by default).

## 2. First onboarding (init): there is no separate "authorization" step

```bash
# First see which hosts on this machine can be onboarded
node apps/cli/dist/one-memory.cjs init

# Pick a source and generate an onboarding plan (plan only by default)
node apps/cli/dist/one-memory.cjs init --source codex

# Onboard and import history at the same time
node apps/cli/dist/one-memory.cjs init --source codex --import

# See what would happen, without doing it
node apps/cli/dist/one-memory.cjs init --source codex --import --dry-run
```

**Onboarding is authorization**: there is no separate authorize step.

## 3. Import history and watch progress

```bash
node apps/cli/dist/one-memory.cjs setup plan --source codex      # generate the import plan
node apps/cli/dist/one-memory.cjs sources import --plan-file <plan-file> --confirm
node apps/cli/dist/one-memory.cjs progress                       # watch import progress
```

Observed on real hardware: after authorizing codex the task ran and kept persisting (stored grew from 14,437 to 62,634).

## 4. Scan (two-phase)

```bash
node apps/cli/dist/one-memory.cjs scan            # preview
node apps/cli/dist/one-memory.cjs scan --yes --source codex   # execute (--source is required)
```

Without `--yes` you only get a plan; passing a bare path at execution time is explicitly rejected (the Runtime addresses by source).

## 5. Register scan directories (path)

```bash
mkdir -p /tmp/scanme
node apps/cli/dist/one-memory.cjs path list
node apps/cli/dist/one-memory.cjs path add --root /tmp/scanme --source codex --yes
node apps/cli/dist/one-memory.cjs path list          # shows the entry just registered, revision 0->1
node apps/cli/dist/one-memory.cjs path remove --root /tmp/scanme --yes
```

Verified behavior: re-registering returns **409**; removing a nonexistent entry returns **404**.

## 6. Inspect host adapters (adapter)

```bash
node apps/cli/dist/one-memory.cjs adapter list
```

Prints each harness's available / authorized / live_sync / recall capabilities.

`adapter probe` **currently rejects** (the Runtime exposes no probe endpoint); it is not a parameter issue.

## 7. Write memories and check tasks (write -> task)

```bash
node apps/cli/dist/one-memory.cjs write --session <session-ID> --from 1 --to 2 --evidence <evidence-ID> --json
# returns a request_id (state=accepted)

node apps/cli/dist/one-memory.cjs task --id <request_id> --json
```

Writes finish asynchronously; `unknown` means the outcome is pending, and **must not be treated as grounds to resend**.

## 8. Recall

Both `recall` paths (raw / memory) were verified on real hardware earlier, and the commands remain usable.

## 9. Backup

```bash
node apps/cli/dist/one-memory.cjs backup create --target <path-inside-the-data-dir> --yes --json
```

The target must be inside the data directory (otherwise it is rejected); the artifact was verified at 540,672 bytes with owner-only permissions.

## 10. Host onboarding (host)

```bash
node apps/cli/dist/one-memory.cjs host list
node apps/cli/dist/one-memory.cjs host status --host codex
node apps/cli/dist/one-memory.cjs host install --host codex          # preview
node apps/cli/dist/one-memory.cjs host install --host codex --approve <fingerprint>   # execute
node apps/cli/dist/one-memory.cjs host uninstall --host codex --approve <fingerprint>
```

Two-phase confirmation: without `--approve` it only previews.

## 11. Continuous sync control

```bash
node apps/cli/dist/one-memory.cjs sync status --host codex
node apps/cli/dist/one-memory.cjs sync pause --host codex
node apps/cli/dist/one-memory.cjs sync resume --host codex
```

## 12. Uninstall (data preserved)

```bash
node apps/cli/dist/one-memory.cjs uninstall --install-root <root> --home <home> --yes
```

Verified: removes the current link, version directories, and CLI link; **the data directory is kept exactly as is**.

## Known explicit rejections (not bugs)

- `adapter probe`: the Runtime has no probe endpoint.
- `service-install/service-uninstall --yes` (in an isolated environment): needs a real user domain (launchctl); on failure it rolls back correctly and leaves no half-finished state.
- Some commands do not accept `--json` (e.g. `start`, `install`, `uninstall`): the contract has no such option for those commands, and passing one exits with code 2.

## Relationship with the desktop App

- Read-only/control commands are verified in CLI-only scenarios: two independent CLI processes connect to the same daemon (same instance_id) and read the same data; `start` does not start twice.
- The App-side half (real-hardware acceptance of the App and the CLI sharing one daemon) depends on the desktop App running; the App is not running right now, so it is unverified.
- Note: `~/.one-memory/runtime.json` holds a stale receipt (the recorded pid/port no longer exist), which makes the CLI show stopped; receipt maintenance lives on the App side and was not changed in this round.
