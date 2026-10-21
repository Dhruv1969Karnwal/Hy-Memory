# One Memory Runtime Unified Architecture and Implementation Contract

Updated: 2026-10-12. Related to D02, D06, D07, D08, D26, D40, RT-D41, D56.

> Status note (2026-10-16): this document keeps the historical `One Memory` / `one-memory` contract names.
> The outward product name is MemHub, and the memory SDK/engine is Hy Memory. The cross-platform goals in this
> document are not the current support matrix; Windows is not supported, and whether other platforms are supported
> follows the corresponding GitHub Release manifest and native-platform acceptance receipts. The current release
> contract follows [`../INSTALLATION.md`](../INSTALLATION.md).

## Goals

One Memory provides two main installation entry points to the outside:

- macOS graphical App;
- the cross-platform `one-memory`-compatible CLI from the design goals; the current public entry points and platform scope follow the installation docs.

The two are not two separate products. Whichever entry point is used to install, in the end they MUST use the same One Memory
Runtime, the same user data, the same permission semantics, and the same
Evidence -> Hy Memory -> Governance -> Matters/Wiki pipeline.

The product invariant is "one installation, one Runtime, one data store, multiple clients".

## Layering

### Kernel / Core

Python/FastAPI Core is the only memory data plane and the only memory writer, responsible for Evidence, import tasks,
Hy Memory, Governance, retrieval, Q&A, graph projection, and Matters/Wiki. Core is not responsible for installing desktop
windows, modifying Host configuration, or registering operating-system services.

### Runtime

The Runtime is a complete local service that can stay resident, composed of the following parts:

- SQLite, the single authoritative business database of MemHub, written exclusively by Core;
- Core;
- Host Gateway, including the Recall Broker, Hook notifications, and Host authorization checks;
- Runtime discovery, local authentication, version, and health status;
- Web Console static assets;
- logging, upgrades, rollback, and service lifecycle control.

The Runtime may be installed by the graphical App or the CLI, but the installation source MUST NOT change its data contract or behavior.

### CLI

`one-memory` is the command-line installer, controller, and client of the Runtime. The final public commands cover at least
`install/start/stop/restart/status/doctor/ui/logs/upgrade/uninstall`, and reuse the same Source, model, and Host authorization
contracts. The internal Kernel entry point is later fixed as `one-memory-core`; during migration, the no-argument
`one-memory` stays compatible with launching Core directly.

### Desktop

Electron only provides graphical installation, native windows, directory selection, system display preferences, and Runtime
client capabilities. It MUST NOT own a second database or a second set of Host authorization semantics. On startup, Desktop
first discovers and connects to an existing Runtime; it starts or installs the Runtime only when no verifiable instance exists.
Closing the window does not stop a standalone Runtime by default.

### Web Console

The React UI stays a single set. Electron and the browser each use the same Runtime contract through a constrained Control
Client. The browser MUST NOT hold a bare Core token long-term; the formal standalone Web Console uses a one-time local
login in exchange for an HttpOnly, SameSite=Strict loopback session.

## Installation Equivalence Definition

"App and CLI equivalence" means:

1. Same platform, same architecture, and same version consume the same Runtime artifact;
2. Use the same One Memory Home, database schema, and migrations;
3. Use the same Core API, Host Gateway, and authorization contracts;
4. At most one writing Runtime per data directory;
5. The App can connect to a Runtime already started by the CLI, and the CLI can also manage a Runtime installed by the App;
6. The three authorization classes, Source, model, and Host, remain independent;
7. Upgrade, revocation, deletion, and rollback results do not vary by installation entry point.

Equivalence does not mean the available Hosts are completely identical on every platform. macOS, Linux, and Windows MUST
show Source and Host capabilities per the real support matrix; fixtures or cross-platform branch tests cannot replace real-device conclusions.

## One Memory Home

The default logical location stays `~/.one-memory`, overridable via `ONE_MEMORY_DATA_DIR`. It is a data instance, not the
private directory of some App version. The formal directory contract converges step by step to:

```text
One Memory Home/
  data/                 # one-memory.sqlite3, HY data, and derived indexes
  migrations/           # SQLite schema upgrade receipts and backup index
  runtime.json          # secret-free runtime instance discovery receipt
  runtime-token         # current instance short-lived local credential
  host-permissions/     # Host installation identity and authorization
  integrations/         # Broker discovery
  logs/
```

The 2026-09-26 D56 already overrides D53: the formal product continues to use SQLite as MemHub's only business database, and
does not install or maintain MySQL in parallel. Evidence, tasks, cursor, session, configuration, permissions, Governance, and
audit all take the same SQLite data instance as authoritative; an Adapter may still read the Host's own SQLite file read-only, and
HY SDK internal storage is handled separately per the HY contract. The Runtime is responsible for the single writing instance per
data directory, schema upgrades, WAL, backup and restore, disk budget, and keeping data on uninstall.

`runtime.json` MUST NOT contain a token. `runtime-token` MUST be owner-only and deleted when the instance exits; before
Windows is formally supported it MUST implement and verify a user SID/DACL on real hardware, and cannot pass by ignoring the POSIX mode.

## Runtime discovery v1

The current first batch first upgrades the Core receipt into the minimal contract a trusted local client can connect to. POSIX
platforms also publish an owner-only token; before the SID/DACL implementation and real-device verification, Windows does not publish
an attach token, so in this batch Windows can still only use the credential held directly by the launcher and cannot claim CLI/App attach:

```json
{
  "schema_version": 1,
  "service": "one-memory-core",
  "instance_id": "...",
  "pid": 1234,
  "parent_pid": null,
  "host": "127.0.0.1",
  "port": 8765,
  "started_at": "..."
}
```

The connecting side MUST simultaneously satisfy: the receipt schema/service is correct, the host is an exact loopback, the port is
valid, the PID is still alive, the owner-only token is readable, and the health service and instance ID match. On any failure it MUST
NOT treat an arbitrary loopback service as One Memory. If the process is still alive but its identity cannot be verified, it MUST fail
explicitly and cannot kill the process, take over the data directory, or fall back to anonymous access.

## First Batch Development Scope

The real pipeline:

```text
CLI/terminal starts Core
  -> Core locks the One Memory Home
  -> writes runtime-token and runtime.json
  -> Desktop reads and verifies receipt/token/health
  -> Desktop attaches the same Core
  -> Desktop exits
  -> the standalone Core keeps running
```

This batch changes:

- records the unified Runtime layering and the later implementation roadmap;
- Core publishes a secret-free discovery receipt and an owner-only short-lived token;
- Desktop tries to connect to a verifiable Core in the same data directory before spawn;
- Desktop stops only the Core it started itself.

This batch does not change:

- HY, Governance, Evidence, the database schema, and Source permissions;
- the current state of the Host Gateway still living inside Electron;
- daily installation packages, real Host configuration, and user profiles;
- Linux/systemd, macOS LaunchAgent, and Windows user services;
- the Runtime hosting the Web Console and browser session;
- publishing, signing, auto-update, and public downloads.

## First Batch Failing Tests and Completion Criteria

1. Tests fail when the Core receipt omits schema/service or the token file permissions are unsafe.
2. Tests fail when Desktop still spawns in the face of a healthy, instance-matched standalone Core.
3. When the receipt points to a non-loopback, a dead PID, a missing token, or a health instance mismatch, it MUST NOT attach.
4. After Desktop attaches, calling stop MUST NOT send a signal to a standalone Core.
5. The Core Desktop started itself MUST still stop gracefully on exit and keep the original orphan-recovery behavior.
6. Core runtime, targeted Desktop supervisor tests, type, and format checks pass.

This batch can only be marked `implemented` / automatic contract `verified`. Without an actual joint test against a live CLI process and
a packaged App over the same database, it cannot be marked `accepted`, let alone called a completed cross-platform one-click install.

## Later Stages

### R2: Runtime Supervisor

Move the Recall Broker, HostPermissions, Source-change notifications, and runtime discovery out of Electron into a
standalone Runtime Supervisor. After the App exits, Host recall and continuous sync stay available.

The first two R2 slices already established `packages/runtime-control`: it uniformly parses the secret-free receipt, verifies the
owner-only token, validates the standalone process and protected identity, and carries Core startup, attach, orphan-recovery, and stop
semantics. Desktop keeps only the App resource path and data-directory selection and consumes that package; the Host Gateway and Host
permissions have still not moved out of Electron.

The third R2 slice added an Electron-independent `RuntimeSupervisor` combinator that starts or attaches Core in order and then
starts the sidecar, releases this run's Core ownership when the sidecar fails, and unifies the shutdown of both on stop. Desktop already
uses it to compose the existing Recall Broker. Because the Broker process still shares a process with Electron, this step is a
migratable lifecycle boundary, not a completed background daemon.

The fourth R2 slice already moved HostPermissions, sync Hook asset verification, and the Source-change handler into
`packages/runtime-control`; the original Desktop path keeps only a compatibility re-export to preserve the existing IPC tests and call
boundaries. The permission receipt, installation transaction, confirmation fingerprint, revocation, and Hook rollback semantics are
unchanged; Windows still explicitly refuses Host authorization control for lack of an SID/DACL implementation.

The fifth R2 slice added `RecallBrokerSupervisor`: if the same Runtime instance already has a live, health-matched Broker, the client
only attaches and does not stop it on exit; a new Broker is created only when it is missing, stopped, or has a dead PID; when one is
alive but belongs to another instance or cannot be verified, it fails closed and cannot overwrite discovery or start a second Broker.
Desktop composes this layer through `RuntimeSupervisor`.

The sixth R2 slice added the internal `one-memory-runtime` foreground entry point and an outer `runtime-service.json`/lease. After
this process acquires the One Memory Home single-instance lease, it builds HostPermissions in order, starts Core and the Broker, and
publishes the outer identity; on SIGINT/SIGTERM it cleans up the Core receipt/token, the outer receipt, and the lease, leaving the
Broker an explicit stopped receipt. Desktop attaches a parent-owned Core only when the outer PID, Core parent PID, instance, and port
all agree and both processes are alive; an ordinary child of another Desktop is still not taken over. This entry point is not yet registered as a system service.

The seventh R2 slice added `packages/runtime-control/dist` to the Desktop formal resource manifest and made the Desktop build build the
Host Bridge and Runtime Control first. The installation package therefore carries a same-version Core, Host Bridge, Web, and
`runtime/runtime.cjs` together; the current Desktop startup logic still runs in embedded mode and does not presumptuously register or
switch a background service before install/upgrade/uninstall transactions are defined.

The eighth R2 slice makes the packaged Desktop run the shared launcher by default: it first verifies the existing outer Runtime and,
when missing, launches the same-package `runtime/runtime.cjs` detached in Electron Node mode, waiting until the outer receipt, Core
token, and identity all hold before entering the original Desktop attach flow. Exiting Desktop only detaches and does not stop the
external Runtime. Development mode and the explicit `ONE_MEMORY_DESKTOP_RUNTIME_MODE=embedded` keep the old in-process combinator
mode as a development and failure fallback; this is still not login autostart or an operating-system service install.

### R3: Full CLI and Service Management

Implement unified commands, adding macOS LaunchAgent, Linux systemd user service, a foreground mode without systemd, and a
Windows current-user login startup. Stop/uninstall and data deletion MUST be confirmed separately.

The first R3 slice implemented the read-only `service-plan`: from an explicit platform, home, and install root it generates the file
content, target path, and argument array of a macOS LaunchAgent or Linux systemd user service, writing no files and running no
commands. It is blocked on macOS when uid is missing; on Windows it is always blocked until the SID/DACL and current-user startup
mechanism are verified on real hardware. A later apply MUST consume the same plan and do atomic writes, preview confirmation, failure rollback, and current-state re-check.

The second R3 slice established `apps/cli`, producing the public `one-memory` command and directly reusing Runtime Control's
serve/status/doctor/service-plan; the Python package adds the `one-memory-core` formal Kernel entry point, and the old Python
`one-memory` stays temporarily compatible. At release only the Node CLI should be exposed as the user command, to avoid two implementations contending for the command name.

The third R3 slice added start/stop/restart to the public CLI. start only launches the Runtime entry in an explicitly installed layout
and waits until the outer receipt, Core token, and identity all verify; stop, after the same verification completes, sends SIGTERM
only to the outer Runtime PID and waits for the parent process and the outer/Core receipt and token cleanup; on an identity mismatch
it sends no signal. restart strictly serializes stop and start, to avoid the old Core racing for the same data directory while still exiting.

The fourth R3 slice added the public `cli/one-memory.cjs` to the Desktop formal extraResources and made the Desktop build build the
Host Bridge, Runtime Control, and CLI first. The artifact hash manifest MUST cover the CLI, Runtime, Core, Host Bridge, and Web
together; this guarantees the command-line entry carried by the Mac App and its background Runtime come from the same source build,
but it does not yet automatically link the CLI onto PATH.

The fifth R3 slice added a CLI distribution-directory builder: it assembles a fixed directory from the already-built CLI, Node,
Runtime, Core, Host Bridge, and Web resources, lists a complete SHA-256 manifest, and uses a stage directory before atomically
replacing the target. It fails and keeps the old distribution directory when any key file is missing or an out-of-bounds symlink
exists. This builder does not download artifacts or install into user directories.

The sixth R3 slice added a shared local distribution install transaction and the public CLI `install`. By default it only outputs a
plan after verifying the manifest, the target platform/architecture, the complete file set, and per-file SHA-256; only an explicit
`--yes` copies into `versions/<version-platform-arch>` and atomically switches `current`. Version/arch characters, source-and-install-root
overlap, out-of-bounds links, a plan modified by the callee, an existing unmanaged current, and same-version-different-content are all
rejected. At this stage it does not download artifacts, delete old versions, or write PATH or service files.

The seventh R3 slice added the `service-install` two-phase operation: by default it only shows `planUserService`, and only after an
explicit `--yes` does it require the current platform to match, the Runtime/Core/Host Bridge/Hook key files to be present, the service
directory to be safe, and the target file not to exist, then it writes with 0600 and calls the service manager by argument array. On
failure it deletes only the file it just created, and does not guess and disable a possibly long-existing same-named service; potential
manager residue is reported later by doctor. Upgrading and uninstalling an existing service are implemented separately.

The eighth R3 slice added the `service-uninstall` two-phase operation: by default it only previews, and after an explicit `--yes` it
first checks the existing service file against the current plan character-for-character, and only executes bootout/disable and deletes
when they match exactly; a user-modified file or a same-named file created by another tool is rejected outright. Uninstalling the
service does not delete `versions/current` or the One Memory Home, and data deletion MUST be another explicit operation.

The ninth R3 slice composes the first `install` into a single transaction: the same preview returns both the distribution directory
and the user service plan; after an explicit `--yes` it installs the version and switches current first, then registers the service for
the first time. When the service step fails it restores the old pointer only on the premise that current still points to this version,
and deletes the version directory it just created. apply re-validates the source manifest and plan, rejecting caller tampering,
source/target overlap, path escape, and unmanaged current. Overwriting an existing service file is not supported now, nor is auto-cleaning old versions or data.

The tenth R3 slice adds the managed CLI entry point to the same transaction: on POSIX it defaults to `~/.local/bin/one-memory` pointing
at `<install-root>/current/bin/one-memory`, overridable with `--bin-dir`. A regular file, a link pointing at another target, and an unsafe
directory are all refused overwrite; when already correctly linked it is idempotent. On service-registration failure it deletes the
just-created link in reverse order and restores the old current and this version. Windows PATH installation is still blocked.

The eleventh R3 slice fixes the distribution-directory POSIX launcher: when invoked from the PATH symlink it resolves its own real
location layer by layer, then runs the same distribution directory's `bin/node` and `bin/one-memory.cjs`. The real temporary symlink
execution test MUST produce the bundled Node path, and cannot judge success by exit 0 alone.

The twelfth R3 slice makes the public CLI in the distribution directory self-locate its install source and provides a platform-default
install root: macOS uses `~/Library/Application Support/One Memory`, Linux uses `~/.local/share/one-memory`, and Windows uses
`LOCALAPPDATA/One Memory` or a home-directory fallback. An explicit source/install-root/home can still be used for isolation and
enterprise deployment; Windows apply is still blocked by the existing ACL gate.

The thirteenth R3 slice adds the `upgrade` two-phase transaction: it requires the existing CLI entry to still point precisely at the
managed current and the service file to match the current plan character-for-character, and only then installs and switches to the new
version before running a service restart. When restart fails it restores the old current, deletes this new version, and best-effort restarts the old version; it does not rewrite the service file or delete data or old versions.

### R4: Standalone Web Console

The Runtime provides same-origin Web static assets and a local session; add the Browser Control Client so that after a CLI install the
full interface is available via `one-memory ui`.

### R5: Desktop Thin Shell

Desktop only does discover -> verify -> attach; when the Runtime is missing it goes through the shared installer. The App and CLI use
the same release manifest and Runtime artifact hash.

### R6: Cross-Platform Distribution and Real-Device Acceptance

Build macOS arm64/x64, Linux x64/arm64, and Windows x64 separately; complete the real-device matrix for permission storage, Source
reading, service lifecycle, upgrade rollback, and Host calls. Docker is an extra deployment form for Linux/NAS/CVM, and
cannot replace desktop user-level CLI install acceptance.
