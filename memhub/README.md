# MemHub / One Memory

MemHub is a local-first shared memory product: it consolidates the conversation history and new
content of the multiple AI products a user has authorized into one shared memory, and lets those
AIs reuse it after their own separate authorization. Hy Memory is the Python SDK and memory engine used by MemHub. One Memory / `one-memory` is the legacy module, data directory, service, and compatibility command name.

## Releases and platform contract

The public Python SDK is `hy-memory==2.0.3`. The MemHub source version is `1.0.3`
(build 7), the GitHub repository is
[`Tencent-Hunyuan/Hy-Memory`](https://github.com/Tencent-Hunyuan/Hy-Memory), and the npm package name is
`memhub-cli`. The current code architecture is described in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

The release targets of `v1.0.3` are the Linux x64/arm64 Runtime and the macOS arm64/x64 App and
Runtime. Linux artifacts are built on GitHub Actions native-architecture runners and pass lifecycle
acceptance; macOS artifacts are manually uploaded after the release pipeline completes Developer ID signing, notarization, and acceptance. Windows is not supported.
Only artifacts that appear in both the GitHub Release and `release-manifest.json`, and that pass
public verification, are inside this version's formally supported scope.

The install flow, the future npm Quick Start, recorded sizes, and the data lifecycle are in
[`docs/INSTALLATION.md`](docs/INSTALLATION.md). The repeatable verification entry point for the local HTTPS fake Release is
[`scripts/platform/linux/verify-fake-release-bootstrap.mjs`](scripts/platform/linux/verify-fake-release-bootstrap.mjs).

## Core principles

- One install, one Runtime, one dataset, many clients.
- Python Core is the sole memory data plane and the only writer to the business database.
- Source reads, model processing, and host reads are independent authorization boundaries.
- The formal memory pipeline stays `Evidence -> Hy Memory -> Governance -> Matters/Wiki`.
- The product contains no screen, Accessibility / AXTree, OCR, keyboard/mouse, clipboard, or background audio capture.

## Repository layout

- `apps/core`: Python / FastAPI kernel for Evidence, Hy Memory, governance, retrieval, and the API.
- `apps/cli`: the `one-memory` CLI that installs, controls, and diagnoses the shared Runtime.
- `apps/web`: React / TypeScript admin UI.
- `apps/desktop`: Electron desktop shell and graphical install entry.
- `packages/contracts`: public API types.
- `packages/host-bridge`: Skill, Hook, plugin, and MCP integration for AI Apps.
- `packages/runtime-control`: lifecycle control for Core, Host Gateway, and Runtime.

Architecture details are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and
[`docs/contracts/runtime-unification.md`](docs/contracts/runtime-unification.md).
Product, testing, and frontline support troubleshoot imports and memory sedimentation with
[`docs/PRODUCT_LOG_TROUBLESHOOTING.md`](docs/PRODUCT_LOG_TROUBLESHOOTING.md).

## Development environment

Requires Node.js 24, pnpm 11, Python 3.12, and uv.

```bash
pnpm install
uv sync --directory apps/core --all-groups --locked
pnpm check
```

Common development commands:

```bash
pnpm dev:web
pnpm dev:desktop
pnpm typecheck
pnpm test
pnpm lint
```

## CLI and Runtime

```text
memhub serve
memhub start
memhub stop
memhub restart
memhub status
memhub doctor
memhub install
memhub upgrade
memhub uninstall
memhub delete-data
```

`memhub serve` starts the full Runtime in the foreground. `memhub` is the primary distribution command; the
npm package `memhub-cli` exposes the same command name, and `one-memory` is kept as a legacy compatibility command.
Desktop and CLI use the same Runtime, data directory, and permission contract; closing the window or quitting the
Desktop UI does not stop the Runtime managed by the system service. The shipping macOS App and the CLI share the single LaunchAgent; on App deletion only an
App-owned service stops automatically, while a CLI-owned service keeps running and stays managed by `memhub uninstall`.

The npm package carries only a ~384 KB Bootstrap CLI and contains no Python Core, SQLite data, Web, bundled Node,
Host Bridge, or models. Only after the user explicitly runs `memhub install` does the CLI download and validate the
matching full Runtime; npm `postinstall` downloads no Runtime, registers no service, and modifies no shell profile.

Python Core and the Python dependencies it needs live in the MemHub-managed Runtime and are never written to the
system Python. MemHub's SQLite data lives in `~/.one-memory` by default, not in the npm package or a Runtime version
directory. `npm uninstall` removes only the npm CLI; `memhub uninstall` removes the service, the managed Runtime, and
the CLI link, and keeps Memory by default. Only the separate `memhub delete-data` deletes data, and only after the
Runtime has stopped and the exact path has been confirmed a second time.

Current platform capabilities and install boundaries are governed by the source, CI, and release notes. A platform
that has not passed real-hardware acceptance is never declared usable on cross-platform tests alone.

## Hy Memory

The current dev dependency is pinned to the public PyPI package `hy-memory==2.0.3`. SDK configuration, permission
boundaries, and the search return format are in [`MEMORY.md`](MEMORY.md). The `content` returned by HY search already contains the logical time and evolution context the SDK generates uniformly; callers must not re-convert timestamps
or append `When` a second time.

## Privacy and security

- Privacy notice: [`PRIVACY.md`](PRIVACY.md)
- Security reporting: [`SECURITY.md`](SECURITY.md)
- Contributing guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Code of conduct: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)

## License

MemHub is licensed under the Apache License 2.0. The Tencent copyright notice is in [`NOTICE`](NOTICE); third-party
attributions and licenses are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
[`SBOM.spdx.json`](SBOM.spdx.json).
