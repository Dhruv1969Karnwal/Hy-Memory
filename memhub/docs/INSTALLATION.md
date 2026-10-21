# MemHub installation and distribution

Release targets for MemHub `v1.0.3` (build 7):

| Platform | Artifact | Build and acceptance |
| --- | --- | --- |
| Linux x64 | Runtime tar.gz | Native x64 runner on GitHub Actions |
| Linux arm64 | Runtime tar.gz | Native arm64 runner on GitHub Actions |
| macOS arm64 | DMG plus Runtime tar.gz | Developer ID signing, notarization, and native arm64 acceptance |
| macOS x64 | DMG plus Runtime tar.gz | Developer ID signing, notarization, and native Intel acceptance |
| Windows | None | Currently not supported |

Formal support is defined by the `release-manifest.json` of the matching GitHub Release. The manifest records version,
build, Git commit, protocol range, file size, and SHA-256; a platform missing an entry is outside the support scope of
that version. Running the repository's `Linux lifecycle` GitHub Actions workflow manually on a `v*` tag uploads the
Linux Runtime and SHA-256 files automatically to the matching Draft Release; runs on ordinary branches keep only the
Actions Artifact. macOS artifacts are uploaded to the same Draft Release after the release pipeline has signed and
notarized them; GitHub never holds the Apple private key. The Release is published only once every artifact is checked.

The minimum supported macOS version is 13. The current update method is to
manually download the new version from GitHub Releases; there is no auto-updater.

## Public installation

Once the npm package and the same-version GitHub Release are live, the standard entry point is:

```bash
npm install -g memhub-cli
memhub install
memhub init
memhub status
```

`npm install` installs the Bootstrap CLI only. The first `memhub install` run is what:

1. Fetches the `release-manifest.json` for the exact same version and channel as the CLI;
2. Selects the Runtime for the current operating system and CPU architecture;
3. Validates the Git commit, the Runtime manifest schema, and the two-way CLI/Runtime protocol compatibility range;
4. Validates the HTTPS URL, redirect domains, file size, and SHA-256;
5. Extracts safely and cross-validates the inner manifest's version, build, commit, protocol, and per-file hashes;
6. Calls the existing install transaction, registering the user-level service and the managed CLI link;
7. Rolls back on failure, preserving the current Runtime and Memory data.

`memhub install` without `--yes` only downloads the smaller Release manifest and shows a plan; the
full Runtime is downloaded only after explicit confirmation. npm `postinstall` does not download the Runtime, register a service, or modify the shell profile.

## Offline installation

Developers who already hold a Runtime distribution can install offline; the repository itself does not ship that distribution:

```bash
/path/to/memhub-runtime/bin/one-memory install \
  --source /path/to/memhub-runtime \
  --yes
```

With `--source` the installer is fully offline and requests no Release manifest and no artifact.
By default it first generates an install plan; the `--yes` in the example above applies the plan,
but it does not skip manifest, commit, protocol, platform, architecture, or file-hash validation. Legacy offline packages with no protocol declaration are not accepted as a source for new installs.

For isolated testing you can pass explicitly:

```text
--install-root <path>
--home <path>
--bin-dir <path>
--data-dir <path>
--xdg-config-home <path>
--no-shell-path
```

A Linux user-level service requires a usable systemd user session for the current user. To run
in the foreground only, use `memhub serve` from the full Runtime, but that is not a completed service install.

## macOS App and background Runtime

The current macOS App and npm CLI reuse the same Runtime, default data directory, install root,
receipts, ports, protocols, and the single `com.tencent.onememory.runtime` LaunchAgent:

- On its first launch from `/Applications`, the shipping App installs the stable Runtime and the unified LaunchAgent;
- After login the LaunchAgent starts Runtime/Core/Broker directly, without Electron and without opening a window;
- Closing, quitting, or force-quitting the Electron window only disconnects the UI; the background Runtime keeps working;
- "Launch background service at login" controls the private marker used by launchd `KeepAlive.PathState`;
  turning it off does not interrupt the running Runtime and only affects future logins/reboots;
- App-owned services keep checking the App bundle. Once the user deletes `/Applications/MemHub.app`, the
  marker is revoked, Runtime/Core/Broker stop and are not relaunched by KeepAlive; Memory data is kept by default;
- A full `memhub install --yes` or a successful upgrade can transactionally take ownership as CLI-owned.
  After that, deleting the GUI App does not affect the service, which is managed via `memhub uninstall`;
- App-first/CLI-second and CLI-first/App-second both use the same service declaration and the single-writer Runtime.

The full process, owner, and uninstall matrix is in [`ARCHITECTURE.md`](ARCHITECTURE.md). Real logout/login and full
machine reboots are part of release acceptance; source tests or isolated launchctl runs cannot replace that verification.

The downloader uses the standard `HTTPS_PROXY`/`https_proxy` and `NO_PROXY`/`no_proxy` environment variables. The proxy
only carries HTTPS CONNECT tunnels; the checks on the Release URL, redirect domains, size, SHA-256, commit, and protocol
are never skipped because a proxy is in play. If the Release host is listed in `NO_PROXY`, requests go direct.

## Upgrade, uninstall, and data

- `npm uninstall -g memhub-cli`: removes only the npm CLI; it does not stop or remove the managed Runtime and does not delete
  Memory. The full fake Release acceptance runs a real npm uninstall while the Runtime is up, then verifies again that
  systemd, `current`, the managed `memhub`, doctor, and SQLite are all still usable.
- `memhub upgrade`: downloads and validates a new Runtime and restores the old version if the health check fails.
- `memhub uninstall`: stops and removes the managed service, the Runtime versions/current, and the managed CLI link;
  data is kept by default.
- `memhub delete-data`: the only command that deletes Memory; the Runtime must already be stopped, and it requires two
  confirmations, `--yes` plus `--confirm <exact data path>`.

The default data directory is `~/.one-memory`. Python Core is the only business data writer, and SQLite data
never lives in the npm package, an App temp directory, or a managed Runtime version directory. Python Core and
its dependencies are carried by the Runtime and never install packages into the user's system Python.

## Proxies, mirrors, and offline sources

- `--source <path>`: fully offline, highest priority.
- `--release-manifest-url <https-url>`: an explicit Release manifest.
- `MEMHUB_RELEASE_MANIFEST_URL`: pins the manifest URL.
- `MEMHUB_RELEASE_MANIFEST_URL_TEMPLATE`: builds a same-version URL from `{version}`.
- `MEMHUB_DEFAULT_RELEASE_MANIFEST_URL_TEMPLATE`: injects the default template only at release build time; it is not a
  user runtime setting. The source default is empty, so no private repository address is hardcoded.
- `MEMHUB_RELEASE_ALLOWED_HOSTS`: comma-separated extra redirect hostnames for the allow-list, for the case where the
  Release platform redirects to object storage.

Both the manifest and the artifact accept HTTPS only. Cross-hostname redirects are rejected by default and must be explicitly added to the allow-list. On a network interruption, a size/hash mismatch, a secure-extraction failure, or low disk
space, the installer stops before entering the local install transaction.
