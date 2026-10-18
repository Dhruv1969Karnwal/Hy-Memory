#!/usr/bin/env python3

from pathlib import Path


REQUIRED_PATHS = (
    "NOTICE",
    "docs/ARCHITECTURE.md",
    "docs/DEVELOPMENT_LOOP.md",
    "apps/core/pyproject.toml",
    "apps/cli/package.json",
    "apps/web/package.json",
    "apps/desktop/package.json",
    "packages/contracts/package.json",
    "scripts/audit_core_test_design.py",
    "scripts/generate_core_test_index.py",
    "scripts/summarize_core_test_report.py",
    "pnpm-workspace.yaml",
    "SECURITY.md",
    "PRIVACY.md",
    "CONTRIBUTING.md",
)

#: Paths that must stay gone, with the reason they were removed.
#:
#: A required-paths list cannot express "this was deleted on purpose", so a
#: deleted entry point reads as an oversight and comes back. `deploy/docker`
#: could not build for three independent reasons — pip does not read
#: `[tool.uv.sources]` so the private HY dependency never installed, the web
#: UI was never copied into the image, and the compose port bound loopback
#: only, so a remote deploy was unreachable even if it had built.
FORBIDDEN_PATHS = {
    "deploy/docker": "Deploy entry that could not be built, removed by RL-31 on 2026-09-09; the desktop shell uses pyinstaller + extraResources",
    "AGENTS.md": "Internal agent collaboration notes are not part of the public source",
    "HANDSOFF.md": "Internal handoff and acceptance state is not part of the public source",
    "docs/mvp/PREREQUISITES.md": "Internal release prerequisites and approval tracking are not part of the public source",
    "docs/mvp/SCOPE.md": "Historical open-source planning snapshot is not part of the public source",
    "docs/mvp/packaging-smoke-checklist.md": "Internal packaging acceptance records are not part of the public source",
}

FORBIDDEN_ROOT_GLOBS = {
    "_tmp_*": "Temporary files must not enter the repository root",
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    missing = [
        relative for relative in REQUIRED_PATHS if not (root / relative).exists()
    ]
    if missing:
        for relative in missing:
            print(f"missing: {relative}")
        raise SystemExit(1)

    returned = [
        (relative, reason)
        for relative, reason in FORBIDDEN_PATHS.items()
        if (root / relative).exists()
    ]
    if returned:
        for relative, reason in returned:
            print(f"should not exist: {relative} — {reason}")
        raise SystemExit(1)

    matched_globs = [
        (path.relative_to(root), reason)
        for pattern, reason in FORBIDDEN_ROOT_GLOBS.items()
        for path in root.glob(pattern)
    ]
    if matched_globs:
        for relative, reason in matched_globs:
            print(f"should not exist: {relative} — {reason}")
        raise SystemExit(1)

    print(
        f"structure gate passed: {len(REQUIRED_PATHS)} required paths exist, "
        f"{len(FORBIDDEN_PATHS)} removed paths stayed removed, "
        f"{len(FORBIDDEN_ROOT_GLOBS)} forbidden root patterns stayed empty"
    )


if __name__ == "__main__":
    main()
