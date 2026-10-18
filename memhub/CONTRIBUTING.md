# Contributing

Thanks for taking part in One Memory.

## Development environment

- Node.js 24
- pnpm 11
- Python 3.12
- uv

Install the dependencies and run the checks:

```bash
pnpm install
uv sync --directory apps/core --all-groups --locked
pnpm check
```

## Submission flow

1. Create a short-lived feature branch from `main`.
2. When fixing a defect or adding behavior, first add tests that prove the problem and the result.
3. Keep changes focused; do not mix in generated files, real user data, or unrelated formatting.
4. Open a Pull Request describing the user impact, test evidence, privacy impact, and rollback path.
5. Wait for CI and code review to pass before merging.

When external source code is involved, record the project, pinned version or commit, license, files adopted, and modification notes. Never commit keys, internal addresses, real conversations,
personal paths, model weights, or signing certificates.
