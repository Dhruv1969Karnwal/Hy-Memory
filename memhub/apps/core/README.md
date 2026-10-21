# One Memory Core

The Python / FastAPI data plane.
It is the sole owner of Evidence, import jobs, Hy Memory, retrieval, and Graph.

Core can run standalone as the Runtime kernel:

```bash
one-memory-core
```

The legacy `one-memory` command in the Python package stays compatible for now and still offers
`serve/status/doctor`; future public user commands come from `apps/cli`, keeping the kernel process and the full Runtime as separate layers.

For the shared Runtime layering, install boundaries, and acceptance contract between the macOS App
and the future cross-platform CLI, see `docs/contracts/runtime-unification.md`.
