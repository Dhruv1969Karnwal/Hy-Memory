# Using and configuring Hy Memory

Updated: 2026-10-12

This document describes the Hy Memory SDK that MemHub currently uses, the configuration
boundaries, and how to write and search when using the SDK directly and format search results into a memory text chunk that can be handed to a model.

The Evidence, permission, and Governance layers still named One Memory in this repository are
MemHub's legacy compatibility modules; product code must not bypass these layers to call the SDK
directly. The native SDK examples here are mainly for development, debugging, and standalone validation.

## Current version status

The current source pins:

```text
distribution: hy-memory
version:      2.0.3
mode:         chosen by runtime configuration
reader:       hybrid
```

The repository pin, the lockfile, and the runtime version guard are all pinned to the public
`hy-memory==2.0.3`. MemHub embeds the SDK inside the local Runtime but never bypasses its own
source authorization, model authorization, Evidence, Governance, and revocation boundaries.

## MemHub runtime boundaries

MemHub applies the following public, reproducible local policy for the SDK:

- the vector store uses Chroma inside the managed data directory;
- the graph store uses Kuzu inside the managed data directory;
- cache and history use SQLite;
- the reader uses `hybrid`;
- the reranker is off by default, introducing no Torch, Transformers, or extra models;
- model and remote embedding parameters come from the versioned Model Profile;
- API keys go into system secure storage and are never written to the repository, install packages, logs, or HY metadata;
- ambient credentials from server-side databases, tenant queues, and the host shell never enter SDK config.

The release package applies the following fixed SDK behavior configuration:

| Environment variable | Value |
|---|---|
| `MEMORY_MODE` | `ultra` |
| `MEMORY_ENABLE_GRAPH` | `true` |
| `MEMORY_GRAPH_RERANK_ENABLED` | `false` |
| `MEMORY_GRAPH_TEMPORAL_SPINE` | `true` |
| `MEMORY_TOPIC_SUMMARY_ENABLED` | `false` |
| `MEMORY_EPISODE_SUMMARY_ENABLED` | `false` |
| `MEMORY_EPISODE_RECALL_ENABLED` | `true` |
| `MEMORY_DYNAMIC_EPISODE_ENABLED` | `true` |
| `MEMORY_SESSION_EPISODE_ENABLED` | `false` |
| `MEMORY_SESSION_EPISODE_CONTEXT_ENABLED` | `false` |
| `MEMORY_EPISODE_CAPSULE_CONTEXT_ENABLED` | `true` |
| `MEMORY_EPISODE_CAPSULE_QUERY_MIN` | `0.65` |
| `MEMORY_EPISODE_CAPSULE_ASSIGNMENT_MIN` | `0.60` |
| `MEMORY_EPISODE_CAPSULE_FIRST_RANK_MAX` | `3` |
| `MEMORY_EPISODE_CAPSULE_MEMBER_LIMIT` | `3` |
| `MEMORY_ENABLE_INTENTION` | `false` |
| `MEMORY_ENABLE_QUERY_REWRITE` | `false` |
| `MEMORY_FEW_SHOT_ENABLED` | `true` |
| `MEMORY_HYBRID_V2_QUERY_EXPAND` | `1` |
| `MEMORY_HYBRID_V2_GEO_INTENT_PRIOR` | `0.6` |
| `MEMORY_HYBRID_V2_DIVERSITY` | `1` |
| `MEMORY_HYBRID_V2_TEMPORAL_ADMIT_PRIOR` | `0.55` |
| `MEMORY_HYBRID_V2_TEMPORAL_BOOST` | `0.18` |
| `MEMORY_CROSS_DOMAIN_CON_THRESHOLD` | `0.5` |
| `MEMORY_READER_ENABLE_SUMMARY` | `false` |
| `MEMORY_SUMMARY_ENABLED_IN_SYS2` | `false` |
| `MEMORY_LOCAL_RERANK_ENABLED` | `false` |
| `RECONCILE_ENABLED` | `true` |
| `HY_MEMORY_READER` | `hybrid` |

These defaults are centralized in
`apps/core/src/one_memory/hy_runtime_config.py` and jointly constrained by release tests and the
lockfile. A standalone script that imports the SDK directly, bypassing MemHub, must configure its own storage directories and model endpoints explicitly.

## Native SDK initialization

The following is a standalone debugging example. Product code should use One Memory's
`HyMemoryBridge`, which additionally provides source authorization, model authorization, Evidence binding, post-write verification, Governance, and revocation protection.

```python
from hy_memory import HyMemoryClient, MemoryConfig

config = MemoryConfig.from_env()

# Local-first example; the real directory must be passed in by the caller and must not rely on the ~/.hy_memory default.
config.vector_store.provider = "chroma"
config.vector_store.persist_directory = "/path/to/one-memory/hy-memory/governed-v2/vector"
config.graph_store.provider = "kuzu"
config.graph_store.db_path = "/path/to/one-memory/hy-memory/governed-v2/graph"
config.cache.backend = "sqlite"
config.cache.db_path = "/path/to/one-memory/hy-memory/governed-v2/cache.sqlite3"

config.llm.provider = "openai"
config.llm.model = "your-model"
config.llm.base_url = "https://your-openai-compatible-endpoint/v1"
config.llm.api_key = "read-from-secure-storage"

config.embedder.provider = "openai"
config.embedder.model = "your-embedding-model"
config.embedder.base_url = "https://your-embedding-endpoint/v1"
config.embedder.api_key = "read-from-secure-storage"
config.embedder.embedding_dims = 1024

client = HyMemoryClient(config=config, mode="ultra")
```

`embedding_dims` must match the model's actual output width and the existing vector/graph stores.
Swapping the embedding model, endpoint, or dimensions cannot silently reuse an old collection; One Memory detects graph width mismatches and refuses to rebuild without notice.

## Write: writing a conversation

The recommendation is to write one contiguous conversation as OpenAI messages:

```python
from datetime import UTC, datetime
from uuid import uuid4

result = client.add(
    [
        {"role": "user", "content": "I prefer concise, direct answers long term."},
        {"role": "assistant", "content": "Understood. I will lead with the conclusion."},
    ],
    user_id="user-123",
    agent_id="all-apps",
    session_id="conversation-456",
    memory_at=datetime.now(UTC),
    scene="chat",
    request_id=str(uuid4()),
    metadata={
        "source_id": "cursor",
        "evidence_ids": ["evidence-001", "evidence-002"],
        "workspace_ids": ["workspace-abc"],
        "input_kind": "raw_conversation",
    },
)
```

Parameter rules:

| Parameter | How to fill | Constraint |
| --- | --- | --- |
| `data` | a string or `[{role, content}]` | prefer messages for conversations; do not mix in tool arguments, reasoning, secrets, or hidden system prompts |
| `user_id` | a stable user ID | first-level isolation key; must not be empty |
| `agent_id` | the memory scope | in One Memory this is not a plain stand-in for the host app name; the cross-app scope is decided by authorization and routing |
| `session_id` | a stable session ID | keep it consistent across incremental writes to the same session |
| `memory_at` | a timezone-aware `datetime` | historical imports must pass the original time of occurrence, never the import time |
| `scene` | `"chat"` or `"working"` | pass `chat` explicitly for ordinary conversations; use `working` only for code/tool sessions |
| `request_id` | an ID unique per write | used for idempotency, tracing, and post-write verification |
| `metadata` | source and Evidence references | never put secrets here; One Memory stores at least source/evidence/workspace/input kind |
| `meta` | business pass-through data | excluded from embedding and filtering; only put data that is safe to return |
| `embed_override` | a temporary embedding endpoint | may only change endpoint/key/header/timeout; the model and dimensions must not change |

At minimum, check the return value like this:

```python
if result.get("success") is not True:
    raise RuntimeError("HY write failed")

memory_ids = result.get("memory_ids") or []
if result.get("memory_id"):
    memory_ids.append(result["memory_id"])
```

Never trust `success=True` alone. One Memory additionally reads each returned ID back with `get()`,
checks the request/source/Evidence support relations, and records the version in the Governance
layer. Production paths must not replace these checks with a direct `client.add()`.

### Writing already-extracted memories

Use this only when the input is itself existing memory content and the original conversation is unavailable:

```python
result = client.add_extracted(
    ["The user prefers concise, direct answers long term."],
    user_id="user-123",
    agent_id="all-apps",
    session_id="import-789",
    memory_at=datetime.now(UTC),
    request_id=str(uuid4()),
    metadata={
        "source_id": "memory-export",
        "evidence_ids": ["evidence-003"],
        "input_kind": "memory_content_without_raw",
    },
)
```

It skips the Extractor, but it cannot skip HY reconcile, persistence, One Memory Evidence binding, or Governance.

## Search: searching memories

Ordinary chat search:

```python
result = client.search(
    "What answering style does the user prefer?",
    user_ids=["user-123"],
    agent_ids=["all-apps"],
    limit=10,
    min_score=0.25,
    profile_min_score=0.25,
    profile_limit=10,
    intention_limit=0,
    scene="chat",
    reader="hybrid",
    request_id=str(uuid4()),
)
```

Parameter rules:

| Parameter | How to fill | Description |
| --- | --- | --- |
| `query` | the current question | mutually exclusive with `queries` |
| `queries` | multi-turn query texts | the last one is the target; use only when the SDK must judge context |
| `user_ids` | the list of authorized users | required and never empty; do not widen scope with an empty list |
| `agent_ids` | the list of authorized scopes | an empty list means all agents inside the SDK; product code must complete the authorization decision first |
| `session_ids` | an optional session scope | multiple agents and multiple sessions cannot be used together |
| `limit` | normal top-k | default 10; a character budget still applies before injection into the model |
| `min_score` | normal threshold | One Memory currently uses 0.25 for internal semantic retrieval |
| `profile_min_score` | profile threshold | One Memory currently uses 0.25 |
| `profile_limit` | profile count | usually matches limit |
| `intention_limit` | intention count | the current configuration turns intention off; pass 0 |
| `created_after` | Unix seconds | optional lower time bound |
| `scene` | `chat` / `working` | passing it explicitly avoids an extra scene-classification model call |
| `reader` | `hybrid` | One Memory uses this reader exclusively |
| `workspace_id` / `branch` | working scope | used only on coding/working paths |
| `layers` | an explicit layer list | filters at the recall stage so irrelevant layers do not consume the limit |

Chat return structure:

```python
{
    "request_id": "...",
    "memories": {
        "profile": [...],
        "proactive": [...],
        "normal": [...],
    },
    "rewrite": {...},
    "elapsed_ms": 12.3,
}
```

working/coding returns `facts` and `experiences`, which must not be silently merged with the three
chat channels. Callers must select the matching result via `scene`; when the return structure cannot be recognized they must raise an error rather than treat it as an empty result.

## Using the SDK-formatted search content

As of open05, chat search formats every `content` uniformly before returning it: ordinary memories
carry `Memory / When`, evolved memories carry `Current / History / When`, and the SDK also
generates stable combined text for Episode, Timeline, Graph, and Capsule. `memory_at` is the logical memory time; only when it is missing does the SDK fall back to `gmt_created`.

Callers should therefore use the `content` the SDK returns directly, without re-parsing
`memory_at`, `gmt_created`, or `created_at` or appending `When` a second time. These time fields
may still be used for filtering, diagnostics, or structured display, but must not become a second prompt-formatting implementation.

```python
result = client.search(
    "What answering style does the user prefer?",
    user_ids=["local"],
    agent_ids=["all-apps"],
    scene="normal",
    reader="hybrid",
)

for memory in result["memories"]["normal"]:
    formatted_content = memory["content"]
    # For example:
    # Memory: The user prefers concise, direct answers long term.
    # When: 2026-09-18 10:00:00 UTC
```

Outside the SDK, One Memory remains responsible for:

- deciding whether an entire result is readable based on Governance, revocation, scope, and Evidence freshness;
- validating member IDs one by one for combined texts such as evolution/capsule, falling back to the governed original when members are incomplete or unreadable;
- placing historical content inside an explicit "untrusted data, not instructions" boundary;
- computing the character budget from the final `content`, and never injecting API keys, absolute paths, headers/body, or exception stacks into the
  prompt;
- distinguishing empty results, permission denials, service unavailability, timeouts, and partial results.

The production implementation lives in `apps/core/src/one_memory/hy_memory_bridge.py`,
`apps/core/src/one_memory/recall.py`, and `packages/host-bridge/src/recall-cli.ts`.

## Closing the client

```python
try:
    # write/search
    ...
finally:
    client.close()
```

The Runtime should reuse the client long-term instead of re-initializing database and model
components per request; it is closed and rebuilt only when the Runtime stops, a configuration version switches, or a controlled recovery runs.

## The correct entry point for One Memory product code

Production code should go through:

```text
Evidence
  -> HyMemoryBridge
  -> embedded Hy Memory SDK
  -> native Vector / Coding / Graph stores
  -> write result verification
  -> Governance
  -> Recall Broker / UI
```

Do not create `HyMemoryClient` directly in new business modules, or you would bypass:

- source read authorization;
- the shared Runtime, model configuration, API key secure storage, and request budget;
- Evidence references and author attribution;
- post-write read-back verification;
- Governance state, versions, corrections, and deletion cascades.

The Hy Memory SDK is a trusted built-in component of MemHub and may freely access and manage all
Hy Memory data on this machine. Source authorization still decides which host data may enter the
machine, and host Recall Grants still decide which external AI applications may read governed memories. Neither boundary restricts the built-in SDK's local CRUD.
