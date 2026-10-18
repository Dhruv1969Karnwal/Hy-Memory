# Hy Memory Usage Guide

This guide describes only the methods and return structures that `HyMemoryClient` currently exposes.

## Installation and Quick Start

```bash
pip install hy-memory==2.0.3

# Optional backends
pip install 'hy-memory[qdrant]==2.0.1'
pip install 'hy-memory[faiss]==2.0.1'
pip install 'hy-memory[graph]==2.0.1'     # Neo4j; Kuzu is already a core dependency
pip install 'hy-memory[mysql]==2.0.1'
pip install 'hy-memory[rerank]==2.0.1'    # Local multilingual cross-encoder rerank
```

The LLM and the Embedder both default to OpenAI-compatible interfaces. With OpenAI this is enough:

```bash
export OPENAI_API_KEY="sk-your-key"
```

```python
from hy_memory import HyMemoryClient

with HyMemoryClient(mode="pro") as client:
    client.add(
        "The user likes science-fiction movies, especially Interstellar",
        user_id="user_001",
    )
    result = client.search(
        "What does the user like?",
        user_ids=["user_001"],
    )
    for channel, memories in result["memories"].items():
        for memory in memories:
            print(channel, memory["score"], memory["content"])
```

## Configuration

The SDK loads `.env` from the project directory. The full list of settings is in [env_reference.md](env_reference.md).

```bash
MEMORY_MODE=pro
MEMORY_LLM_MODEL=deepseek-chat
MEMORY_LLM_API_KEY=sk-your-key
MEMORY_LLM_BASE_URL=https://api.deepseek.com
MEMORY_EMBEDDER_MODEL=text-embedding-v4
MEMORY_EMBEDDER_API_KEY=sk-your-key
MEMORY_EMBEDDER_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MEMORY_EMBEDDING_DIMS=1024
```

The currently public `from_config()` works as well:

```python
client = HyMemoryClient.from_config({
    "mode": "pro",
    "llm": {
        "model": "deepseek-chat",
        "api_key": "sk-xxx",
        "base_url": "https://api.deepseek.com",
    },
    "embedder": {
        "model": "text-embedding-v4",
        "api_key": "sk-xxx",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "embedding_dims": 1024,
    },
    "vector_store": {
        "provider": "chroma",
        "embedding_dims": 1024,
    },
})
```

When you configure through `from_config()`, the Embedder and Vector Store dimensions are two separate fields
that must be kept equal; with environment variables a single `MEMORY_EMBEDDING_DIMS` covers both.

Supported vector backends are `chroma`, `qdrant` and `faiss`. Graph backends are usually
`kuzu` or `neo4j`.

## Client API

Synchronous and asynchronous methods take the same arguments. Asynchronous methods are prefixed with `async_`.

### `add()` / `async_add()`

```python
result = client.add(
    [
        {"role": "user", "content": "Recommend a movie"},
        {"role": "assistant", "content": "I recommend Interstellar"},
    ],
    user_id="user_001",
    agent_id="default_agent",
    session_id="default_session",
    source="openclaw",
    scene=None,
    memory_at=None,
    metadata=None,
    meta=None,
    embed_override=None,
)
```

Public values of `scene`:

- `None`: plain text goes straight to chat; messages that contain tools trigger a quick scene judgment.
- `chat`: forces the ordinary memory pipeline.
- `working`: forces the productivity memory pipeline.
- `coding`: forces the software-engineering variant and maintains the version evolution chain.

Forcing `working` / `coding` requires tool messages in the input, otherwise it falls back to chat. A normal chat
write returns fields such as `success`, `scene`, `memory_id`, `request_id` and `elapsed_ms`; the productivity
pipeline returns its two-tier reconcile operations, `experience_ids`, `fact_ids` and similar fields, and the
full structure is documented in [scene_coding_usage.md](scene_coding_usage.md#4-return-structure).

`mode="lite"` runs no LLM extraction and stores written content directly as `L1_RAW`. Search on lite is
strictly raw-only vector retrieval: whatever `layers`, `scene` or `search_mode` the caller passes, only
`L1_RAW` comes back, the working/coding, Schema and Intention pipelines are never entered, and no scene
judgment LLM is called. pro / ultra still honor their own layer and scene parameters and do not expose raw conversations by default.

`source` identifies the calling application, for example `openclaw`, `cursor` or `codex`. It is orthogonal to
`agent_id`: the first says which product or entry point wrote the memory, the second is still the agent
identity used for memory isolation. source is stored after `strip().lower()` and returned with chat, working and coding memories.

### `search()` / `async_search()`

Normal chat retrieval:

```python
result = client.search(
    query="What does the user like?",
    user_ids=["user_001"],
    agent_ids=None,
    session_ids=None,
    include_sources=["openclaw", "codex"],
    exclude_sources=["cursor"],
    limit=10,
    min_score=0.4,
    schema_limit=10,
    schema_min_score=0.4,
    intention_limit=0,
    reader="",                  # HY_MEMORY_READER; defaults to legacy
    layers=None,
    created_after=None,
    search_mode="default",
)
```

Source filtering means:

```text
source IN include_sources AND source NOT IN exclude_sources
```

An empty list drops the matching condition; when the two conflict, the deny list wins. Chroma/Qdrant/Graph
filter before the database candidate list is truncated, and FAISS narrows candidates with an ID selector before
the index search instead of using multi-way recall or filtering after a wider top-k. Aggregate memories derived
from evidence of several sources are marked `mixed`; any source filter excludes `mixed` by default and returns it only when `mixed` is explicitly whitelisted.

The chat success path returns:

```python
{
    "request_id": "...",
    "memories": {
        "normal": [],      # L2/L3 ordinary memories; L4/L5 are currently unused
        "schema": [],      # L6 Schema
        "proactive": [],   # L7 Intention; off by default
    },
    "rewrite": {
        "used": False,
        "original_query": "What does the user like?",
        "effective_query": "What does the user like?",
        "created_after": None,
        "scene": "normal",
    },
    "elapsed_ms": 12.3,
}
```

If the Reader fails, an empty three-bucket `memories` plus `request_id` / `elapsed_ms` is returned; the current
failure fallback does not include `rewrite`.

The L0 user profile is not part of Search. `schema_limit` only bounds L6 and is never taken up by L0.

Search can also route to the separate productivity memory store:

```python
result = client.search(
    "How was this exception eventually fixed?",
    user_ids=["user_001"],
    scene="coding",
    workspace_id="repo-id",
    branch="main",
)
```

The `working` / `coding` paths return `scene="productivity"`, `facts` and `experiences` instead of the chat
three-bucket structure; `coding` also returns `chain_block` and `chain_used`.

#### `search_mode`

| `search_mode` | `scene` | Behavior |
|---|---|---|
| `default` | `None` + a single `query` | Fixed chat/normal, zero LLM |
| `default` | `None` + `queries=[...]` | Decided by the Coding and Query Rewrite switches together, see the table below |
| `agentic` | `None` + single/multiple query | One LLM call does scene judgment, rewrite and time resolution |
| Any | Explicit `chat/working/coding` | Routed directly, no judge/rewrite LLM call |

Pass either `query` or `queries`; the last item of `queries` is the target query and the earlier ones provide
context for resolution. If the LLM or the parsing fails, it falls back to chat/normal and the original query.

Default multi-query input whose total character count stays within `MEMORY_QUERY_REWRITE_MAX_CHARS`:

| `MEMORY_CODING_ENABLED` | `MEMORY_ENABLE_QUERY_REWRITE` | Behavior |
|---|---|---|
| `true` | Any | One LLM call does `chat / working / coding` classification and query rewrite |
| `false` | `true` | One LLM call does rewrite and time resolution, no scene classification |
| `false` | `false` | Zero LLM, fixed chat/normal, uses the last query |

Past the length guard no rewrite runs; with Coding on only a bounded scene classification runs, with Coding off there is zero LLM.

### `get_user_profile()` / `async_get_user_profile()`

The user profile is determined by `user_id` alone and accepts neither `agent_id` nor `session_id`:

```python
result = client.get_user_profile(user_id="user_001")
profile = result["profile"]
```

```python
{
    "request_id": "...",
    "profile": {
        "memory_id": "...",
        "user_id": "user_001",
        "content": "...",
        "layer": "l0_basic_info",
        "basic_info": {"name": "Alice"},
        "catalog": {"attached": False},
        "supplement": {"attached": False},
        "tags": [],
        "memory_at": 1234567890,
        "gmt_created": 1234567890,
    },
    "elapsed_ms": 1.2,
}
```

When no L0 exists and there is no balanced catalog to attach, `profile` is `None`. This call uses neither
Embedding nor LLM. Once the matching switches are enabled, the active catalog and supplement are attached to the
same profile; a catalog can also form a read-only synthetic profile when there is no base L0.

### CRUD, Listing and Cloning

All of the following are currently public methods:

```python
memory = client.get("memory_id")
result = client.update("memory_id", "updated content")
result = client.delete("memory_id")
result = client.delete_all(user_id="user_001")

page = client.list_memories(
    user_id="user_001",
    limit=100,
    offset=0,
    order="desc",
    layers=None,
    with_evolution=False,
)

result = client.clone_user("user_001", "user_002", agent_id=None)
```

The matching async methods are `async_get()`, `async_update()`, `async_delete()`,
`async_delete_all()`, `async_list_memories()` and `async_clone_user()`.

Plain `list_memories()` returns the `vdb` bucket and adds `graph` whenever Graph has results; when a
`workspace_id` is passed it returns productivity memory as `scene="productivity"` and `experiences`.

### `evolve()` / `async_evolve()` (Ultra)

System 2 is not triggered automatically by `add()`, and there is no `get_write_status()` interface either. After
a batch of writes is complete, the caller runs it explicitly:

```python
result = client.evolve(
    user_id="user_001",
    agent_id="default_agent",
)
```

`evolve()` is only allowed with `mode="ultra"`; the synchronous method blocks until this round of cognitive
processing is done. It reads the full current VDB, creates or updates L6 Schema, and runs the System 2 processing the configuration allows.

### Other Currently Public Methods

- `finalize_session()` / `async_finalize_session()`: archives a given session when the feature switch is on.
- `get_user_basic_info()` / `async_get_user_basic_info()`: reads the L0 basic info record.
- `sweep()` / `async_sweep()`: explicitly runs the cross-domain sweeper on Ultra.
- `get_metrics()` / `async_get_metrics()`: reads SDK metrics.
- `build_memory_graph()` / `async_build_memory_graph()`: backfills the ordinary memory relation graph.
- `build_entity_store()` / `async_build_entity_store()`: backfills the separate Entity Store.
- `history()` / `async_history()` and `get_recent_history()` / `async_get_recent_history()`.

## Embedding Endpoint Override

`add()` / `async_add()` / `search()` / `async_search()` accept a per-call
`embed_override`:

```python
override = {
    "base_url": "http://embedding-host:8000/v1",
    "api_key": "sk-xxx",
    "extra_headers": {"X-Route": "direct"},
    "timeout": 15,
}
client.add("The user likes basketball", user_id="u1", embed_override=override)
client.search("hobbies", user_ids=["u1"], embed_override=override)
```

It overrides only the access endpoint, never the model, the dimensions or `extra_body`. The new endpoint must
run an Embedding model compatible with the data that is already stored.

## Data Isolation and Processing Modes

Ordinary memory uses the three-level scope `user_id → agent_id → session_id`. Search requires a non-empty
`user_ids`; when `agent_ids` / `session_ids` are omitted the search spans those dimensions.

The user profile is the exception: L0, the balanced catalog and the specific supplement are all determined by
`user_id` alone, from generation through to reading.

- `lite`: the raw text is vectorized, no extraction LLM is called.
- `pro`: Embedding + LLM extraction + reconcile; Graph can be turned on with `MEMORY_ENABLE_GRAPH`.
- `ultra`: everything pro offers with Graph forced on; System 2 runs when the caller explicitly calls `evolve()`.

Reader, Rerank and Search parameters are covered in [hy_memory_search_skill.md](hy_memory_search_skill.md).
