# Working and Coding Scenes

The public `scene` parameter of Hy Memory accepts three values:

| `scene` | Purpose | Storage and retrieval path |
|---|---|---|
| `"chat"` | Casual chat, personal facts, general questions | General memory pipeline |
| `"working"` | Productivity tasks such as documents, office work, research, finance, content creation and computer operation | working two-layer memory |
| `"coding"` | Software engineering tasks such as coding, debugging, testing, building, API/SDK, scripts and SQL | coding two-layer memory, plus the version evolution chain |

working and coding reuse the same productivity pipeline structure and data schema, but use different extraction specs, SQLite files and vector collections, so the data never mixes.

## 1. Write Routing

### Explicit scene

A productivity write needs a tool message or assistant `tool_calls`:

```python
from hy_memory import HyMemoryClient

client = HyMemoryClient(mode="pro")

messages = [
    {"role": "user", "content": "Fix the timeout in the search client."},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "arguments": '{"path":"src/search.py"}'
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "apply_patch",
        "content": "Patch applied successfully",
    },
]

result = client.add(
    messages,
    scene="coding",
    user_id="user_123",
    workspace_id="github.com/example/service",
    branch="fix/search-timeout",
)
```

Public behavior:

- `scene="chat"`: skips the scene judge; the tool trace is stripped before the chat pipeline.
- `scene="working"`: skips the scene judge and enters the working pipeline.
- `scene="coding"`: skips the scene judge, enters the coding pipeline and tries to record the version evolution chain.
- Explicit `working/coding` while the input has no tool/tool_calls: the current implementation falls back to the chat pipeline.
- An invalid scene raises `ValueError`.

### Omitting scene

With `scene=None`:

- Input without tool/tool_calls: goes straight to the chat pipeline, with no scene judge LLM call.
- Input with tool/tool_calls: one three-way `chat / working / coding` classification over a bounded summary of the user query and the tool names.
- Judge failure or unparseable output: falls back to chat.
- Once the result is chat, the tool trace is stripped before the chat pipeline.

Classification boundaries:

- `coding`: code or development artifacts are the object of the work or the medium used to do it, including source code, scripts, SQL, notebooks, tests, builds, compile/runtime errors, API/SDK and code review.
- `working`: a clear productivity operation or deliverable, but code is not the core working medium, for example documents, PPT, spreadsheets, translation, finance/business analysis, research reports, email and calendar, project management, design, file organization, software installation and device operation.
- `chat`: casual chat, personal information, advice and general factual questions. Using only search/browser tools does not automatically become working.

## 2. Search Routing

Passing `scene` explicitly selects the matching retrieval path directly, with no judge/rewrite LLM call:

```python
result = client.search(
    "Which timeout implementation was finally kept?",
    scene="coding",
    user_ids=["user_123"],
    workspace_id="github.com/example/service",
    branch="fix/search-timeout",
    limit=5,
)
```

Default and agentic behavior:

| Call | Current behavior |
|---|---|
| `search(query="...", search_mode="default")` | Always chat/normal light search, zero LLM |
| `search(queries=[...], search_mode="default")` | Classifies and rewrites when Coding is on; only rewrite/time when only Query Rewrite is on; zero LLM when both are off |
| `search(query="...", search_mode="agentic")` | LLM three-way classification, query rewrite and time-range resolution |
| `search(queries=[...], search_mode="agentic")` | Same as above, with the preceding queries as context |
| Any mode with an explicit `scene` | scene wins, zero judge/rewrite LLM |

`search_mode="agentic"` uses a bounded query view, so both single and multiple queries go through judge/rewrite. A single query under `search_mode="default"` never calls that LLM.

Under default, multiple queries over `MEMORY_QUERY_REWRITE_MAX_CHARS` are not rewritten; with Coding on a
bounded scene classification still runs, and with Coding off the last query is used directly.

If the coding feature is disabled with `MEMORY_CODING_ENABLED=false`, productivity search falls back safely to chat/normal retrieval.

## 3. Extractor Input Window

The working/coding extractor splits the input into:

- `LAST 5 USER TURNS`: history from earlier writes, used only for coreference resolution and context, never extracted again.
- `NEW MESSAGES`: every message passed to this `add()` call, all of them extraction subjects, not trimmed by the last-5 window.

Current compression rules:

- A tool result is capped at `2048` bytes by default, keeping the head and the tail and eliding the middle; configured by `MEMORY_CODING_TOOL_RESULT_MAX_BYTES`.
- A single assistant text over `6000` characters keeps the first `3000` and the last `3000` characters by default.
- History is trimmed by user-turn window; one user-turn is a user message plus the assistant/tool messages before the next user message.
- `MEMORY_EXTRACT_HISTORY_TURNS` controls how much history the store reads back, default 5; the extractor injects at most the 5 most recent user-turns, so a value above 5 does not widen the history window in the prompt.

For example, when this write contains 10 user-turns, all 10 go into `NEW MESSAGES`; the history section separately supplies at most the 5 most recent user-turns.

working/coding ship both zh and en templates for the Extractor, the experience Reconcile, the fact Reconcile and the cross-session progress summary.
Routing first checks the first 10 effective language characters of the new input; the Extractor only looks at the NEW user
messages of this call and never uses history or tool results to pick the language. Input that starts in Chinese and then
carries a lot of code, paths or error text still takes the zh template; English input that occasionally uses Chinese terms keeps the en template.

## 4. Return Structure

A working/coding write returns a two-layer result:

```python
{
    "success": True,
    "scene": "productivity",
    "request_id": "...",
    "ops": [...],
    "ops_summary": {...},
    "experience_ids": [...],
    "fact_ops": [...],
    "fact_ops_summary": {...},
    "fact_ids": [...],
    "memory_id": "..." or None,
    "elapsed_ms": 123.4,
}
```

`scene="coding"` also returns `chain_rounds`. For historical compatibility that field currently counts the diff hunks
recognized and processed in this call, and is not guaranteed to equal the number of unique evolution rounds after merging.
`0` means no recordable code change was recognized, not that the two-layer memory write failed.

A working/coding search returns:

```python
{
    "request_id": "...",
    "scene": "productivity",
    "facts": [...],
    "experiences": [...],
    "elapsed_ms": 123.4,
}
```

`limit` caps facts and experiences separately, so each layer returns at most `limit` items.

`scene="coding"` additionally returns `chain_block` and `chain_used`:

- `chain_block` is version evolution context aligned with the recalled facts; it is `""` when nothing can be chained.
- `chain_used` holds the indexes into `facts` that already made it into the chain block.
- The SDK never removes those entries from `facts`. Callers that want "chain block + remaining facts" must filter by `chain_used` themselves.

## 5. Storage and Scope

Default local files:

```text
<coding db directory>/
├── coding_memory.db          # working experience/fact metadata
├── coding_memory_cv.db       # coding experience/fact metadata
├── coding_memory_trace.db    # history messages shared by working/coding
└── hy_code_chain/
    └── hy_code.db            # coding version evolution chain
```

Vector collections:

| Variant | Experience layer | Fact layer |
|---|---|---|
| working | `{base}_coding_keys` | `{base}_working_facts` |
| coding | `{base}_coding_keys_cv` | `{base}_working_facts_cv` |

One client can alternate between working and coding; the implementation caches the writer/store per variant.

`workspace_id` and `branch` scope productivity memory:

- Without `workspace_id`: no workspace filtering.
- With `workspace_id`: exact match on that workspace.
- `branch` only takes effect when `workspace_id` is also passed.
- Branch filtering keeps historical rows with `branch IS NULL` so generic experiences recorded without a branch are not dropped.

Use the same normalized `workspace_id` for writes and searches. The owner of a version chain is `user_id` plus `workspace_id`, so patches from different users or repositories cannot chain into each other.

## 6. Coding Version Evolution Chain

The version evolution chain lives in `hy_memory/coding/chain_store.py`. It calls no LLM and no embedding, and recognizes changes from:

- With `MEMORY_CODING_CHAIN_REPO_ROOT` set, diffs are read from the local Git working tree or its changes.
- Deterministic replace actions with path, old and new, recognized from the tool trace.

Edit forms that cannot be parsed reliably are skipped. The chain layer is best-effort: a recording failure never blocks the working/coding two-layer memory write or retrieval.

## 7. Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MEMORY_CODING_ENABLED` | `true` | Enables the working/coding pipeline |
| `MEMORY_CODING_DB_PATH` | `{DATA_DIR}/data/coding_memory.db` | Base SQLite path for working; the coding/trace paths derive from it |
| `MEMORY_CODING_TOOL_RESULT_MAX_BYTES` | `2048` | Byte cap for a single tool result |
| `MEMORY_CODING_VERBATIM_FALLBACK_ENABLED` | `false` | Whether to save a verbatim fact when the model returns nothing; off by default so assistant/tool trace is not surfaced directly |
| `MEMORY_EXTRACT_HISTORY_TURNS` | `5` | Historical user-turns read back; the prompt ultimately uses at most 5 |
| `MEMORY_CODING_CHAIN_REPO_ROOT` | empty | Repository path from which the coding version chain reads local Git diffs |

The `HCM_*` variables are advanced tuning knobs of the version chain, not a stable public configuration contract; ordinary integrations should not depend on their defaults.

## 8. Targeted Verification

```bash
python -m pytest -q \
  tests/coding/test_write_scene_routing.py \
  tests/coding/test_search_scene_routing.py \
  tests/coding/test_scene_coding_variant.py \
  tests/coding/test_scene_coding_search.py \
  tests/coding/test_coding_pipeline_log.py
```
