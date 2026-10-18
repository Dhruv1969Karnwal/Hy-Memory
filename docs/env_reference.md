# Hy Memory SDK Environment Variable Reference

This page lists only the configuration that the current code actually reads and that directly affects public
SDK functionality. Every variable can be set through the process environment or `.env`; when unset, the default value from the table applies.

## Core Configuration

| Variable | Default | Description |
|---|---|---|
| `MEMORY_MODE` | `pro` | `lite` / `pro` / `ultra` |
| `MEMORY_DATA_DIR` | `~/.hy_memory` | Root directory for local data, cache and logs |

- `lite`: embedding-only writes, Graph is always off.
- `pro`: LLM extraction and reconcile; Graph is controlled by `MEMORY_ENABLE_GRAPH`.
- `ultra`: everything `pro` offers plus Graph forced on; System 2 runs only when the caller explicitly
  calls `client.evolve()`, and is never auto-enqueued after `add()`.

## LLM

| Variable | Default | Description |
|---|---|---|
| `MEMORY_LLM_PROVIDER` | `openai` | OpenAI-compatible Chat Completions protocol |
| `MEMORY_LLM_MODEL` | `gpt-4.1-nano` | Model name |
| `MEMORY_LLM_API_KEY` | — | Fallbacks: `LLM_API_KEY`, `OPENAI_API_KEY` |
| `MEMORY_LLM_BASE_URL` | — | Falls back to `LLM_BASE_URL` |
| `MEMORY_LLM_TEMPERATURE` | `0.1` | Shared temperature; `none` / `omit` / an empty value means do not send it |
| `MEMORY_LLM_OMIT_TEMPERATURE` | `false` | When `true`, `temperature` is never sent at all |
| `MEMORY_LLM_MAX_TOKENS` | `1024` | General generation cap |
| `MEMORY_AGENT_MAX_TOKENS` | `2000` | Cap for agent calls such as Extract/Reconcile/System 2 |
| `MEMORY_LLM_TIMEOUT` | `180` | Per-attempt timeout, seconds |
| `MEMORY_LLM_MAX_RETRIES` | `5` | Maximum number of retries |
| `MEMORY_LLM_RETRY_DELAY` | `1.0` | Retry interval, seconds |
| `MEMORY_LLM_EXTRA_HEADERS` | — | JSON object with extra HTTP headers |
| `MEMORY_LLM_EXTRA_BODY` | — | JSON object with extra request body fields |
| `MEMORY_FEW_SHOT_ENABLED` | `false` | Attach few-shot examples to the built-in Extractor/Reconcile prompts |
| `MEMORY_ENABLE_INTENTION` | `true` | Whether to extract and write L7 Intention |
| `MEMORY_LOG_LLM_CONTENT` | `false` | Only `true` allows logs to record the full prompt/response |

## Embedding

| Variable | Default | Description |
|---|---|---|
| `MEMORY_EMBEDDER_PROVIDER` | `openai` | OpenAI-compatible Embeddings protocol |
| `MEMORY_EMBEDDER_MODEL` | `text-embedding-3-small` | Model name |
| `MEMORY_EMBEDDER_API_KEY` | — | Fallbacks: `EMBEDDING_API_KEY`, `OPENAI_API_KEY` |
| `MEMORY_EMBEDDER_BASE_URL` | — | Falls back to `EMBEDDING_BASE_URL` |
| `MEMORY_EMBEDDING_DIMS` | `1536` | Collection dimensions; should match the model output |
| `MEMORY_EMBEDDER_TIMEOUT` | `15` | Per-attempt timeout, seconds |
| `MEMORY_EMBEDDER_MAX_RETRIES` | `2` | Maximum number of retries |
| `MEMORY_EMBEDDER_RETRY_DELAY` | `0.3` | Retry interval, seconds |
| `MEMORY_EMBEDDER_EXTRA_HEADERS` | — | JSON object with extra HTTP headers |
| `MEMORY_EMBEDDER_EXTRA_BODY` | — | JSON object with extra request body fields |

## Vector Store

| Variable | Default | Description |
|---|---|---|
| `MEMORY_VECTOR_STORE` | `chroma` | `chroma` / `qdrant` / `faiss` |
| `MEMORY_COLLECTION_NAME` | `agent_memories` | Name of the main collection |
| `MEMORY_PERSIST_DIR` | `{DATA_DIR}/data/vector_db` | Local persistence directory |
| `MEMORY_VECTOR_HOST` | — | Host of a remote backend such as Qdrant |
| `MEMORY_VECTOR_PORT` | — | Port of the remote backend |
| `MEMORY_VECTOR_API_KEY` | — | API key for the remote backend |
Keyword candidates for `hybrid` depend on the backend implementation. Chroma degrades to pure vectors;
Qdrant can use sparse/BM25 candidates.

## Graph Store

| Variable | Default | Description |
|---|---|---|
| `MEMORY_ENABLE_GRAPH` | `false` | Only takes effect on `pro`; forced off on `lite`, forced on on `ultra` |
| `MEMORY_GRAPH_PROVIDER` | `kuzu` | `kuzu` / `neo4j` |
| `MEMORY_GRAPH_DB_PATH` | `{DATA_DIR}/data/kuzu_db` | Kuzu database path |
| `NEO4J_URL` | `bolt://localhost:7687` | Neo4j Bolt URL; may fall back to `MEMGRAPH_URL` |
| `NEO4J_USERNAME` | — | May fall back to `MEMGRAPH_USERNAME` |
| `NEO4J_PASSWORD` | — | May fall back to `MEMGRAPH_PASSWORD` |
| `NEO4J_DATABASE` | `neo4j` | Neo4j database |

Advanced switches for Graph writes and read-side Rerank are documented in [pro_graph_design.md](pro_graph_design.md).

## Cache

| Variable | Default | Description |
|---|---|---|
| `MEMORY_CACHE_BACKEND` | `sqlite` | `sqlite` / `mysql`; Redis is not supported |
| `MEMORY_CACHE_DB_PATH` | `{DATA_DIR}/data/cache.db` | SQLite file path |
| `MEMORY_MYSQL_HOST` | `localhost` | MySQL host |
| `MEMORY_MYSQL_PORT` | `3306` | MySQL port |
| `MEMORY_MYSQL_USER` | `root` | MySQL user |
| `MEMORY_MYSQL_PASSWORD` | — | MySQL password |
| `MEMORY_MYSQL_DATABASE` | `hy_memory` | MySQL database |
| `MEMORY_MYSQL_POOL_SIZE` | `10` | Connection pool size |
| `MEMORY_MYSQL_POOL_RECYCLE` | `3600` | Connection recycle period, seconds |

## Write, Scene and Query Rewrite

| Variable | Default | Description |
|---|---|---|
| `RECONCILE_ENABLED` | `true` | Whether candidate retrieval and reconcile run |
| `MEMORY_ENABLE_SEARCH_QUERY` | `false` | Whether an LLM additionally generates a retrieval query before reconcile |
| `MEMORY_EXTRACT_HISTORY_TURNS` | `5` | Number of earlier user turns injected into the Extractor besides the current input |
| `MEMORY_CODING_ENABLED` | `true` | Whether the separate working/coding pipeline is enabled |
| `MEMORY_CODING_DB_PATH` | `{DATA_DIR}/data/coding_memory.db` | SQLite path for productivity memory |
| `MEMORY_CODING_TOOL_RESULT_MAX_BYTES` | `2048` | Maximum bytes per tool result |
| `MEMORY_CODING_VERBATIM_FALLBACK_ENABLED` | `false` | Whether to store the rendered conversation as a working fact when nothing is extracted |
| `MEMORY_ENABLE_QUERY_REWRITE` | `false` | With Coding off, controls whether default multi-query input is rewritten; with Coding on, multi-query input is always classified and rewritten whatever this value is |
| `MEMORY_QUERY_REWRITE_MAX_CHARS` | `1000` | Length guard for default multi-query rewrite |

By default a single-query Search is a zero-LLM light search. Under `search_mode=agentic`, both single and
multi-query input go through one merged pass of scene judgment, rewrite and time resolution. Default
multi-query input past the length guard is not rewritten; with Coding on only classification runs, with it off there is zero LLM.

## System 2 and User Profile

| Variable | Default | Description |
|---|---|---|
| `SWEEPER_ENABLED` | `true` | Whether Ultra evolve runs the cross-domain sweeper |
| `MEMORY_SUMMARY_ENABLED_IN_SYS2` | `false` | Whether System 2 material includes L3 Summary |
| `MEMORY_EVOLING_BALANCED_CATALOG_ENABLED` | `false` | Whether Ultra evolve generates a balanced catalog |
| `MEMORY_EVOLING_BALANCED_CATALOG_MAX_FACTS` | `120` | Maximum number of L2 entries reviewed in one run |
| `MEMORY_EVOLING_BALANCED_CATALOG_CONTENT_CHARS` | `180` | Character cap per L2 entry in the prompt |
| `MEMORY_EVOLING_BALANCED_CATALOG_MAX_TOKENS` | `512` | Output cap for catalog selection |
| `MEMORY_EVOLING_SPECIFIC_SUPPLEMENT_ENABLED` | `false` | Whether Ultra evolve generates a supplement |
| `MEMORY_USER_PROFILE_BALANCED_CATALOG_ENABLED` | `false` | Whether `get_user_profile()` attaches the active catalog |
| `MEMORY_USER_PROFILE_SPECIFIC_SUPPLEMENT_ENABLED` | `false` | Whether the active supplement is attached |

Catalog and supplement are determined by `user_id` alone; they never appear in Search as ordinary L6 and do
not consume `schema_limit`.

### Cross-domain Sweeper

| Variable | Default | Description |
|---|---|---|
| `MEMORY_CROSS_DOMAIN_ENABLE` | `true` | Whether the cross-domain collision scan runs |
| `MEMORY_CROSS_DOMAIN_MIN_BASICS` | `5` | Minimum number of schemas needed to enter the general collision scan |
| `MEMORY_CROSS_DOMAIN_BEH_THRESHOLD` | `0.70` | Behavior-vector similarity threshold |
| `MEMORY_CROSS_DOMAIN_CON_THRESHOLD` | `0.60` | Cross-domain difference threshold for the content vector |

## Reader and Local Rerank

| Variable | Default | Description |
|---|---|---|
| `HY_MEMORY_READER` | `legacy` | `legacy` (vector) / `hybrid` (vector + backend keyword candidates) |
| `MEMORY_HYBRID_V2_VDB_WEIGHT_BM25` | `0.35` | Hybrid BM25 boost weight; the old prefix is still kept |
| `MEMORY_LOCAL_RERANK_ENABLED` | `false` | Local cross-encoder Rerank shared by both Readers |
| `MEMORY_LOCAL_RERANK_MODEL` | `nreimers/mmarco-mMiniLMv2-L6-H384-v1` | Model name or local path of the small multilingual Hugging Face model |
| `MEMORY_LOCAL_RERANK_DEVICE` | `auto` | A device such as `auto` / `cpu` / `cuda` |
| `MEMORY_LOCAL_RERANK_BATCH_SIZE` | `16` | Batch per forward pass, not a request concurrency limit |
| `MEMORY_LOCAL_RERANK_MAX_LENGTH` | `512` | Maximum tokenizer length |
| `MEMORY_LOCAL_RERANK_POOL_FACTOR` | `3` | Candidate pool multiplier over the final limit |
| `HY_MEMORY_TAG_INDEX_WRITE` | `true` | Whether the write-side Tag Index is maintained |

The SDK sets no concurrency cap for Rerank Search. Once enabled, the model is loaded and warmed up once during
`HyMemoryClient` initialization, and a failed initialization keeps the Client from becoming ready. A single
inference failure during operation still fails open. See [hy_memory_search_skill.md](hy_memory_search_skill.md).

## Episode and Summaries

| Variable | Default | Description |
|---|---|---|
| `MEMORY_EPISODE_GAP_SECONDS` | `1800` | A new episode starts once this gap is exceeded |
| `MEMORY_EPISODE_MAX_MEMORIES` | `50` | User-side memory cap per episode |
| `MEMORY_EPISODE_SUMMARY_ENABLED` | `false` | Whether an L3 Summary is generated when an episode ends |
| `MEMORY_EPISODE_SUMMARY_MIN_MEMORIES` | `3` | Minimum number of memories for a Summary |
| `MEMORY_EPISODE_RECALL_ENABLED` | `false` | Whether Search weights Episode relationships |
| `MEMORY_EPISODE_RECALL_WEIGHT` | `0.7` | Weight of Episode structural support |
| `MEMORY_TOPIC_SUMMARY_ENABLED` | `false` | Whether Topic Summaries are generated |
| `MEMORY_TOPIC_SUMMARY_MIN_MEMORIES` | `5` | Minimum number of memories for a Topic Summary |
| `MEMORY_TOPIC_SUMMARY_STEP` | `5` | How many memories a Topic gains before its Summary is rebuilt |
| `MEMORY_SESSION_EPISODE_ENABLED` | `false` | Whether `finalize_session()` may archive a given session |
| `MEMORY_SESSION_EPISODE_CONTEXT_ENABLED` | `false` | Whether the Reader recalls Session Episode context |
| `MEMORY_SESSION_EPISODE_LIMIT` | `5` | Candidate cap for Session Episodes |
| `MEMORY_SESSION_EPISODE_FACT_LIMIT` | `2` | Number of facts attached to each Session Episode |

## Supplementary Recall Features

| Variable | Default | Description |
|---|---|---|
| `MEMORY_RECALL_STRENGTH_ENABLED` | `false` | Whether normal results are reranked by access count and idle time |
| `MEMORY_RECALL_STRENGTH_TAU` | `180` | Decay time constant, days |
| `MEMORY_RECALL_ACCESS_TRACKING` | `true` | Whether access information is written back when Strength is on |
| `MEMORY_ENTITY_STORE_ENABLED` | `false` | Whether a separate entity collection is maintained; the current Reader does not read it directly |

For older data, the currently public `client.build_entity_store(user_id)` builds the Entity Store afterwards.

## Logging, Trace and History

| Variable | Default | Description |
|---|---|---|
| `MEMORY_LOG_LEVEL` | `INFO` | Runtime log level |
| `MEMORY_LOG_FORMAT` | `text` | `text` / `json` |
| `MEMORY_LOG_DIR` | — | Unified log root directory |
| `MEMORY_TRACE_LOG_DIR` | — | Legacy alias for the JSONL log directory; prefer `MEMORY_LOG_DIR` |
| `MEMORY_PIPELINE_TRACE_ENABLED` | `true` | Whether rows are written to the pipeline trace table in cache |
| `MEMORY_MEMORY_OPERATIONS_ENABLED` | `false` | Whether memory operation audit records are written |
| `MEMORY_HISTORY_ENABLE` | `true` | Whether history recording is enabled |
| `MEMORY_HISTORY_DB_PATH` | `{DATA_DIR}/data/history.db` | History SQLite path |
| `MEMORY_HISTORY_RECORD_SEARCHES` | `false` | Whether Search operations are recorded |
| `MEMORY_CODING_CHAIN_REPO_ROOT` | — | Repository root from which the coding version chain reads local Git diffs |

## HTTP Server

| Variable | Default | Description |
|---|---|---|
| `HY_MEMORY_SERVER_HOST` | `127.0.0.1` | Address the Server listens on |
| `HY_MEMORY_SERVER_PORT` | `19527` | Port the Server listens on |

Legacy settings have no effect; the table on this page and `.env.example` define what is currently supported.
Cache backends currently support only `sqlite` and `mysql`.
