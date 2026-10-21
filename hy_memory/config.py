"""
Agent Memory - Configuration management

Provides flexible configuration options via environment variables and config files
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List
from pathlib import Path


def _load_dotenv():
    """Load the .env file (if present), searching parent directories."""
    # Search upward from this file for .env
    cur = Path(__file__).resolve().parent
    env_path = None
    for _ in range(5):
        candidate = cur / ".env"
        if candidate.exists():
            env_path = candidate
            break
        cur = cur.parent
    if env_path is None:
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Do not overwrite environment variables that already exist
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()


def _get_env_float(key: str, default: float) -> float:
    """Get a float environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_int(key: str, default: int) -> int:
    """Get an int environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_bool(key: str, default: bool) -> bool:
    """Get a bool environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def _default_data_dir() -> str:
    """
    Return the data root directory.

    Priority:
      1. Env var MEMORY_DATA_DIR (mount path for cloud deployments, e.g. /data/memory)
      2. ~/.hy_memory (default path for local pip installs)

    All local storage (VDB, graph database, SQLite, logs) is derived from this directory.
    When a remote service is configured (e.g. Qdrant host, Neo4j url), the matching local path is unused.
    """
    return os.getenv("MEMORY_DATA_DIR", os.path.join(Path.home(), ".hy_memory"))


@dataclass
class VectorStoreConfig:
    """Vector store configuration."""

    provider: str = None  # qdrant, chroma, faiss
    collection_name: str = None  # Collection name
    tag_index_collection_name: Optional[str] = None  # env: MEMORY_TAG_INDEX_COLLECTION_NAME
    persist_directory: str = None  # Persistence directory
    embedding_dims: int = None  # Embedding dimensions
    on_disk: bool = True  # Whether to persist to disk

    # Connection settings (remote storage)
    host: Optional[str] = None
    port: Optional[int] = None
    api_key: Optional[str] = None

    def __post_init__(self):
        if self.provider is None:
            self.provider = os.getenv("MEMORY_VECTOR_STORE", "chroma")
        if self.collection_name is None:
            self.collection_name = os.getenv("MEMORY_COLLECTION_NAME", "agent_memories")
        if self.tag_index_collection_name is None:
            self.tag_index_collection_name = os.getenv(
                "MEMORY_TAG_INDEX_COLLECTION_NAME"
            )
        if self.persist_directory is None:
            self.persist_directory = os.getenv(
                "MEMORY_PERSIST_DIR",
                os.path.join(_default_data_dir(), "data", "vector_db"),
            )
        if self.embedding_dims is None:
            self.embedding_dims = _get_env_int("MEMORY_EMBEDDING_DIMS", 1536)
        if self.host is None:
            self.host = os.getenv("MEMORY_VECTOR_HOST")
        if self.port is None:
            port_str = os.getenv("MEMORY_VECTOR_PORT")
            self.port = int(port_str) if port_str else None
        if self.api_key is None:
            self.api_key = os.getenv("MEMORY_VECTOR_API_KEY")


@dataclass
class LLMConfig:
    """LLM configuration (used for memory extraction and processing)."""

    # The provider is uniformly openai for now; third-party services connect through a compatible base_url.
    provider: str = None
    model: str = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    # Generation parameters
    temperature: float = None
    # True = **omit temperature from the request entirely**.
    # Some gateways (e.g. Opus 5) accept only temperature=1 or nothing at all; 0/0.1/0.3 returns 400.
    # env: MEMORY_LLM_OMIT_TEMPERATURE=true, or MEMORY_LLM_TEMPERATURE=none/omit
    omit_temperature: bool = None
    max_tokens: int = None  # Default max_tokens (general)
    agent_max_tokens: int = None  # Shared max_tokens for agent scenarios

    # Call parameters
    timeout: int = None  # LLM request timeout (seconds)
    max_retries: int = None  # Maximum number of retries
    retry_delay: float = None  # Retry interval (seconds)

    # Feature flags
    few_shot_enabled: bool = None  # Whether to append few-shot examples to the extractor / reconcile prompts (off by default, env: MEMORY_FEW_SHOT_ENABLED)
    structure_metadata_enabled: bool = None  # env: MEMORY_STRUCTURE_METADATA_ENABLED

    # In default mode this applies only when the caller explicitly passes queries=[...]; the flag decides whether the
    # legacy multi-query route is upgraded to "classify + rewrite + time parsing". A single query is never affected.
    # search_mode="agentic" ignores this flag and runs judge + rewrite for both single and multi query.
    enable_query_rewrite: bool = None
    query_rewrite_max_chars: int = None  # Skip rewrite when the joined length of queries exceeds this (env MEMORY_QUERY_REWRITE_MAX_CHARS, default 1000)

    # Intention (L7) extraction switch: when off, the extractor prompt no longer asks for intentions
    # and the writer stops persisting L7. Future plans worth recording belong in memory (L2). On by default.
    enable_intention: bool = None  # env MEMORY_ENABLE_INTENTION, default true

    # Custom extractor system prompt (client level, configured only via from_config/LLMConfig, no env var).
    # None = use the built-in default; when non-empty, that language's extractor uses this system prompt directly (no built-in few-shot appended).
    # The user template (data injection: content/last_messages/memory_at/current_date) still uses the built-in one.
    extract_system_prompt_zh: Optional[str] = None  # Extractor system prompt override for Chinese input
    extract_system_prompt_en: Optional[str] = None  # Extractor system prompt override for English input

    # OpenAI-compatible extensions (extra_headers / extra_body passed through as-is)
    extra_headers: Optional[Dict[str, str]] = None
    extra_body: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.provider is None:
            self.provider = os.getenv("MEMORY_LLM_PROVIDER", "openai")
        if self.model is None:
            self.model = os.getenv("MEMORY_LLM_MODEL", "gpt-4.1-nano")
        if self.api_key is None:
            self.api_key = os.getenv(
                "MEMORY_LLM_API_KEY",
                os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY")),
            )
        if self.base_url is None:
            self.base_url = os.getenv("MEMORY_LLM_BASE_URL", os.getenv("LLM_BASE_URL"))
        if self.temperature is None:
            # MEMORY_LLM_TEMPERATURE=none/omit/null/empty -> do not send this parameter
            _raw_temp = os.getenv("MEMORY_LLM_TEMPERATURE")
            if _raw_temp is not None and _raw_temp.strip().lower() in (
                "", "none", "null", "omit", "unset",
            ):
                if self.omit_temperature is None:
                    self.omit_temperature = True
                self.temperature = 0.1   # Placeholder; never sent while omit_temperature is true
            else:
                self.temperature = _get_env_float("MEMORY_LLM_TEMPERATURE", 0.1)
        if self.omit_temperature is None:
            self.omit_temperature = _get_env_bool("MEMORY_LLM_OMIT_TEMPERATURE", False)
        if self.max_tokens is None:
            self.max_tokens = _get_env_int("MEMORY_LLM_MAX_TOKENS", 1024)
        if self.agent_max_tokens is None:
            self.agent_max_tokens = _get_env_int("MEMORY_AGENT_MAX_TOKENS", 2000)
        if self.timeout is None:
            self.timeout = _get_env_int("MEMORY_LLM_TIMEOUT", 180)
        if self.max_retries is None:
            self.max_retries = _get_env_int("MEMORY_LLM_MAX_RETRIES", 5)
        if self.retry_delay is None:
            self.retry_delay = _get_env_float("MEMORY_LLM_RETRY_DELAY", 1.0)
        if self.few_shot_enabled is None:
            # Few-shot is off by default; when enabled, extractor / reconcile prompts append examples.
            self.few_shot_enabled = _get_env_bool("MEMORY_FEW_SHOT_ENABLED", False)
        if self.structure_metadata_enabled is None:
            self.structure_metadata_enabled = _get_env_bool(
                "MEMORY_STRUCTURE_METADATA_ENABLED", False
            )
        if self.enable_query_rewrite is None:
            self.enable_query_rewrite = _get_env_bool("MEMORY_ENABLE_QUERY_REWRITE", False)
        if self.query_rewrite_max_chars is None:
            self.query_rewrite_max_chars = _get_env_int("MEMORY_QUERY_REWRITE_MAX_CHARS", 1000)
        if self.enable_intention is None:
            self.enable_intention = _get_env_bool("MEMORY_ENABLE_INTENTION", True)
        # extra_headers / extra_body: passed through to the OpenAI-compatible API.
        # If the env vars are never actually read,
        # "disable thinking" settings such as MEMORY_LLM_EXTRA_BODY='{"reasoning_effort":"none"}'
        # fail silently -- the reasoning model's chain of thought still consumes max_tokens.
        for _attr, _env in (
            ("extra_headers", "MEMORY_LLM_EXTRA_HEADERS"),
            ("extra_body", "MEMORY_LLM_EXTRA_BODY"),
        ):
            if getattr(self, _attr) is not None:
                continue
            _val = os.getenv(_env)
            if not _val:
                continue
            import json as _json

            try:
                _parsed = _json.loads(_val)
            except (ValueError, TypeError):
                _parsed = None
            # Invalid JSON or non-dict values are ignored (stay None); a config error must not break client construction
            if isinstance(_parsed, dict):
                setattr(self, _attr, _parsed)


@dataclass
class EmbedderConfig:
    """Embedder configuration (always over the OpenAI-compatible protocol)."""

    provider: str = None  # openai (compatible with DashScope, DeepSeek, etc.)
    model: str = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    embedding_dims: int = None

    # Retry parameters
    max_retries: int = None  # Maximum number of retries
    retry_delay: float = None  # Retry interval (seconds)
    timeout: int = None  # Request timeout (seconds)

    # Internal platform extensions (extra_headers / extra_body are passed to the OpenAI client)
    extra_headers: Optional[Dict[str, str]] = None
    extra_body: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.provider is None:
            self.provider = os.getenv("MEMORY_EMBEDDER_PROVIDER", "openai")
        if self.model is None:
            self.model = os.getenv("MEMORY_EMBEDDER_MODEL", "text-embedding-3-small")
        if self.api_key is None:
            self.api_key = os.getenv(
                "MEMORY_EMBEDDER_API_KEY",
                os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY")),
            )
        if self.base_url is None:
            self.base_url = os.getenv(
                "MEMORY_EMBEDDER_BASE_URL", os.getenv("EMBEDDING_BASE_URL")
            )
        if self.embedding_dims is None:
            val = os.getenv("MEMORY_EMBEDDING_DIMS")
            self.embedding_dims = int(val) if val else None
        if self.max_retries is None:
            # Retry count: retry only on timeout/connection errors, raise business errors immediately. Default 2.
            # Worst-case wall time ≈ timeout × (max_retries+1) + retry_delay × max_retries
            #           = 3 × 3 + 0.3 × 2 ≈ 9.6s
            self.max_retries = _get_env_int("MEMORY_EMBEDDER_MAX_RETRIES", 2)
        if self.retry_delay is None:
            self.retry_delay = _get_env_float("MEMORY_EMBEDDER_RETRY_DELAY", 0.3)
        if self.timeout is None:
            # Timeout for a single embed request. Long-running production samples already reach ~8s at p50; 3s would
            # cut off requests that could have succeeded and cause retry storms. SDK-internal retries are disabled, so
            # 15s is the real ceiling per explicit attempt; deployments can override it via MEMORY_EMBEDDER_TIMEOUT.
            self.timeout = _get_env_int("MEMORY_EMBEDDER_TIMEOUT", 15)
        if self.extra_headers is None:
            val = os.getenv("MEMORY_EMBEDDER_EXTRA_HEADERS")
            if val:
                import json as _json

                try:
                    self.extra_headers = _json.loads(val)
                except Exception:
                    pass
        if self.extra_body is None:
            val = os.getenv("MEMORY_EMBEDDER_EXTRA_BODY")
            if val:
                import json as _json

                try:
                    self.extra_body = _json.loads(val)
                except Exception:
                    pass


@dataclass
class RecallConfig:
    """Retrieval configuration."""

    # Default retrieval limits
    default_limit: int = 10
    max_limit: int = 100
    default_limit_per_layer: int = 5

    # Scoring weights
    semantic_weight: float = 0.5
    recency_weight: float = 0.3
    importance_weight: float = 0.15
    access_weight: float = 0.05

    # Recency decay
    recency_decay_days: int = 30  # Decay period (days)
    recency_decay_factor: float = 0.9  # Decay factor

    # Memory Strength (time-decay ranking based on idle time)
    # strength = (1 + log(access_count)) * exp(-idle_days / tau), multiplied into the normal channel score
    strength_enabled: bool = False        # Master switch (off by default): when off, no strength reranking and no access write-back
    strength_tau: float = 180.0           # Decay time constant (days)
    access_tracking_enabled: bool = True  # Whether to write back access_count / last_accessed_at after search (gated by the master switch)

    # Entity Store (separate {collection}_entities collection)
    # This switch **controls the write side only**: when on, persisting L2_FACT also extracts entities and refreshes the store.
    # The shared read path may boost candidates using graph entity hits; the underlying entity collection stays
    # available for the write side and migration tools.
    entity_store_enabled: bool = False    # Write-side switch (off by default), env: MEMORY_ENTITY_STORE_ENABLED

    # ── Episode (internal session) ──
    # Groups "one thing" across write batches under the same episode_id; the read side uses it to scope candidates for weighting.
    episode_gap_seconds: float = None   # env: MEMORY_EPISODE_GAP_SECONDS
    episode_max_memories: int = None    # env: MEMORY_EPISODE_MAX_MEMORIES
    episode_recall_enabled: bool = None # env: MEMORY_EPISODE_RECALL_ENABLED (read-side switch)
    episode_recall_weight: float = None # env: MEMORY_EPISODE_RECALL_WEIGHT
    # Summarize once when an episode closes (replaces the every-N-turns rewrite of rolling summary, saving tokens)
    episode_summary_enabled: bool = None      # env: MEMORY_EPISODE_SUMMARY_ENABLED
    episode_summary_min_memories: int = None  # env: MEMORY_EPISODE_SUMMARY_MIN_MEMORIES
    # First-class session episode: finalized explicitly by the caller, producing a full narrative straight from
    # that session's L1_RAW; the read side leads with the episode plus a few atomic facts.
    session_episode_enabled: bool = None       # env: MEMORY_SESSION_EPISODE_ENABLED
    session_episode_context_enabled: bool = None  # env: MEMORY_SESSION_EPISODE_CONTEXT_ENABLED
    session_episode_limit: int = None          # env: MEMORY_SESSION_EPISODE_LIMIT
    session_episode_fact_limit: int = None     # env: MEMORY_SESSION_EPISODE_FACT_LIMIT
    dynamic_episode_enabled: bool = None       # env: MEMORY_DYNAMIC_EPISODE_ENABLED
    dynamic_episode_summary_concurrency: int = None  # env: MEMORY_DYNAMIC_EPISODE_SUMMARY_CONCURRENCY
    series_episode_enabled: bool = None        # env: MEMORY_SERIES_EPISODE_ENABLED
    episode_capsule_context_enabled: bool = None  # env: MEMORY_EPISODE_CAPSULE_CONTEXT_ENABLED
    episode_capsule_query_min: float = None    # env: MEMORY_EPISODE_CAPSULE_QUERY_MIN
    episode_capsule_assignment_min: float = None  # env: MEMORY_EPISODE_CAPSULE_ASSIGNMENT_MIN
    episode_capsule_first_rank_max: int = None # env: MEMORY_EPISODE_CAPSULE_FIRST_RANK_MAX
    episode_capsule_member_limit: int = None   # env: MEMORY_EPISODE_CAPSULE_MEMBER_LIMIT
    episode_capsule_render_mode: str = None    # env: MEMORY_EPISODE_CAPSULE_RENDER_MODE
    # Compact L3 is a retrieval-only container: retrieve its referenced L2
    # members into the local-rerank pool, but never return the L3 body itself.
    episode_member_expansion_enabled: bool = None  # env: MEMORY_EPISODE_MEMBER_EXPANSION_ENABLED
    episode_member_expansion_mode: str = None  # env: MEMORY_EPISODE_MEMBER_EXPANSION_MODE
    episode_member_expansion_episode_limit: int = None  # env: MEMORY_EPISODE_MEMBER_EXPANSION_EPISODE_LIMIT
    episode_member_expansion_member_limit: int = None  # env: MEMORY_EPISODE_MEMBER_EXPANSION_MEMBER_LIMIT
    episode_member_expansion_per_episode: int = None  # env: MEMORY_EPISODE_MEMBER_EXPANSION_PER_EPISODE
    episode_member_expansion_episode_min_similarity: float = None  # env: MEMORY_EPISODE_MEMBER_EXPANSION_EPISODE_MIN_SIMILARITY
    episode_member_expansion_min_similarity: float = None  # env: MEMORY_EPISODE_MEMBER_EXPANSION_MIN_SIMILARITY
    same_series_timeline_enabled: bool = None  # env: MEMORY_SAME_SERIES_TIMELINE_ENABLED
    local_rerank_enabled: bool = None          # env: MEMORY_LOCAL_RERANK_ENABLED
    local_rerank_model: str = None             # env: MEMORY_LOCAL_RERANK_MODEL
    local_rerank_pool_factor: int = None        # env: MEMORY_LOCAL_RERANK_POOL_FACTOR
    local_rerank_max_length: int = None         # env: MEMORY_LOCAL_RERANK_MAX_LENGTH
    local_rerank_batch_size: int = None         # env: MEMORY_LOCAL_RERANK_BATCH_SIZE
    local_rerank_device: str = None             # env: MEMORY_LOCAL_RERANK_DEVICE
    dynamic_entity_tag_embed_enabled: bool = None  # env: MEMORY_HYBRID_V2_DYNAMIC_ENTITY_TAG_EMBED_ENABLED
    tag_index_nominee_enabled: bool = None       # env: MEMORY_TAG_INDEX_NOMINEE_ENABLED
    tag_index_nominee_topk: int = None           # env: MEMORY_TAG_INDEX_NOMINEE_TOPK
    tag_index_nominee_min_score: float = None    # env: MEMORY_TAG_INDEX_NOMINEE_MIN_SCORE
    tag_index_nominee_min_matches: int = None    # env: MEMORY_TAG_INDEX_NOMINEE_MIN_MATCHES
    tag_index_nominee_search_limit: int = None   # env: MEMORY_TAG_INDEX_NOMINEE_SEARCH_LIMIT
    tag_index_nominee_limit: int = None          # env: MEMORY_TAG_INDEX_NOMINEE_LIMIT
    tag_index_nominee_weight: float = None       # env: MEMORY_TAG_INDEX_NOMINEE_WEIGHT
    tag_index_nominee_max_tag_df: int = None     # env: MEMORY_TAG_INDEX_NOMINEE_MAX_TAG_DF
    hierarchy_bundle_enabled: bool = None       # env: MEMORY_HIERARCHY_BUNDLE_ENABLED
    hierarchy_bundle_max_members: int = None    # env: MEMORY_HIERARCHY_BUNDLE_MAX_MEMBERS
    hierarchy_bundle_token_budget: int = None   # env: MEMORY_HIERARCHY_BUNDLE_TOKEN_BUDGET
    hierarchy_bundle_topic_limit: int = None    # env: MEMORY_HIERARCHY_BUNDLE_TOPIC_LIMIT
    hierarchy_bundle_node_beam: int = None      # env: MEMORY_HIERARCHY_BUNDLE_NODE_BEAM
    hierarchy_bundle_member_pool: int = None    # env: MEMORY_HIERARCHY_BUNDLE_MEMBER_POOL
    hierarchy_bundle_allowed_node_types: str = None  # env: MEMORY_HIERARCHY_BUNDLE_ALLOWED_NODE_TYPES
    hierarchy_bundle_novel_floor_margin: float = None  # env: MEMORY_HIERARCHY_BUNDLE_NOVEL_FLOOR_MARGIN
    multi_seed_bundle_enabled: bool = None     # env: MEMORY_MULTI_SEED_BUNDLE_ENABLED
    multi_seed_bundle_mode: str = None         # env: MEMORY_MULTI_SEED_BUNDLE_MODE
    multi_seed_bundle_alpha: float = None      # env: MEMORY_MULTI_SEED_BUNDLE_ALPHA
    multi_seed_bundle_beta: float = None       # env: MEMORY_MULTI_SEED_BUNDLE_BETA
    multi_seed_bundle_gamma: float = None      # env: MEMORY_MULTI_SEED_BUNDLE_GAMMA
    multi_seed_bundle_member_floor: float = None  # env: MEMORY_MULTI_SEED_BUNDLE_MEMBER_FLOOR
    multi_seed_bundle_group_floor: float = None   # env: MEMORY_MULTI_SEED_BUNDLE_GROUP_FLOOR
    multi_seed_bundle_tag_s1_floor: float = None  # env: MEMORY_MULTI_SEED_BUNDLE_TAG_S1_FLOOR
    multi_seed_bundle_tau: float = None        # env: MEMORY_MULTI_SEED_BUNDLE_TAU
    multi_seed_bundle_lexical_weight: float = None  # env: MEMORY_MULTI_SEED_BUNDLE_LEXICAL_WEIGHT
    multi_seed_bundle_max_tag_df: int = None   # env: MEMORY_MULTI_SEED_BUNDLE_MAX_TAG_DF
    multi_seed_bundle_scan_limit: int = None   # env: MEMORY_MULTI_SEED_BUNDLE_SCAN_LIMIT
    multi_seed_bundle_max_members: int = None  # env: MEMORY_MULTI_SEED_BUNDLE_MAX_MEMBERS
    multi_seed_bundle_max_bundles: int = None  # env: MEMORY_MULTI_SEED_BUNDLE_MAX_BUNDLES
    multi_seed_bundle_min_episode_seeds: int = None  # env: MEMORY_MULTI_SEED_BUNDLE_MIN_EPISODE_SEEDS
    multi_seed_bundle_single_tag_seed_ceiling: float = None  # env: MEMORY_MULTI_SEED_BUNDLE_SINGLE_TAG_SEED_CEILING
    multi_seed_bundle_token_budget: int = None  # env: MEMORY_MULTI_SEED_BUNDLE_TOKEN_BUDGET
    retrieval_clauses_enabled: bool = None       # env: MEMORY_RETRIEVAL_CLAUSES_ENABLED
    retrieval_clauses_min_words: int = None      # env: MEMORY_RETRIEVAL_CLAUSES_MIN_WORDS
    retrieval_clauses_min_score: float = None    # env: MEMORY_RETRIEVAL_CLAUSES_MIN_SCORE
    retrieval_clauses_margin: float = None       # env: MEMORY_RETRIEVAL_CLAUSES_MARGIN
    retrieval_clauses_cap: int = None            # env: MEMORY_RETRIEVAL_CLAUSES_CAP
    retrieval_clauses_require_keyword: bool = None  # env: MEMORY_RETRIEVAL_CLAUSES_REQUIRE_KEYWORD
    retrieval_clauses_min_keyword_score: float = None  # normalized BM25 floor
    supplemental_propositions_enabled: bool = None  # env: MEMORY_SUPPLEMENTAL_PROPOSITIONS_ENABLED

    # Minimum score threshold
    min_score_threshold: float = 0.0

    def __post_init__(self):
        self.default_limit = _get_env_int("MEMORY_RECALL_DEFAULT_LIMIT", 10)
        self.max_limit = _get_env_int("MEMORY_RECALL_MAX_LIMIT", 100)
        self.semantic_weight = _get_env_float("MEMORY_RECALL_SEMANTIC_WEIGHT", 0.5)
        self.recency_weight = _get_env_float("MEMORY_RECALL_RECENCY_WEIGHT", 0.3)
        self.importance_weight = _get_env_float("MEMORY_RECALL_IMPORTANCE_WEIGHT", 0.15)
        self.access_weight = _get_env_float("MEMORY_RECALL_ACCESS_WEIGHT", 0.05)
        self.recency_decay_days = _get_env_int("MEMORY_RECALL_DECAY_DAYS", 30)
        self.strength_enabled = _get_env_bool("MEMORY_RECALL_STRENGTH_ENABLED", False)
        self.strength_tau = _get_env_float("MEMORY_RECALL_STRENGTH_TAU", 180.0)
        self.access_tracking_enabled = _get_env_bool("MEMORY_RECALL_ACCESS_TRACKING", True)
        self.entity_store_enabled = _get_env_bool("MEMORY_ENTITY_STORE_ENABLED", False)
        if self.episode_gap_seconds is None:
            # 30 minutes: across writes but close in time → same discussion
            self.episode_gap_seconds = _get_env_float("MEMORY_EPISODE_GAP_SECONDS", 1800.0)
        if self.episode_max_memories is None:
            # The cap keeps an always-open session from becoming one giant episode
            self.episode_max_memories = _get_env_int("MEMORY_EPISODE_MAX_MEMORIES", 50)
        if self.episode_recall_enabled is None:
            self.episode_recall_enabled = _get_env_bool("MEMORY_EPISODE_RECALL_ENABLED", False)
        if self.episode_recall_weight is None:
            self.episode_recall_weight = _get_env_float("MEMORY_EPISODE_RECALL_WEIGHT", 0.7)
        if self.episode_summary_enabled is None:
            self.episode_summary_enabled = _get_env_bool("MEMORY_EPISODE_SUMMARY_ENABLED", False)
        if self.episode_summary_min_memories is None:
            self.episode_summary_min_memories = _get_env_int(
                "MEMORY_EPISODE_SUMMARY_MIN_MEMORIES", 3)
        if self.session_episode_enabled is None:
            self.session_episode_enabled = _get_env_bool(
                "MEMORY_SESSION_EPISODE_ENABLED", False
            )
        if self.session_episode_context_enabled is None:
            self.session_episode_context_enabled = _get_env_bool(
                "MEMORY_SESSION_EPISODE_CONTEXT_ENABLED", False
            )
        if self.session_episode_limit is None:
            self.session_episode_limit = _get_env_int(
                "MEMORY_SESSION_EPISODE_LIMIT", 5
            )
        if self.session_episode_fact_limit is None:
            self.session_episode_fact_limit = _get_env_int(
                "MEMORY_SESSION_EPISODE_FACT_LIMIT", 2
            )
        if self.dynamic_episode_enabled is None:
            self.dynamic_episode_enabled = _get_env_bool(
                "MEMORY_DYNAMIC_EPISODE_ENABLED", False
            )
        if self.dynamic_episode_summary_concurrency is None:
            self.dynamic_episode_summary_concurrency = max(1, min(16, _get_env_int(
                "MEMORY_DYNAMIC_EPISODE_SUMMARY_CONCURRENCY", 3
            )))
        if self.series_episode_enabled is None:
            self.series_episode_enabled = _get_env_bool(
                "MEMORY_SERIES_EPISODE_ENABLED", False
            )
        if self.episode_capsule_context_enabled is None:
            self.episode_capsule_context_enabled = _get_env_bool(
                "MEMORY_EPISODE_CAPSULE_CONTEXT_ENABLED", False
            )
        if self.episode_capsule_query_min is None:
            self.episode_capsule_query_min = _get_env_float(
                "MEMORY_EPISODE_CAPSULE_QUERY_MIN", 0.65
            )
        if self.episode_capsule_assignment_min is None:
            self.episode_capsule_assignment_min = _get_env_float(
                "MEMORY_EPISODE_CAPSULE_ASSIGNMENT_MIN", 0.60
            )
        if self.episode_capsule_first_rank_max is None:
            self.episode_capsule_first_rank_max = _get_env_int(
                "MEMORY_EPISODE_CAPSULE_FIRST_RANK_MAX", 3
            )
        if self.episode_capsule_member_limit is None:
            self.episode_capsule_member_limit = _get_env_int(
                "MEMORY_EPISODE_CAPSULE_MEMBER_LIMIT", 3
            )
        if self.episode_capsule_render_mode is None:
            self.episode_capsule_render_mode = os.getenv(
                "MEMORY_EPISODE_CAPSULE_RENDER_MODE", "full"
            ).strip().lower()
        if self.episode_capsule_render_mode not in {"full", "l2_visible"}:
            self.episode_capsule_render_mode = "full"
        if self.episode_member_expansion_enabled is None:
            self.episode_member_expansion_enabled = _get_env_bool(
                "MEMORY_EPISODE_MEMBER_EXPANSION_ENABLED", False
            )
        if self.episode_member_expansion_mode is None:
            self.episode_member_expansion_mode = os.getenv(
                "MEMORY_EPISODE_MEMBER_EXPANSION_MODE", "anchored_attach"
            ).strip().lower()
        if self.episode_member_expansion_mode not in {
            "anchored_attach", "anchored", "direct", "series_direct",
        }:
            self.episode_member_expansion_mode = "anchored_attach"
        if self.episode_member_expansion_episode_limit is None:
            self.episode_member_expansion_episode_limit = _get_env_int(
                "MEMORY_EPISODE_MEMBER_EXPANSION_EPISODE_LIMIT", 3
            )
        if self.episode_member_expansion_member_limit is None:
            self.episode_member_expansion_member_limit = _get_env_int(
                "MEMORY_EPISODE_MEMBER_EXPANSION_MEMBER_LIMIT", 6
            )
        if self.episode_member_expansion_per_episode is None:
            self.episode_member_expansion_per_episode = _get_env_int(
                "MEMORY_EPISODE_MEMBER_EXPANSION_PER_EPISODE", 3
            )
        if self.episode_member_expansion_episode_min_similarity is None:
            self.episode_member_expansion_episode_min_similarity = _get_env_float(
                "MEMORY_EPISODE_MEMBER_EXPANSION_EPISODE_MIN_SIMILARITY", 0.35
            )
        if self.episode_member_expansion_min_similarity is None:
            self.episode_member_expansion_min_similarity = _get_env_float(
                "MEMORY_EPISODE_MEMBER_EXPANSION_MIN_SIMILARITY", 0.20
            )
        if self.same_series_timeline_enabled is None:
            self.same_series_timeline_enabled = _get_env_bool(
                "MEMORY_SAME_SERIES_TIMELINE_ENABLED", False
            )
        if self.local_rerank_enabled is None:
            self.local_rerank_enabled = _get_env_bool(
                "MEMORY_LOCAL_RERANK_ENABLED", False
            )
        if self.local_rerank_model is None:
            self.local_rerank_model = os.getenv(
                "MEMORY_LOCAL_RERANK_MODEL",
                "nreimers/mmarco-mMiniLMv2-L6-H384-v1",
            )
        if self.local_rerank_pool_factor is None:
            self.local_rerank_pool_factor = _get_env_int(
                "MEMORY_LOCAL_RERANK_POOL_FACTOR", 3
            )
        if self.local_rerank_max_length is None:
            self.local_rerank_max_length = _get_env_int(
                "MEMORY_LOCAL_RERANK_MAX_LENGTH", 512
            )
        if self.local_rerank_batch_size is None:
            self.local_rerank_batch_size = _get_env_int(
                "MEMORY_LOCAL_RERANK_BATCH_SIZE", 16
            )
        if self.local_rerank_device is None:
            self.local_rerank_device = os.getenv(
                "MEMORY_LOCAL_RERANK_DEVICE", "auto"
            )
        if self.dynamic_entity_tag_embed_enabled is None:
            self.dynamic_entity_tag_embed_enabled = _get_env_bool(
                "MEMORY_HYBRID_V2_DYNAMIC_ENTITY_TAG_EMBED_ENABLED", False
            )
        if self.tag_index_nominee_enabled is None:
            self.tag_index_nominee_enabled = _get_env_bool(
                "MEMORY_TAG_INDEX_NOMINEE_ENABLED", False
            )
        if self.tag_index_nominee_topk is None:
            self.tag_index_nominee_topk = _get_env_int(
                "MEMORY_TAG_INDEX_NOMINEE_TOPK", 10
            )
        if self.tag_index_nominee_min_score is None:
            self.tag_index_nominee_min_score = _get_env_float(
                "MEMORY_TAG_INDEX_NOMINEE_MIN_SCORE", 0.4
            )
        if self.tag_index_nominee_min_matches is None:
            self.tag_index_nominee_min_matches = _get_env_int(
                "MEMORY_TAG_INDEX_NOMINEE_MIN_MATCHES", 2
            )
        if self.tag_index_nominee_search_limit is None:
            self.tag_index_nominee_search_limit = _get_env_int(
                "MEMORY_TAG_INDEX_NOMINEE_SEARCH_LIMIT", 40
            )
        if self.tag_index_nominee_limit is None:
            self.tag_index_nominee_limit = _get_env_int(
                "MEMORY_TAG_INDEX_NOMINEE_LIMIT", 1
            )
        if self.tag_index_nominee_weight is None:
            self.tag_index_nominee_weight = _get_env_float(
                "MEMORY_TAG_INDEX_NOMINEE_WEIGHT", 0.3
            )
        if self.tag_index_nominee_max_tag_df is None:
            self.tag_index_nominee_max_tag_df = _get_env_int(
                "MEMORY_TAG_INDEX_NOMINEE_MAX_TAG_DF", 8
            )
        if self.hierarchy_bundle_enabled is None:
            self.hierarchy_bundle_enabled = _get_env_bool(
                "MEMORY_HIERARCHY_BUNDLE_ENABLED", False
            )
        if self.hierarchy_bundle_max_members is None:
            self.hierarchy_bundle_max_members = _get_env_int(
                "MEMORY_HIERARCHY_BUNDLE_MAX_MEMBERS", 8
            )
        if self.hierarchy_bundle_token_budget is None:
            self.hierarchy_bundle_token_budget = _get_env_int(
                "MEMORY_HIERARCHY_BUNDLE_TOKEN_BUDGET", 400
            )
        if self.hierarchy_bundle_topic_limit is None:
            self.hierarchy_bundle_topic_limit = _get_env_int(
                "MEMORY_HIERARCHY_BUNDLE_TOPIC_LIMIT", 12
            )
        if self.hierarchy_bundle_node_beam is None:
            self.hierarchy_bundle_node_beam = _get_env_int(
                "MEMORY_HIERARCHY_BUNDLE_NODE_BEAM", 4
            )
        if self.hierarchy_bundle_member_pool is None:
            self.hierarchy_bundle_member_pool = _get_env_int(
                "MEMORY_HIERARCHY_BUNDLE_MEMBER_POOL", 48
            )
        if self.hierarchy_bundle_allowed_node_types is None:
            self.hierarchy_bundle_allowed_node_types = os.getenv(
                "MEMORY_HIERARCHY_BUNDLE_ALLOWED_NODE_TYPES", ""
            )
        if self.hierarchy_bundle_novel_floor_margin is None:
            raw_novel_floor_margin = os.getenv(
                "MEMORY_HIERARCHY_BUNDLE_NOVEL_FLOOR_MARGIN", ""
            ).strip()
            self.hierarchy_bundle_novel_floor_margin = (
                float(raw_novel_floor_margin)
                if raw_novel_floor_margin else None
            )
        if self.multi_seed_bundle_enabled is None:
            self.multi_seed_bundle_enabled = _get_env_bool(
                "MEMORY_MULTI_SEED_BUNDLE_ENABLED", False
            )
        if self.multi_seed_bundle_mode is None:
            self.multi_seed_bundle_mode = os.getenv(
                "MEMORY_MULTI_SEED_BUNDLE_MODE", "episode_single"
            )
        if self.multi_seed_bundle_alpha is None:
            self.multi_seed_bundle_alpha = _get_env_float(
                "MEMORY_MULTI_SEED_BUNDLE_ALPHA", 0.6
            )
        if self.multi_seed_bundle_beta is None:
            self.multi_seed_bundle_beta = _get_env_float(
                "MEMORY_MULTI_SEED_BUNDLE_BETA", 0.4
            )
        if self.multi_seed_bundle_gamma is None:
            self.multi_seed_bundle_gamma = _get_env_float(
                "MEMORY_MULTI_SEED_BUNDLE_GAMMA", 0.4
            )
        if self.multi_seed_bundle_member_floor is None:
            self.multi_seed_bundle_member_floor = _get_env_float(
                "MEMORY_MULTI_SEED_BUNDLE_MEMBER_FLOOR", 0.20
            )
        if self.multi_seed_bundle_group_floor is None:
            self.multi_seed_bundle_group_floor = _get_env_float(
                "MEMORY_MULTI_SEED_BUNDLE_GROUP_FLOOR", 0.35
            )
        if self.multi_seed_bundle_tag_s1_floor is None:
            self.multi_seed_bundle_tag_s1_floor = _get_env_float(
                "MEMORY_MULTI_SEED_BUNDLE_TAG_S1_FLOOR", 0.0
            )
        if self.multi_seed_bundle_tau is None:
            self.multi_seed_bundle_tau = _get_env_float(
                "MEMORY_MULTI_SEED_BUNDLE_TAU", 0.45
            )
        if self.multi_seed_bundle_lexical_weight is None:
            self.multi_seed_bundle_lexical_weight = _get_env_float(
                "MEMORY_MULTI_SEED_BUNDLE_LEXICAL_WEIGHT", 0.35
            )
        if self.multi_seed_bundle_max_tag_df is None:
            self.multi_seed_bundle_max_tag_df = _get_env_int(
                "MEMORY_MULTI_SEED_BUNDLE_MAX_TAG_DF", 8
            )
        if self.multi_seed_bundle_scan_limit is None:
            self.multi_seed_bundle_scan_limit = _get_env_int(
                "MEMORY_MULTI_SEED_BUNDLE_SCAN_LIMIT", 50
            )
        if self.multi_seed_bundle_max_members is None:
            self.multi_seed_bundle_max_members = _get_env_int(
                "MEMORY_MULTI_SEED_BUNDLE_MAX_MEMBERS", 8
            )
        if self.multi_seed_bundle_max_bundles is None:
            self.multi_seed_bundle_max_bundles = _get_env_int(
                "MEMORY_MULTI_SEED_BUNDLE_MAX_BUNDLES", 0
            )
        if self.multi_seed_bundle_min_episode_seeds is None:
            self.multi_seed_bundle_min_episode_seeds = _get_env_int(
                "MEMORY_MULTI_SEED_BUNDLE_MIN_EPISODE_SEEDS", 1
            )
        if self.multi_seed_bundle_single_tag_seed_ceiling is None:
            self.multi_seed_bundle_single_tag_seed_ceiling = _get_env_float(
                "MEMORY_MULTI_SEED_BUNDLE_SINGLE_TAG_SEED_CEILING", 1.01
            )
        if self.multi_seed_bundle_token_budget is None:
            self.multi_seed_bundle_token_budget = _get_env_int(
                "MEMORY_MULTI_SEED_BUNDLE_TOKEN_BUDGET", 400
            )
        if self.retrieval_clauses_enabled is None:
            self.retrieval_clauses_enabled = _get_env_bool(
                "MEMORY_RETRIEVAL_CLAUSES_ENABLED", False
            )
        if self.retrieval_clauses_min_words is None:
            self.retrieval_clauses_min_words = _get_env_int(
                "MEMORY_RETRIEVAL_CLAUSES_MIN_WORDS", 70
            )
        if self.retrieval_clauses_min_score is None:
            self.retrieval_clauses_min_score = _get_env_float(
                "MEMORY_RETRIEVAL_CLAUSES_MIN_SCORE", 0.50
            )
        if self.retrieval_clauses_margin is None:
            self.retrieval_clauses_margin = _get_env_float(
                "MEMORY_RETRIEVAL_CLAUSES_MARGIN", 0.07
            )
        if self.retrieval_clauses_cap is None:
            self.retrieval_clauses_cap = _get_env_int(
                "MEMORY_RETRIEVAL_CLAUSES_CAP", 3
            )
        if self.retrieval_clauses_require_keyword is None:
            self.retrieval_clauses_require_keyword = _get_env_bool(
                "MEMORY_RETRIEVAL_CLAUSES_REQUIRE_KEYWORD", True
            )
        if self.retrieval_clauses_min_keyword_score is None:
            self.retrieval_clauses_min_keyword_score = _get_env_float(
                "MEMORY_RETRIEVAL_CLAUSES_MIN_KEYWORD_SCORE", 0.10
            )
        if self.supplemental_propositions_enabled is None:
            self.supplemental_propositions_enabled = _get_env_bool(
                "MEMORY_SUPPLEMENTAL_PROPOSITIONS_ENABLED", False
            )


@dataclass
class TimeWindowConfig:
    """Time window configuration."""

    # Rolling deletion
    enable_rolling_delete: bool = False
    rolling_window_days: int = 90  # Rolling window (days)
    rolling_check_interval: int = 3600  # Check interval (seconds)

    # Per-layer window configuration
    layer_windows: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.enable_rolling_delete = _get_env_bool("MEMORY_ROLLING_DELETE", False)
        self.rolling_window_days = _get_env_int("MEMORY_ROLLING_WINDOW_DAYS", 90)

        # Default windows per layer
        if not self.layer_windows:
            self.layer_windows = {
                "raw": _get_env_int("MEMORY_RAW_WINDOW_DAYS", 30),
                "dialogue": _get_env_int("MEMORY_DIALOGUE_WINDOW_DAYS", 60),
                "summary": _get_env_int("MEMORY_SUMMARY_WINDOW_DAYS", 180),
                "profile": _get_env_int("MEMORY_PROFILE_WINDOW_DAYS", 365),
                "knowledge": _get_env_int("MEMORY_KNOWLEDGE_WINDOW_DAYS", 365),
            }


@dataclass
class ExtractorConfig:
    """Memory extractor configuration."""

    # Automatic routing
    enable_auto_routing: bool = True

    # Automatic extraction
    enable_auto_extract: bool = False
    extract_entities: bool = True  # Extract entities
    extract_summary: bool = True  # Extract summary
    extract_profile: bool = True  # Extract profile

    # Extraction thresholds
    min_content_length: int = 10  # Minimum content length
    summary_trigger_length: int = 500  # Length that triggers a summary

    # Number of history context turns injected at extract time (k dialogue turns = k*2 messages).
    # The client can override it; env: MEMORY_EXTRACT_HISTORY_TURNS
    history_turns: int = None

    def __post_init__(self):
        self.enable_auto_routing = _get_env_bool("MEMORY_AUTO_ROUTING", True)
        self.enable_auto_extract = _get_env_bool("MEMORY_AUTO_EXTRACT", False)
        self.min_content_length = _get_env_int("MEMORY_MIN_CONTENT_LENGTH", 10)
        if self.history_turns is None:
            self.history_turns = _get_env_int("MEMORY_EXTRACT_HISTORY_TURNS", 5)


@dataclass
class GraphStoreConfig:
    """Graph store configuration (Kuzu / Neo4j / Memgraph)."""

    provider: str = None  # "kuzu" | "neo4j" | "memgraph"
    db_path: str = None  # Kuzu database path
    url: str = None  # Neo4j/Memgraph bolt URL
    username: str = None  # Neo4j/Memgraph username
    password: str = None  # Neo4j/Memgraph password
    database: str = None  # Neo4j database name

    # ── Memory relation graph (pro mode) ──
    # Write side: project L2_FACT into the graph and build RELATED_TO / TAGGED_WITH edges
    memory_graph_write: bool = None   # env: MEMORY_GRAPH_WRITE, default True (takes effect only when graph is on)
    # Explicit Subject→Topic→Subtopic hierarchy projection. Off by default; wired into Search only after modelling validation.
    hierarchy_topic_write: bool = None  # env: MEMORY_HIERARCHY_TOPIC_WRITE
    # Read-side master switch: **off** by default, enabled only after canary validation
    graph_rerank_enabled: bool = None  # env: MEMORY_GRAPH_RERANK_ENABLED

    # Usage A: in-pool support propagation
    propagate_beta: float = None       # env: MEMORY_GRAPH_PROPAGATE_BETA
    propagate_iters: int = None        # env: MEMORY_GRAPH_PROPAGATE_ITERS
    support_cap: float = None          # env: MEMORY_GRAPH_SUPPORT_CAP
    # Usage B: threshold relaxation
    relax_gamma: float = None          # env: MEMORY_GRAPH_RELAX_GAMMA, 0 disables usage B
    seed_floor: float = None           # env: MEMORY_GRAPH_SEED_FLOOR
    max_expand: int = None             # env: MEMORY_GRAPH_MAX_EXPAND
    # Usage C: relation paths
    path_enabled: bool = None          # env: MEMORY_GRAPH_PATH_ENABLED
    path_high_floor: float = None      # env: MEMORY_GRAPH_PATH_HIGH_FLOOR
    # Write-side edge-building thresholds
    edge_min_score: float = None       # env: MEMORY_GRAPH_EDGE_MIN_SCORE
    co_occurs_max_batch: int = None    # env: MEMORY_GRAPH_CO_OCCURS_MAX_BATCH

    # ── Phase 5 (all off by default, each one adds cost) ──
    # Temporal spine: chain directed TEMPORAL_NEXT edges within one Topic by memory_at (write side, no LLM)
    temporal_spine: bool = None        # env: MEMORY_GRAPH_TEMPORAL_SPINE
    # Topic-scoped summary: split summaries by topic instead of by time window (write side, **requires an LLM**)
    topic_summary_enabled: bool = None      # env: MEMORY_TOPIC_SUMMARY_ENABLED
    topic_summary_min_memories: int = None  # env: MEMORY_TOPIC_SUMMARY_MIN_MEMORIES
    topic_summary_step: int = None          # env: MEMORY_TOPIC_SUMMARY_STEP
    topic_summary_concurrency: int = None   # env: MEMORY_TOPIC_SUMMARY_CONCURRENCY
    # Usage B: expansion + rescoring (funnel, see _graph/rerank.py)
    expand_hops: int = None            # env: MEMORY_GRAPH_EXPAND_HOPS
    expand_prefilter_top_n: int = None # env: MEMORY_GRAPH_EXPAND_PREFILTER_TOP_N
    expand_boost_beta: float = None    # env: MEMORY_GRAPH_EXPAND_BOOST_BETA
    # Usage B hard gate: sim(cand) ≥ seed_sim − delta (admission test replacing the old gamma relaxation threshold)
    seed_sim_delta: float = None       # env: MEMORY_GRAPH_SEED_SIM_DELTA, default 0.2
    # Whether expansions that pass the gate are merged into the seed in time order (\\n\\n); when True they take no separate top-k slot
    expand_merge_into_seed: bool = None  # env: MEMORY_GRAPH_EXPAND_MERGE_SEED
    # Multi-channel scoring weights (semantic / BM25 / tag∩keyword / temporal / co_occur / other structural edges)
    expand_w_sem: float = None         # env: MEMORY_GRAPH_EXPAND_W_SEM
    expand_w_bm25: float = None        # env: MEMORY_GRAPH_EXPAND_W_BM25
    expand_w_tag: float = None         # env: MEMORY_GRAPH_EXPAND_W_TAG (recommended highest)
    expand_w_temporal: float = None    # env: MEMORY_GRAPH_EXPAND_W_TEMPORAL
    expand_w_co_occur: float = None    # env: MEMORY_GRAPH_EXPAND_W_CO_OCCUR (recommended low)
    expand_w_struct: float = None      # env: MEMORY_GRAPH_EXPAND_W_STRUCT (typed edges)
    # Discount factor when episode support is stacked on graph-edge support (bounded stacking, not max)
    episode_blend_lambda: float = None  # env: MEMORY_GRAPH_EPISODE_BLEND_LAMBDA
    multi_seed_bonus: float = None     # env: MEMORY_GRAPH_MULTI_SEED_BONUS
    temporal_tau_days: float = None    # env: MEMORY_GRAPH_TEMPORAL_TAU_DAYS
    temporal_affinity_floor: float = None  # env: MEMORY_GRAPH_TEMPORAL_AFFINITY_FLOOR

    def __post_init__(self):
        if self.provider is None:
            self.provider = os.getenv("MEMORY_GRAPH_PROVIDER", "kuzu")
        if self.db_path is None:
            self.db_path = os.getenv(
                "MEMORY_GRAPH_DB_PATH",
                os.path.join(_default_data_dir(), "data", "kuzu_db"),
            )
        if self.url is None:
            self.url = os.getenv(
                "NEO4J_URL", os.getenv("MEMGRAPH_URL", "bolt://localhost:7687")
            )
        if self.username is None:
            self.username = os.getenv(
                "NEO4J_USERNAME", os.getenv("MEMGRAPH_USERNAME", "")
            )
        if self.password is None:
            self.password = os.getenv(
                "NEO4J_PASSWORD", os.getenv("MEMGRAPH_PASSWORD", "")
            )
        if self.database is None:
            self.database = os.getenv("NEO4J_DATABASE", "neo4j")

        # ── Memory relation graph parameters (defaults in docs/pro_graph_design.md) ──
        if self.memory_graph_write is None:
            self.memory_graph_write = _get_env_bool("MEMORY_GRAPH_WRITE", True)
        if self.hierarchy_topic_write is None:
            self.hierarchy_topic_write = _get_env_bool(
                "MEMORY_HIERARCHY_TOPIC_WRITE", False
            )
        if self.graph_rerank_enabled is None:
            # Read side off by default: wiring in an empty/sparse graph brings no gain, run a canary first (incl. null test)
            self.graph_rerank_enabled = _get_env_bool("MEMORY_GRAPH_RERANK_ENABLED", False)
        if self.propagate_beta is None:
            self.propagate_beta = _get_env_float("MEMORY_GRAPH_PROPAGATE_BETA", 0.15)
        if self.propagate_iters is None:
            self.propagate_iters = _get_env_int("MEMORY_GRAPH_PROPAGATE_ITERS", 1)
        if self.support_cap is None:
            self.support_cap = _get_env_float("MEMORY_GRAPH_SUPPORT_CAP", 1.0)
        if self.relax_gamma is None:
            self.relax_gamma = _get_env_float("MEMORY_GRAPH_RELAX_GAMMA", 0.3)
        if self.seed_floor is None:
            # 0.5 rather than 0.6: measured over 15 real recall pools the median top vector score is only 0.636,
            # so at 0.6 the median number of "hits usable as seeds" is **0** → expansion never fires,
            # and multi-seed corroboration never engages. At 0.5 the median is 18 reachable candidates and 80% of queries have >=2 seeds.
            self.seed_floor = _get_env_float("MEMORY_GRAPH_SEED_FLOOR", 0.5)
        if self.max_expand is None:
            self.max_expand = _get_env_int("MEMORY_GRAPH_MAX_EXPAND", 20)
        if self.path_enabled is None:
            self.path_enabled = _get_env_bool("MEMORY_GRAPH_PATH_ENABLED", True)
        if self.path_high_floor is None:
            # Recalibrated like seed_floor. Paths require **both ends** above the line; at 0.6 typically only top-1
            # clears it and no pair can be formed → 262 measured retrievals produced 0 paths.
            self.path_high_floor = _get_env_float("MEMORY_GRAPH_PATH_HIGH_FLOOR", 0.5)
        if self.edge_min_score is None:
            self.edge_min_score = _get_env_float("MEMORY_GRAPH_EDGE_MIN_SCORE", 0.55)
        if self.co_occurs_max_batch is None:
            self.co_occurs_max_batch = _get_env_int("MEMORY_GRAPH_CO_OCCURS_MAX_BATCH", 8)

        # Phase 5: each adds cost (LLM or extra retrieval), all off by default
        if self.temporal_spine is None:
            self.temporal_spine = _get_env_bool("MEMORY_GRAPH_TEMPORAL_SPINE", False)
        if self.topic_summary_enabled is None:
            self.topic_summary_enabled = _get_env_bool("MEMORY_TOPIC_SUMMARY_ENABLED", False)
        if self.topic_summary_min_memories is None:
            self.topic_summary_min_memories = _get_env_int(
                "MEMORY_TOPIC_SUMMARY_MIN_MEMORIES", 5
            )
        if self.topic_summary_step is None:
            self.topic_summary_step = _get_env_int("MEMORY_TOPIC_SUMMARY_STEP", 5)
        if self.topic_summary_concurrency is None:
            self.topic_summary_concurrency = max(1, min(16, _get_env_int(
                "MEMORY_TOPIC_SUMMARY_CONCURRENCY", 3
            )))
        if self.expand_hops is None:
            self.expand_hops = _get_env_int("MEMORY_GRAPH_EXPAND_HOPS", 2)
        if self.expand_prefilter_top_n is None:
            self.expand_prefilter_top_n = _get_env_int(
                "MEMORY_GRAPH_EXPAND_PREFILTER_TOP_N", 50
            )
        if self.expand_boost_beta is None:
            self.expand_boost_beta = _get_env_float("MEMORY_GRAPH_EXPAND_BOOST_BETA", 0.1)
        if self.seed_sim_delta is None:
            self.seed_sim_delta = _get_env_float("MEMORY_GRAPH_SEED_SIM_DELTA", 0.2)
        if self.expand_merge_into_seed is None:
            self.expand_merge_into_seed = _get_env_bool(
                "MEMORY_GRAPH_EXPAND_MERGE_SEED", True
            )
        if self.expand_w_sem is None:
            self.expand_w_sem = _get_env_float("MEMORY_GRAPH_EXPAND_W_SEM", 1.0)
        if self.expand_w_bm25 is None:
            self.expand_w_bm25 = _get_env_float("MEMORY_GRAPH_EXPAND_W_BM25", 0.3)
        if self.expand_w_tag is None:
            self.expand_w_tag = _get_env_float("MEMORY_GRAPH_EXPAND_W_TAG", 0.5)
        if self.expand_w_temporal is None:
            self.expand_w_temporal = _get_env_float(
                "MEMORY_GRAPH_EXPAND_W_TEMPORAL", 0.2
            )
        if self.expand_w_co_occur is None:
            self.expand_w_co_occur = _get_env_float(
                "MEMORY_GRAPH_EXPAND_W_CO_OCCUR", 0.15
            )
        if self.expand_w_struct is None:
            self.expand_w_struct = _get_env_float("MEMORY_GRAPH_EXPAND_W_STRUCT", 0.25)
        if self.episode_blend_lambda is None:
            self.episode_blend_lambda = _get_env_float(
                "MEMORY_GRAPH_EPISODE_BLEND_LAMBDA", 0.3)
        if self.multi_seed_bonus is None:
            self.multi_seed_bonus = _get_env_float("MEMORY_GRAPH_MULTI_SEED_BONUS", 0.5)
        if self.temporal_tau_days is None:
            self.temporal_tau_days = _get_env_float("MEMORY_GRAPH_TEMPORAL_TAU_DAYS", 7.0)
        if self.temporal_affinity_floor is None:
            self.temporal_affinity_floor = _get_env_float(
                "MEMORY_GRAPH_TEMPORAL_AFFINITY_FLOOR", 0.3
            )


@dataclass
class CacheConfig:
    """
    Cache backend configuration

    Supported backends:
    - "sqlite"   : Local SQLite database (default, zero dependencies)
    - "mysql"    : Standard MySQL persistence (requires the aiomysql package)

    Fine-grained control: enable/disable individual cache features
    - enable_profile_cache   : Profile hot cache (read optimization)
    - enable_system2_queue   : System2 task queue (background tasks)
    - enable_intention_cache : Intention queue (triggers)
    - enable_write_records   : Write records (status queries)
    """

    backend: str = None  # "sqlite" | "mysql"
    db_path: str = None  # SQLite path

    # MySQL connection parameters (used when backend="mysql")
    mysql_host: str = None
    mysql_port: int = None
    mysql_user: str = None
    mysql_password: str = None
    mysql_database: str = None
    mysql_pool_size: int = None
    mysql_pool_recycle: int = None

    # Fine-grained cache control (supported by all backends)
    enable_profile_cache: bool = None  # Profile cache
    enable_system2_queue: bool = None  # System2 task queue
    enable_intention_cache: bool = None  # Intention queue
    enable_write_records: bool = None  # Write records

    def __post_init__(self):
        if self.backend is None:
            self.backend = os.getenv("MEMORY_CACHE_BACKEND", "sqlite")
        backend_norm = str(self.backend).lower().strip()
        if backend_norm not in ("sqlite", "mysql"):
            raise ValueError(
                f"Unsupported cache backend {self.backend!r}; "
                f"must be one of: sqlite / mysql"
            )
        self.backend = backend_norm
        if self.db_path is None:
            self.db_path = os.getenv(
                "MEMORY_CACHE_DB_PATH",
                os.path.join(_default_data_dir(), "data", "cache.db"),
            )

        # MySQL connection parameters
        if self.mysql_host is None:
            self.mysql_host = os.getenv("MEMORY_MYSQL_HOST", "localhost")
        if self.mysql_port is None:
            self.mysql_port = _get_env_int("MEMORY_MYSQL_PORT", 3306)
        if self.mysql_user is None:
            self.mysql_user = os.getenv("MEMORY_MYSQL_USER", "root")
        if self.mysql_password is None:
            self.mysql_password = os.getenv("MEMORY_MYSQL_PASSWORD", "")
        if self.mysql_database is None:
            self.mysql_database = os.getenv("MEMORY_MYSQL_DATABASE", "hy_memory")
        if self.mysql_pool_size is None:
            self.mysql_pool_size = _get_env_int("MEMORY_MYSQL_POOL_SIZE", 10)
        if self.mysql_pool_recycle is None:
            self.mysql_pool_recycle = _get_env_int("MEMORY_MYSQL_POOL_RECYCLE", 3600)

        # Fine-grained cache control - all default to True (every feature enabled)
        if self.enable_profile_cache is None:
            self.enable_profile_cache = _get_env_bool(
                "MEMORY_CACHE_ENABLE_PROFILE", True
            )
        if self.enable_system2_queue is None:
            self.enable_system2_queue = _get_env_bool(
                "MEMORY_CACHE_ENABLE_SYSTEM2_QUEUE", True
            )
        if self.enable_intention_cache is None:
            self.enable_intention_cache = _get_env_bool(
                "MEMORY_CACHE_ENABLE_INTENTION_CACHE", True
            )
        if self.enable_write_records is None:
            self.enable_write_records = _get_env_bool(
                "MEMORY_CACHE_ENABLE_WRITE_RECORDS", True
            )


@dataclass
class APIConfig:
    """API configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    prefix: str = "/api/v1/memory"

    # Authentication
    enable_auth: bool = False
    api_keys: List[str] = field(default_factory=list)

    # Rate limiting
    enable_rate_limit: bool = False
    rate_limit_per_minute: int = 100

    # CORS
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    def __post_init__(self):
        self.host = os.getenv("MEMORY_API_HOST", "0.0.0.0")
        self.port = _get_env_int("MEMORY_API_PORT", 8000)
        self.prefix = os.getenv("MEMORY_API_PREFIX", "/api/v1/memory")
        self.enable_auth = _get_env_bool("MEMORY_API_AUTH", False)

        # API Keys
        api_keys_str = os.getenv("MEMORY_API_KEYS", "")
        if api_keys_str:
            self.api_keys = [k.strip() for k in api_keys_str.split(",")]


@dataclass
class HistoryConfig:
    """History record configuration (SQLite audit trail)."""

    enable: bool = None  # Whether history records are enabled
    db_path: str = None  # SQLite database path
    record_searches: bool = None  # Whether to record search operations

    def __post_init__(self):
        if self.enable is None:
            self.enable = _get_env_bool("MEMORY_HISTORY_ENABLE", True)
        if self.db_path is None:
            self.db_path = os.getenv(
                "MEMORY_HISTORY_DB_PATH",
                os.path.join(_default_data_dir(), "data", "history.db"),
            )
        if self.record_searches is None:
            self.record_searches = _get_env_bool(
                "MEMORY_HISTORY_RECORD_SEARCHES", False
            )


@dataclass
class CodingConfig:
    """
    Coding Memory configuration (standalone pipeline for productivity / coding scenarios)

    See docs/scene_coding_usage.md for external usage.
    """

    enable: bool = None  # Whether the coding pipeline is enabled (default True; when off every add uses the chat pipeline)
    db_path: str = None  # coding_memory.db path
    tool_result_max_bytes: int = None  # Truncation byte cap for a single tool_result
    # Whether to store the raw dialogue as a working fact when the LLM fails or extracts no fact.
    # Off by default so assistant/tool traces are not exposed to consumers as statements.
    verbatim_fallback_enabled: bool = None

    def __post_init__(self):
        if self.enable is None:
            self.enable = _get_env_bool("MEMORY_CODING_ENABLED", True)
        if self.db_path is None:
            self.db_path = os.getenv(
                "MEMORY_CODING_DB_PATH",
                os.path.join(_default_data_dir(), "data", "coding_memory.db"),
            )
        if self.tool_result_max_bytes is None:
            self.tool_result_max_bytes = _get_env_int(
                "MEMORY_CODING_TOOL_RESULT_MAX_BYTES", 2048
            )
        if self.verbatim_fallback_enabled is None:
            self.verbatim_fallback_enabled = _get_env_bool(
                "MEMORY_CODING_VERBATIM_FALLBACK_ENABLED", False
            )


@dataclass
class BasicProfileConfig:
    """
    Field schema configuration for the user base profile (L0_BASIC_INFO).

    Design principles:
    - No LLM function-calling tool (it makes weak models fill fields arbitrarily).
    - The extractor formats the fields table straight into the EXTRACT prompt and asks the LLM
      to return it as the `basic_info` field of its JSON output (filled only when the dialogue states it explicitly).
    - Different clients can configure different schemas (e.g. car systems care about vehicle_model,
      medical scenarios about medical_history); the SDK provides 5 generic fields by default.
    - If the configuration is empty (fields == {}), the SDK falls back to the 5 defaults from default_factory.

    Usage:
        config.basic_profile.fields["hobby"] = "User's main hobby."
        # Or replace the whole schema
        config.basic_profile.fields = {
            "name": "User's full or preferred name.",
            "vehicle_model": "User's primary vehicle model.",
        }
    """

    fields: Dict[str, str] = field(
        default_factory=lambda: {
            "name": "User's full or preferred name.",
            "age": "User's age in years (integer).",
            "location": "User's primary city or region of residence.",
            "occupation": "User's job title or role.",
            "employer": "User's employer / company name.",
        }
    )

    def effective_fields(self) -> Dict[str, str]:
        """Fall back to the defaults when the configuration is empty, otherwise use the configured value."""
        if not self.fields:
            return BasicProfileConfig().fields
        return dict(self.fields)


@dataclass
class PipelineRouteConfig:
    """
    Pipeline routing configuration

    Controls which Pipeline version (Lite, Pro, ...) a given business party or request uses.
    """

    # Default Pipeline version
    default_version: str = None

    # Business party → Pipeline version mapping
    # Example: {"business_a": "pro", "experiment_group": "pro"}
    business_version_map: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.default_version is None:
            self.default_version = os.getenv("MEMORY_PIPELINE_DEFAULT_VERSION", "lite")

        # Load the business party mapping from an environment variable
        # Format: MEMORY_PIPELINE_BUSINESS_MAP="business_a:pro,business_b:lite"
        map_str = os.getenv("MEMORY_PIPELINE_BUSINESS_MAP", "")
        if map_str and not self.business_version_map:
            for pair in map_str.split(","):
                pair = pair.strip()
                if ":" in pair:
                    biz, ver = pair.split(":", 1)
                    self.business_version_map[biz.strip()] = ver.strip()


@dataclass
class MemoryConfig:
    """
    Agent Memory main configuration

    Aggregates all sub-configurations behind one unified configuration entry point
    """

    # Sub-configurations
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    graph_store: GraphStoreConfig = field(default_factory=GraphStoreConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    recall: RecallConfig = field(default_factory=RecallConfig)
    time_window: TimeWindowConfig = field(default_factory=TimeWindowConfig)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    api: APIConfig = field(default_factory=APIConfig)
    pipeline: PipelineRouteConfig = field(default_factory=PipelineRouteConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    coding: CodingConfig = field(default_factory=CodingConfig)
    basic_profile: BasicProfileConfig = field(default_factory=BasicProfileConfig)

    # Global configuration
    enable_graph: bool = False
    enable_agent: bool = False
    debug: bool = False
    metrics_enabled: bool = True
    # Runtime-only HTTP transport factory. MemHub uses this for request-cost
    # budgeting without restricting the embedded SDK's local store access.
    http_transport_factory: Optional[Callable[[], Any]] = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self):
        # enable_graph follows mode: enabled automatically when mode=ultra, no separate env var
        # from_dict() can still override it explicitly
        self.enable_agent = _get_env_bool("MEMORY_ENABLE_AGENT", False)
        self.debug = _get_env_bool("MEMORY_DEBUG", False)
        self.metrics_enabled = _get_env_bool("MEMORY_METRICS", True)

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """Create the configuration from environment variables."""
        return cls()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MemoryConfig":
        """Create the configuration from a dict."""
        config = cls()

        _sections = [
            "vector_store",
            "graph_store",
            "cache",
            "llm",
            "embedder",
            "recall",
            "time_window",
            "extractor",
            "api",
            "pipeline",
            "history",
            "coding",
        ]
        for section in _sections:
            if section in config_dict:
                sub = getattr(config, section)
                for key, value in config_dict[section].items():
                    if hasattr(sub, key):
                        setattr(sub, key, value)

        # basic_profile: fields is Dict[str, str], replaced wholesale (when non-empty)
        if "basic_profile" in config_dict:
            bp = config_dict["basic_profile"] or {}
            fields = bp.get("fields")
            if isinstance(fields, dict) and fields:
                # Force str→str; drop empty descriptions
                config.basic_profile.fields = {
                    str(k): str(v).strip()
                    for k, v in fields.items()
                    if str(k).strip() and str(v).strip()
                }

        config.enable_graph = config_dict.get("enable_graph", config.enable_graph)
        config.enable_agent = config_dict.get("enable_agent", config.enable_agent)
        config.debug = config_dict.get("debug", config.debug)
        config.metrics_enabled = config_dict.get(
            "metrics_enabled", config.metrics_enabled
        )

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dict."""
        return {
            "vector_store": {
                "provider": self.vector_store.provider,
                "collection_name": self.vector_store.collection_name,
                "persist_directory": self.vector_store.persist_directory,
                "embedding_dims": self.vector_store.embedding_dims,
            },
            "graph_store": {
                "provider": self.graph_store.provider,
                "db_path": self.graph_store.db_path,
                "url": self.graph_store.url,
                "database": self.graph_store.database,
            },
            "cache": {
                "backend": self.cache.backend,
                "db_path": self.cache.db_path,
            },
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "base_url": self.llm.base_url,
            },
            "embedder": {
                "provider": self.embedder.provider,
                "model": self.embedder.model,
                "base_url": self.embedder.base_url,
                "embedding_dims": self.embedder.embedding_dims,
            },
            "recall": {
                "default_limit": self.recall.default_limit,
                "semantic_weight": self.recall.semantic_weight,
                "recency_weight": self.recall.recency_weight,
                "importance_weight": self.recall.importance_weight,
                "supplemental_propositions_enabled": (
                    self.recall.supplemental_propositions_enabled
                ),
            },
            "pipeline": {
                "default_version": self.pipeline.default_version,
                "business_version_map": self.pipeline.business_version_map,
            },
            "history": {
                "enable": self.history.enable,
                "db_path": self.history.db_path,
                "record_searches": self.history.record_searches,
            },
            "basic_profile": {
                "fields": dict(self.basic_profile.fields),
            },
            "debug": self.debug,
            "enable_graph": self.enable_graph,
            "enable_agent": self.enable_agent,
            "metrics_enabled": self.metrics_enabled,
        }

    def print_config(self) -> None:
        """Print the configuration."""
        print("=" * 60)
        print("Agent Memory Configuration")
        print("=" * 60)
        print("[Vector Store]")
        print(f"  Provider: {self.vector_store.provider}")
        print(f"  Collection: {self.vector_store.collection_name}")
        print(f"  Persist Dir: {self.vector_store.persist_directory}")
        print(f"  Embedding Dims: {self.vector_store.embedding_dims}")
        print("[Graph Store]")
        print(f"  Provider: {self.graph_store.provider}")
        if self.graph_store.provider == "neo4j":
            print(f"  URL: {self.graph_store.url}")
            print(f"  Database: {self.graph_store.database}")
        else:
            print(f"  DB Path: {self.graph_store.db_path}")
        print("[Cache]")
        print(f"  Backend: {self.cache.backend}")
        if self.cache.backend == "sqlite":
            print(f"  DB Path: {self.cache.db_path}")
        else:
            print(f"  MySQL: {self.cache.mysql_host}:{self.cache.mysql_port}/{self.cache.mysql_database}")
        print("[LLM]")
        print(f"  Provider: {self.llm.provider}")
        print(f"  Model: {self.llm.model}")
        print(f"  Base URL: {self.llm.base_url}")
        print("[Embedder]")
        print(f"  Provider: {self.embedder.provider}")
        print(f"  Model: {self.embedder.model}")
        print(f"  Base URL: {self.embedder.base_url}")
        print("[Recall]")
        print(f"  Default Limit: {self.recall.default_limit}")
        print(
            f"  Weights: semantic={self.recall.semantic_weight}, "
            f"recency={self.recall.recency_weight}, "
            f"importance={self.recall.importance_weight}"
        )
        print("[Pipeline]")
        print(f"  Default Version: {self.pipeline.default_version}")
        print(f"  Business Map: {self.pipeline.business_version_map}")
        print("[API]")
        print(f"  Prefix: {self.api.prefix}")
        print(f"  Auth: {self.api.enable_auth}")
        print("[History]")
        print(f"  Enable: {self.history.enable}")
        print(f"  DB Path: {self.history.db_path}")
        print(f"  Record Searches: {self.history.record_searches}")
        print("=" * 60)
