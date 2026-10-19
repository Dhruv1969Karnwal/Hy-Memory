import type {
  ConversationBrowserPage,
  ConversationSearchEmptyReason,
  ConversationSearchHit,
  ConversationTranscript,
  GovernedMemory,
} from "@one-memory/contracts";
import { isAxiosError } from "axios";
import { ChevronRight, RefreshCw, Search } from "lucide-react";
import type { ReactNode } from "react";
import { useEffect, useId, useRef, useState } from "react";

import {
  apiErrorCode,
  getConversation,
  getConversationBrowser,
  getConversationMemories,
  searchConversations,
} from "./api";
import { AppMark } from "./app-mark";
import "./conversation-search.css";
import { countLabel } from "./count-label";

type OpenedConversation = {
  hit: ConversationSearchHit;
  phase: "loading" | "ready" | "error";
  transcript: ConversationTranscript | null;
  memories: GovernedMemory[];
  error: string | null;
};

type ConversationListHit = ConversationSearchHit;

/**
 * Searching imported history before any generation model exists.
 *
 * Original user, assistant and unknown-author turns remain distinct from
 * governed memories. Each asynchronous result belongs to its initiating view.
 */
export function ConversationSearch({
  onOpenMemory,
}: {
  onOpenMemory?(memoryId: string): void;
} = {}) {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<ConversationSearchHit[] | null>(null);
  const [browser, setBrowser] = useState<ConversationBrowserPage | null>(null);
  const [sourceFilter, setSourceFilter] = useState("");
  const [sourceOptions, setSourceOptions] = useState<string[]>([]);
  const [emptyReason, setEmptyReason] =
    useState<ConversationSearchEmptyReason | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [mobileMemories, setMobileMemories] = useState(false);
  const [opened, setOpened] = useState<OpenedConversation | null>(null);
  const id = useId();
  const mounted = useRef(false);
  const searchVersion = useRef(0);
  const readVersion = useRef(0);
  const searchFlight = useRef<string | null>(null);
  const activeHit = useRef<ConversationSearchHit | null>(null);
  const readFlight = useRef<number | null>(null);
  const lastQuery = useRef("");
  const matchedTurn = useRef<HTMLElement | null>(null);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      searchVersion.current += 1;
      readVersion.current += 1;
      searchFlight.current = null;
      readFlight.current = null;
      activeHit.current = null;
    };
  }, []);

  useEffect(() => {
    if (!mounted.current) return;
    let cancelled = false;
    void Promise.resolve()
      .then(() =>
        getConversationBrowser({ sourceId: sourceFilter || undefined }),
      )
      .then(
        (page) => {
          if (cancelled || !mounted.current) return;
          setBrowser(page);
          setSourceOptions((current) => [
            ...new Set([
              ...current,
              ...page.items.map((item) => item.source_id),
            ]),
          ]);
        },
        () => {
          if (!cancelled && mounted.current)
            setError(
              "The conversation list cannot be read at the moment. Please retry.",
            );
        },
      );
    return () => {
      cancelled = true;
    };
  }, [sourceFilter]);

  function closeOriginal() {
    readVersion.current += 1;
    readFlight.current = null;
    activeHit.current = null;
    setOpened(null);
  }

  async function run(text: string) {
    if (!text || searchFlight.current === text) return;
    const request = ++searchVersion.current;
    searchFlight.current = text;
    lastQuery.current = text;
    setBusy(true);
    setError(null);
    setHits(null);
    setEmptyReason(null);
    closeOriginal();
    try {
      const found = await searchConversations(text);
      if (!mounted.current || request !== searchVersion.current) return;
      setHits(found.hits);
      setEmptyReason(found.hits.length ? null : found.empty_reason);
    } catch (reason) {
      if (!mounted.current || request !== searchVersion.current) return;
      setHits(null);
      setEmptyReason(null);
      setError(failureWording(reason));
    } finally {
      if (mounted.current && request === searchVersion.current) {
        searchFlight.current = null;
        setBusy(false);
      }
    }
  }

  async function read(hit: ConversationSearchHit) {
    if (sameHit(activeHit.current, hit) && readFlight.current !== null) return;
    const request = ++readVersion.current;
    const searchRequest = searchVersion.current;
    readFlight.current = request;
    activeHit.current = hit;
    setMobileMemories(false);
    setOpened({
      hit,
      phase: "loading",
      transcript: null,
      memories: [],
      error: null,
    });
    const isCurrent = () =>
      mounted.current &&
      request === readVersion.current &&
      searchRequest === searchVersion.current &&
      sameHit(activeHit.current, hit);
    try {
      const [result, related] = await Promise.all([
        hit.turn_offset
          ? getConversation(hit.evidence_id, {
              offset: hit.turn_offset,
              limit: 20,
            })
          : getConversation(hit.evidence_id),
        getConversationMemories(hit.evidence_id),
      ]);
      if (!isCurrent()) return;
      if (!matchesTranscript(result, hit))
        throw new Error("conversation_response_mismatch");
      setOpened({
        hit,
        phase: "ready",
        transcript: result,
        memories: related.items,
        error: null,
      });
    } catch (reason) {
      if (!isCurrent()) return;
      if (permissionUnavailable(reason)) {
        closeOriginal();
        setHits(null);
        setEmptyReason(null);
        setError(
          "The source text can no longer be read; authorization may have been revoked or the record is unavailable. Old content was collapsed. Please search again to verify.",
        );
      } else {
        setOpened({
          hit,
          phase: "error",
          transcript: null,
          memories: [],
          error: "The source text read did not complete. Please retry.",
        });
      }
    } finally {
      if (request === readVersion.current) readFlight.current = null;
    }
  }

  function toggle(hit: ConversationSearchHit) {
    if (sameHit(activeHit.current, hit)) closeOriginal();
    else void read(hit);
  }

  async function loadOlder() {
    if (!opened?.transcript?.has_more || readFlight.current !== null) return;
    const request = ++readVersion.current;
    readFlight.current = request;
    try {
      const older = await getConversation(opened.hit.evidence_id, {
        offset:
          (opened.transcript.offset ?? 0) + (opened.transcript.limit ?? 20),
        limit: opened.transcript.limit ?? 20,
      });
      if (!mounted.current || request !== readVersion.current) return;
      setOpened((current) =>
        current?.transcript
          ? {
              ...current,
              transcript: {
                ...older,
                messages: [...older.messages, ...current.transcript.messages],
              },
            }
          : current,
      );
    } catch {
      if (mounted.current && request === readVersion.current) {
        setOpened((current) =>
          current
            ? {
                ...current,
                error:
                  "Older conversations cannot be read at the moment. Please retry.",
              }
            : current,
        );
      }
    } finally {
      if (request === readVersion.current) readFlight.current = null;
    }
  }

  const listedHits: ConversationListHit[] =
    hits ??
    browser?.items.map((item) => ({
      evidence_id: item.evidence_id,
      conversation_id: item.conversation_id,
      source_id: item.source_id,
      workspace: item.workspace,
      project: item.project,
      created_at: item.latest_at,
      excerpt: item.title,
      turn_count: item.turn_count,
      memory_count: item.memory_count,
      turn_offset: 0,
    })) ??
    [];
  const turns = opened?.transcript ? groupTurns(opened.transcript) : [];
  const matchId = opened?.phase === "ready" ? opened.hit.evidence_id : null;
  useEffect(() => {
    if (!matchId) return;
    const turn = matchedTurn.current;
    // A hit can sit far down a long transcript. Opening it has to bring the
    // matching turn into view, not only render it somewhere below the fold.
    // Layout-less hosts (jsdom) have no scrollIntoView to call.
    if (typeof turn?.scrollIntoView === "function")
      turn.scrollIntoView({ block: "center" });
  }, [matchId]);

  return (
    <section className="conversation-search">
      <form
        className="memory-query-bar"
        onSubmit={(event) => {
          event.preventDefault();
          if (!query.trim()) return;
          void run(query.trim());
        }}
      >
        <label htmlFor={`${id}-query`}>Search conversation history</label>
        <div>
          <input
            id={`${id}-query`}
            name="conversation-query"
            autoComplete="off"
            value={query}
            placeholder="e.g. packaging, extraction, weekend"
            onChange={(event) => setQuery(event.target.value)}
          />
          <button
            type="submit"
            disabled={
              !query.trim() || (busy && searchFlight.current === query.trim())
            }
          >
            <Search size={16} aria-hidden="true" />
            {busy ? "Searching..." : "Search"}
          </button>
        </div>
      </form>

      <p className="conversation-search-note">
        This searches the original conversations you imported, not memories.
        Memories start being generated once a model is configured.
      </p>

      <label className="conversation-source-filter">
        <span>App</span>
        <select
          value={sourceFilter}
          onChange={(event) => setSourceFilter(event.target.value)}
        >
          <option value="">All apps</option>
          {sourceOptions.map((source) => (
            <option value={source} key={source}>
              {source}
            </option>
          ))}
        </select>
      </label>

      {error ? (
        <div className="conversation-search-error">
          <p role="alert">{error}</p>
          <button
            type="button"
            className="quiet"
            disabled={busy}
            onClick={() => void run(lastQuery.current)}
          >
            <RefreshCw size={16} aria-hidden="true" />
            Search again
          </button>
        </div>
      ) : null}

      {emptyReason ? (
        <p className="conversation-search-empty" role="status">
          {emptyReason === "nothing_imported"
            ? "Import history first, then you can search"
            : "No matching conversations were found"}
        </p>
      ) : null}

      {listedHits.length ? (
        <section
          className="conversation-browser"
          aria-label="Imported conversation search results"
        >
          <section
            className="conversation-browser-list"
            aria-label="Conversation list"
          >
            {listedHits.map((hit) => (
              <article
                key={`${hit.source_id}:${hit.conversation_id}:${hit.workspace ?? ""}`}
                className="conversation-hit"
                data-selected={sameHit(opened?.hit, hit)}
                data-evidence-id={hit.evidence_id}
              >
                <button
                  type="button"
                  className="conversation-hit-open"
                  aria-label={
                    sameHit(opened?.hit, hit)
                      ? "Collapse source text"
                      : "View source text"
                  }
                  aria-expanded={sameHit(opened?.hit, hit)}
                  aria-controls={`${id}-transcript`}
                  onClick={() => toggle(hit)}
                >
                  <span>
                    <strong>{hit.title ?? hit.excerpt}</strong>
                    {hit.title && hit.title !== hit.excerpt ? (
                      <span className="conversation-hit-match">
                        {hit.excerpt}
                      </span>
                    ) : null}
                    <small className="conversation-hit-meta">
                      <span className="conversation-source-name">
                        <SourceIcon sourceId={hit.source_id} />
                        {sourceLabel(hit.source_id)}
                      </span>
                      {hit.project ? <span>{hit.project}</span> : null}
                      {typeof hit.turn_count === "number" ? (
                        <span>{countLabel(hit.turn_count, "turn")}</span>
                      ) : null}
                      {hit.memory_count ? (
                        <span>
                          {countLabel(hit.memory_count, "memory")} distilled
                        </span>
                      ) : null}
                      <OriginalTime value={hit.created_at} />
                    </small>
                  </span>
                  <ChevronRight size={16} aria-hidden="true" />
                  <span className="sr-only">View source text</span>
                </button>
              </article>
            ))}
          </section>
          <section
            id={`${id}-transcript`}
            className="conversation-transcript"
            aria-label="Complete original conversation"
            aria-busy={opened?.phase === "loading"}
          >
            {!opened ? (
              <p className="conversation-search-empty">Select a conversation</p>
            ) : null}
            {opened?.phase === "ready" ? (
              <button
                type="button"
                className="conversation-memory-toggle quiet"
                aria-expanded={mobileMemories}
                aria-controls={`${id}-memories`}
                onClick={() => setMobileMemories((visible) => !visible)}
              >
                Linked memories {opened.memories.length}
              </button>
            ) : null}
            {opened?.phase === "loading" ? (
              <p role="status">Reading the source text...</p>
            ) : null}
            {opened?.error ? (
              <div className="conversation-read-error">
                <p role="alert">{opened.error}</p>
                <button
                  type="button"
                  className="quiet"
                  onClick={() => void read(opened.hit)}
                >
                  <RefreshCw size={16} aria-hidden="true" />
                  Retry reading the source text
                </button>
              </div>
            ) : null}
            {opened?.transcript?.has_more ? (
              <button
                type="button"
                className="quiet"
                onClick={() => void loadOlder()}
              >
                Load older conversations
              </button>
            ) : null}
            {opened?.phase === "ready" ? (
              <ol className="conversation-turns">
                {turns.map((turn, index) => (
                  <li key={turn.key} className="conversation-round">
                    {turn.system.map((item) => (
                      <details
                        key={item.evidence_id}
                        className="conversation-system"
                      >
                        <summary>System context</summary>
                        <p>{item.verbatim}</p>
                      </details>
                    ))}
                    {turn.messages.map((item) => {
                      const matched =
                        item.evidence_id === opened.hit.evidence_id;
                      return item.role === "tool" ? (
                        <details
                          key={item.evidence_id}
                          className="conversation-tool"
                        >
                          <summary>Tool record</summary>
                          <p>{item.verbatim}</p>
                        </details>
                      ) : (
                        <section
                          key={item.evidence_id}
                          ref={matched ? matchedTurn : undefined}
                          data-role={item.role}
                          data-match={matched}
                        >
                          <header className="conversation-turn-meta">
                            <span>{authorLabel(item.written_by)}</span>
                            <OriginalTime value={item.created_at} />
                          </header>
                          <p className="conversation-turn-verbatim">
                            {matched
                              ? markTerm(item.verbatim, lastQuery.current)
                              : item.verbatim}
                          </p>
                        </section>
                      );
                    })}
                    {index === turns.length - 1 &&
                    opened.transcript?.trailing_turn_unanswered ? (
                      <p className="conversation-round-unanswered">
                        This turn has not received an answer yet, so it is left
                        out of memory processing.
                      </p>
                    ) : null}
                  </li>
                ))}
              </ol>
            ) : null}
          </section>
          <aside
            id={`${id}-memories`}
            className="conversation-memories"
            aria-label="Linked memories"
            data-mobile-visible={mobileMemories}
          >
            <h3>Memories distilled from this conversation</h3>
            {!opened ? <p>Select a conversation to view</p> : null}
            {opened?.phase === "ready" && opened.memories.length === 0 ? (
              <p>This conversation has not produced memories yet</p>
            ) : null}
            {opened?.memories.map((memory) => (
              <article key={memory.memory_id}>
                <button
                  type="button"
                  onClick={() => onOpenMemory?.(memory.memory_id)}
                >
                  {memory.content}
                  <ChevronRight size={14} aria-hidden="true" />
                </button>
              </article>
            ))}
          </aside>
        </section>
      ) : null}
    </section>
  );
}

/** Wraps every occurrence of the searched term so the hit is visible in place. */
function markTerm(text: string, term: string): ReactNode {
  const needle = term.trim();
  if (!needle) return text;
  const haystack = text.toLowerCase();
  const target = needle.toLowerCase();
  const parts: ReactNode[] = [];
  let from = 0;
  for (
    let at = haystack.indexOf(target);
    at !== -1;
    at = haystack.indexOf(target, from)
  ) {
    if (at > from) parts.push(text.slice(from, at));
    parts.push(
      <mark className="conversation-hit-term" key={at}>
        {text.slice(at, at + needle.length)}
      </mark>,
    );
    from = at + needle.length;
  }
  if (!parts.length) return text;
  if (from < text.length) parts.push(text.slice(from));
  return parts;
}

function groupTurns(transcript: ConversationTranscript) {
  const grouped = new Map<
    number,
    {
      key: string;
      messages: ConversationTranscript["messages"];
      system: ConversationTranscript["messages"];
    }
  >();
  let fallback = 0;
  for (const message of transcript.messages) {
    if (
      message.role === "user" &&
      message.written_by === "person" &&
      !message.system_context
    ) {
      fallback += 1;
    }
    const index = message.turn_index ?? fallback;
    const turn = grouped.get(index) ?? {
      key: `turn-${index}`,
      messages: [],
      system: [],
    };
    (message.system_context ? turn.system : turn.messages).push(message);
    grouped.set(index, turn);
  }
  return [...grouped.values()];
}

function sourceLabel(sourceId: string) {
  const labels: Record<string, string> = {
    workbuddy: "WorkBuddy",
    cursor: "Cursor",
    codex: "Codex",
    "claude-code": "Claude Code",
  };
  return labels[sourceId] ?? sourceId;
}

function SourceIcon({ sourceId }: { sourceId: string }) {
  return <AppMark id={sourceId} size={16} />;
}

function failureWording(reason: unknown): string {
  if (permissionUnavailable(reason))
    return "Read permission or records for the conversation history are unavailable; old content was collapsed. Search again to verify.";
  if (apiErrorCode(reason) === "conversation_search_unavailable") {
    return "The search index is unavailable. This does not mean nothing was found: local search is not running, so check Core status in Health.";
  }
  return "The search did not complete. Please try again.";
}

function OriginalTime({ value }: { value: string | null }) {
  return value && Number.isFinite(Date.parse(value)) ? (
    <time dateTime={value}>
      {new Date(value).toLocaleString("zh-CN", { hour12: false })}
    </time>
  ) : (
    <span>Time unknown</span>
  );
}

function authorLabel(value: string) {
  return value === "person"
    ? "User's own words"
    : value === "agent"
      ? "Assistant's own words"
      : value === "tool"
        ? "Tool record"
        : "Author unknown";
}

function sameHit(
  left: ConversationSearchHit | null | undefined,
  right: ConversationSearchHit,
) {
  return (
    !!left &&
    left.evidence_id === right.evidence_id &&
    left.source_id === right.source_id &&
    left.conversation_id === right.conversation_id
  );
}

function matchesTranscript(
  result: ConversationTranscript,
  hit: ConversationSearchHit,
) {
  return (
    result?.source_id === hit.source_id &&
    result.conversation_id === hit.conversation_id &&
    Array.isArray(result.messages) &&
    result.messages.some((item) => item.evidence_id === hit.evidence_id) &&
    result.messages.every((item) => typeof item.verbatim === "string")
  );
}

function permissionUnavailable(reason: unknown) {
  return (
    [
      "source_not_authorized",
      "conversation_not_found",
      "invalid_local_token",
    ].includes(apiErrorCode(reason) ?? "") ||
    (isAxiosError(reason) && [401, 403].includes(reason.response?.status ?? 0))
  );
}
