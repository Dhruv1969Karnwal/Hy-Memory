import {
  ArrowLeft,
  ChevronLeft,
  ChevronRight,
  RefreshCw,
  Trash2,
} from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";
import { getConversation } from "./api";
import {
  CALL_HOSTS,
  CALL_TOOLS,
  type CallEvent,
  type CallHistoryPage,
  callDeliveryLabel,
  callDetailHeadline,
  callReturnDeliveryLabel,
  callRowDetail,
  callRowHeadline,
  clearAllCallEvents,
  clearCallEvents,
  getCallHistory,
  getCallReferences,
  type ResolvedCallReference,
} from "./call-history-client";
import { clockStamp } from "./clock-stamp";
import { R6Button } from "./R6Controls";
import { traceUiAction } from "./ui-trace";
import { isDefiniteRefusal } from "./write-outcome";
import "./call-history.css";

const PAGE_SIZE = 20;
const stateLabels: Record<CallEvent["state"], string> = {
  grounded: "Query succeeded",
  empty: "Query succeeded, no related memory found",
  partial: "Query succeeded, some content is still being organized",
  timed_out: "Query timed out, no memory was provided",
  unavailable: "One Memory is unavailable right now",
  denied: "The current authorization does not allow this query",
};

function toolName(tool: CallEvent["tool"]) {
  return CALL_TOOLS[tool];
}

function sourceLabel(id: string | null) {
  return id === null ? "Not recorded" : (CALL_HOSTS[id] ?? id);
}

function CallReferences({
  event,
  onOpenMemory,
  onOpenEvidence,
}: {
  event: CallEvent;
  onOpenMemory?(id: string): void;
  onOpenEvidence?(id: string): void;
}) {
  const [result, setResult] = useState<{
    event: CallEvent;
    items: ResolvedCallReference[];
  } | null>(null);
  const [failure, setFailure] = useState<CallEvent | null>(null);
  const [transcript, setTranscript] = useState<{
    id: string;
    status: "loading" | "ready" | "failed";
    messages: { evidence_id: string; verbatim: string }[];
  } | null>(null);
  useEffect(() => {
    let stopped = false;
    setResult(null);
    setFailure(null);
    setTranscript(null);
    if (event.references.length) {
      void getCallReferences(event)
        .then((items) => {
          if (!stopped) setResult({ event, items });
        })
        .catch(() => {
          if (!stopped) setFailure(event);
        });
    }
    return () => {
      stopped = true;
    };
  }, [event]);
  const resolved = result?.event === event ? result.items : null;
  const failed = failure === event;
  const loading = event.references.length > 0 && !resolved && !failed;
  const references = new Map(
    event.references.map((reference, index) => [
      JSON.stringify([
        reference.kind,
        reference.id,
        reference.revision,
        reference.source_id,
      ]),
      { reference, current: resolved?.[index] },
    ]),
  );
  return (
    <section
      className="call-history-references"
      aria-label="Related content"
      aria-busy={loading}
    >
      <h3>Related content</h3>
      {loading && <p role="status">Loading related content</p>}
      {failed && (
        <p role="alert">
          Related content can't be read right now. Refresh and try again.
        </p>
      )}
      {event.references.length ? (
        <ul>
          {Array.from(references, ([key, { reference, current }]) => (
            <li key={key}>
              <div>
                <h4>
                  {reference.kind === "raw" ? "Source text" : "Memory"} ·{" "}
                  {current?.available && current.reason === null
                    ? current.title
                    : current?.reason === "changed"
                      ? "Version changed"
                      : loading
                        ? "Pending verification"
                        : "Unavailable"}
                </h4>
                {reference.source_id && (
                  <p>{sourceLabel(reference.source_id)}</p>
                )}
              </div>
              {current?.available && current.reason === null && (
                <R6Button
                  variant="quiet"
                  onClick={() => {
                    if (reference.kind === "memory") {
                      onOpenMemory?.(reference.id);
                      return;
                    }
                    onOpenEvidence?.(reference.id);
                    setTranscript({
                      id: reference.id,
                      status: "loading",
                      messages: [],
                    });
                    void getConversation(reference.id)
                      .then((value) => {
                        setTranscript({
                          id: reference.id,
                          status: "ready",
                          messages: value.messages.map((message) => ({
                            evidence_id: message.evidence_id,
                            verbatim: message.verbatim,
                          })),
                        });
                      })
                      .catch(() => {
                        setTranscript({
                          id: reference.id,
                          status: "failed",
                          messages: [],
                        });
                      });
                  }}
                >
                  {reference.kind === "raw"
                    ? "View source text"
                    : "View memory"}
                </R6Button>
              )}
            </li>
          ))}
        </ul>
      ) : (
        <p>No related content found</p>
      )}
      {transcript && (
        <aside
          aria-label="Source text body"
          aria-busy={transcript.status === "loading"}
        >
          {transcript.status === "loading" && (
            <p role="status">Loading source text</p>
          )}
          {transcript.status === "failed" && (
            <p role="alert">
              The source text is no longer available or its version changed.
            </p>
          )}
          {transcript.status === "ready" &&
            transcript.messages.map((message) => (
              <p key={message.evidence_id}>{message.verbatim}</p>
            ))}
        </aside>
      )}
    </section>
  );
}

function shortTime(value: string) {
  return clockStamp(value);
}

function ClearConfirmation({
  title,
  description,
  confirmLabel,
  pending,
  onCancel,
  onConfirm,
}: {
  title: string;
  description: string;
  confirmLabel: string;
  pending: boolean;
  onCancel(): void;
  onConfirm(): void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  const descriptionId = useId();
  useEffect(() => {
    const previous = document.activeElement;
    dialog.current?.showModal();
    return () => {
      if (previous instanceof HTMLElement && previous.isConnected)
        previous.focus();
    };
  }, []);
  return (
    <dialog
      ref={dialog}
      className="call-history-confirm"
      aria-labelledby={titleId}
      aria-describedby={descriptionId}
      onCancel={(event) => {
        event.preventDefault();
        if (!pending) onCancel();
      }}
    >
      <h2 id={titleId}>{title}</h2>
      <p id={descriptionId}>{description}</p>
      <div className="call-history-actions">
        <R6Button disabled={pending} onClick={onCancel}>
          Cancel
        </R6Button>
        <R6Button variant="primary" pending={pending} onClick={onConfirm}>
          {confirmLabel}
        </R6Button>
      </div>
    </dialog>
  );
}

export function CallHistory({
  onBack,
  onOpenMemory,
  onOpenEvidence,
}: {
  onBack(): void;
  onOpenMemory?(id: string): void;
  onOpenEvidence?(id: string): void;
}) {
  const [request, setRequest] = useState({
    hostId: "",
    offset: 0,
    revision: 0,
  });
  const [result, setResult] = useState<{
    request: typeof request;
    page: CallHistoryPage;
  } | null>(null);
  const [failure, setFailure] = useState<typeof request | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [confirmation, setConfirmation] = useState<
    { kind: "page"; ids: string[] } | { kind: "all" } | null
  >(null);
  const [clearing, setClearing] = useState(false);
  const [clearNotice, setClearNotice] = useState("");
  const mounted = useRef(false);
  const clearFlight = useRef(false);
  const heading = useRef<HTMLHeadingElement>(null);
  const list = useRef<HTMLUListElement>(null);
  const previousSelection = useRef<string | null>(null);
  const titleId = useId();
  const filterId = useId();

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    let stopped = false;
    setResult(null);
    setFailure(null);
    void getCallHistory({
      limit: PAGE_SIZE,
      offset: request.offset,
      ...(request.hostId ? { hostId: request.hostId } : {}),
    })
      .then((page) => {
        if (stopped) return;
        setResult({ request, page });
      })
      .catch(() => {
        if (!stopped) setFailure(request);
      });
    return () => {
      stopped = true;
    };
  }, [request]);

  useEffect(() => {
    if (selectedId !== null) heading.current?.focus();
    else if (previousSelection.current !== null) {
      const row = Array.from(
        list.current?.querySelectorAll<HTMLButtonElement>("[data-event-id]") ??
          [],
      ).find((button) => button.dataset.eventId === previousSelection.current);
      (row ?? heading.current)?.focus();
    }
    previousSelection.current = selectedId;
  }, [selectedId]);

  const page = result?.request === request && !clearing ? result.page : null;
  const failed = failure === request;
  const loading = !page && !failed && !clearing;
  const events = page?.available ? page.items : [];
  const selected =
    selectedId === null
      ? null
      : events.find((item) => item.event_id === selectedId);
  const returnedFrom =
    selected?.references.find((item) => item.source_id)?.source_id ?? null;
  const refresh = () =>
    setRequest((current) => ({ ...current, revision: current.revision + 1 }));

  async function confirmClear() {
    if (!confirmation || clearFlight.current) return;
    const clearingAll = confirmation.kind === "all";
    clearFlight.current = true;
    setClearing(true);
    setClearNotice("");
    const trace = traceUiAction("calls", "confirm");
    try {
      if (clearingAll) await clearAllCallEvents(trace.traceId);
      else await clearCallEvents([...confirmation.ids], trace.traceId);
      trace.done("succeeded");
      if (!mounted.current) return;
      setClearNotice(
        clearingAll ? "All records cleared." : "Selected records cleared.",
      );
    } catch (reason) {
      // A 4xx (e.g. `409 call_history_disabled`) is Core answering and
      // refusing: nothing was deleted, so it is a definite failure, not an
      // unconfirmed one. Everything else (timeout/5xx/transport) may or may
      // not have landed and stays `unknown` with the no-auto-retry notice.
      const refused = isDefiniteRefusal(reason);
      trace.done(refused ? "failed" : "unknown", {
        reason: refused
          ? clearingAll
            ? "clear_all_failed"
            : "clear_selected_failed"
          : clearingAll
            ? "clear_all_unconfirmed"
            : "clear_selected_unconfirmed",
      });
      if (!mounted.current) return;
      setClearNotice(
        refused
          ? clearingAll
            ? "Couldn't clear the usage records. The existing records are unchanged. Check them and try again."
            : "Couldn't clear the selected records. The existing records are unchanged. Check them and try again."
          : "The clear result is unconfirmed. Check the records after rereading. The clear was not retried automatically.",
      );
    } finally {
      clearFlight.current = false;
      if (mounted.current) {
        setClearing(false);
        setConfirmation(null);
        setSelectedId(null);
        setRequest((current) => ({
          ...current,
          offset: clearingAll ? 0 : current.offset,
          revision: current.revision + 1,
        }));
      }
    }
  }

  return (
    <section className="call-history" aria-labelledby={titleId}>
      <header className="call-history-heading">
        <h1 id={titleId} ref={heading} tabIndex={-1}>
          {selectedId === null ? "Usage records" : "This memory lookup"}
        </h1>
        <R6Button
          className="call-history-back"
          disabled={clearing}
          onClick={selectedId === null ? onBack : () => setSelectedId(null)}
        >
          <ArrowLeft size={16} aria-hidden="true" />
          {selectedId === null ? "Back to apps" : "Back to records"}
        </R6Button>
      </header>
      <div
        className={`call-history-toolbar${selectedId !== null ? " call-history-detail-toolbar" : ""}`}
      >
        {selectedId === null ? (
          <div className="call-history-filter">
            <label className="call-history-sr-only" htmlFor={filterId}>
              Filter by app
            </label>
            <select
              id={filterId}
              value={request.hostId}
              disabled={clearing}
              onChange={(event) => {
                setSelectedId(null);
                setRequest((current) => ({
                  hostId: event.target.value,
                  offset: 0,
                  revision: current.revision + 1,
                }));
              }}
            >
              <option value="">All apps</option>
              {Object.entries(CALL_HOSTS).map(([host, label]) => (
                <option key={host} value={host}>
                  {label}
                </option>
              ))}
            </select>
          </div>
        ) : (
          <span />
        )}
        <div className="call-history-actions">
          {selectedId === null && <span>Recent records</span>}
          <R6Button
            className="call-history-icon"
            variant="quiet"
            title="Refresh records"
            aria-label="Refresh records"
            disabled={clearing}
            onClick={refresh}
          >
            <RefreshCw size={18} aria-hidden="true" />
          </R6Button>
          {selectedId === null && (
            <>
              <R6Button
                className="call-history-icon"
                variant="quiet"
                title="Clear this page"
                aria-label="Clear this page"
                disabled={!events.length || clearing}
                onClick={() =>
                  setConfirmation({
                    kind: "page",
                    ids: events.map((item) => item.event_id),
                  })
                }
              >
                <Trash2 size={18} aria-hidden="true" />
              </R6Button>
              <R6Button
                variant="quiet"
                title="Clear all records"
                aria-label="Clear all records"
                disabled={!page?.total || clearing}
                onClick={() => setConfirmation({ kind: "all" })}
              >
                Clear all
              </R6Button>
            </>
          )}
        </div>
      </div>
      {clearNotice && (
        <p className="call-history-notice" role="status">
          {clearNotice}
        </p>
      )}
      <div aria-busy={loading || clearing}>
        {loading && (
          <p className="call-history-state" role="status">
            Loading usage records
          </p>
        )}
        {clearing && (
          <p className="call-history-state" role="status">
            Clearing the selected records
          </p>
        )}
        {failed && (
          <p className="call-history-state" role="alert">
            Couldn't read the records. Refresh and try again.
          </p>
        )}
        {page && !page.available && (
          <p className="call-history-state" role="alert">
            Usage records are unavailable right now. Refresh in a moment.
          </p>
        )}
        {page?.recording_error && (
          <p className="call-history-notice" role="alert">
            Some calls weren't recorded, so the list may be incomplete.
          </p>
        )}
        {page?.available &&
          (selectedId !== null ? (
            selected ? (
              <article
                className="call-history-detail"
                aria-label="Call details"
              >
                <h2>{`${sourceLabel(selected.host_id)} · ${shortTime(selected.occurred_at)}`}</h2>
                <p className="call-history-summary">
                  {callDetailHeadline(selected)}
                  {returnedFrom ? ` · From ${sourceLabel(returnedFrom)}` : ""}
                </p>
                <p className="call-history-stage">
                  {stateLabels[selected.state]}
                </p>
                <p className="call-history-stage">
                  Found {selected.raw_count} source excerpts · memories{" "}
                  {selected.memory_count} · elapsed{" "}
                  {(selected.elapsed_ms / 1000).toFixed(1)} seconds
                </p>
                {callDeliveryLabel(selected) && (
                  <p className="call-history-stage">
                    {callDeliveryLabel(selected)}
                  </p>
                )}
                <CallReferences
                  event={selected}
                  onOpenMemory={onOpenMemory}
                  onOpenEvidence={onOpenEvidence}
                />
                {callReturnDeliveryLabel(selected) && (
                  <p className="call-history-adoption">
                    {callReturnDeliveryLabel(selected)}
                  </p>
                )}
                <section
                  className="call-history-metadata"
                  aria-label="Record details"
                >
                  <h3>Record details</h3>
                  <p>
                    For privacy, this doesn't keep the words you asked or the
                    answer, only the time, source and result.
                  </p>
                  <dl>
                    <div>
                      <dt>Exact time</dt>
                      <dd>
                        <time dateTime={selected.occurred_at}>
                          {selected.occurred_at}
                        </time>
                      </dd>
                    </div>
                    <div>
                      <dt>App</dt>
                      <dd>{sourceLabel(selected.host_id)}</dd>
                    </div>
                    <div>
                      <dt>Type</dt>
                      <dd>{toolName(selected.tool)}</dd>
                    </div>
                    <div>
                      <dt>Result state</dt>
                      {/* The tool above is already humanized in this same
                        panel ("Memory lookup", never `om_recall`); printing the raw
                        `grounded`/`partial` enum here leaks the internal state
                        vocabulary the rest of the page translates. */}
                      <dd>{stateLabels[selected.state]}</dd>
                    </div>
                  </dl>
                </section>
              </article>
            ) : (
              <p className="call-history-state" role="status">
                This record is no longer on the current page. Go back to the
                records to view it.
              </p>
            )
          ) : (
            <>
              <ul
                ref={list}
                className="call-history-list"
                aria-label="Call records"
              >
                {events.map((item) => (
                  <li key={item.event_id}>
                    <time dateTime={item.occurred_at} title={item.occurred_at}>
                      {shortTime(item.occurred_at)}
                    </time>
                    <div className="call-history-row-content">
                      <h2>
                        <button
                          type="button"
                          data-event-id={item.event_id}
                          onClick={() => setSelectedId(item.event_id)}
                        >
                          {callRowHeadline(item)}
                        </button>
                      </h2>
                      <p>{callRowDetail(item)}</p>
                    </div>
                  </li>
                ))}
              </ul>
              {!events.length && (
                <p className="call-history-state" role="status">
                  {request.offset
                    ? "No records on this page"
                    : "No usage records yet"}
                </p>
              )}
              <nav
                className="call-history-pagination"
                aria-label="Record pagination"
              >
                <R6Button
                  aria-label="Previous page"
                  title="Previous page"
                  disabled={request.offset === 0}
                  onClick={() =>
                    setRequest((current) => ({
                      ...current,
                      offset: Math.max(0, current.offset - PAGE_SIZE),
                    }))
                  }
                >
                  <ChevronLeft size={18} aria-hidden="true" />
                </R6Button>
                <span>
                  {events.length
                    ? `${page.offset + 1}–${page.offset + events.length}`
                    : "0"}{" "}
                  / {page.total}
                </span>
                <R6Button
                  aria-label="Next page"
                  title="Next page"
                  disabled={
                    page.offset + PAGE_SIZE >= page.total || !events.length
                  }
                  onClick={() =>
                    setRequest((current) => ({
                      ...current,
                      offset: current.offset + PAGE_SIZE,
                    }))
                  }
                >
                  <ChevronRight size={18} aria-hidden="true" />
                </R6Button>
              </nav>
            </>
          ))}
      </div>
      {page?.available &&
        page.retention_days !== null &&
        page.max_events !== null && (
          <p className="call-history-policy">
            Retained for {page.retention_days} days, up to{" "}
            {new Intl.NumberFormat("zh-CN").format(page.max_events)} records
          </p>
        )}
      {confirmation && (
        <ClearConfirmation
          title={
            confirmation.kind === "all"
              ? "Clear all usage records?"
              : `Clear ${confirmation.ids.length} records on this page?`
          }
          description={
            confirmation.kind === "all"
              ? "This deletes every retained call record, not just this page or the current filter. It doesn't delete source text or memories, and it doesn't change authorization or retention rules. This can't be undone."
              : "This clears only these call records. It doesn't delete source text or memories, and it doesn't change authorization or retention rules. This can't be undone."
          }
          confirmLabel={
            confirmation.kind === "all"
              ? "Confirm clearing all"
              : "Confirm clearing"
          }
          pending={clearing}
          onCancel={() => setConfirmation(null)}
          onConfirm={() => void confirmClear()}
        />
      )}
    </section>
  );
}
