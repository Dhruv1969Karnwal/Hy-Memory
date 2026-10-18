import type { MemoryDetail, SourceResponse } from "@one-memory/contracts";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { apiErrorCode, correctMemory, getMemory, mutateMemory } from "./api";
import { clockStamp } from "./clock-stamp";
import { MemoryRecallScope } from "./MemoryRecallScope";
import { R6Button } from "./R6Controls";
import { sourceNameOr } from "./source-label";
import { traceUiAction } from "./ui-trace";
import { isDefiniteRefusal, writeTraceStatus } from "./write-outcome";
import "./selected-memory.css";

type Page = "read" | "edit" | "original" | "delete";

export function SelectedMemoryDetail({
  detail,
  sources,
  onChanged,
  onSaveScopes,
  suspended = false,
  hideActions = false,
  conversationAt = null,
}: {
  detail: MemoryDetail;
  sources: SourceResponse[];
  onChanged(id: string): void;
  onSaveScopes?(scopes: string[]): Promise<void>;
  suspended?: boolean;
  hideActions?: boolean;
  /** D99: when the cited conversation happened, in the list's own terms. */
  conversationAt?: { at: string; basis: "conversation" | "import" } | null;
}) {
  const [page, setPage] = useState<Page>("read");
  const [draft, setDraft] = useState(detail.memory.content);
  const [base, setBase] = useState(detail.memory.revision);
  const [conflict, setConflict] = useState<MemoryDetail | null>(null);
  const [conflictFound, setConflictFound] = useState(false);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const [showHistory, setShowHistory] = useState(false);
  const flight = useRef(false),
    alive = useRef(true),
    request = useRef<string | null>(null),
    trigger = useRef<HTMLButtonElement | null>(null),
    focus = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  const memory = detail.memory;
  const evidence = detail.evidence[0];
  const statusLabel =
    memory.status === "confirmed"
      ? "Stored"
      : memory.status === "deleted"
        ? "Deleted"
        : memory.status === "superseded"
          ? "Superseded"
          : "Status pending verification";
  const source = sourceNameOr(
    sources,
    evidence?.source_id,
    "Source pending verification",
  );
  const dirty = page === "edit" && draft !== memory.content;
  useEffect(() => {
    if (page === "edit" && memory.revision > base) setConflictFound(true);
  }, [page, memory.revision, base]);
  useEffect(() => {
    if (page === "read" || suspended) return;
    const previousFocus = trigger.current;
    const root = document.querySelector<HTMLElement>(".r6-shell");
    const previous = root?.inert;
    if (root) root.inert = true;
    focus.current?.focus({ preventScroll: true });
    return () => {
      if (root) root.inert = previous ?? false;
      if (previousFocus instanceof HTMLElement && previousFocus.isConnected)
        previousFocus.focus({ preventScroll: true });
    };
  }, [page, suspended]);
  useEffect(() => {
    if (!dirty) return;
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);
  function close() {
    if (busy) return;
    if (dirty && !window.confirm("Discard unsaved changes?")) return;
    setPage("read");
    setError("");
    setConflictFound(false);
    setConflict(null);
    request.current = null;
  }
  async function refreshConflict() {
    setBusy(true);
    try {
      const latest = await getMemory(memory.memory_id);
      if (alive.current) setConflict(latest);
    } catch {
      if (alive.current)
        setError(
          "The new version cannot be read right now; the draft is kept.",
        );
    } finally {
      if (alive.current) setBusy(false);
    }
  }
  async function save() {
    if (flight.current || conflictFound || !draft.trim()) return;
    flight.current = true;
    setBusy(true);
    setError("");
    request.current ??= crypto.randomUUID();
    const trace = traceUiAction("memory", "submit");
    try {
      await correctMemory(
        memory.memory_id,
        base,
        draft,
        undefined,
        request.current,
        trace.traceId,
      );
      trace.done("succeeded");
      if (!alive.current) return;
      request.current = null;
      setPage("read");
      onChanged(memory.memory_id);
    } catch (reason) {
      trace.done(writeTraceStatus(reason), {
        reason:
          apiErrorCode(reason) === "memory_revision_conflict"
            ? "memory_revision_conflict"
            : isDefiniteRefusal(reason)
              ? "memory_correct_failed"
              : "memory_correct_result_unknown",
      });
      if (alive.current) {
        if (apiErrorCode(reason) === "memory_revision_conflict") {
          setConflictFound(true);
          setError("");
        } else if (isDefiniteRefusal(reason))
          setError(
            "The change was not submitted; the draft is kept. Fix what the message reports and retry.",
          );
        else
          setError(
            "The change is not confirmed yet; the draft is kept. Refresh and check first to avoid submitting twice.",
          );
      }
    } finally {
      flight.current = false;
      if (alive.current) setBusy(false);
    }
  }
  async function remove() {
    if (flight.current) return;
    flight.current = true;
    setBusy(true);
    setError("");
    request.current ??= crypto.randomUUID();
    const trace = traceUiAction("memory", "confirm");
    try {
      await mutateMemory(
        memory.memory_id,
        "delete",
        memory.revision,
        request.current,
        trace.traceId,
      );
      trace.done("succeeded");
      if (alive.current) {
        setPage("read");
        request.current = null;
        onChanged(memory.memory_id);
      }
    } catch (reason) {
      const refused = isDefiniteRefusal(reason);
      trace.done(refused ? "failed" : "unknown", {
        reason: refused ? "memory_delete_failed" : "memory_delete_unconfirmed",
      });
      if (alive.current)
        setError(
          refused
            ? "The delete was not submitted. Check the status of this memory and retry."
            : "The delete is not confirmed yet. Re-check the memory status; the delete scope never widens on its own.",
        );
    } finally {
      flight.current = false;
      if (alive.current) setBusy(false);
    }
  }
  async function govern(action: "undo") {
    if (flight.current) return;
    flight.current = true;
    setBusy(true);
    setError("");
    const trace = traceUiAction("memory", "retry");
    try {
      await mutateMemory(
        memory.memory_id,
        action,
        memory.revision,
        crypto.randomUUID(),
        trace.traceId,
      );
      trace.done("succeeded");
      if (alive.current) onChanged(memory.memory_id);
    } catch (reason) {
      const refused = isDefiniteRefusal(reason);
      trace.done(refused ? "failed" : "unknown", {
        reason: refused ? "memory_undo_failed" : "memory_undo_unconfirmed",
      });
      if (alive.current)
        setError(
          refused
            ? "The undo was not submitted. Fix what the message reports and retry."
            : "The undo is not confirmed yet. Check the current memory version first to avoid undoing twice.",
        );
    } finally {
      flight.current = false;
      if (alive.current) setBusy(false);
    }
  }
  const author =
    evidence?.written_by === "person" || evidence?.written_by === "user"
      ? "You said"
      : evidence?.written_by === "agent" || evidence?.written_by === "assistant"
        ? "Assistant quote"
        : "Author pending verification";
  const timestamp = clockStamp(evidence?.created_at);
  const split = memory.content.search(/[\u3002\n]/);
  const heading =
    split > 0 && split < 85
      ? memory.content.slice(0, split + 1)
      : memory.content;
  const body =
    split > 0 && split < 85 ? memory.content.slice(split + 1).trim() : "";
  return (
    <>
      <article
        className="selected-memory-detail"
        data-hide-actions={hideActions ? "true" : "false"}
      >
        <p className="selected-memory-status">
          {statusLabel} · revision {memory.revision}
        </p>
        <h3>{heading}</h3>
        {body && <p className="selected-memory-body">{body}</p>}
        <p className="selected-memory-source">
          Source {source} · {timestamp}
        </p>
        <p className="selected-memory-times">
          Conversation time{" "}
          {clockStamp(conversationAt?.at ?? evidence?.created_at)}
          {" · "}
          Saved time {clockStamp(memory.updated_at)}
        </p>
        {!hideActions && (
          <R6Button
            onClick={(event) => {
              trigger.current = event.currentTarget;
              setDraft(memory.content);
              setBase(memory.revision);
              setConflictFound(false);
              setConflict(null);
              request.current = null;
              setPage("edit");
            }}
          >
            Edit
          </R6Button>
        )}
        <section className="selected-memory-evidence">
          <h4>Original evidence</h4>
          <small>
            {author} · Original conversation {timestamp}
          </small>
          <blockquote>
            {evidence?.available
              ? evidence.verbatim
              : "This source text is not available right now."}
          </blockquote>
          {!hideActions && (
            <R6Button
              onClick={(event) => {
                trigger.current = event.currentTarget;
                setPage("original");
              }}
            >
              View source text
            </R6Button>
          )}
        </section>
        {!hideActions && (
          <R6Button variant="quiet" onClick={() => setShowHistory((v) => !v)}>
            Version history
          </R6Button>
        )}
        {!hideActions && showHistory && (
          <ol className="selected-memory-history">
            {detail.history.map((v) => (
              <li key={v.revision}>
                <small>revision {v.revision}</small>
                <p>{v.content}</p>
              </li>
            ))}
          </ol>
        )}
        {!hideActions && (
          <details className="selected-memory-more">
            <summary>More management</summary>
            {onSaveScopes && (
              <MemoryRecallScope
                detail={detail}
                busy={busy}
                onSave={onSaveScopes}
              />
            )}
            {detail.history.length > 1 && (
              <R6Button pending={busy} onClick={() => void govern("undo")}>
                Undo last change
              </R6Button>
            )}
            <R6Button
              onClick={(event) => {
                trigger.current = event.currentTarget;
                request.current = null;
                setPage("delete");
              }}
            >
              Delete memory
            </R6Button>
          </details>
        )}
        {error && page === "read" && <p role="alert">{error}</p>}
      </article>
      {page !== "read" &&
        createPortal(
          <section
            className="selected-memory-page"
            hidden={suspended}
            aria-label={
              page === "edit"
                ? "Edit memory"
                : page === "original"
                  ? "Original conversation"
                  : "Delete memory"
            }
            onKeyDown={(event) => {
              if (event.key === "Tab") {
                const controls = Array.from(
                  event.currentTarget.querySelectorAll<HTMLElement>(
                    'button:not(:disabled),textarea:not(:disabled),[tabindex="0"]',
                  ),
                );
                const first = controls[0],
                  last = controls.at(-1);
                if (
                  event.shiftKey &&
                  (document.activeElement === first ||
                    document.activeElement === focus.current)
                ) {
                  event.preventDefault();
                  last?.focus();
                } else if (!event.shiftKey && document.activeElement === last) {
                  event.preventDefault();
                  first?.focus();
                }
              }
              if (event.key === "Escape") {
                event.preventDefault();
                close();
              }
            }}
          >
            <header>
              <h1 ref={focus} tabIndex={-1}>
                {page === "edit"
                  ? "Edit memory"
                  : page === "original"
                    ? "Original conversation"
                    : "Delete memory"}
              </h1>
              {page === "original" && (
                <R6Button onClick={close}>Back to memory</R6Button>
              )}
            </header>
            {page === "edit" ? (
              <div className="selected-editor">
                <p>
                  Edit this memory; the original conversation stays unchanged.
                </p>
                <textarea
                  aria-label="Memory content"
                  name="memory-content"
                  value={draft}
                  disabled={busy}
                  maxLength={1000000}
                  onChange={(e) => {
                    setDraft(e.target.value);
                    request.current = null;
                  }}
                />
                <p>The usage scope stays unchanged</p>
                {conflictFound && (
                  <section className="selected-conflict">
                    <h2>A newer version exists</h2>
                    <p>
                      Your draft is kept. Look at the new version before
                      deciding how to change things.
                    </p>
                    <R6Button
                      pending={busy}
                      onClick={() => void refreshConflict()}
                    >
                      View the new version
                    </R6Button>
                    {conflict && (
                      <>
                        <blockquote>{conflict.memory.content}</blockquote>
                        <R6Button
                          onClick={() => {
                            setBase(conflict.memory.revision);
                            setConflictFound(false);
                            request.current = null;
                          }}
                        >
                          Keep editing from this version
                        </R6Button>
                      </>
                    )}
                  </section>
                )}
                <div className="selected-page-actions">
                  <R6Button disabled={busy} onClick={close}>
                    Cancel
                  </R6Button>
                  <R6Button
                    variant="primary"
                    pending={busy}
                    disabled={!draft.trim() || conflictFound}
                    onClick={() => void save()}
                  >
                    Save changes
                  </R6Button>
                </div>
              </div>
            ) : page === "original" ? (
              <div className="selected-original">
                <small>
                  {source} · {timestamp}
                </small>
                <p>{author}</p>
                {detail.evidence.map((item) => (
                  <section key={item.evidence_id}>
                    <small>
                      {sourceNameOr(
                        sources,
                        item.source_id,
                        "Source pending verification",
                      )}{" "}
                      ·{" "}
                      {item.written_by === "person"
                        ? "You"
                        : item.written_by === "agent"
                          ? "Assistant"
                          : "Author pending verification"}
                    </small>
                    <blockquote>
                      {item.available
                        ? item.verbatim
                        : "Source access changed or the source text is unavailable."}
                    </blockquote>
                  </section>
                ))}
                <small>Matching memory</small>
                <h2>{heading}</h2>
                <R6Button onClick={close}>View memory</R6Button>
              </div>
            ) : (
              <div className="selected-delete">
                <h2>Delete this memory?</h2>
                <p>{heading}</p>
                <p>
                  Once deleted it stops feeding new recall; the original
                  conversation in the source app is not deleted.
                  <br />
                  Content already sent to an app cannot be recalled remotely.
                </p>
                <div className="selected-page-actions">
                  <R6Button disabled={busy} onClick={close}>
                    Cancel
                  </R6Button>
                  <R6Button
                    variant="primary"
                    pending={busy}
                    onClick={() => void remove()}
                  >
                    Delete memory
                  </R6Button>
                </div>
              </div>
            )}
            {error && (
              <p className="selected-page-error" role="alert">
                {error}
              </p>
            )}
          </section>,
          document.body,
        )}
    </>
  );
}
