import type {
  ConnectSourcesResponse,
  FileImportReceipt,
  SourceResponse,
} from "@one-memory/contracts";
import { Check, Clipboard, FileUp, RefreshCw, Save, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  apiErrorCode,
  commitHistoryFile,
  discardHistoryFile,
  getHistoryFile,
  historyImportIssue,
  reconcileHistoryUpload,
  uploadHistoryFile,
} from "./api";
import { countLabel } from "./count-label";
import { readPendingFileUpload } from "./pending-file-upload";
import { traceUiAction } from "./ui-trace";

const errors: Record<string, string> = {
  unsupported_export_schema:
    "The conversation format of this file is not supported yet; nothing was imported.",
  export_size_exceeded:
    "The file or its total unpacked content exceeds 128 MB. Select conversations.json instead.",
  ambiguous_export_payload:
    "The archive has no single conversations.json. Select that file directly.",
  unsafe_export_archive:
    "The archive failed the security check; it was neither unpacked nor imported.",
  invalid_export_json:
    "This JSON file is incomplete or uses an unsupported encoding.",
  export_has_no_text_conversations:
    "The file has no importable text conversations.",
  duplicate_export_identity:
    "The file has duplicate conversation or message identities; nothing was imported.",
  export_record_invalid:
    "The file contains overlong or unreadable conversation records; nothing was imported.",
  imports_disabled: "Imports are disabled on this instance; use the full app.",
  generic_import_invalid:
    "The conversation format is incomplete; nothing was imported.",
  unsupported_memory_content_schema:
    "This is not a supported memory file format. Use TXT, Markdown or one-memory-content-v1 JSON; ZIP and vendor-proprietary formats are not supported yet.",
  invalid_memory_content_record:
    "The existing memory record is empty or has unsupported fields; nothing was imported.",
  invalid_memory_content_time:
    "The existing memory time lacks a date or timezone; keep unknown times as null.",
};

export function HistoryImport({
  onConnect,
  onRefresh,
  disabled = false,
}: {
  disabled?: boolean;
  onConnect(
    ids: string[],
    snapshot: SourceResponse[],
    options?: { retryFailed?: boolean; importUploadId?: string },
  ): Promise<ConnectSourcesResponse>;
  onRefresh(): Promise<{ sources: SourceResponse[] }>;
}) {
  const [open, setOpen] = useState(false);
  const [receipt, setReceipt] = useState<FileImportReceipt | null>(null);
  const [selectionValid, setSelectionValid] = useState(false);
  const [consent, setConsent] = useState(false);
  const [candidate, setCandidate] = useState<SourceResponse | null>(null);
  const [saveUnknown, setSaveUnknown] = useState(false);
  const [pendingUpload, setPendingUpload] = useState<string | null>(null);
  const [uploadStorageFailed, setUploadStorageFailed] = useState(false);
  const [connection, setConnection] = useState<"new" | "failed" | "unknown">(
    "new",
  );
  const [operation, setOperation] = useState<
    "upload" | "save" | "refresh" | "connect" | "discard" | null
  >(null);
  const busy = operation !== null;
  const inputLocked =
    busy || saveUnknown || !!pendingUpload || uploadStorageFailed;
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [mode, setMode] = useState<"file" | "paste">("file");
  const [inputKind, setInputKind] = useState<
    "raw_conversation" | "memory_content"
  >("raw_conversation");
  const isMemoryContent = inputKind === "memory_content";
  const [paste, setPaste] = useState("");
  const [issue, setIssue] =
    useState<ReturnType<typeof historyImportIssue>>(null);
  const panel = useRef<HTMLElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const wasOpen = useRef(false);
  const scopeConsent = useRef<HTMLInputElement>(null);
  const alive = useRef(true);
  const generation = useRef(0);
  const flight = useRef<number | null>(null);
  const closing = useRef(false);
  const discards = useRef(new Map<string, Promise<void>>());
  const tracked = useRef<{
    receipt: FileImportReceipt;
    uncertain: boolean;
  } | null>(null);
  const discardStaged = useCallback((id: string) => {
    const pending = discards.current.get(id);
    if (pending) return pending;
    const request = discardHistoryFile(id).finally(() =>
      discards.current.delete(id),
    );
    discards.current.set(id, request);
    return request;
  }, []);
  const updateUploadPending = useCallback(() => {
    try {
      const pending = readPendingFileUpload();
      setPendingUpload(pending);
      setUploadStorageFailed(false);
      return pending;
    } catch {
      setUploadStorageFailed(true);
      throw new Error("file_upload_storage_unavailable");
    }
  }, []);
  useEffect(() => {
    alive.current = true;
    try {
      updateUploadPending();
    } catch {
      /* Preserve the local recovery fence. */
    }
    return () => {
      alive.current = false;
      generation.current += 1;
      const previous = tracked.current;
      tracked.current = null;
      if (previous && !previous.receipt.committed && !previous.uncertain) {
        void discardStaged(previous.receipt.upload_id).catch(() => {});
      }
    };
  }, [discardStaged, updateUploadPending]);
  useEffect(() => {
    if (open)
      panel.current
        ?.querySelector<HTMLInputElement>("input")
        ?.focus({ preventScroll: true });
    else if (wasOpen.current) trigger.current?.focus({ preventScroll: true });
    wasOpen.current = open;
  }, [open]);
  useEffect(() => {
    if (candidate && !busy) scopeConsent.current?.focus();
  }, [candidate, busy]);

  function start(next: NonNullable<typeof operation>) {
    if (flight.current !== null || !alive.current) return null;
    const token = ++generation.current;
    flight.current = token;
    setOperation(next);
    setError("");
    setIssue(null);
    return token;
  }

  function active(token: number) {
    return alive.current && token === generation.current;
  }

  function finish(token: number) {
    if (!active(token)) return;
    flight.current = null;
    setOperation(null);
  }

  function invalidateSelection() {
    setConsent(false);
    setSelectionValid(false);
    setCandidate(null);
    setConnection("new");
  }

  function remember(next: FileImportReceipt, uncertain = false) {
    tracked.current = { receipt: next, uncertain };
    setReceipt(next);
    setSaveUnknown(uncertain);
  }

  async function discardPrevious() {
    const previous = tracked.current;
    if (previous && !previous.receipt.committed && !previous.uncertain) {
      await discardStaged(previous.receipt.upload_id);
    }
    if (tracked.current === previous) tracked.current = null;
  }

  async function reviewSource(saved: FileImportReceipt, token: number) {
    setOperation("refresh");
    const result = await onRefresh();
    if (!active(token)) return;
    const matches = result.sources.filter(
      (source) => source.source_id === saved.source_id,
    );
    const source = matches[0];
    if (
      matches.length !== 1 ||
      !source.available ||
      !source.candidate_root_id?.trim() ||
      !source.candidate_path?.trim() ||
      !["default", "environment_override"].includes(source.candidate_kind) ||
      (source.location_state && source.location_state !== "found")
    )
      throw new Error("source_identity_unavailable");
    // Copy the displayed candidate, never replace it during the later consent click.
    setCandidate({ ...source });
    setConsent(false);
    setConnection(
      candidate?.source_id === source.source_id &&
        candidate.candidate_root_id === source.candidate_root_id &&
        candidate.candidate_path === source.candidate_path &&
        candidate.candidate_kind === source.candidate_kind
        ? connection
        : "new",
    );
  }

  async function select(file: File | undefined) {
    if (!file || inputLocked) return;
    const token = start("upload");
    if (token === null) return;
    invalidateSelection();
    const trace = traceUiAction("sources", "open");
    let settled = false;
    const settle = (outcome: "succeeded" | "failed" | "unknown") => {
      if (settled) return;
      settled = true;
      trace.done(outcome);
    };
    try {
      await discardPrevious();
      if (!active(token)) return;
      setReceipt(null);
      setSaveUnknown(false);
      if (file.size > 128 * 1024 * 1024) {
        setError(errors.export_size_exceeded);
        settle("failed");
        return;
      }
      const next = await uploadHistoryFile(file, inputKind, trace.traceId);
      settle("succeeded");
      if (!active(token)) {
        if (!next.committed)
          await discardStaged(next.upload_id).catch(() => {});
        return;
      }
      remember(next);
      setSelectionValid(true);
    } catch (reason) {
      settle(apiErrorCode(reason) ? "failed" : "unknown");
      if (!active(token)) return;
      try {
        updateUploadPending();
      } catch {
        /* Keep controls locked. */
      }
      setIssue(historyImportIssue(reason));
      setError(
        errors[apiErrorCode(reason) ?? ""] ??
          "This history package could not be read. Check the file and retry.",
      );
    } finally {
      // An early return before any request (e.g. the selection was invalidated)
      // would otherwise leave a `ui.action` with no matching `ui.result`, which
      // the native verifier reports as an incomplete chain. `settle` is
      // idempotent, so this is a no-op once a real outcome was recorded.
      settle("unknown");
      finish(token);
    }
  }

  async function changeInput(update: () => void) {
    if (inputLocked) return;
    const token = start("discard");
    if (token === null) return;
    invalidateSelection();
    try {
      await discardPrevious();
      if (!active(token)) return;
      setReceipt(null);
      setSaveUnknown(false);
      update();
    } catch {
      if (active(token))
        setError("The staged file could not be cleaned up. Retry the switch.");
    } finally {
      finish(token);
    }
  }

  async function recoverUpload() {
    const token = start("refresh");
    if (token === null) return;
    invalidateSelection();
    try {
      const next = await reconcileHistoryUpload();
      if (!active(token)) {
        if (next && !next.committed)
          await discardStaged(next.upload_id).catch(() => {});
        return;
      }
      updateUploadPending();
      if (next) {
        remember(next);
        setInputKind(next.content_kind ?? "raw_conversation");
        setSelectionValid(true);
        setNotice(
          "The upload result was recovered; no new read authorization was granted. Review the content before continuing.",
        );
        if (next.committed) await reviewSource(next, token);
      }
    } catch {
      if (active(token)) {
        let pending: string | null = null;
        try {
          pending = updateUploadPending();
        } catch {
          /* Storage failure is separate. */
        }
        setError(
          pending
            ? "The upload result is still pending verification; it is not reuploaded. Check again later."
            : "The upload result is currently unavailable; check the file status. No new read authorization was submitted.",
        );
      }
    } finally {
      finish(token);
    }
  }

  async function save() {
    if (
      !receipt ||
      receipt.committed ||
      saveUnknown ||
      !selectionValid ||
      !consent
    )
      return;
    const token = start("save");
    if (token === null) return;
    setConsent(false);
    // A lost POST response is not evidence that publishing failed.
    remember(receipt, true);
    let saved = false;
    const trace = traceUiAction("sources", "confirm");
    let settled = false;
    const settle = (outcome: "succeeded" | "failed" | "unknown") => {
      if (settled) return;
      settled = true;
      trace.done(outcome);
    };
    try {
      const next = await commitHistoryFile(receipt.upload_id, trace.traceId);
      if (!active(token)) return;
      assertFileReceipt(receipt, next);
      if (!next.committed) throw new Error("unconfirmed_commit");
      remember(next);
      saved = true;
      await reviewSource(next, token);
      settle("succeeded");
    } catch (reason) {
      settle(saved ? "unknown" : apiErrorCode(reason) ? "failed" : "unknown");
      if (active(token))
        setError(
          saved
            ? "The file was saved, but the source location or identity was not verified. Recheck the source; no read authorization was submitted."
            : "The file save result is not confirmed yet. Check the file receipt first; the save or authorization is not simply repeated.",
        );
    } finally {
      // See `select()`: an early return after the request still needs a
      // `ui.result`, and `settle` makes this idempotent.
      settle("unknown");
      finish(token);
    }
  }

  async function recover() {
    if (!receipt) return;
    const token = start("refresh");
    if (token === null) return;
    setCandidate(null);
    setConsent(false);
    setConnection("new");
    try {
      const next = await getHistoryFile(receipt.upload_id);
      if (!active(token)) return;
      assertFileReceipt(receipt, next);
      if (receipt.committed && !next.committed)
        throw new Error("commit_receipt_regressed");
      remember(next);
      if (next.committed) await reviewSource(next, token);
    } catch {
      if (active(token))
        setError(
          receipt.committed
            ? "The file was saved, but the file receipt or source identity could not be verified. The saved file is kept; retry verification."
            : "The file save result cannot be verified right now. The original receipt is kept; the save or authorization is not repeated.",
        );
    } finally {
      finish(token);
    }
  }

  async function confirm() {
    if (!receipt?.committed || !candidate || !selectionValid || !consent)
      return;
    const token = start("connect");
    if (token === null) return;
    setConsent(false);
    try {
      const result = await onConnect([receipt.source_id], [{ ...candidate }], {
        importUploadId: receipt.upload_id,
        ...(connection === "failed" ? { retryFailed: true } : {}),
      });
      if (!active(token)) return;
      const accepted = result.accepted_source_ids;
      const failed = Object.keys(result.failed_sources);
      if (
        accepted.length === 0 &&
        failed.length === 1 &&
        failed[0] === receipt.source_id
      ) {
        setConnection("failed");
        setError(
          "This source could not be connected, but the file was saved. After reviewing the directory above you can explicitly retry this source; if the location changed, recheck the source first.",
        );
        return;
      }
      if (
        accepted.length !== 1 ||
        accepted[0] !== receipt.source_id ||
        failed.length
      )
        throw new Error("invalid_connection_receipt");
      setNotice(
        isMemoryContent
          ? "Source connection receipt obtained. Content verification, model processing and usage scope for the existing memory still need separate confirmation."
          : "Source connection receipt obtained; import progress is tracked by the corresponding source task. Model processing and usage permissions are unchanged.",
      );
      tracked.current = null;
      setReceipt(null);
      invalidateSelection();
      setOpen(false);
    } catch {
      if (active(token)) {
        setConnection("unknown");
        setError(
          "The connection result is not confirmed yet, but the file was saved. Check the connection receipt; if the source changed you must review and confirm again; authorization is not renewed automatically.",
        );
      }
    } finally {
      finish(token);
    }
  }

  async function close() {
    if (closing.current || operation === "discard") return;
    closing.current = true;
    const previousOperation = operation;
    const previous = tracked.current;
    const token = ++generation.current;
    flight.current = token;
    setOperation("discard");
    invalidateSelection();
    try {
      await discardPrevious();
      if (!active(token)) return;
      setReceipt(null);
      setSaveUnknown(false);
      setError("");
      setIssue(null);
      if (previous?.receipt.committed || previous?.uncertain) {
        setNotice(
          previousOperation === "connect" || connection === "unknown"
            ? "Confirmation closed. A connection request already sent may still proceed; check the result on the corresponding source. Closing does not revoke authorization or cancel tasks."
            : previous?.uncertain
              ? "Confirmation closed. The file save request may have been submitted; check it in the source. No new read authorization was submitted."
              : "The file is kept on this device; no new read authorization was submitted.",
        );
      }
      setOpen(false);
    } catch {
      if (active(token))
        setError("The staged file could not be cleaned up. Retry closing.");
    } finally {
      closing.current = false;
      finish(token);
    }
  }

  return (
    <>
      <button
        type="button"
        className="secondary history-import-trigger"
        ref={trigger}
        disabled={open || disabled}
        onClick={() => {
          try {
            updateUploadPending();
          } catch {
            /* Show the recovery state on open. */
          }
          setOpen(true);
          setNotice("");
        }}
      >
        <FileUp size={16} />
        Import history package
      </button>
      {notice && (
        <p role="status" className="history-import-notice">
          {notice}
        </p>
      )}
      {open && (
        <div className="consent-backdrop">
          <section
            className="consent-sheet history-import-sheet"
            role="dialog"
            aria-modal="true"
            aria-labelledby="history-import-title"
            ref={panel}
            onKeyDown={(event) => {
              if (event.key === "Escape") {
                event.preventDefault();
                void close();
              }
              if (event.key !== "Tab") return;
              const controls = [
                ...(panel.current?.querySelectorAll<HTMLElement>(
                  "button:not(:disabled), input:not(:disabled), textarea:not(:disabled), summary",
                ) ?? []),
              ];
              const first = controls[0],
                last = controls.at(-1);
              if (event.shiftKey && document.activeElement === first) {
                event.preventDefault();
                last?.focus();
              } else if (!event.shiftKey && document.activeElement === last) {
                event.preventDefault();
                first?.focus();
              }
            }}
          >
            <header className="memory-note-heading">
              <h2 id="history-import-title">Import history package</h2>
              <button
                type="button"
                className="quiet"
                aria-label="Close history import"
                title="Close"
                disabled={operation === "discard"}
                onClick={() => void close()}
              >
                <X size={18} />
              </button>
            </header>
            {(pendingUpload || uploadStorageFailed) && (
              <div role="status" className="model-refresh-notice">
                <p>
                  {uploadStorageFailed
                    ? "The local upload recovery record cannot be read; no new upload is started."
                    : "The last upload result is pending verification; there is no reupload or automatic import."}
                </p>
                <button
                  type="button"
                  disabled={busy || uploadStorageFailed}
                  onClick={() => void recoverUpload()}
                >
                  Check upload result
                </button>
              </div>
            )}
            <fieldset
              className="history-content-kind"
              aria-label="Content type"
              disabled={inputLocked}
            >
              <label>
                <input
                  type="radio"
                  name="history-kind"
                  value="raw_conversation"
                  checked={!isMemoryContent}
                  onChange={() =>
                    void changeInput(() => {
                      setInputKind("raw_conversation");
                      setPaste("");
                    })
                  }
                />
                Historical conversations
              </label>
              <label>
                <input
                  type="radio"
                  name="history-kind"
                  value="memory_content"
                  checked={isMemoryContent}
                  onChange={() =>
                    void changeInput(() => {
                      setInputKind("memory_content");
                      setPaste("");
                    })
                  }
                />
                Existing memory
              </label>
            </fieldset>
            <fieldset
              className="map-axis import-input-mode"
              aria-label="Import method"
            >
              <button
                type="button"
                disabled={inputLocked}
                aria-pressed={mode === "file"}
                onClick={() => {
                  if (mode !== "file") void changeInput(() => setMode("file"));
                }}
              >
                <FileUp size={15} aria-hidden="true" />
                Select file
              </button>
              <button
                type="button"
                disabled={inputLocked}
                aria-pressed={mode === "paste"}
                onClick={() => {
                  if (mode !== "paste")
                    void changeInput(() => setMode("paste"));
                }}
              >
                <Clipboard size={15} aria-hidden="true" />
                {isMemoryContent ? "Paste content" : "Paste conversation"}
              </button>
            </fieldset>
            {mode === "file" ? (
              <label className="history-file-label">
                History data file
                <input
                  key={inputKind}
                  type="file"
                  accept={
                    isMemoryContent ? ".json,.md,.txt" : ".json,.zip,.md,.txt"
                  }
                  name="history-file"
                  disabled={inputLocked}
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    event.target.value = "";
                    void select(file);
                  }}
                />
              </label>
            ) : selectionValid ? (
              <button
                type="button"
                className="quiet"
                disabled={inputLocked}
                onClick={() => {
                  invalidateSelection();
                }}
              >
                Edit source text
              </button>
            ) : (
              <div className="history-paste">
                <label htmlFor="history-paste-text">
                  {isMemoryContent
                    ? "Existing memory source text"
                    : "Conversation source text"}
                </label>
                <textarea
                  id="history-paste-text"
                  name="history-paste-text"
                  autoComplete="off"
                  spellCheck={false}
                  rows={7}
                  disabled={inputLocked}
                  value={paste}
                  placeholder={
                    isMemoryContent
                      ? "Existing memory content..."
                      : "User: original message...\nAssistant: original reply..."
                  }
                  onChange={(event) => {
                    setPaste(event.target.value);
                    setSelectionValid(false);
                    setConsent(false);
                    setError("");
                    setIssue(null);
                  }}
                />
                <button
                  type="button"
                  className="secondary"
                  disabled={inputLocked || !paste.trim()}
                  onClick={() =>
                    void select(
                      new File(
                        [paste],
                        isMemoryContent
                          ? paste.trimStart().startsWith("{")
                            ? "pasted-memory.json"
                            : "pasted-memory.txt"
                          : paste.trimStart().startsWith("[")
                            ? "pasted-conversation.json"
                            : "pasted-conversation.txt",
                        { type: "text/plain" },
                      ),
                    )
                  }
                >
                  Preview content
                </button>
              </div>
            )}
            <p className="history-format-note">
              {isMemoryContent
                ? "TXT / Markdown or one-memory-content-v1 JSON. The original conversation is not provided; importing does not mean facts are confirmed, and it grants no extra read access for other apps."
                : "ChatGPT / Claude JSON, ZIP, Markdown with role markers, TXT or JSON turns. Text conversations only; attachments are excluded and imports are snapshots."}
            </p>
            <p className="history-format-note">
              This confirmation does not grant model processing, continuous
              sync, external calls or installation permissions; existing
              permissions are not changed here.
            </p>
            {busy && (
              <p role="status">
                {operation === "upload"
                  ? "Reading the file locally..."
                  : operation === "save"
                    ? "Saving the file..."
                    : operation === "refresh"
                      ? "Checking the file and source..."
                      : operation === "connect"
                        ? "Checking the connection receipt..."
                        : "Cleaning up the staged file..."}
              </p>
            )}
            {receipt && selectionValid && (
              <>
                <dl className="history-import-summary">
                  <div>
                    <dt>File</dt>
                    <dd>{receipt.filename}</dd>
                  </div>
                  <div>
                    <dt>Source</dt>
                    <dd>
                      {receipt.vendor === "memory-content"
                        ? "Manually provided existing memory"
                        : receipt.vendor === "generic"
                          ? "Manually provided conversation"
                          : receipt.vendor === "chatgpt"
                            ? "ChatGPT"
                            : "Claude"}
                    </dd>
                  </div>
                  <div>
                    <dt>File contents</dt>
                    <dd>
                      {receipt.messages}{" "}
                      {isMemoryContent
                        ? "existing memory records"
                        : "text messages"}
                    </dd>
                  </div>
                  <div>
                    <dt>Author markers</dt>
                    <dd>
                      User {receipt.human_messages} - Agent{" "}
                      {receipt.agent_messages}
                      {(receipt.unknown_author_messages ?? 0) > 0 &&
                        ` - Unknown ${receipt.unknown_author_messages}`}
                    </dd>
                  </div>
                  <div>
                    <dt>
                      {isMemoryContent
                        ? "File snapshot"
                        : "Conversations and branches"}
                    </dt>
                    <dd>{receipt.conversations}</dd>
                  </div>
                  {(receipt.skipped_records > 0 ||
                    receipt.ignored_files > 0) && (
                    <div>
                      <dt>Excluded</dt>
                      <dd>
                        {countLabel(
                          receipt.skipped_records,
                          "non-conversation record",
                        )}{" "}
                        - {countLabel(receipt.ignored_files, "other file")}
                      </dd>
                    </div>
                  )}
                </dl>
                {!!receipt.preview?.length && (
                  <details className="history-preview" open>
                    <summary>
                      Source preview - {receipt.preview.length} /{" "}
                      {countLabel(receipt.messages, "record")}
                    </summary>
                    <ol>
                      {receipt.preview.map((item, index) => (
                        // biome-ignore lint/suspicious/noArrayIndexKey: This read-only preview is immutable and keyed by its upload identity.
                        <li key={`${receipt.upload_id}-${index}`}>
                          <div>
                            <strong>
                              {item.written_by === "unknown" ||
                              (item.role === "memory" && !item.written_by)
                                ? "Author unknown"
                                : item.written_by === "person" ||
                                    item.role === "user"
                                  ? "User"
                                  : "Agent"}
                            </strong>
                            <time>
                              {item.created_at
                                ? new Date(item.created_at).toLocaleString(
                                    "zh-CN",
                                  )
                                : "Time unknown"}
                            </time>
                          </div>
                          <p>
                            {item.text}
                            {item.truncated ? "…" : ""}
                          </p>
                        </li>
                      ))}
                    </ol>
                  </details>
                )}
                {!receipt.committed && !saveUnknown && (
                  <label className="history-import-consent">
                    <input
                      type="checkbox"
                      checked={consent}
                      disabled={busy}
                      onChange={(event) => setConsent(event.target.checked)}
                    />
                    Confirm saving this file to this device; source read
                    authorization is confirmed separately
                  </label>
                )}
                {receipt.committed && candidate && (
                  <>
                    <dl
                      className="history-import-summary"
                      aria-label="Source scope pending authorization"
                    >
                      <div>
                        <dt>Source</dt>
                        <dd>{candidate.display_name}</dd>
                      </div>
                      <div>
                        <dt>Source ID</dt>
                        <dd>{candidate.source_id}</dd>
                      </div>
                      <div>
                        <dt>Actual directory</dt>
                        <dd>{candidate.candidate_path}</dd>
                      </div>
                      <div>
                        <dt>Directory identity</dt>
                        <dd>{candidate.candidate_root_id}</dd>
                      </div>
                      <div>
                        <dt>Location type</dt>
                        <dd>
                          {candidate.candidate_kind === "environment_override"
                            ? "Custom location"
                            : "Default location"}
                        </dd>
                      </div>
                      <div>
                        <dt>Existing authorization</dt>
                        <dd>
                          {candidate.authorized
                            ? "Authorized"
                            : "Not authorized"}
                        </dd>
                      </div>
                    </dl>
                    <p className="history-format-note">
                      The read scope covers all committed history packages in
                      the directory above, including files saved earlier, not
                      just this file. The message counts and source preview
                      above belong to this file only and are not an inventory of
                      the whole directory.
                    </p>
                    <label className="history-import-consent">
                      <input
                        ref={scopeConsent}
                        type="checkbox"
                        checked={consent}
                        disabled={busy}
                        onChange={(event) => setConsent(event.target.checked)}
                      />
                      Allow one-time read of history packages in the source
                      directory above
                    </label>
                  </>
                )}
              </>
            )}
            {error && (
              <div className="error-banner" role="alert">
                <p>{error}</p>
                {issue && (
                  <>
                    <p>
                      {issueLocation(issue.location)}:{" "}
                      {issueLabels[issue.code] ??
                        "This content could not be read"}
                    </p>
                    <pre>{issue.excerpt}</pre>
                  </>
                )}
              </div>
            )}
            <div className="consent-actions" style={{ flexWrap: "wrap" }}>
              <button
                type="button"
                className="secondary"
                disabled={operation === "discard"}
                onClick={() => void close()}
              >
                <X size={16} aria-hidden="true" />
                {operation === "connect" || connection === "unknown"
                  ? "Close"
                  : receipt?.committed
                    ? "Skip authorization"
                    : "Cancel"}
              </button>
              {receipt &&
                selectionValid &&
                (saveUnknown || receipt.committed) && (
                  <button
                    type="button"
                    className="secondary"
                    disabled={busy}
                    onClick={() => void recover()}
                  >
                    <RefreshCw size={16} aria-hidden="true" />
                    {saveUnknown ? "Check file save result" : "Recheck source"}
                  </button>
                )}
              {!receipt?.committed && !saveUnknown && (
                <button
                  type="button"
                  disabled={busy || !receipt || !selectionValid || !consent}
                  onClick={() => void save()}
                >
                  <Save size={16} aria-hidden="true" />
                  Save file and check source
                </button>
              )}
              {receipt?.committed && candidate && selectionValid && (
                <button
                  type="button"
                  disabled={busy || !consent}
                  onClick={() => void confirm()}
                >
                  <Check size={16} aria-hidden="true" />
                  {connection === "failed"
                    ? "Retry this source"
                    : connection === "unknown"
                      ? "Check connection result"
                      : "Authorize and start import"}
                </button>
              )}
            </div>
          </section>
        </div>
      )}
    </>
  );
}

function assertFileReceipt(
  expected: FileImportReceipt,
  actual: FileImportReceipt,
) {
  if (
    !actual ||
    actual.upload_id !== expected.upload_id ||
    actual.source_id !== expected.source_id ||
    actual.checksum !== expected.checksum ||
    actual.vendor !== expected.vendor ||
    actual.filename !== expected.filename ||
    (actual.content_kind ?? "raw_conversation") !==
      (expected.content_kind ?? "raw_conversation") ||
    typeof actual.committed !== "boolean"
  ) {
    throw new Error("file_receipt_mismatch");
  }
}

const issueLabels: Record<string, string> = {
  missing_role: "Missing speaker marker",
  unknown_role: "Cannot identify the speaker",
  empty_message: "This message has no body",
  invalid_time:
    "The time needs an explicit date and timezone; use null when unknown",
  invalid_record: "Message fields should be role, content, time",
  message_too_long: "A single message is too long",
  unclosed_code: "The code block has no closing marker",
};
function issueLocation(value: string) {
  const [kind, number] = value.split(":");
  return kind === "line" ? `Line ${number}` : `Message ${number}`;
}
