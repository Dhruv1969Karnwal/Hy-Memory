import type {
  SourceDirectoryCandidate,
  SourceResponse,
} from "@one-memory/contracts";
import { ArrowLeft, Check, FolderSearch, RefreshCw, X } from "lucide-react";
import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { previewSourceDirectory } from "./api";
import { saveSourceRoot } from "./source-state";
import { traceUiAction } from "./ui-trace";
import { isDefiniteJobRejection } from "./use-import-sources";
import { useMediaPreference } from "./use-media-preference";
import "./source-directory-picker.css";

export interface SourceDirectoryPickerProps {
  source: SourceResponse;
  disabled?: boolean;
  onSaved(): Promise<unknown>;
  onBusyChange?(active: boolean): void;
}

const prioritySources = new Set([
  "workbuddy",
  "cursor",
  "claude-code",
  "codex",
]);
const changedMessage =
  "The source location or candidate version changed. Cancel and choose the directory again. An old confirmation is never applied to a changed source.";
type Attempt = {
  source: SourceResponse;
  key: string;
  epoch: number;
  trigger: HTMLElement;
  path: string;
};

export function SourceDirectoryPicker(props: SourceDirectoryPickerProps) {
  const { source, disabled = false } = props;
  const [picking, setPicking] = useState(false);
  const [attempt, setAttempt] = useState<Attempt | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const mounted = useRef(false);
  const flight = useRef(false);
  const active = useRef(false);
  const generation = useRef(0);
  const current = useRef(props);
  current.current = props;
  const key = sourceKey(source);
  const identity = useRef({ key, epoch: 0 });
  if (identity.current.key !== key)
    identity.current = { key, epoch: identity.current.epoch + 1 };
  const block = blockedReason(source);
  const supported = typeof window.oneMemory?.pickSourceDirectory === "function";
  const id = useId();

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      generation.current += 1;
      if (active.current) current.current.onBusyChange?.(false);
      active.current = false;
    };
  }, []);

  function finish() {
    generation.current += 1;
    active.current = false;
    flight.current = false;
    setPicking(false);
    setAttempt(null);
    current.current.onBusyChange?.(false);
  }

  async function pick(trigger: HTMLElement) {
    const bridge = window.oneMemory;
    if (
      flight.current ||
      active.current ||
      disabled ||
      !bridge?.pickSourceDirectory ||
      document.querySelector('[role="dialog"][aria-modal="true"]')
    )
      return;
    if (block) {
      setNotice(block);
      return;
    }
    const token = ++generation.current;
    const selected: Attempt = {
      source: Object.freeze({ ...source }),
      key,
      epoch: identity.current.epoch,
      trigger,
      path: "",
    };
    flight.current = true;
    active.current = true;
    setPicking(true);
    setError(null);
    setNotice(null);
    props.onBusyChange?.(true);
    try {
      const result = await bridge.pickSourceDirectory(source.source_id);
      if (!mounted.current || generation.current !== token) return;
      if (
        selected.key !== identity.current.key ||
        selected.epoch !== identity.current.epoch ||
        current.current.disabled ||
        window.oneMemory !== bridge
      ) {
        setError(changedMessage);
        finish();
        return;
      }
      if (result?.cancelled === true) {
        finish();
        return;
      }
      if (result?.cancelled !== false || !absolutePath(result.path))
        throw new Error("Directory picking didn't return a valid location.");
      setAttempt({ ...selected, path: result.path });
      flight.current = false;
      setPicking(false);
    } catch (reason) {
      if (mounted.current && generation.current === token) {
        setError(
          reason instanceof Error
            ? `Directory picking didn't finish: ${reason.message}`
            : "Directory picking didn't finish. Try again.",
        );
        finish();
      }
    }
  }

  if (!prioritySources.has(source.source_id)) return null;
  const explanation = !supported
    ? "This desktop version doesn't support directory picking. Update the desktop app."
    : null;
  return (
    <div className="source-directory-picker">
      <button
        type="button"
        className="source-directory-trigger"
        aria-label={`Choose ${source.display_name} history directory`}
        title={`Choose ${source.display_name} history directory`}
        aria-describedby={explanation ? id : undefined}
        disabled={disabled || !!explanation || picking || attempt !== null}
        onClick={(event) => void pick(event.currentTarget)}
      >
        <FolderSearch size={16} aria-hidden="true" />
        {picking ? "Choosing directory..." : "Choose directory"}
      </button>
      {explanation && (
        <p id={id} className="source-directory-meta">
          {explanation}
        </p>
      )}
      {error && (
        <p role="alert" className="source-directory-error">
          {error}
        </p>
      )}
      {notice && (
        <p role="status" className="source-directory-meta">
          {notice}
        </p>
      )}
      {attempt && (
        <CandidateSheet
          attempt={attempt}
          currentSource={source}
          currentEpoch={identity.current.epoch}
          disabled={disabled}
          onRefresh={() => current.current.onSaved()}
          onClose={(saved) => {
            if (saved)
              setNotice(
                "Candidate location saved. You still need to choose the source and confirm history reading.",
              );
            finish();
          }}
        />
      )}
    </div>
  );
}

function CandidateSheet({
  attempt,
  currentSource,
  currentEpoch,
  disabled,
  onRefresh,
  onClose,
}: {
  attempt: Attempt;
  currentSource: SourceResponse;
  currentEpoch: number;
  disabled: boolean;
  onRefresh(): Promise<unknown>;
  onClose(saved: boolean): void;
}) {
  const id = useId();
  const panel = useRef<HTMLElement>(null);
  const cover = useRef<HTMLDivElement>(null);
  const alive = useRef(false);
  const generation = useRef(0);
  const flight = useRef(false);
  const committed = useRef(false);
  const closingRef = useRef(false);
  const invalidated = useRef(false);
  const [preview, setPreview] = useState<SourceDirectoryCandidate | null>(null);
  const [phase, setPhase] = useState<
    | "preview"
    | "review"
    | "saving"
    | "refreshing"
    | "refresh-error"
    | "unknown"
    | "saved-changed"
  >("preview");
  const [confirmed, setConfirmed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stale, setStale] = useState(false);
  const [closing, setClosing] = useState(false);
  const reducedMotion = useMediaPreference("(prefers-reduced-motion: reduce)");
  const changed =
    sourceKey(currentSource) !== attempt.key ||
    currentEpoch !== attempt.epoch ||
    !!blockedReason(currentSource);
  const pending =
    phase === "preview" || phase === "saving" || phase === "refreshing";
  const writePending = phase === "saving" || phase === "refreshing";
  const latest = useRef({
    changed,
    disabled,
    onRefresh,
    onClose,
    reducedMotion,
    phase,
  });
  latest.current = {
    changed,
    disabled,
    onRefresh,
    onClose,
    reducedMotion,
    phase,
  };

  useEffect(() => {
    if (changed && !committed.current) {
      invalidated.current = true;
      setStale(true);
      setConfirmed(false);
    }
  }, [changed]);

  useEffect(() => {
    if (!closing) return;
    if (reducedMotion) {
      latest.current.onClose(committed.current);
      return;
    }
    const timer = window.setTimeout(
      () => latest.current.onClose(committed.current),
      160,
    );
    return () => window.clearTimeout(timer);
  }, [closing, reducedMotion]);

  function close(force = false) {
    if (
      closingRef.current ||
      (!force &&
        (latest.current.phase === "saving" ||
          latest.current.phase === "refreshing"))
    )
      return;
    closingRef.current = true;
    generation.current += 1;
    if (latest.current.reducedMotion) latest.current.onClose(committed.current);
    else setClosing(true);
  }

  async function check() {
    if (
      flight.current ||
      closingRef.current ||
      invalidated.current ||
      latest.current.changed ||
      latest.current.disabled
    )
      return;
    const token = ++generation.current;
    flight.current = true;
    setPhase("preview");
    setPreview(null);
    setConfirmed(false);
    setError(null);
    try {
      const value = await previewSourceDirectory(
        attempt.source.source_id,
        attempt.path,
      );
      if (!alive.current || token !== generation.current) return;
      if (latest.current.changed || invalidated.current) return;
      if (
        !validCandidate(value, attempt.source.source_id) ||
        value.candidate_revision !== (attempt.source.candidate_revision ?? 0)
      )
        throw new Error(
          "The directory check receipt is invalid or the candidate version changed. Choose again.",
        );
      setPreview(Object.freeze({ ...value }));
    } catch (reason) {
      if (alive.current && token === generation.current)
        setError(directoryError(reason));
    } finally {
      if (alive.current && token === generation.current) {
        flight.current = false;
        setPhase("review");
      }
    }
  }

  async function refresh() {
    if (flight.current || closingRef.current) return;
    const token = ++generation.current;
    flight.current = true;
    setPhase("refreshing");
    setError(null);
    try {
      await latest.current.onRefresh();
      if (alive.current && token === generation.current) close(true);
    } catch {
      if (alive.current && token === generation.current) {
        setError(
          committed.current
            ? "Candidate location saved, but the source list failed to refresh. Refreshing won't save or authorize again."
            : "The save result still needs verification. Refresh the source status; the save won't be submitted again.",
        );
        setPhase(committed.current ? "refresh-error" : "unknown");
      }
    } finally {
      flight.current = false;
    }
  }

  async function save() {
    if (
      flight.current ||
      closingRef.current ||
      invalidated.current ||
      latest.current.changed ||
      latest.current.disabled ||
      !confirmed ||
      !preview
    )
      return;
    const token = ++generation.current;
    flight.current = true;
    setPhase("saving");
    setError(null);
    setConfirmed(false);
    // Saving the chosen folder is a real Core write (PUT /sources/{id}/candidate
    // under `/api/v1/sources`, so Core emits kernel-call/kernel-return). Without
    // its own pair those kernel legs belong to no `ui.action`, and the log says
    // the write happened with nothing the user did. Every other non-GET write
    // opens a chain; this one was missed.
    const trace = traceUiAction("sources", "confirm");
    try {
      const value = await saveSourceRoot([
        attempt.source.source_id,
        {
          path: preview.path,
          expected_revision: preview.candidate_revision,
          expected_root_id: preview.root_id,
          user_confirmed: true,
        },
        trace.traceId,
      ]);
      if (!alive.current || token !== generation.current) return;
      const noChange =
        attempt.source.candidate_kind === "user_selected" &&
        attempt.source.candidate_path === preview.path &&
        attempt.source.candidate_root_id === preview.root_id &&
        value?.candidate_revision === preview.candidate_revision;
      if (
        !validCandidate(value, preview.source_id) ||
        value.path !== preview.path ||
        value.root_id !== preview.root_id ||
        (value.candidate_revision !== preview.candidate_revision + 1 &&
          !noChange)
      )
        throw new Error("candidate_save_unconfirmed");
      committed.current = true;
      trace.done("succeeded");
      if (invalidated.current || latest.current.changed) {
        setPhase("saved-changed");
        setError(
          "Candidate location saved, but the source status changed. Cancel and check again; an old receipt is never applied to a new source.",
        );
        return;
      }
      flight.current = false;
      await refresh();
    } catch (reason) {
      // The copy below already says the save is *unconfirmed*, so the log must
      // not claim `failed` unless Core actually refused it.
      const refused = isDefiniteJobRejection(reason);
      trace.done(refused ? "failed" : "unknown", {
        reason: refused
          ? "source_directory_save_failed"
          : "source_directory_save_unconfirmed",
      });
      if (alive.current && token === generation.current) {
        setPhase("unknown");
        setError(
          "The save result is pending verification. Refresh the source status and reopen; the save won't be submitted again.",
        );
      }
    } finally {
      flight.current = false;
    }
  }

  // biome-ignore lint/correctness/useExhaustiveDependencies: Each mounted sheet owns one immutable picker attempt.
  useEffect(() => {
    alive.current = true;
    void check();
    return () => {
      alive.current = false;
      generation.current += 1;
      flight.current = false;
    };
  }, [attempt]);

  useLayoutEffect(() => {
    const element = panel.current,
      overlay = cover.current;
    if (!element || !overlay) return;
    const prior = new Map<
      HTMLElement,
      { inert: boolean; attribute: string | null }
    >();
    const makeInert = () => {
      for (const child of document.body.children) {
        if (
          !(child instanceof HTMLElement) ||
          child === overlay ||
          prior.has(child)
        )
          continue;
        prior.set(child, {
          inert: !!child.inert,
          attribute: child.getAttribute("inert"),
        });
        child.inert = true;
        child.setAttribute("inert", "");
      }
    };
    const focusInside = () =>
      (
        element.querySelector<HTMLElement>(
          "button:not(:disabled), input:not(:disabled)",
        ) ?? element
      ).focus({ preventScroll: true });
    const contain = (event: FocusEvent) => {
      if (event.target instanceof Node && !element.contains(event.target))
        focusInside();
    };
    makeInert();
    const observer = new MutationObserver(makeInert);
    observer.observe(document.body, { childList: true });
    document.addEventListener("focusin", contain);
    focusInside();
    return () => {
      observer.disconnect();
      document.removeEventListener("focusin", contain);
      for (const [child, value] of prior) {
        child.inert = value.inert;
        if (value.attribute === null) child.removeAttribute("inert");
        else child.setAttribute("inert", value.attribute);
      }
      queueMicrotask(() => {
        if (
          panel.current?.isConnected ||
          document.querySelector('[role="dialog"][aria-modal="true"]')
        )
          return;
        const target =
          attempt.trigger.isConnected && !attempt.trigger.matches(":disabled")
            ? attempt.trigger
            : document.querySelector<HTMLElement>("#cold-start-phase");
        if (target && !target.closest("[inert], [hidden]"))
          target.focus({ preventScroll: true });
      });
    };
  }, [attempt]);

  useLayoutEffect(() => {
    if (writePending || closing) panel.current?.focus();
  }, [writePending, closing]);

  const invalid = !committed.current && (stale || changed);
  const refreshOnly =
    phase === "unknown" ||
    phase === "refresh-error" ||
    phase === "refreshing" ||
    phase === "saved-changed";
  return createPortal(
    <div
      ref={cover}
      className="source-directory-cover"
      data-motion={reducedMotion ? "reduced" : "full"}
    >
      <section
        ref={panel}
        className="source-directory-sheet"
        role="dialog"
        aria-modal="true"
        aria-busy={pending}
        aria-labelledby={`${id}-title`}
        aria-describedby={`${id}-description`}
        data-closing={closing}
        tabIndex={-1}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            event.preventDefault();
            event.stopPropagation();
            close();
          }
          if (event.key !== "Tab") return;
          const controls = panel.current?.querySelectorAll<HTMLElement>(
            "button:not(:disabled), input:not(:disabled)",
          );
          const first = controls?.[0],
            last = controls?.[controls.length - 1];
          if (!first || writePending || closing) {
            event.preventDefault();
            panel.current?.focus();
          } else if (
            event.shiftKey &&
            (document.activeElement === first ||
              document.activeElement === panel.current)
          ) {
            event.preventDefault();
            last?.focus();
          } else if (
            !event.shiftKey &&
            (document.activeElement === last ||
              document.activeElement === panel.current)
          ) {
            event.preventDefault();
            first.focus();
          }
          event.stopPropagation();
        }}
      >
        <header className="source-directory-heading">
          <h2 id={`${id}-title`}>
            Save {attempt.source.display_name}'s candidate location
          </h2>
          <button
            type="button"
            className="source-directory-icon"
            aria-label="Close directory confirmation"
            title="Close directory confirmation"
            disabled={writePending || closing}
            onClick={() => close()}
          >
            <X size={18} aria-hidden="true" />
          </button>
        </header>
        <div className="source-directory-reading">
          <p id={`${id}-description`}>
            This only saves the candidate directory for this source. It doesn't
            read history, turn on continuous sync, process model content, or
            install a host connection. After saving, you still choose the source
            and confirm history reading separately.
          </p>
          <strong>{attempt.source.display_name}</strong>
          <code>{preview?.path ?? attempt.path}</code>
          {phase === "preview" && (
            <p role="status">Checking the directory location...</p>
          )}
          {writePending && (
            <p role="status">
              {phase === "saving"
                ? "Saving the candidate location..."
                : "Refreshing the source status..."}
            </p>
          )}
          {preview && !refreshOnly && (
            <label className="source-directory-consent">
              <input
                type="checkbox"
                checked={confirmed && !invalid}
                disabled={pending || closing || invalid || disabled}
                onChange={(event) => setConfirmed(event.target.checked)}
              />
              <span>
                Only this candidate location is saved, with no history read
                authorization
              </span>
            </label>
          )}
          {(invalid || error) && (
            <p role="alert" className="source-directory-error">
              {invalid ? changedMessage : error}
            </p>
          )}
        </div>
        <footer className="source-directory-actions">
          <button
            type="button"
            className="source-directory-secondary"
            disabled={writePending || closing}
            onClick={() => close()}
          >
            <ArrowLeft size={16} aria-hidden="true" />
            Cancel
          </button>
          {phase === "saved-changed" ? null : refreshOnly ? (
            <button
              type="button"
              disabled={pending || closing}
              onClick={() => void refresh()}
            >
              <RefreshCw size={16} aria-hidden="true" />
              Refresh source status
            </button>
          ) : preview ? (
            <button
              type="button"
              disabled={!confirmed || pending || closing || invalid || disabled}
              onClick={() => void save()}
            >
              <Check size={16} aria-hidden="true" />
              Save candidate location
            </button>
          ) : (
            phase !== "preview" && (
              <button
                type="button"
                disabled={closing || invalid || disabled}
                onClick={() => void check()}
              >
                <RefreshCw size={16} aria-hidden="true" />
                Check the directory again
              </button>
            )
          )}
        </footer>
      </section>
    </div>,
    document.body,
  );
}

function blockedReason(source: SourceResponse): string | null {
  if (
    source.candidate_kind === "environment_override" ||
    source.candidate_change_reason?.includes("environment")
  )
    return "This location comes from an environment variable, so it can't be changed in the app.";
  if (
    source.authorized ||
    (source.grant_revision ?? 0) > 0 ||
    source.imported_messages > 0 ||
    source.candidate_change_allowed === false
  )
    return "This source already has history, authorization or jobs, so changing its directory isn't supported.";
  if (source.candidate_change_allowed !== true)
    return "The current Core hasn't confirmed whether the directory can be changed. Refresh or update, then try again.";
  if (source.capabilities.historical_import !== "implemented")
    return "This source doesn't support history directory import yet.";
  return null;
}

function sourceKey(source: SourceResponse) {
  return JSON.stringify([
    source.source_id,
    source.display_name,
    source.candidate_path,
    source.candidate_root_id ?? null,
    source.candidate_kind,
    source.candidate_revision ?? 0,
    source.candidate_change_allowed ?? null,
    source.candidate_change_reason ?? null,
    source.authorized,
    source.grant_revision ?? 0,
    source.imported_messages > 0,
    source.capabilities.historical_import,
  ]);
}

function absolutePath(value: unknown): value is string {
  return (
    typeof value === "string" &&
    value.length > 0 &&
    !Array.from(value).some((character) => character.charCodeAt(0) < 32) &&
    (/^\//.test(value) ||
      /^[A-Za-z]:[\\/]/.test(value) ||
      /^\\\\[^\\]+\\/.test(value))
  );
}

function validCandidate(
  value: SourceDirectoryCandidate | undefined,
  sourceId: string,
): value is SourceDirectoryCandidate {
  return (
    !!value &&
    value.source_id === sourceId &&
    absolutePath(value.path) &&
    typeof value.root_id === "string" &&
    value.root_id.trim().length > 0 &&
    Number.isSafeInteger(value.candidate_revision) &&
    value.candidate_revision >= 0
  );
}

function directoryError(reason: unknown) {
  return reason instanceof Error
    ? `Directory check failed: ${reason.message}`
    : "Directory check failed. Try again or choose again.";
}
