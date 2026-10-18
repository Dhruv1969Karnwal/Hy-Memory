import type { SourceResponse } from "@one-memory/contracts";
import { RefreshCw, ShieldCheck, ShieldOff, X } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type {
  RecallHostStatus,
  RecallPermissionReceipt,
  RecallPermissionStatus,
  RecallPermissionTool,
  SetRecallPermissionInput,
} from "./global";
import { hostPermissionBridge } from "./host-state";
import {
  eligibleSource,
  fullGrantSelection,
  grantableSources,
  selectableTools,
  selectionIsValid,
} from "./recall-grant";
import "./recall-permission-control.css";

export interface RecallPermissionControlProps {
  host: RecallHostStatus;
  sources: SourceResponse[];
  onChanged(): void;
  disabled?: boolean;
  /** Existing Core-issued scope IDs only. This control never derives a scope. */
  workspaceScopes?: readonly { scope: string; label: string }[];
  compact?: boolean;
}

type WorkspaceScope = { scope: string; label: string };
type Snapshot = {
  hostId: string;
  label: string;
  installed: boolean;
  permission: RecallPermissionStatus & { receipt: RecallPermissionReceipt };
  sources: SourceResponse[];
  workspaces: WorkspaceScope[];
  action: "grant" | "revoke";
  returnFocus: HTMLElement;
  fallbackFocus: HTMLElement | null;
};
type Pending = "check" | "write" | "refresh";
const changedMessage =
  "The installation identity, grant revision or selected source scope has changed. Close and reopen to review. This confirmation is not applied to a changed grant.";
const toolNames: Record<RecallPermissionTool, string> = {
  om_recall: "Retrieve memories",
  om_ask: "Memory Q&A (optional)",
  om_conversations: "Read raw conversations",
  om_matters: "Matters (already authorized)",
  om_diagnostic_trace: "Read redacted diagnostic records",
};
const focusSelector =
  'button:not(:disabled), input:not(:disabled), [tabindex="0"]';

export function RecallPermissionControl(props: RecallPermissionControlProps) {
  const {
    host,
    sources,
    disabled = false,
    workspaceScopes = [],
    compact = false,
  } = props;
  const [local, setLocal] = useState<{
    baseline: string;
    status: RecallPermissionStatus;
  } | null>(null);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [pending, setPending] = useState(false);
  const [granting, setGranting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const generation = useRef(0);
  const flight = useRef(false);
  const mounted = useRef(false);
  const control = useRef<HTMLElement>(null);
  const currentHost = useRef(host.hostId);
  currentHost.current = host.hostId;
  const baseline = JSON.stringify(freezeStatus(host.permission) ?? null);
  const localIsNewer =
    local?.status.receipt &&
    host.permission?.receipt &&
    local.status.hostId === host.hostId &&
    local.status.receipt.installationId ===
      host.permission.receipt.installationId &&
    local.status.receipt.revision > host.permission.receipt.revision &&
    host.permission.reason === null &&
    ["approved", "revoked", "install-only"].includes(host.permission.state);
  const status = freezeStatus(
    local &&
      local.status.hostId === host.hostId &&
      (local.baseline === baseline || localIsNewer)
      ? local.status
      : host.permission,
  );
  const bridge = permissionBridge();
  const receipt = validReceipt(status, host.hostId);
  const canGrant = !!receipt && canApprove(status) && host.installed;
  const canRevoke =
    !!receipt &&
    receipt.status === "approved" &&
    ["approved", "relink-required"].includes(status?.state ?? "");

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      generation.current += 1;
    };
  }, []);

  function receive(value: RecallPermissionStatus) {
    if (value.hostId === currentHost.current)
      setLocal({ baseline, status: freezeStatus(value) });
  }

  async function refresh() {
    const api = permissionBridge();
    if (!api || disabled || flight.current || snapshot) return;
    const hostId = host.hostId;
    const token = ++generation.current;
    flight.current = true;
    setPending(true);
    setError(null);
    try {
      const value = await api.read(hostId);
      if (
        !mounted.current ||
        token !== generation.current ||
        currentHost.current !== hostId
      )
        return;
      if (value?.hostId !== hostId)
        throw new Error("host_permission_invalid_receipt");
      receive(value);
    } catch {
      if (
        mounted.current &&
        token === generation.current &&
        currentHost.current === hostId
      )
        setError(
          "The call authorization state could not be read. Refresh and check later. Permissions were not changed.",
        );
    } finally {
      if (mounted.current && token === generation.current) {
        flight.current = false;
        setPending(false);
      }
    }
  }

  /** The frozen picture a write is checked against, taken the moment the user
   * commits to one. Built for both routes into a grant: the sheet, and the
   * one-click allow that skips it, so neither can drift from the other.
   */
  function pictureOf(
    action: Snapshot["action"],
    returnFocus: HTMLElement,
    fallbackFocus: HTMLElement | null,
  ): Snapshot | null {
    if (!status || !receipt) return null;
    return {
      hostId: host.hostId,
      label: host.label,
      installed: host.installed,
      permission: { ...status, receipt },
      sources: grantableSources(sources),
      workspaces: knownWorkspaces(workspaceScopes),
      action,
      returnFocus,
      fallbackFocus,
    };
  }

  function open(action: Snapshot["action"], trigger: HTMLElement) {
    if (
      !bridge ||
      !status ||
      !receipt ||
      disabled ||
      flight.current ||
      snapshot ||
      (action === "grant" ? !canGrant : !canRevoke) ||
      document.querySelector('[role="dialog"][aria-modal="true"]')
    )
      return;
    const picture = pictureOf(action, trigger, control.current);
    if (!picture) return;
    setError(null);
    setSnapshot(picture);
  }

  /** Allow everything, without making the user open the scope sheet first.
   *
   * The sheet opens with every box already ticked, so its only remaining
   * content was a second confirmation of a choice nobody had made. The row
   * button now commits the same selection the sheet would have submitted,
   * through the same guard; the sheet stays reachable for anyone who wants to
   * narrow the scope. Nothing is granted before the installation identity and
   * revision are re-read and matched, which is what stops an approval from
   * being spent on a connection that changed underneath it.
   */
  async function grantAll(trigger: HTMLElement) {
    const api = bridge;
    if (
      !api ||
      !status ||
      !receipt ||
      disabled ||
      flight.current ||
      snapshot ||
      granting ||
      !canGrant ||
      receipt.status === "approved"
    )
      return;
    const picture = pictureOf("grant", trigger, control.current);
    if (!picture) return;
    setGranting(true);
    setNotice(null);
    setError(null);
    const hostId = host.hostId;
    try {
      const selection = fullGrantSelection({
        sources,
        workspaces: picture.workspaces,
        keep: receipt.allowedTools,
      });
      if (!selectionIsValid(selection))
        throw new Error("host_permission_invalid_selection");
      const review: SetRecallPermissionInput = Object.freeze({
        installationId: receipt.installationId,
        expectedRevision: receipt.revision,
        allowedTools: Object.freeze([...selection.allowedTools]),
        allowedScopes: Object.freeze([...selection.allowedScopes].sort()),
        userConfirmed: true,
      });
      const latest = await api.read(hostId);
      if (!mounted.current || currentHost.current !== hostId) return;
      if (!sameStatus(picture.permission, latest)) {
        receive(latest);
        setError(changedMessage);
        return;
      }
      const saved = await api.grant(hostId, review);
      if (!mounted.current || currentHost.current !== hostId) return;
      if (!validMutationResult(saved, picture, review))
        throw new Error("host_permission_invalid_receipt");
      receive(saved);
      setNotice(`Allowed ${host.label} to use memory.`);
      // The write already landed: a failed list refresh is reported as such,
      // never as a failed grant, so nobody re-submits an approval that stuck.
      try {
        await props.onChanged();
      } catch {
        if (mounted.current && currentHost.current === hostId)
          setError(refreshAfterCommitMessage("grant"));
      }
    } catch (reason) {
      if (!mounted.current || currentHost.current !== hostId) return;
      // No automatic re-read here: a refresh clears the inline error, and the
      // row has its own refresh control for finding out what actually landed.
      setError(
        isStaleError(reason)
          ? changedMessage
          : "This grant could not be confirmed. Refresh, check and try again; do not resubmit.",
      );
    } finally {
      if (mounted.current) setGranting(false);
    }
  }

  return (
    <section
      className={`recall-permission-control${compact ? " recall-permission-compact" : ""}`}
      aria-label={`${host.label} call permissions`}
      ref={control}
      tabIndex={-1}
    >
      <p className="recall-permission-status" role="status" hidden={compact}>
        {statusLabel(status, host.hostId)}
      </p>
      {!bridge && (
        <p className="recall-permission-meta">
          The current desktop connection does not support full permission
          management; it cannot grant or revoke here.
        </p>
      )}
      {status?.reason === "installation_incomplete" && (
        <p className="recall-permission-meta">
          The connection install is not complete, so memory use cannot be
          authorized.
        </p>
      )}
      {compact ? (
        <label className="recall-permission-switch">
          <span>Allow memory use</span>
          <input
            type="checkbox"
            aria-label="Allow memory use"
            checked={receipt?.status === "approved"}
            disabled={
              disabled ||
              !bridge ||
              pending ||
              granting ||
              !!snapshot ||
              (!canGrant && !canRevoke)
            }
            onChange={(event) => {
              if (event.target.checked) void grantAll(event.currentTarget);
              else open("revoke", event.currentTarget);
            }}
          />
        </label>
      ) : (
        <div className="recall-permission-entry-actions">
          {receipt?.status === "approved" ? (
            <button
              type="button"
              className="secondary"
              disabled={
                disabled || !bridge || !canGrant || pending || !!snapshot
              }
              onClick={(event) => open("grant", event.currentTarget)}
            >
              <ShieldCheck size={16} aria-hidden="true" />
              Adjust call permissions
            </button>
          ) : (
            <>
              {/* The row's allow is the whole flow: the sheet's default is every
                tool and every eligible source, so showing it only to have the
                same choice confirmed twice was two clicks for one decision. */}
              <button
                type="button"
                className="secondary"
                disabled={
                  disabled ||
                  !bridge ||
                  !canGrant ||
                  pending ||
                  granting ||
                  !!snapshot
                }
                onClick={(event) => void grantAll(event.currentTarget)}
              >
                <ShieldCheck size={16} aria-hidden="true" />
                {granting ? "Allowing..." : "Allow memory use"}
              </button>
              <button
                type="button"
                className="quiet"
                disabled={
                  disabled ||
                  !bridge ||
                  !canGrant ||
                  pending ||
                  granting ||
                  !!snapshot
                }
                onClick={(event) => open("grant", event.currentTarget)}
              >
                Adjust scope
              </button>
            </>
          )}
          {canRevoke && (
            <button
              type="button"
              className="quiet"
              disabled={disabled || !bridge || pending || !!snapshot}
              onClick={(event) => open("revoke", event.currentTarget)}
            >
              <ShieldOff size={16} aria-hidden="true" />
              Revoke call permissions
            </button>
          )}
          <button
            type="button"
            className="quiet recall-permission-icon"
            aria-label="Refresh call permissions"
            title="Refresh call permissions"
            disabled={disabled || !bridge || pending || granting || !!snapshot}
            onClick={() => void refresh()}
          >
            <RefreshCw size={16} aria-hidden="true" />
          </button>
        </div>
      )}
      {pending && <p role="status">Reading the call authorization state...</p>}
      {granting && <p role="status">Allowing {host.label} to use memory...</p>}
      {notice && (
        <p className="recall-permission-meta" role="status">
          {notice}
        </p>
      )}
      {error && (
        <p className="recall-permission-error" role="alert">
          {error}
        </p>
      )}
      {snapshot && (
        <PermissionDialog
          snapshot={snapshot}
          currentHost={host}
          currentStatus={status}
          sources={sources}
          workspaces={knownWorkspaces(workspaceScopes)}
          disabled={disabled}
          onClose={() => setSnapshot(null)}
          onStatus={receive}
          onChanged={props.onChanged}
        />
      )}
    </section>
  );
}

function PermissionDialog({
  snapshot,
  currentHost,
  currentStatus,
  sources,
  workspaces,
  disabled,
  onClose,
  onStatus,
  onChanged,
}: {
  snapshot: Snapshot;
  currentHost: RecallHostStatus;
  currentStatus: RecallPermissionStatus | undefined;
  sources: SourceResponse[];
  workspaces: WorkspaceScope[];
  disabled: boolean;
  onClose(): void;
  onStatus(value: RecallPermissionStatus): void;
  onChanged(): void;
}) {
  const id = useId();
  const panel = useRef<HTMLElement>(null);
  const receipt = snapshot.permission.receipt;
  const editing = receipt.status === "approved";
  const defaults = fullGrantSelection({
    sources: snapshot.sources,
    workspaces: snapshot.workspaces,
    keep: receipt.allowedTools,
  });
  const [tools, setTools] = useState<readonly RecallPermissionTool[]>(
    editing ? receipt.allowedTools : defaults.allowedTools,
  );
  const [scopes, setScopes] = useState<readonly string[]>(
    editing ? receipt.allowedScopes : defaults.allowedScopes,
  );
  const [phase, setPhase] = useState<"select" | "revoke" | "receipt">(
    snapshot.action === "revoke" ? "revoke" : "select",
  );
  // The input of the write in flight, set in the same tick as the write. State
  // would only be readable by the next render, and the write's own guards need
  // it right away: reading a stale value there would report the very receipt
  // the write produced as a change that invalidates it.
  const submitted = useRef<SetRecallPermissionInput | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [pending, setPending] = useState<Pending | null>(null);
  const [stale, setStale] = useState(false);
  const [needsRefresh, setNeedsRefresh] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RecallPermissionStatus | null>(null);
  const committed = useRef(false);
  const mounted = useRef(false);
  const generation = useRef(0);
  const operation = useRef<Pending | null>(null);
  const invalidated = useRef(false);
  const current = useRef({
    disabled,
    currentHost,
    currentStatus,
    sources,
    workspaces,
  });
  current.current = {
    disabled,
    currentHost,
    currentStatus,
    sources,
    workspaces,
  };
  const unverified = receipt.allowedScopes.filter(
    (scope) => !knownScope(scope, snapshot),
  );
  const selectionValid =
    selectionIsValid({
      allowedTools: [...tools],
      allowedScopes: [...scopes],
    }) && scopes.every((scope) => knownScope(scope, snapshot));
  // While a write is in flight the parent may report back the very receipt
  // this sheet is producing; that is the expected outcome, not a change.
  const changed =
    !committed.current && !matchesCurrent(operation.current !== null);
  const blocked =
    stale || changed || needsRefresh || disabled || !permissionBridge();

  function matchesCurrent(allowCommitted = false) {
    const value = current.current;
    return (
      value.currentHost.hostId === snapshot.hostId &&
      value.currentHost.installed === snapshot.installed &&
      (sameStatus(snapshot.permission, value.currentStatus) ||
        (allowCommitted &&
          !!value.currentStatus &&
          validMutationResult(
            value.currentStatus,
            snapshot,
            submitted.current,
          ))) &&
      (snapshot.action === "revoke" ||
        scopes.every((scope) => {
          if (scope.startsWith("source:")) {
            const saved = snapshot.sources.find(
              (source) => `source:${source.source_id}` === scope,
            );
            if (!saved) return true;
            return (
              value.sources.filter(
                (source) => source.source_id === saved.source_id,
              ).length === 1 &&
              sameSource(
                saved,
                value.sources.find(
                  (source) => source.source_id === saved.source_id,
                ),
              )
            );
          }
          if (scope.startsWith("workspace:v1:")) {
            const saved = snapshot.workspaces.find(
              (option) => option.scope === scope,
            );
            return (
              !saved ||
              value.workspaces.some(
                (option) =>
                  option.scope === scope && option.label === saved.label,
              )
            );
          }
          return true;
        }))
    );
  }

  useEffect(() => {
    if (changed) {
      invalidated.current = true;
      setStale(true);
      setConfirmed(false);
    }
  }, [changed]);

  useEffect(() => {
    mounted.current = true;
    const before = snapshot.returnFocus;
    const prior = new Map<HTMLElement, boolean>();
    const root = panel.current?.parentElement;
    const makeInert = () => {
      for (const child of document.body.children) {
        if (
          !(child instanceof HTMLElement) ||
          child === root ||
          prior.has(child)
        )
          continue;
        prior.set(child, child.inert);
        child.inert = true;
      }
    };
    makeInert();
    const observer = new MutationObserver(makeInert);
    observer.observe(document.body, { childList: true });
    const focusInside = () =>
      (
        panel.current?.querySelector<HTMLElement>(focusSelector) ??
        panel.current
      )?.focus();
    const containFocus = (event: FocusEvent) => {
      if (
        event.target instanceof Node &&
        !panel.current?.contains(event.target)
      )
        focusInside();
    };
    document.addEventListener("focusin", containFocus);
    focusInside();
    return () => {
      mounted.current = false;
      generation.current += 1;
      observer.disconnect();
      document.removeEventListener("focusin", containFocus);
      for (const [element, inert] of prior) element.inert = inert;
      if (before.isConnected && !before.matches(":disabled")) before.focus();
      else snapshot.fallbackFocus?.focus();
    };
  }, [snapshot]);

  // A completed action or phase switch can remove or disable the focused control.
  // biome-ignore lint/correctness/useExhaustiveDependencies: Focus follows phase changes as well as pending state.
  useEffect(() => {
    if (pending) panel.current?.focus();
    else if (
      !panel.current?.contains(document.activeElement) ||
      document.activeElement?.matches(":disabled")
    )
      panel.current?.querySelector<HTMLElement>(focusSelector)?.focus();
  }, [pending, phase]);

  const active = (token: number) =>
    mounted.current && generation.current === token;
  function begin(kind: Pending) {
    if (operation.current || !mounted.current) return null;
    operation.current = kind;
    setPending(kind);
    setError(null);
    return ++generation.current;
  }
  function finish(token: number) {
    if (!active(token)) return;
    operation.current = null;
    setPending(null);
  }
  function invalidate(value?: RecallPermissionStatus) {
    invalidated.current = true;
    setStale(true);
    setConfirmed(false);
    setError(changedMessage);
    if (value?.hostId === snapshot.hostId) onStatus(value);
  }
  function close() {
    if (operation.current === "write") return;
    generation.current += 1;
    onClose();
  }
  function toggleTool(tool: RecallPermissionTool) {
    if (pending || blocked) return;
    setConfirmed(false);
    setTools((values) => toggle(values, tool));
  }
  function toggleScope(scope: string) {
    if (pending || blocked) return;
    setConfirmed(false);
    setScopes((values) => toggle(values, scope));
  }
  /** Commit the sheet as it stands.
   *
   * There used to be a second screen here: a summary plus a "I confirm the
   * tools and scope above" checkbox, reached by a "review scope" button. The
   * scope is already the box the user is looking at, so the review screen only
   * restated the same selection — reported from real use as an unnecessary
   * step under a sheet that should have been self-evident. The button is now
   * the confirmation, and the desktop still refuses the write unless the
   * connection identity and revision are what they were when the sheet opened.
   */
  function approve() {
    if (
      operation.current ||
      blocked ||
      !selectionValid ||
      invalidated.current ||
      !matchesCurrent()
    )
      return;
    const input: SetRecallPermissionInput = Object.freeze({
      installationId: receipt.installationId,
      expectedRevision: receipt.revision,
      allowedTools: Object.freeze([...tools]),
      allowedScopes: Object.freeze([...scopes].sort()),
      userConfirmed: true,
    });
    submitted.current = input;
    void commit(input);
  }

  async function refresh() {
    const api = permissionBridge();
    if (!api || disabled) return;
    const token = begin("refresh");
    if (token === null) return;
    setConfirmed(false);
    try {
      const value = await api.read(snapshot.hostId);
      if (!active(token)) return;
      const expected = result ?? snapshot.permission;
      if (
        !sameStatus(expected, value) ||
        (!committed.current && (invalidated.current || !matchesCurrent()))
      ) {
        invalidate(value);
        return;
      }
      setNeedsRefresh(false);
      if (committed.current) {
        onStatus(value);
        try {
          await onChanged();
        } catch {
          if (active(token))
            setError(refreshAfterCommitMessage(snapshot.action));
        }
      }
    } catch {
      if (active(token)) {
        setNeedsRefresh(true);
        setError(
          committed.current
            ? refreshAfterCommitMessage(snapshot.action)
            : "The authorization state could not be refreshed. Try again later. No permission change was resubmitted.",
        );
      }
    } finally {
      finish(token);
    }
  }

  /** Write the grant or the revocation the sheet is showing.
   *
   * `grant` is the exact selection the user just approved; `revoke` carries no
   * selection and still needs its own checkbox, because withdrawing a
   * permission is not something a single click should be able to do by
   * accident.
   */
  async function commit(grant: SetRecallPermissionInput | null) {
    const api = permissionBridge();
    if (
      !api ||
      blocked ||
      invalidated.current ||
      committed.current ||
      (snapshot.action === "revoke" ? !confirmed : !grant)
    )
      return;
    const token = begin("check");
    if (token === null) return;
    try {
      const latest = await api.read(snapshot.hostId);
      if (!active(token)) return;
      if (
        invalidated.current ||
        !matchesCurrent() ||
        !sameStatus(snapshot.permission, latest)
      ) {
        invalidate(latest);
        return;
      }
      if (current.current.disabled || permissionBridge() !== api) return;
      if (snapshot.action === "grant" && !grant) return;
      operation.current = "write";
      setPending("write");
      let saved: RecallPermissionStatus;
      if (snapshot.action === "revoke") {
        saved = await api.revoke(snapshot.hostId, {
          installationId: receipt.installationId,
          expectedRevision: receipt.revision,
          userConfirmed: true,
        });
      } else {
        if (!grant) return;
        saved = await api.grant(snapshot.hostId, grant);
      }
      if (!active(token)) return;
      if (!validMutationResult(saved, snapshot, grant))
        throw new Error("host_permission_invalid_receipt");
      const changedDuringCommit = invalidated.current || !matchesCurrent(true);
      committed.current = true;
      setResult(freezeStatus(saved));
      setPhase("receipt");
      setConfirmed(false);
      setNeedsRefresh(false);
      setStale(changedDuringCommit);
      onStatus(saved);
      operation.current = "refresh";
      setPending("refresh");
      try {
        await onChanged();
      } catch {
        if (active(token)) setError(refreshAfterCommitMessage(snapshot.action));
      }
    } catch (reason) {
      if (!active(token)) return;
      setConfirmed(false);
      setNeedsRefresh(true);
      if (isStaleError(reason)) invalidate();
      else
        setError(
          snapshot.action === "revoke"
            ? "The revocation result is not confirmed yet. Refresh, check and confirm again; do not resubmit."
            : "The save result is not confirmed yet. Refresh, check and confirm again; do not resubmit.",
        );
    } finally {
      finish(token);
    }
  }

  const removed = [
    ...receipt.allowedTools.filter((tool) => !tools.includes(tool)),
    ...receipt.allowedScopes.filter((scope) => !scopes.includes(scope)),
  ];
  const visibleError = stale || changed ? changedMessage : error;
  const offeredTools = selectableTools(receipt.allowedTools);

  return createPortal(
    <div className="consent-backdrop recall-permission-backdrop">
      <section
        className="consent-sheet recall-permission-sheet"
        role="dialog"
        aria-modal="true"
        aria-busy={pending !== null}
        aria-labelledby={`${id}-title`}
        ref={panel}
        tabIndex={-1}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            event.preventDefault();
            event.stopPropagation();
            close();
          }
          if (event.key !== "Tab") return;
          const controls =
            panel.current?.querySelectorAll<HTMLElement>(focusSelector);
          const first = controls?.[0],
            last = controls?.[controls.length - 1];
          if (!first) {
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
        }}
      >
        <header className="recall-permission-heading">
          <div>
            <p className="eyebrow">Separate call authorization</p>
            <h2 id={`${id}-title`}>
              {snapshot.label} ·{" "}
              {snapshot.action === "revoke"
                ? "Revoke call permissions"
                : "Use memory"}
            </h2>
          </div>
          <button
            type="button"
            className="quiet recall-permission-icon"
            aria-label="Close"
            title="Close"
            disabled={pending === "write"}
            onClick={close}
          >
            <X size={18} aria-hidden="true" />
          </button>
        </header>
        <div className="recall-permission-body">
          <p className="recall-permission-meta">
            Installation: <code>{receipt.installationId}</code> - grant revision{" "}
            {result?.receipt?.revision ?? receipt.revision}
          </p>
          {phase === "select" && (
            <>
              <p>
                The connection install and memory use are authorized separately.
                The selected permissions apply only to this install of this app.
              </p>
              <fieldset
                className="recall-permission-options"
                disabled={!!pending || blocked}
              >
                <legend>Allowed tools</legend>
                {offeredTools.map((tool) => (
                  <label className="recall-permission-choice" key={tool}>
                    <input
                      type="checkbox"
                      aria-label={toolNames[tool]}
                      checked={tools.includes(tool)}
                      onChange={() => toggleTool(tool)}
                    />
                    <span>
                      {toolNames[tool]}
                      {tool === "om_ask" && (
                        <small>
                          May use the configured model for processing. This
                          grants only this app's memory read scope.
                        </small>
                      )}
                    </span>
                  </label>
                ))}
              </fieldset>
              <fieldset
                className="recall-permission-options"
                disabled={!!pending || blocked}
              >
                <legend>Allowed memory scopes</legend>
                <label className="recall-permission-choice">
                  <input
                    type="checkbox"
                    checked={scopes.includes("all-apps")}
                    onChange={() => toggleScope("all-apps")}
                  />
                  <span>Cross-app shared memory (all-apps)</span>
                </label>
                <p className="recall-permission-meta">
                  Only memories already allowed to be shared; this is not all
                  raw conversations and does not widen source read
                  authorization.
                </p>
                {snapshot.workspaces.map((option) => (
                  <label
                    className="recall-permission-choice"
                    key={option.scope}
                  >
                    <input
                      type="checkbox"
                      aria-label={`Workspace: ${option.label}`}
                      checked={scopes.includes(option.scope)}
                      onChange={() => toggleScope(option.scope)}
                    />
                    <span>
                      Workspace: {option.label}
                      <small>{option.scope}</small>
                    </span>
                  </label>
                ))}
              </fieldset>
              <fieldset
                className="recall-permission-options"
                disabled={!!pending || blocked}
              >
                <legend>Raw conversation sources</legend>
                <p className="recall-permission-meta">
                  Reading raw conversations requires the raw conversation tool,
                  all-apps and an explicitly selected authorized source. Other
                  sources are never included automatically.
                </p>
                {!snapshot.sources.length && (
                  <p>
                    No authorized source is available to check, so no raw
                    conversation read permission can be added.
                  </p>
                )}
                {snapshot.sources.map((source) => (
                  <label
                    className="recall-permission-choice"
                    key={source.source_id}
                  >
                    <input
                      type="checkbox"
                      aria-label={`${source.display_name} raw conversations`}
                      checked={scopes.includes(`source:${source.source_id}`)}
                      onChange={() => toggleScope(`source:${source.source_id}`)}
                    />
                    <span>
                      {source.display_name}
                      <small>
                        source:{source.source_id} - source grant revision{" "}
                        {source.grant_revision}
                      </small>
                    </span>
                  </label>
                ))}
              </fieldset>
              {unverified.length > 0 && (
                <fieldset
                  className="recall-permission-options"
                  disabled={!!pending || blocked}
                >
                  <legend>Existing scopes that could not be verified</legend>
                  <p>
                    The scopes below are never dropped automatically. To
                    continue updating you must explicitly remove them, or cancel
                    and reopen once the source and workspace information is
                    complete.
                  </p>
                  {unverified.map((scope) => (
                    <label className="recall-permission-choice" key={scope}>
                      <input
                        type="checkbox"
                        checked={!scopes.includes(scope)}
                        aria-label={`Remove unverified scope: ${scope}`}
                        onChange={() => toggleScope(scope)}
                      />
                      <span>
                        Remove this scope<small>{scope}</small>
                      </span>
                    </label>
                  ))}
                </fieldset>
              )}
            </>
          )}
          {editing && phase === "select" && (
            <section
              className="recall-permission-summary"
              aria-label="Current authorization"
            >
              <h3>Current authorization</h3>
              <PermissionSummary
                tools={receipt.allowedTools}
                scopes={receipt.allowedScopes}
                snapshot={snapshot}
              />
              <p>
                This confirmation replaces all current tools and scopes and is
                not merged with the old grant. Unselected permissions are
                removed.
              </p>
            </section>
          )}
          {/* The scope is on this screen rather than on a second one: the sheet
              already opens with everything ticked, so restating the same list
              behind another button was a step with no decision in it. */}
          {phase === "select" && (
            <section
              className="recall-permission-summary"
              aria-label="This authorization"
            >
              <h3>
                {editing
                  ? "This grant"
                  : "This grant (all selected by default)"}
              </h3>
              <PermissionSummary
                tools={tools}
                scopes={scopes}
                snapshot={snapshot}
              />
              {removed.length > 0 && (
                <p>This grant removes: {removed.join(", ")}</p>
              )}
            </section>
          )}
          {phase === "revoke" && (
            <>
              <p>
                This ends the call authorization for this install and blocks
                further memory reads. It does not delete sources or uninstall
                the connection, and it does not disable other apps'
                authorization.
              </p>
              <p>It cannot take back what this app has already read.</p>
              <PermissionSummary
                tools={receipt.allowedTools}
                scopes={receipt.allowedScopes}
                snapshot={snapshot}
              />
            </>
          )}
          {phase === "revoke" && (
            <label className="recall-permission-choice recall-permission-final">
              <input
                type="checkbox"
                checked={confirmed && !blocked}
                disabled={!!pending || blocked}
                onChange={(event) => setConfirmed(event.target.checked)}
              />
              <span>
                I confirm revoking the call authorization for this install
              </span>
            </label>
          )}
          {phase === "receipt" && result?.receipt && (
            <>
              <p role="status">
                {snapshot.action === "revoke"
                  ? "The call authorization for this install has been revoked."
                  : "The call authorization was saved."}
              </p>
              <p className="recall-permission-meta">
                Grant revision {result.receipt.revision}
              </p>
              <PermissionSummary
                tools={result.receipt.allowedTools}
                scopes={result.receipt.allowedScopes}
                snapshot={snapshot}
              />
            </>
          )}
          {snapshot.action === "grant" && (
            <p className="recall-permission-meta">
              This does not install the connection, import history or enable
              continuous sync, and it does not change the built-in SDK local
              memory processing. Authorization does not mean the app has
              actually called or adopted memories.
            </p>
          )}
          {pending && (
            <p role="status">
              {pending === "write"
                ? "Submitting call permissions..."
                : "Checking the installation identity and grant revision..."}
            </p>
          )}
          {visibleError && (
            <p className="recall-permission-error" role="alert">
              {visibleError}
            </p>
          )}
        </div>
        <footer className="consent-actions recall-permission-actions">
          <button
            type="button"
            className="secondary"
            disabled={pending === "write"}
            onClick={close}
          >
            {phase === "receipt" ? "Done" : "Cancel"}
          </button>
          {(needsRefresh || error) && !stale && !changed && (
            <button
              type="button"
              className="secondary"
              disabled={!!pending || disabled || !permissionBridge()}
              onClick={() => void refresh()}
            >
              <RefreshCw size={16} aria-hidden="true" />
              Refresh and check
            </button>
          )}
          {phase === "select" && (
            <button
              type="button"
              disabled={!!pending || blocked || !selectionValid}
              onClick={approve}
            >
              <ShieldCheck size={16} aria-hidden="true" />
              {editing ? "Save this scope" : "Allow and save"}
            </button>
          )}
          {phase === "revoke" && (
            <button
              type="button"
              disabled={!!pending || blocked || !confirmed}
              onClick={() => void commit(null)}
            >
              <ShieldOff size={16} aria-hidden="true" />
              Confirm revocation
            </button>
          )}
        </footer>
      </section>
    </div>,
    document.body,
  );
}

function PermissionSummary({
  tools,
  scopes,
  snapshot,
}: {
  tools: readonly RecallPermissionTool[];
  scopes: readonly string[];
  snapshot: Snapshot;
}) {
  return (
    <>
      <p>Tools: {tools.map((tool) => toolNames[tool]).join(", ") || "None"}</p>
      <ul className="recall-permission-scope-list" aria-label="Granted scopes">
        {scopes.length ? (
          scopes.map((scope) => (
            <li key={scope}>
              <code>{scope}</code>
              {scope.startsWith("source:") && (
                <span>
                  {snapshot.sources.find(
                    (source) => `source:${source.source_id}` === scope,
                  )?.display_name ?? "Source not verified"}
                </span>
              )}
              {scope.startsWith("workspace:") && (
                <span>
                  {snapshot.workspaces.find((option) => option.scope === scope)
                    ?.label ?? "Workspace not verified"}
                </span>
              )}
            </li>
          ))
        ) : (
          <li>No granted scopes</li>
        )}
      </ul>
    </>
  );
}

const permissionBridge = hostPermissionBridge;

function validReceipt(
  status: RecallPermissionStatus | undefined,
  hostId: string,
) {
  const receipt = status?.receipt;
  if (
    !receipt ||
    status?.hostId !== hostId ||
    receipt.hostId !== hostId ||
    typeof receipt.installationId !== "string" ||
    !receipt.installationId.trim() ||
    !Number.isSafeInteger(receipt.revision) ||
    receipt.revision < 1 ||
    !["install-only", "approved", "revoked"].includes(receipt.status) ||
    !Array.isArray(receipt.allowedTools) ||
    !receipt.allowedTools.every((tool) => Object.hasOwn(toolNames, tool)) ||
    !Array.isArray(receipt.allowedScopes) ||
    !receipt.allowedScopes.every(
      (scope) => typeof scope === "string" && validScope(scope),
    ) ||
    new Set(receipt.allowedTools).size !== receipt.allowedTools.length ||
    new Set(receipt.allowedScopes).size !== receipt.allowedScopes.length ||
    (["install-only", "approved", "revoked"].includes(status.state) &&
      receipt.status !== status.state)
  )
    return null;
  return receipt;
}

function freezeStatus(status: RecallPermissionStatus): RecallPermissionStatus;
function freezeStatus(
  status: RecallPermissionStatus | undefined,
): RecallPermissionStatus | undefined;
function freezeStatus(
  status: RecallPermissionStatus | undefined,
): RecallPermissionStatus | undefined {
  if (!status) return undefined;
  const receipt = validReceipt(status, status.hostId);
  // Explicitly copy display fields only; neither credentials nor unknown IPC fields belong in UI state.
  return Object.freeze({
    hostId: status.hostId,
    state: status.state,
    reason: status.reason,
    receipt: receipt
      ? Object.freeze({
          hostId: receipt.hostId,
          installationId: receipt.installationId,
          revision: receipt.revision,
          status: receipt.status,
          allowedTools: Object.freeze([...receipt.allowedTools]),
          allowedScopes: Object.freeze([...receipt.allowedScopes]),
        })
      : null,
  });
}

function canApprove(status: RecallPermissionStatus | undefined) {
  return (
    status?.reason === null &&
    ["install-only", "approved"].includes(status.state)
  );
}

function statusLabel(
  status: RecallPermissionStatus | undefined,
  hostId: string,
) {
  if (!status || status.hostId !== hostId)
    return "Call authorization state not verified yet";
  if (status.state === "not-installed")
    return "The connection is not installed, so memory use cannot be authorized";
  if (status.state === "relink-required") {
    if (status.reason === "legacy_installation")
      return "An old connection was detected; reconnect first, then allow memory use";
    if (status.reason === "managed_config_changed")
      return "The connection configuration changed; review and reconnect first, then allow memory use";
    return "The connection credentials need updating; reconnect first, then allow memory use";
  }
  if (status.state === "unavailable")
    return "The call authorization service is unavailable";
  if (status.state === "unsupported")
    return "This connection does not support call permission management";
  if (!validReceipt(status, hostId))
    return "Call authorization details are incomplete; refresh and check";
  if (status.state === "approved")
    return "Memory use is allowed for the selected scopes";
  if (status.state === "revoked")
    return "The call authorization for this install was revoked; reconnect and grant it again";
  return "Installed; memory use is not allowed yet";
}

function sameStatus(
  snapshot: RecallPermissionStatus,
  value: RecallPermissionStatus | undefined,
) {
  const receipt = validReceipt(value, snapshot.hostId);
  const saved = snapshot.receipt;
  return (
    !!receipt &&
    !!saved &&
    snapshot.state === value?.state &&
    snapshot.reason === value.reason &&
    saved.installationId === receipt.installationId &&
    saved.revision === receipt.revision &&
    saved.status === receipt.status &&
    sameSet(saved.allowedTools, receipt.allowedTools) &&
    sameSet(saved.allowedScopes, receipt.allowedScopes)
  );
}

function sameSet(a: readonly string[], b: readonly string[]) {
  return a.length === b.length && a.every((value) => b.includes(value));
}

function toggle<T>(values: readonly T[], value: T): T[] {
  return values.includes(value)
    ? values.filter((item) => item !== value)
    : [...values, value];
}

function validScope(scope: string) {
  return (
    scope === "all-apps" ||
    /^workspace:v1:[0-9a-f]{64}$/.test(scope) ||
    /^source:[a-z0-9][a-z0-9-]{0,127}$/.test(scope)
  );
}

function knownWorkspaces(
  options: RecallPermissionControlProps["workspaceScopes"],
): WorkspaceScope[] {
  const counts = new Map<string, number>();
  for (const option of options ?? [])
    counts.set(option.scope, (counts.get(option.scope) ?? 0) + 1);
  return (options ?? [])
    .filter(
      (option) =>
        /^workspace:v1:[0-9a-f]{64}$/.test(option.scope) &&
        typeof option.label === "string" &&
        !!option.label.trim() &&
        counts.get(option.scope) === 1,
    )
    .map((option) => ({ ...option }));
}

function sameSource(
  saved: SourceResponse,
  current: SourceResponse | undefined,
) {
  return (
    !!current &&
    eligibleSource(current) &&
    saved.root_id === current.root_id &&
    saved.grant_revision === current.grant_revision &&
    saved.candidate_path === current.candidate_path &&
    saved.root_label === current.root_label &&
    saved.candidate_kind === current.candidate_kind
  );
}

function knownScope(scope: string, snapshot: Snapshot) {
  return (
    scope === "all-apps" ||
    snapshot.sources.some((source) => `source:${source.source_id}` === scope) ||
    snapshot.workspaces.some((option) => option.scope === scope)
  );
}

function validMutationResult(
  status: RecallPermissionStatus,
  snapshot: Snapshot,
  review: SetRecallPermissionInput | null,
) {
  const receipt = validReceipt(status, snapshot.hostId);
  return (
    !!receipt &&
    status.reason === null &&
    receipt.installationId === snapshot.permission.receipt.installationId &&
    receipt.revision === snapshot.permission.receipt.revision + 1 &&
    (snapshot.action === "revoke"
      ? status.state === "revoked" &&
        receipt.allowedTools.length === 0 &&
        receipt.allowedScopes.length === 0
      : status.state === "approved" &&
        !!review &&
        sameSet(receipt.allowedTools, review.allowedTools) &&
        sameSet(receipt.allowedScopes, review.allowedScopes))
  );
}

function isStaleError(error: unknown) {
  const message = error instanceof Error ? error.message : "";
  return [
    "host_grant_revision_conflict",
    "host_grant_revoked",
    "host_permission_installation_mismatch",
    "host_permission_relink_required",
  ].some((code) => message.includes(code));
}

function refreshAfterCommitMessage(action: Snapshot["action"]) {
  return action === "revoke"
    ? "The authorization was revoked, but the list refresh failed. Refreshing to check does not revoke again."
    : "The authorization was saved, but the list refresh failed. Refreshing to check does not grant again.";
}
