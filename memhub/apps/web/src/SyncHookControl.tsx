import { Download, RefreshCw, Trash2, X } from "lucide-react";
import { useCallback, useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type {
  ApplySyncHookInput,
  RecallHostStatus,
  SyncHookPreview,
  SyncHookStatus,
} from "./global";
import "./sync-hook-control.css";

export type { SyncHookPreview, SyncHookStatus } from "./global";

interface SyncHookBridge {
  getSyncHookStatus(hostId: string): Promise<SyncHookStatus>;
  previewSyncHook(
    hostId: string,
    operation: "install" | "uninstall",
  ): Promise<SyncHookPreview>;
  applySyncHook(
    hostId: string,
    input: ApplySyncHookInput,
  ): Promise<SyncHookStatus>;
}
interface Snapshot {
  host: RecallHostStatus;
  key: string;
  operation: "install" | "uninstall";
  updating: boolean;
  bridge: SyncHookBridge;
  trigger: HTMLElement;
}
const statusNames: Record<SyncHookStatus["state"], string> = {
  unsupported: "This app does not support the sync Hook yet",
  "connection-required": "The MCP connection needs to be relinked",
  "not-installed": "Sync Hook not installed",
  installed: "Sync Hook installed",
  modified: "Sync Hook files have changed",
  unavailable: "Sync Hook status is unavailable right now",
};
const changedMessage =
  "The connection has changed. Close and reopen it to verify again. This confirmation will not be used for the changed connection.";
const focusSelector =
  'button:not(:disabled), input:not(:disabled), [tabindex="0"]';

function hookBridge(): SyncHookBridge | null {
  const api = window.oneMemory as
    | (Window["oneMemory"] & Partial<SyncHookBridge>)
    | undefined;
  return api &&
    typeof api.getSyncHookStatus === "function" &&
    typeof api.previewSyncHook === "function" &&
    typeof api.applySyncHook === "function"
    ? (api as SyncHookBridge)
    : null;
}
function hostKey(host: RecallHostStatus): string {
  const permission = host.permission;
  return JSON.stringify([
    host.hostId,
    host.present,
    host.installed,
    permission?.state,
    permission?.reason,
    permission?.receipt?.installationId,
    permission?.receipt?.revision,
    permission?.receipt?.status,
  ]);
}
function checkedStatus(value: SyncHookStatus, hostId: string): SyncHookStatus {
  if (
    !value ||
    value.hostId !== hostId ||
    !Object.hasOwn(statusNames, value.state) ||
    (value.reason !== null && typeof value.reason !== "string")
  )
    throw new Error("invalid_hook_status");
  return { hostId, state: value.state, reason: value.reason };
}
function checkedPreview(
  value: SyncHookPreview,
  snapshot: Snapshot,
): SyncHookPreview {
  const receipt = snapshot.host.permission?.receipt;
  if (
    !value ||
    value.hostId !== snapshot.host.hostId ||
    value.operation !== snapshot.operation ||
    typeof value.installationId !== "string" ||
    !/^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/.test(
      value.installationId,
    ) ||
    !Number.isSafeInteger(value.revision) ||
    value.revision < 1 ||
    (receipt &&
      (receipt.installationId !== value.installationId ||
        receipt.revision !== value.revision)) ||
    typeof value.fingerprint !== "string" ||
    !/^[a-f0-9]{64}$/.test(value.fingerprint) ||
    !Array.isArray(value.files) ||
    value.files.length > 64 ||
    value.files.some(
      (file) =>
        !file ||
        typeof file.path !== "string" ||
        file.path.length > 4096 ||
        !/^(?:\/|[A-Za-z]:[\\/])/.test(file.path) ||
        /\p{Cc}/u.test(file.path) ||
        !["create", "update", "remove"].includes(file.action) ||
        !["host", "one-memory"].includes(file.owner),
    ) ||
    new Set(value.files.map((file) => file.path)).size !== value.files.length
  )
    throw new Error("invalid_hook_preview");
  return {
    hostId: value.hostId,
    operation: value.operation,
    installationId: value.installationId,
    revision: value.revision,
    fingerprint: value.fingerprint,
    files: value.files.map((file) => ({
      path: file.path,
      action: file.action,
      owner: file.owner,
    })),
  };
}

export function SyncHookControl({
  host,
  disabled = false,
}: {
  host: RecallHostStatus;
  disabled?: boolean;
}) {
  const bridge = hookBridge();
  const visible =
    !!bridge && host.hostId === "claude-code" && host.present && host.installed;
  const key = hostKey(host);
  const currentKey = useRef(key);
  currentKey.current = key;
  const generation = useRef(0);
  const control = useRef<HTMLElement>(null);
  const [local, setLocal] = useState<{
    key: string;
    status: SyncHookStatus | null;
    loading: boolean;
    error: boolean;
  } | null>(null);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const current = local?.key === key ? local : null;

  useEffect(() => {
    if (!visible || !bridge) return;
    const token = ++generation.current;
    setLocal({ key, status: null, loading: true, error: false });
    void bridge
      .getSyncHookStatus(host.hostId)
      .then((value) => {
        const status = checkedStatus(value, host.hostId);
        if (token === generation.current && currentKey.current === key)
          setLocal({ key, status, loading: false, error: false });
      })
      .catch(() => {
        if (token === generation.current && currentKey.current === key)
          setLocal({ key, status: null, loading: false, error: true });
      });
    return () => {
      generation.current++;
    };
  }, [key, visible, bridge, host.hostId]);

  async function refresh() {
    if (!bridge || disabled || snapshot || current?.loading) return;
    const token = ++generation.current;
    setLocal({
      key,
      status: current?.status ?? null,
      loading: true,
      error: false,
    });
    try {
      const status = checkedStatus(
        await bridge.getSyncHookStatus(host.hostId),
        host.hostId,
      );
      if (token === generation.current && currentKey.current === key)
        setLocal({ key, status, loading: false, error: false });
    } catch {
      if (token === generation.current && currentKey.current === key)
        setLocal({
          key,
          status: current?.status ?? null,
          loading: false,
          error: true,
        });
    }
  }
  function open(operation: Snapshot["operation"], trigger: HTMLElement) {
    if (
      !bridge ||
      !visible ||
      disabled ||
      snapshot ||
      current?.loading ||
      current?.error ||
      document.querySelector('[role="dialog"][aria-modal="true"]')
    )
      return;
    setSnapshot({
      host: {
        ...host,
        permission: host.permission
          ? structuredClone(host.permission)
          : undefined,
      },
      key,
      operation,
      updating:
        current?.status?.state === "installed" ||
        current?.status?.state === "modified",
      bridge,
      trigger,
    });
  }
  const installed =
    current?.status?.state === "installed" ||
    current?.status?.state === "modified";
  const canInstall =
    current?.status &&
    !current.error &&
    current.status.reason === null &&
    ["not-installed", "installed", "modified"].includes(current.status.state) &&
    host.permission?.state !== "revoked" &&
    host.permission?.state !== "relink-required";

  if (!visible && !snapshot) return null;
  return (
    <section
      ref={control}
      className="sync-hook-control"
      aria-label={`${host.label} Sync Hook`}
      tabIndex={-1}
    >
      {visible && (
        <>
          <p role="status" className="sync-hook-status">
            {current?.loading
              ? "Loading the sync Hook status..."
              : current?.status
                ? statusNames[current.status.state]
                : "Sync Hook status pending verification"}
          </p>
          <p className="sync-hook-note">
            The plug-in only announces new conversations; reading and continuous
            sync stay controlled by source permissions.
          </p>
          {installed && current?.status?.reason !== null && (
            <p className="sync-hook-note">
              The MCP connection needs verification; an existing sync Hook can
              still be removed.
            </p>
          )}
          <div className="sync-hook-actions">
            {canInstall && (
              <button
                type="button"
                className="secondary"
                disabled={disabled || !!snapshot || current?.loading}
                onClick={(event) => open("install", event.currentTarget)}
              >
                {installed ? (
                  <RefreshCw size={16} aria-hidden="true" />
                ) : (
                  <Download size={16} aria-hidden="true" />
                )}
                {installed ? "Update Sync Hook" : "Install Sync Hook"}
              </button>
            )}
            {installed && (
              <button
                type="button"
                className="quiet"
                disabled={
                  disabled || !!snapshot || current?.loading || current?.error
                }
                onClick={(event) => open("uninstall", event.currentTarget)}
              >
                <Trash2 size={16} aria-hidden="true" />
                Uninstall Sync Hook
              </button>
            )}
            <button
              type="button"
              className="quiet sync-hook-icon"
              title="Refresh sync Hook status"
              aria-label="Refresh sync Hook status"
              disabled={disabled || !!snapshot || current?.loading}
              onClick={() => void refresh()}
            >
              <RefreshCw size={16} aria-hidden="true" />
            </button>
          </div>
          {current?.error && (
            <p role="alert" className="sync-hook-error">
              Could not read the sync Hook status. Refresh to verify again. No
              connection was changed.
            </p>
          )}
        </>
      )}
      {snapshot && (
        <HookDialog
          snapshot={snapshot}
          currentKey={key}
          disabled={disabled || !bridge}
          onClose={() => setSnapshot(null)}
          fallbackFocus={control.current}
          onStatus={(status) => {
            if (currentKey.current === snapshot.key) {
              generation.current++;
              setLocal({
                key: snapshot.key,
                status,
                loading: false,
                error: false,
              });
            }
          }}
        />
      )}
    </section>
  );
}

function HookDialog({
  snapshot,
  currentKey,
  disabled,
  onClose,
  onStatus,
  fallbackFocus,
}: {
  snapshot: Snapshot;
  currentKey: string;
  disabled: boolean;
  onClose(): void;
  onStatus(status: SyncHookStatus): void;
  fallbackFocus: HTMLElement | null;
}) {
  const id = useId();
  const panel = useRef<HTMLElement>(null);
  const mounted = useRef(false);
  const sequence = useRef(0);
  const flight = useRef<"preview" | "write" | "refresh" | null>(null);
  const latest = useRef({ currentKey, disabled });
  latest.current = { currentKey, disabled };
  const invalidated = useRef(false);
  if (currentKey !== snapshot.key) invalidated.current = true;
  const changed = invalidated.current;
  const [preview, setPreview] = useState<SyncHookPreview | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [pending, setPending] = useState<
    "preview" | "write" | "refresh" | null
  >("preview");
  const [error, setError] = useState<string | null>(null);
  const [needsPreview, setNeedsPreview] = useState(false);
  const [result, setResult] = useState<SyncHookStatus | null>(null);
  const resultRef = useRef<SyncHookStatus | null>(null);
  const noChanges = preview !== null && preview.files.length === 0;
  const action =
    snapshot.operation === "uninstall"
      ? "Uninstall"
      : snapshot.updating
        ? "Update"
        : "Install";
  const active = useCallback(
    (token: number) => mounted.current && sequence.current === token,
    [],
  );
  const usable = useCallback(
    () =>
      !invalidated.current &&
      latest.current.currentKey === snapshot.key &&
      !latest.current.disabled &&
      hookBridge() === snapshot.bridge,
    [snapshot],
  );

  const loadPreview = useCallback(async () => {
    if (flight.current || resultRef.current || !usable()) return;
    const token = ++sequence.current;
    flight.current = "preview";
    setPending("preview");
    setError(null);
    setConfirmed(false);
    setNeedsPreview(true);
    try {
      const value = checkedPreview(
        await snapshot.bridge.previewSyncHook(
          snapshot.host.hostId,
          snapshot.operation,
        ),
        snapshot,
      );
      if (!active(token) || !usable()) return;
      setPreview(value);
      setNeedsPreview(false);
    } catch {
      if (active(token)) {
        setPreview(null);
        setError(
          "Could not verify the file changes. Preview again. This operation was not submitted.",
        );
      }
    } finally {
      if (active(token)) {
        flight.current = null;
        setPending(null);
      }
    }
  }, [active, snapshot, usable]);
  useEffect(() => {
    mounted.current = true;
    void loadPreview();
    return () => {
      mounted.current = false;
      sequence.current++;
      flight.current = null;
    };
  }, [loadPreview]);

  useEffect(() => {
    const dialog = panel.current;
    const backdrop = dialog?.parentElement;
    if (!dialog || !backdrop) return;
    const prior = new Map<HTMLElement, boolean>();
    for (const child of document.body.children) {
      if (child instanceof HTMLElement && child !== backdrop) {
        prior.set(child, child.inert);
        child.inert = true;
      }
    }
    const focusInside = () =>
      (dialog.querySelector<HTMLElement>(focusSelector) ?? dialog).focus();
    const keepFocus = (event: FocusEvent) => {
      if (event.target instanceof Node && !dialog.contains(event.target))
        focusInside();
    };
    document.addEventListener("focusin", keepFocus);
    focusInside();
    return () => {
      document.removeEventListener("focusin", keepFocus);
      for (const [element, inert] of prior) element.inert = inert;
      if (
        snapshot.trigger.isConnected &&
        !(snapshot.trigger as HTMLButtonElement).disabled
      )
        snapshot.trigger.focus();
      else if (fallbackFocus?.isConnected) fallbackFocus.focus();
    };
  }, [snapshot, fallbackFocus]);
  useEffect(() => {
    if (pending === "write") panel.current?.focus();
  }, [pending]);
  function close() {
    if (flight.current !== "write") onClose();
  }

  async function refreshStatus() {
    if (flight.current || !resultRef.current || !usable()) return;
    const token = ++sequence.current;
    flight.current = "refresh";
    setPending("refresh");
    setError(null);
    try {
      const value = checkedStatus(
        await snapshot.bridge.getSyncHookStatus(snapshot.host.hostId),
        snapshot.host.hostId,
      );
      if (!active(token) || !usable()) return;
      onStatus(value);
      if (value.state !== resultRef.current.state)
        setError(
          "This operation succeeded; the file status has since changed. Close and verify again.",
        );
    } catch {
      if (active(token))
        setError(
          `${action} succeeded, but the status refresh failed. Refreshing the status will not run ${action} again.`,
        );
    } finally {
      if (active(token)) {
        flight.current = null;
        setPending(null);
      }
    }
  }
  async function submit() {
    if (
      !preview ||
      noChanges ||
      !confirmed ||
      needsPreview ||
      flight.current ||
      resultRef.current ||
      !usable()
    )
      return;
    const token = ++sequence.current;
    flight.current = "write";
    setPending("write");
    setError(null);
    try {
      const value = checkedStatus(
        await snapshot.bridge.applySyncHook(snapshot.host.hostId, {
          operation: preview.operation,
          installationId: preview.installationId,
          revision: preview.revision,
          fingerprint: preview.fingerprint,
          userConfirmed: true,
        }),
        snapshot.host.hostId,
      );
      if (
        snapshot.operation === "install"
          ? value.state !== "installed"
          : !["not-installed", "connection-required"].includes(value.state)
      )
        throw new Error("hook_write_unconfirmed");
      if (!active(token)) return;
      resultRef.current = value;
      setResult(value);
      setConfirmed(false);
      if (usable()) onStatus(value);
      flight.current = null;
      setPending(null);
      if (usable()) await refreshStatus();
    } catch {
      if (active(token)) {
        setError(
          `${action} result is not confirmed yet. Preview and verify again, then retry.`,
        );
        setNeedsPreview(true);
        setConfirmed(false);
      }
    } finally {
      if (active(token)) {
        flight.current = null;
        setPending(null);
      }
    }
  }

  return createPortal(
    <div className="consent-backdrop sync-hook-backdrop">
      <section
        ref={panel}
        role="dialog"
        aria-modal="true"
        aria-labelledby={`${id}-title`}
        aria-describedby={`${id}-scope`}
        aria-busy={!!pending}
        tabIndex={-1}
        className="consent-sheet sync-hook-sheet"
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            event.preventDefault();
            event.stopPropagation();
            close();
          }
          if (event.key === "Tab") {
            const items =
              panel.current?.querySelectorAll<HTMLElement>(focusSelector);
            const first = items?.[0],
              last = items?.[items.length - 1];
            if (!first || !last) {
              event.preventDefault();
              panel.current?.focus();
            } else if (
              event.shiftKey &&
              (document.activeElement === first ||
                document.activeElement === panel.current)
            ) {
              event.preventDefault();
              last.focus();
            } else if (
              !event.shiftKey &&
              (document.activeElement === last ||
                document.activeElement === panel.current)
            ) {
              event.preventDefault();
              first.focus();
            }
          }
        }}
      >
        <header className="sync-hook-heading">
          <div>
            <p className="eyebrow">{snapshot.host.label}</p>
            <h2 id={`${id}-title`}>{action} Sync Hook</h2>
          </div>
          <button
            type="button"
            className="quiet sync-hook-icon"
            title="Close"
            aria-label="Close"
            disabled={pending === "write"}
            onClick={close}
          >
            <X size={18} aria-hidden="true" />
          </button>
        </header>
        <div className="sync-hook-body">
          <p id={`${id}-scope`}>
            The plug-in only announces new conversations; reading and continuous
            sync stay controlled by source permissions. Installing does not
            enable sync or grant model processing permission.
          </p>
          {snapshot.operation === "uninstall" && (
            <p>
              This does not turn off standalone continuous sync and does not
              disconnect the MCP memory connection.
            </p>
          )}
          {preview && (
            <>
              <p className="sync-hook-note">
                Install identity <code>{preview.installationId}</code> -
                revision {preview.revision}
              </p>
              {!noChanges && (
                <ul className="sync-hook-files" aria-label="File changes">
                  {preview.files.map((file) => (
                    <li key={file.path}>
                      <span>
                        {
                          { create: "Add", update: "Update", remove: "Remove" }[
                            file.action
                          ]
                        }{" "}
                        ·{" "}
                        {file.owner === "host"
                          ? `${snapshot.host.label} configuration`
                          : "MemHub file"}
                      </span>
                      <code>{file.path}</code>
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
          {noChanges && (
            <p role="status">
              {snapshot.operation === "install" && snapshot.updating
                ? "The sync Hook is already current; no changes are needed."
                : "No files need to change; no operation was submitted."}
            </p>
          )}
          {preview && !noChanges && !result && (
            <label className="sync-hook-confirm">
              <input
                type="checkbox"
                checked={confirmed && !changed}
                disabled={!!pending || disabled || changed || needsPreview}
                onChange={(event) => setConfirmed(event.target.checked)}
              />
              I confirm the file changes above
            </label>
          )}
          {result && (
            <p className="sync-hook-receipt" role="status">
              {snapshot.operation === "uninstall"
                ? "Sync Hook uninstalled. Standalone continuous sync and the MCP connection are unchanged."
                : `Sync Hook ${action} complete. The source continuous sync settings are unchanged.`}
            </p>
          )}
          {pending && (
            <p role="status">
              {pending === "preview"
                ? "Verifying file changes..."
                : pending === "write"
                  ? `${action} sync Hook in progress...`
                  : "Refreshing status..."}
            </p>
          )}
          {(changed || error) && (
            <p role="alert" className="sync-hook-error">
              {changed ? changedMessage : error}
            </p>
          )}
        </div>
        <footer className="consent-actions sync-hook-footer">
          <button
            type="button"
            className="secondary"
            disabled={pending === "write"}
            onClick={close}
          >
            {result || noChanges ? "Done" : "Cancel"}
          </button>
          {error && !result && !changed && (
            <button
              type="button"
              className="secondary"
              disabled={!!pending || disabled}
              onClick={() => void loadPreview()}
            >
              <RefreshCw size={16} aria-hidden="true" />
              Re-preview
            </button>
          )}
          {result && error && !changed && (
            <button
              type="button"
              className="secondary"
              disabled={!!pending || disabled}
              onClick={() => void refreshStatus()}
            >
              <RefreshCw size={16} aria-hidden="true" />
              Refresh status
            </button>
          )}
          {preview && !noChanges && !result && (
            <button
              type="button"
              disabled={
                !confirmed || !!pending || disabled || changed || needsPreview
              }
              onClick={() => void submit()}
            >
              {snapshot.operation === "uninstall" ? (
                <Trash2 size={16} aria-hidden="true" />
              ) : snapshot.updating ? (
                <RefreshCw size={16} aria-hidden="true" />
              ) : (
                <Download size={16} aria-hidden="true" />
              )}
              Confirm {action}
            </button>
          )}
        </footer>
      </section>
    </div>,
    document.body,
  );
}
