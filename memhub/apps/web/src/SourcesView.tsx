import { RefreshCw, ShieldCheck } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { ConnectionFallback } from "./ConnectionFallback";
import type {
  RecallHostStatus,
  RecallInstallPreview,
  RecallLinkCheck,
  RecallUninstallPreview,
} from "./global";
import { HistoryImport } from "./HistoryImport";
import {
  applyHostSkill,
  grantHostPermission,
  hostBridgeAvailable,
  hostCapabilities,
  hostWriteReason,
  installHost,
  previewHostInstall,
  previewHostSkill,
  previewHostUninstall,
  probeHost,
  readHostPermission,
  refreshRecallHosts,
  uninstallHost,
  useRecallHosts,
} from "./host-state";
import { RecallPermissionControl } from "./RecallPermissionControl";
import { fullGrantSelection, selectionIsValid } from "./recall-grant";
import { LINK_WORDING, SKILL_WORDING } from "./recall-host-wording";
import { SourceDirectoryPicker } from "./SourceDirectoryPicker";
import { SourceImportReceipt } from "./SourceImportReceipt";
import { SourceJobActions } from "./SourceJobActions";
import { SourceSnapshotStatus } from "./SourceSnapshotStatus";
import { SyncHookControl } from "./SyncHookControl";
import { traceUiAction, type UiTrace } from "./ui-trace";
import {
  isDefiniteAccessRefusal,
  type useImportSources,
} from "./use-import-sources";

type ImportState = ReturnType<typeof useImportSources>;

/** One app's part of a bulk connect: what installing it writes, and whether a
 * grant still has to be written after that. */
type BulkStep = {
  hostId: string;
  label: string;
  /** Null when the connection is already in place. */
  install: RecallInstallPreview | null;
  allow: boolean;
  /** Non-null when this host could not be prepared and will be skipped. */
  error: string | null;
};

function requiresHookRemoval(error: unknown) {
  return (
    error instanceof Error && /\bsync_hook_remove_first\b/.test(error.message)
  );
}

function needsRecallReconnect(host: RecallHostStatus) {
  return (
    host.permission?.state === "relink-required" ||
    host.permission?.state === "revoked" ||
    host.permission?.reason === "installation_incomplete"
  );
}

export function SourcesView({ imports }: { imports: ImportState }) {
  const visibleSources = useMemo(
    () =>
      imports.sources.filter(
        (source) =>
          source.available ||
          source.authorized ||
          source.candidate_change_allowed === true,
      ),
    [imports.sources],
  );
  const availableUnconnected = useMemo(
    () =>
      visibleSources.filter((source) => source.available && !source.authorized),
    [visibleSources],
  );
  const [pendingProcessingSource, setPendingProcessingSource] = useState<
    string | null
  >(null);
  const [directoryBusy, setDirectoryBusy] = useState<string | null>(null);
  const [consentPhase, setConsentPhase] = useState<
    "general" | "environment_override" | null
  >(null);
  const [consentSources, setConsentSources] = useState<ImportState["sources"]>(
    [],
  );
  // The trace opened by the switch, carried through the consent dialog so the
  // confirmation's connect request joins the same four-line chain instead of
  // emitting a second, kernel-less pair. The trace object (not just its id) is
  // kept so the dialog can close it exactly once when it resolves.
  const consentTrace = useRef<UiTrace | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [retryFailedConnection, setRetryFailedConnection] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [authorizingSource, setAuthorizingSource] = useState<string | null>(
    null,
  );
  const [authorizationErrors, setAuthorizationErrors] = useState<
    Record<string, string>
  >({});
  const connectionFlight = useRef(false);
  const consentPanel = useRef<HTMLElement>(null);
  const consentOpen = consentPhase !== null;
  const hostRead = useRecallHosts({ read: true });
  // A failed host action shows the list as unreadable until the next read lands.
  const [hostActionFailed, setHostActionFailed] = useState(false);
  const recallHosts: RecallHostStatus[] | "checking" | "error" | "unavailable" =
    !hostBridgeAvailable()
      ? "unavailable"
      : hostActionFailed || (hostRead.isError && !hostRead.data)
        ? "error"
        : hostRead.data
          ? hostRead.data.filter((host) => host.present)
          : "checking";
  const markHostsUnreadable = () => setHostActionFailed(true);
  const rereadHosts = async () => {
    await refreshRecallHosts();
    setHostActionFailed(false);
  };
  const [busyHost, setBusyHost] = useState<string | null>(null);
  const [hostErrors, setHostErrors] = useState<Record<string, string>>({});
  const [hookRemovalRequired, setHookRemovalRequired] = useState<string | null>(
    null,
  );
  const [linkCheck, setLinkCheck] = useState<Record<string, RecallLinkCheck>>(
    {},
  );
  const [skillConflict, setSkillConflict] = useState<{
    hostId: string;
    fingerprint: string;
  } | null>(null);
  const [installPreview, setInstallPreview] =
    useState<RecallInstallPreview | null>(null);
  const [uninstallPreview, setUninstallPreview] =
    useState<RecallUninstallPreview | null>(null);
  const [bulk, setBulk] = useState<BulkStep[] | null>(null);
  const [bulkBusy, setBulkBusy] = useState(false);
  const [bulkErrors, setBulkErrors] = useState<Record<string, string>>({});
  const [bulkDone, setBulkDone] = useState<string[] | null>(null);

  useEffect(() => {
    if (!consentOpen || !consentPanel.current) return;
    const before = document.activeElement;
    const siblings = new Map<HTMLElement, boolean>();
    let branch: HTMLElement =
      consentPanel.current.parentElement ?? consentPanel.current;
    while (branch.parentElement) {
      for (const sibling of branch.parentElement.children) {
        if (!(sibling instanceof HTMLElement) || sibling === branch) continue;
        siblings.set(sibling, sibling.inert);
        sibling.inert = true;
      }
      branch = branch.parentElement;
      if (branch === document.body) break;
    }
    consentPanel.current.querySelector<HTMLButtonElement>("button")?.focus();
    return () => {
      for (const [sibling, wasInert] of siblings) sibling.inert = wasInert;
      if (before instanceof HTMLElement && before.isConnected) before.focus();
    };
  }, [consentOpen]);

  useEffect(() => {
    if (connecting) consentPanel.current?.focus();
  }, [connecting]);

  const connected = visibleSources.filter((source) => source.authorized);
  const syncable = connected.filter(
    (source) =>
      source.available &&
      source.sync_enabled === true &&
      typeof source.grant_revision === "number" &&
      typeof source.root_id === "string" &&
      source.capabilities.live_sync === "implemented" &&
      source.sync_completeness !== "one_time_snapshot",
  );
  const needsConnection = availableUnconnected.length > 0;
  const actionBusy =
    connecting ||
    authorizingSource !== null ||
    directoryBusy !== null ||
    imports.busyKey === "connect:selected" ||
    imports.busyKey === "sync:all";
  const snapshotStale = imports.sourceSnapshot?.stale === true;
  const overrideSources = consentSources.filter(
    (source) => source.candidate_kind !== "default",
  );

  async function connectApproved() {
    if (connectionFlight.current || !consentSources.length) return;
    if (
      consentSources.some(
        (approved) =>
          !imports.sources.some(
            (current) =>
              current.source_id === approved.source_id &&
              current.available &&
              current.authorized === approved.authorized &&
              current.candidate_kind === approved.candidate_kind &&
              (current.candidate_revision ?? 0) ===
                (approved.candidate_revision ?? 0) &&
              current.candidate_path === approved.candidate_path,
          ),
      )
    ) {
      setConnectionError(
        "The source state or path has changed; cancel and verify again. This authorization was not submitted.",
      );
      return;
    }
    connectionFlight.current = true;
    setConnecting(true);
    setConnectionError(null);
    try {
      const result = await imports.connectSelected(
        consentSources.map((source) => source.source_id),
        consentSources,
        {
          ...(retryFailedConnection ? { retryFailed: true } : {}),
          ...(consentTrace.current
            ? { traceId: consentTrace.current.traceId }
            : {}),
        },
      );
      if (!result.accepted_source_ids.length) {
        setRetryFailedConnection(true);
        setConnectionError(
          "The selected sources were not approved for import; check the per-source reasons and retry.",
        );
        return;
      }
      setConsentPhase(null);
    } catch {
      setConnectionError(
        "The connection result is unconfirmed. The selected sources are kept; check and retry.",
      );
      consentTrace.current?.done("unknown", { reason: "connect_unconfirmed" });
      consentTrace.current = null;
    } finally {
      connectionFlight.current = false;
      setConnecting(false);
    }
  }

  /** The one Harness switch: scan + enqueue + schedule, or freeze all three. */
  async function toggleSourceProcessing(
    source: ImportState["sources"][number],
    enabled: boolean,
  ) {
    if (enabled && !source.available && !source.authorized) return;
    setConnectionError(null);
    setRetryFailedConnection(false);
    setAuthorizationErrors((errors) => {
      if (!(source.source_id in errors)) return errors;
      const next = { ...errors };
      delete next[source.source_id];
      return next;
    });
    if (snapshotStale) {
      setConnectionError(
        "Source state is pending verification, so the processing switch cannot change. Refresh the source state first.",
      );
      return;
    }
    setPendingProcessingSource(source.source_id);
    const trace = traceUiAction("sources", "toggle");
    if (source.authorized) {
      try {
        await imports.setSync(source, enabled, trace.traceId);
        trace.done("succeeded");
      } catch (error) {
        const refused = isDefiniteAccessRefusal(error);
        trace.done(refused ? "failed" : "unknown", {
          reason: refused
            ? enabled
              ? "processing_on_failed"
              : "processing_freeze_failed"
            : "processing_result_unconfirmed",
        });
        setAuthorizationErrors((errors) => ({
          ...errors,
          [source.source_id]: refused
            ? enabled
              ? "Could not turn processing on; check and retry."
              : "Could not freeze processing; check and retry."
            : "The processing switch result is unconfirmed; refresh and verify; there is no automatic retry.",
        }));
      } finally {
        setPendingProcessingSource(null);
      }
      return;
    }
    if (!enabled) {
      setPendingProcessingSource(null);
      trace.done("succeeded");
      return;
    }
    setConsentSources([{ ...source }]);
    consentTrace.current = trace;
    if (source.candidate_kind === "environment_override") {
      setPendingProcessingSource(null);
      setConsentPhase("environment_override");
      return;
    }
    setAuthorizingSource(source.source_id);
    setConnecting(true);
    try {
      const result = await imports.connectSelected(
        [source.source_id],
        [{ ...source }],
        { traceId: trace.traceId },
      );
      if (!result.accepted_source_ids.includes(source.source_id)) {
        trace.done("failed", { reason: "connect_not_accepted" });
        setRetryFailedConnection(true);
        const failure = result.failed_sources[source.source_id];
        setAuthorizationErrors((errors) => ({
          ...errors,
          [source.source_id]: failure
            ? `Could not turn processing on: ${failureLabel(failure)}`
            : "The selected sources were not approved for import; check and retry.",
        }));
        setConnectionError(
          result.failed_sources[source.source_id]
            ? `Could not turn processing on: ${failureLabel(result.failed_sources[source.source_id])}`
            : "The selected sources were not approved for import; check the per-source reasons and retry.",
        );
      } else {
        trace.done("succeeded", { count: 1 });
      }
    } catch {
      // The reason string already says `unknown`; the outcome must match, or the
      // log claims a failure the client cannot know about.
      trace.done("unknown", { reason: "connect_result_unknown" });
      setRetryFailedConnection(true);
      const failure = "The connection result is unconfirmed; check and retry.";
      setAuthorizationErrors((errors) => ({
        ...errors,
        [source.source_id]: failure,
      }));
      setConnectionError(`Could not turn processing on: ${failure}`);
    } finally {
      setAuthorizingSource(null);
      setPendingProcessingSource(null);
      setConnecting(false);
    }
  }

  const hostList = Array.isArray(recallHosts) ? recallHosts : [];
  /** Hosts a single bulk connect still has work to do on. */
  const bulkTargets = hostList.filter(
    (host) =>
      !host.installed ||
      needsRecallReconnect(host) ||
      host.permission?.receipt?.status === "install-only",
  );
  const hostBusy = busyHost !== null || bulkBusy;

  /** Allow one host everything on offer, exactly as its sheet would.
   *
   * The installation is re-read immediately before the write, so a connection
   * that changed between the preview and the write is reported rather than
   * overwritten with an approval the user gave for something else.
   */
  async function allowHost(hostId: string) {
    // An older desktop shell may not carry the permission bridge at all; then
    // the install still happened and the grant is simply left to the row.
    if (!hostCapabilities().permission) return;
    const status = await readHostPermission(hostId);
    const receipt = status?.receipt;
    if (!receipt || status?.reason !== null) return;
    if (status.state !== "install-only" || receipt.status === "approved")
      return;
    const selection = fullGrantSelection({
      sources: imports.sources,
      keep: receipt.allowedTools,
    });
    if (!selectionIsValid(selection)) return;
    await grantHostPermission({
      hostId,
      input: {
        installationId: receipt.installationId,
        expectedRevision: receipt.revision,
        allowedTools: selection.allowedTools,
        allowedScopes: selection.allowedScopes,
        userConfirmed: true,
      },
    });
  }

  /** Read what connecting every app would write, before writing any of it. */
  async function prepareBulk() {
    if (!window.oneMemory || bulkBusy || !bulkTargets.length) return;
    setBulkBusy(true);
    setBulkDone(null);
    setBulkErrors({});
    try {
      const steps: BulkStep[] = [];
      for (const host of bulkTargets) {
        const needsInstall = !host.installed || needsRecallReconnect(host);
        let install: RecallInstallPreview | null = null;
        let error: string | null = null;
        if (needsInstall) {
          try {
            install = await previewHostInstall(host.hostId);
          } catch (reason) {
            error = requiresHookRemoval(reason)
              ? "Remove this app's Sync Hook first"
              : "Could not read this app's connection change preview";
          }
        }
        steps.push({
          hostId: host.hostId,
          label: host.label,
          install,
          allow: true,
          error,
        });
      }
      setBulk(steps);
    } finally {
      setBulkBusy(false);
    }
  }

  /** Install and allow every prepared host, one at a time.
   *
   * Sequential rather than parallel: these are other programs' settings files,
   * and a half-finished batch is easier to describe when the writes are
   * ordered and each result is known before the next one starts.
   */
  async function runBulk() {
    if (!window.oneMemory || !bulk || bulkBusy) return;
    setBulkBusy(true);
    const errors: Record<string, string> = {};
    const done: string[] = [];
    try {
      for (const step of bulk) {
        if (step.error) continue;
        setBusyHost(step.hostId);
        try {
          if (step.install)
            await installHost({
              hostId: step.hostId,
              fingerprint: step.install.fingerprint,
            });
          if (step.allow) await allowHost(step.hostId);
          done.push(step.hostId);
        } catch (reason) {
          errors[step.hostId] = requiresHookRemoval(reason)
            ? "Remove this app's Sync Hook first"
            : (hostWriteReason(reason) ?? "Could not finish");
        } finally {
          setBusyHost(null);
        }
      }
      // Installed state and every receipt live on the desktop, so the whole
      // list is re-read rather than patched from what this run thinks it did.
      await rereadHosts();
    } catch {
      markHostsUnreadable();
    } finally {
      setBulkErrors(errors);
      setBulkDone(done);
      setBulkBusy(false);
    }
  }

  return (
    <section className="view sources-view" aria-labelledby="sources-title">
      <header className="view-header">
        <div>
          <h1 id="sources-title">Apps</h1>
          <p className="lede">
            Manage conversation sources and memory use permissions.
          </p>
        </div>
        <button
          type="button"
          disabled={
            actionBusy ||
            snapshotStale ||
            (needsConnection ? true : !syncable.length)
          }
          onClick={() => {
            if (!needsConnection) {
              void imports.syncAll();
            }
          }}
        >
          <RefreshCw size={16} aria-hidden="true" />
          {actionBusy
            ? "Starting..."
            : needsConnection
              ? "Turn on source authorization one by one"
              : "Sync sources"}
        </button>
      </header>

      <SourceSnapshotStatus
        snapshot={imports.sourceSnapshot}
        onRefresh={imports.refresh}
      />

      <div className="history-import-row" data-tour-id="history-import-row">
        <HistoryImport
          onConnect={imports.connectSelected}
          onRefresh={imports.refresh}
        />
      </div>
      <div className="coverage-line" aria-live="polite" hidden>
        {snapshotStale && imports.sourceSnapshot?.confirmedAt === null ? (
          <span>
            The source state has not been verified, so connected, connectable
            and active jobs cannot be confirmed.
          </span>
        ) : (
          <>
            {snapshotStale && <span>Recently confirmed: </span>}
            <strong>{connected.length}</strong> apps connected
            <span>·</span>
            <strong>{availableUnconnected.length}</strong> can connect
            <span>·</span>
            {snapshotStale ? (
              <>
                <strong>{imports.jobs.filter(isActive).length}</strong> active
                jobs pending verification
              </>
            ) : (
              <>
                {imports.jobs.some(
                  (job) => isActive(job) && !isSyncJob(job),
                ) && (
                  <span>
                    <strong>
                      {
                        imports.jobs.filter(
                          (job) => isActive(job) && !isSyncJob(job),
                        ).length
                      }
                    </strong>{" "}
                    importing history
                  </span>
                )}
                {imports.jobs.some(
                  (job) => isActive(job) && isSyncJob(job),
                ) && (
                  <span>
                    <strong>
                      {
                        imports.jobs.filter(
                          (job) => isActive(job) && isSyncJob(job),
                        ).length
                      }
                    </strong>{" "}
                    checking for additions
                  </span>
                )}
                {!imports.jobs.some(isActive) && <span>0 active jobs</span>}
              </>
            )}
          </>
        )}
      </div>

      {imports.error && !consentOpen ? (
        <p className="error-banner" role="alert">
          {imports.error}
        </p>
      ) : null}

      {imports.lastConnectResult ? (
        <section
          className="batch-results"
          aria-label="Bulk connect results"
          aria-live="polite"
        >
          <strong>This authorization and import result</strong>
          <ul>
            {imports.lastConnectResult.accepted_source_ids.map((sourceId) => (
              <li className="batch-success" key={sourceId}>
                <span>{sourceName(visibleSources, sourceId)}</span>
                <span>Authorized, import job started</span>
              </li>
            ))}
            {Object.entries(imports.lastConnectResult.failed_sources).map(
              ([sourceId, reason]) => (
                <li className="batch-failure" key={sourceId}>
                  <span>{sourceName(visibleSources, sourceId)}</span>
                  <span>Could not authorize: {failureLabel(reason)}</span>
                  <button
                    type="button"
                    className="quiet"
                    disabled={
                      actionBusy ||
                      !visibleSources.some(
                        (source) =>
                          source.source_id === sourceId && source.available,
                      )
                    }
                    onClick={() => {
                      const source = visibleSources.find(
                        (item) => item.source_id === sourceId,
                      );
                      if (!source) return;
                      setConsentSources([{ ...source }]);
                      setRetryFailedConnection(true);
                      setConnectionError(null);
                      setConsentPhase("general");
                    }}
                  >
                    <RefreshCw size={14} aria-hidden="true" />
                    Retry this source
                  </button>
                </li>
              ),
            )}
          </ul>
        </section>
      ) : null}

      <h2 className="app-section-title">Import sources</h2>
      <div className="source-table" aria-busy={imports.loading}>
        {visibleSources
          .slice()
          .sort(sourceOrder)
          .map((source) => {
            const job = imports.latestJobBySource.get(source.source_id);
            return (
              <article className="source-row" key={source.source_id}>
                <label className="source-choice">
                  <input
                    type="checkbox"
                    name={`source-${source.source_id}`}
                    aria-label={`Process ${source.display_name}`}
                    checked={source.authorized && source.sync_enabled === true}
                    aria-busy={pendingProcessingSource === source.source_id}
                    disabled={
                      (!source.available && !source.authorized) ||
                      snapshotStale ||
                      authorizingSource === source.source_id ||
                      pendingProcessingSource === source.source_id ||
                      imports.busyKey !== null
                    }
                    onChange={(event) => {
                      void toggleSourceProcessing(source, event.target.checked);
                    }}
                  />
                  <span>
                    <strong>{source.display_name}</strong>
                    <small>
                      {source.sync_enabled
                        ? "Processing: the history backlog and new conversations keep entering the memory pipeline"
                        : "Frozen: pending jobs are kept and new conversations are no longer scanned"}
                    </small>
                    <small title={source.candidate_path}>
                      {source.candidate_path}
                      {source.candidate_kind === "environment_override"
                        ? " · Environment variable override path"
                        : ""}
                    </small>
                  </span>
                </label>
                <p
                  className={`source-state state-${snapshotStale ? "neutral" : stateTone(source, job)}`}
                >
                  {snapshotStale
                    ? "Pending verification"
                    : stateLabel(source, job)}
                </p>
                <p className="source-counts">
                  {snapshotStale ? "Recently confirmed: " : ""}
                  {source.authorized
                    ? `Total ${source.imported_conversations} conversations · ${source.imported_messages} messages${job ? ` · ${job.stored} ${isSyncJob(job) ? "new from this sync" : "new from this import"}` : ""} · ${capabilityLabel(source)}`
                    : job
                      ? `${job.stored} ${isSyncJob(job) ? "new from this sync" : "new from this import"} · ${job.merged} already present · ${job.skipped} skipped`
                      : source.available
                        ? (foundSessionsLabel(source) ??
                          "Saved directory found")
                        : "Not found on this Mac"}
                  {job && job.errors > 0 ? ` · ${job.errors} issues` : ""}
                </p>
                <div className="source-row-actions">
                  {authorizationErrors[source.source_id] ? (
                    <span className="source-authorization-error" role="alert">
                      {authorizationErrors[source.source_id]}
                    </span>
                  ) : null}
                  {source.authorized &&
                    job?.status === "completed" &&
                    job.errors > 0 && (
                      <button
                        type="button"
                        className="secondary"
                        onClick={() =>
                          void imports.startImport(source.source_id)
                        }
                      >
                        Rescan
                      </button>
                    )}
                </div>
                <SourceDirectoryPicker
                  source={source}
                  disabled={
                    connecting ||
                    consentOpen ||
                    installPreview !== null ||
                    uninstallPreview !== null ||
                    hostBusy ||
                    imports.busyKey !== null ||
                    (directoryBusy !== null &&
                      directoryBusy !== source.source_id)
                  }
                  onBusyChange={(active) =>
                    setDirectoryBusy(active ? source.source_id : null)
                  }
                  onSaved={async () => {
                    setConsentSources([]);
                    setConsentPhase(null);
                    await imports.refresh();
                  }}
                />
                {job && job.status !== "completed" && (
                  <SourceJobActions
                    key={job.job_id}
                    job={job}
                    sourceLabel={source.display_name}
                    syncEnabled={source.sync_enabled === true}
                    snapshotStale={snapshotStale}
                    disabled={
                      consentOpen ||
                      installPreview !== null ||
                      uninstallPreview !== null ||
                      hostBusy ||
                      imports.busyKey !== null
                    }
                    onAction={imports.changeJob}
                  />
                )}
                <SourceImportReceipt source={source} />
              </article>
            );
          })}
      </div>

      <section className="host-recall-card" aria-live="polite">
        <div>
          <h2>Use memories</h2>
          <p>
            Read connections and import authorizations are managed separately
            for each app.
          </p>
          {bulkTargets.length > 0 && (
            <button
              type="button"
              className="secondary"
              disabled={
                actionBusy ||
                hostBusy ||
                consentOpen ||
                installPreview !== null ||
                uninstallPreview !== null
              }
              onClick={() => void prepareBulk()}
            >
              <ShieldCheck size={14} aria-hidden="true" />
              {bulkBusy && bulk === null
                ? "Preparing..."
                : `Connect all ${bulkTargets.length} apps in one click`}
            </button>
          )}
        </div>
        {recallHosts === "unavailable" ? (
          <small>Install this in the MemHub desktop app.</small>
        ) : recallHosts === "checking" ? (
          <small>Checking...</small>
        ) : recallHosts === "error" ? (
          <small>Cannot read the installation state.</small>
        ) : recallHosts.length === 0 ? (
          <small>No recall-capable app was detected on this machine.</small>
        ) : (
          <ul className="recall-host-list">
            {recallHosts.map((host) => (
              <li key={host.hostId}>
                <span>{host.label}</span>
                {/* The link's own verdict, kept apart from "installed".
                    Installed only means our files are intact, which was true
                    of the three hosts that pointed at a wrong discovery path
                    and recalled nothing. Worded so that green is the only
                    thing that reads as working. */}
                {needsRecallReconnect(host) ? (
                  <span className="recall-link recall-link-unreachable">
                    Link: needs reconnection
                  </span>
                ) : host.installed && host.selfCheck ? (
                  <span
                    className={`recall-link recall-link-${host.selfCheck.state}`}
                  >
                    Link:{" "}
                    {LINK_WORDING[host.selfCheck.state] ?? "Status unknown"}
                  </span>
                ) : null}
                {host.installed && host.skillDrift ? (
                  <span
                    className={`recall-skill recall-skill-${host.skillDrift.state}`}
                  >
                    Call rule:
                    {SKILL_WORDING[host.skillDrift.state] ?? "Status unknown"}
                  </span>
                ) : null}
                {host.installed &&
                !needsRecallReconnect(host) &&
                host.skillDrift &&
                host.skillDrift.state !== "in_sync" ? (
                  <button
                    type="button"
                    disabled={busyHost === host.hostId || bulkBusy}
                    onClick={async () => {
                      if (!hostCapabilities().skill) return;
                      setBusyHost(host.hostId);
                      try {
                        const preview = await previewHostSkill(host.hostId);
                        if (preview.upToDate) return;
                        if (preview.conflict) {
                          setSkillConflict({
                            hostId: host.hostId,
                            fingerprint: preview.fingerprint,
                          });
                          return;
                        }
                        setInstallPreview(preview);
                      } catch {
                        markHostsUnreadable();
                      } finally {
                        setBusyHost(null);
                      }
                    }}
                  >
                    Update call rule
                  </button>
                ) : null}
                {skillConflict?.hostId === host.hostId ? (
                  <span className="recall-skill-conflict">
                    You edited this call rule by hand. Updating will overwrite
                    your changes.
                    <button
                      type="button"
                      disabled={busyHost === host.hostId || bulkBusy}
                      onClick={async () => {
                        if (!hostCapabilities().skill || !skillConflict) return;
                        setBusyHost(host.hostId);
                        try {
                          await applyHostSkill({
                            hostId: host.hostId,
                            fingerprint: skillConflict.fingerprint,
                            overwrite: true,
                          });
                          setSkillConflict(null);
                          setHostActionFailed(false);
                        } catch {
                          markHostsUnreadable();
                        } finally {
                          setBusyHost(null);
                        }
                      }}
                    >
                      Overwrite anyway
                    </button>
                  </span>
                ) : null}
                {host.installed &&
                !needsRecallReconnect(host) &&
                host.selfCheck &&
                host.selfCheck.state !== "wired" ? (
                  <button
                    type="button"
                    disabled={busyHost === host.hostId || bulkBusy}
                    onClick={async () => {
                      if (!window.oneMemory) return;
                      setBusyHost(host.hostId);
                      try {
                        const check = await probeHost(host.hostId);
                        setLinkCheck((checks) => ({
                          ...checks,
                          [host.hostId]: check,
                        }));
                      } catch {
                        markHostsUnreadable();
                      } finally {
                        setBusyHost(null);
                      }
                    }}
                  >
                    Diagnose
                  </button>
                ) : null}
                {linkCheck[host.hostId]?.repoint ? (
                  <button
                    type="button"
                    disabled={busyHost === host.hostId || bulkBusy}
                    onClick={async () => {
                      if (!window.oneMemory) return;
                      setBusyHost(host.hostId);
                      try {
                        // The same preview install goes through. Repointing
                        // is an install with the right path, so it is shown
                        // and approved like any other edit to their file
                        // rather than written behind their back.
                        setInstallPreview(
                          await previewHostInstall(host.hostId),
                        );
                      } catch (error) {
                        if (requiresHookRemoval(error))
                          setHookRemovalRequired(host.hostId);
                        else markHostsUnreadable();
                      } finally {
                        setBusyHost(null);
                      }
                    }}
                  >
                    Repoint
                  </button>
                ) : null}
                {linkCheck[host.hostId] !== undefined &&
                linkCheck[host.hostId]?.repoint === null &&
                linkCheck[host.hostId]?.probe.state !== "wired" ? (
                  <span className="recall-link-detail">
                    {linkCheck[host.hostId]?.probe.detail}
                  </span>
                ) : null}
                <button
                  type="button"
                  disabled={busyHost === host.hostId || bulkBusy}
                  onClick={async () => {
                    if (!window.oneMemory) return;
                    setBusyHost(host.hostId);
                    setHookRemovalRequired(null);
                    setHostErrors((errors) => ({
                      ...errors,
                      [host.hostId]: "",
                    }));
                    try {
                      // Both directions edit this other program's own settings
                      // file, so both are shown first and applied only on
                      // approval. Disconnecting restoring what was there is
                      // still an edit to a file that is not ours, and the user
                      // may have added a server of their own since.
                      if (!host.installed || needsRecallReconnect(host)) {
                        setInstallPreview(
                          await previewHostInstall(host.hostId),
                        );
                        return;
                      }
                      setUninstallPreview(
                        await previewHostUninstall(host.hostId),
                      );
                    } catch (error) {
                      if (requiresHookRemoval(error))
                        setHookRemovalRequired(host.hostId);
                      else
                        setHostErrors((errors) => ({
                          ...errors,
                          [host.hostId]:
                            "Could not read the connection change preview. You can retry; no write was requested.",
                        }));
                    } finally {
                      setBusyHost(null);
                    }
                  }}
                >
                  {busyHost === host.hostId
                    ? "Connecting..."
                    : needsRecallReconnect(host)
                      ? "Reconnect in one click"
                      : host.installed
                        ? "Installed · click to disconnect"
                        : `Install to ${host.label} in one click`}
                </button>
                {hostErrors[host.hostId] && (
                  <p className="recall-link-detail" role="alert">
                    {hostErrors[host.hostId]}
                  </p>
                )}
                <RecallPermissionControl
                  host={host}
                  sources={imports.sources}
                  disabled={
                    hostBusy ||
                    consentOpen ||
                    installPreview !== null ||
                    uninstallPreview !== null
                  }
                  onChanged={async () => {
                    if (!window.oneMemory)
                      throw new Error(
                        "The desktop connection is unavailable; check again later.",
                      );
                    await rereadHosts();
                  }}
                />
                <SyncHookControl
                  host={host}
                  disabled={
                    hostBusy ||
                    consentOpen ||
                    installPreview !== null ||
                    uninstallPreview !== null
                  }
                />
                {hookRemovalRequired === host.hostId && (
                  <p role="alert" className="sync-hook-error">
                    Remove the Sync Hook before disconnecting the app
                    connection.
                  </p>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>
      <ConnectionFallback />
      <details className="application-coverage">
        <summary>Source support</summary>
        <table>
          <thead>
            <tr>
              <th>App / data</th>
              <th>History import</th>
              <th>Addition sync</th>
              <th>External reads</th>
            </tr>
          </thead>
          <tbody>
            {imports.sources.map((source) => {
              const host = Array.isArray(recallHosts)
                ? recallHosts.find((item) => item.hostId === source.source_id)
                : undefined;
              const file = source.source_id.endsWith("-export");
              return (
                <tr key={source.source_id}>
                  <th>{source.display_name}</th>
                  <td>
                    {file
                      ? "File import"
                      : source.available
                        ? "Found on this machine"
                        : "Not found on this machine"}
                  </td>
                  <td>
                    {file
                      ? "One-time snapshot"
                      : source.capabilities.live_sync === "implemented"
                        ? "File watch / polling"
                        : "Pending integration"}
                  </td>
                  <td>
                    {file
                      ? "No desktop Hook"
                      : host?.installed
                        ? "Installed, needs a real check"
                        : host?.present
                          ? "Installable"
                          : "Not verified"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </details>

      <p className="privacy-note">
        No reading of the screen, accessibility, keyboard, audio or clipboard.
        Each source's authorization can be revoked separately.
      </p>

      {installPreview ? (
        <InstallPreviewSheet
          preview={installPreview}
          busy={busyHost === installPreview.hostId}
          onCancel={() => setInstallPreview(null)}
          onApprove={async () => {
            if (!window.oneMemory) return;
            const { hostId, fingerprint } = installPreview;
            setBusyHost(hostId);
            try {
              await installHost({ hostId, fingerprint });
              setInstallPreview(null);
              // Requirement 8: one click completes the connection. Once the install lands, the call permission
              // is granted with the default full selection right away, instead of asking the user to click "Allow memory use" on the app row again.
              let granted = true;
              try {
                await allowHost(hostId);
              } catch {
                // The connection is already established; a failed grant only shows inline and does not change this connection's result.
                // The failed grant was already re-read once at the state layer, so it is not re-read here.
                granted = false;
              }
              if (granted)
                try {
                  await rereadHosts();
                } catch {
                  // A failed list refresh does not undo a connection or grant that already completed.
                }
            } catch (error) {
              if (requiresHookRemoval(error)) setHookRemovalRequired(hostId);
              else markHostsUnreadable();
              setInstallPreview(null);
            } finally {
              setBusyHost(null);
            }
          }}
        />
      ) : null}

      {bulk ? (
        <BulkConnectSheet
          steps={bulk}
          busy={bulkBusy}
          errors={bulkErrors}
          done={bulkDone}
          onCancel={() => setBulk(null)}
          onApprove={() => void runBulk()}
        />
      ) : null}

      {uninstallPreview ? (
        <UninstallPreviewSheet
          preview={uninstallPreview}
          busy={busyHost === uninstallPreview.hostId}
          onCancel={() => setUninstallPreview(null)}
          onApprove={async () => {
            if (!window.oneMemory) return;
            const { hostId, fingerprint } = uninstallPreview;
            setBusyHost(hostId);
            try {
              await uninstallHost({ hostId, fingerprint });
              setUninstallPreview(null);
            } catch (error) {
              if (requiresHookRemoval(error)) setHookRemovalRequired(hostId);
              else markHostsUnreadable();
              setUninstallPreview(null);
            } finally {
              setBusyHost(null);
            }
          }}
        />
      ) : null}

      {consentPhase ? (
        <div className="consent-backdrop">
          <section
            ref={consentPanel}
            className="consent-sheet"
            role="dialog"
            aria-modal="true"
            aria-busy={connecting}
            tabIndex={-1}
            aria-labelledby={
              consentPhase === "general"
                ? "consent-title"
                : "override-consent-title"
            }
            onKeyDown={(event) => {
              if (event.key === "Escape") {
                event.preventDefault();
                event.stopPropagation();
                if (!connecting) {
                  setConsentPhase(null);
                  consentTrace.current?.done("rejected", {
                    reason: "consent_dismissed",
                  });
                  consentTrace.current = null;
                }
              }
              if (event.key === "Tab") {
                const buttons =
                  consentPanel.current?.querySelectorAll<HTMLButtonElement>(
                    "button:not(:disabled)",
                  );
                const first = buttons?.[0],
                  last = buttons?.[buttons.length - 1];
                if (!first) {
                  event.preventDefault();
                  consentPanel.current?.focus();
                } else if (
                  event.shiftKey &&
                  (document.activeElement === first ||
                    document.activeElement === consentPanel.current)
                ) {
                  event.preventDefault();
                  last?.focus();
                } else if (!event.shiftKey && document.activeElement === last) {
                  event.preventDefault();
                  first.focus();
                }
              }
            }}
          >
            {consentPhase === "general" ? (
              <>
                <p className="eyebrow">First confirmation</p>
                <h2 id="consent-title">
                  Allow reading the text conversations of the selected sources?
                </h2>
                <p>
                  MemHub reads the text conversations already saved in the
                  sources below and keeps them on this machine. Model
                  processing, continuous sync and external app reads each need
                  separate authorization.
                </p>
                <ConsentSourceList sources={consentSources} />
                <p className="consent-exclusion">
                  No reading of the screen, keyboard, audio, clipboard or live
                  content of other apps.
                </p>
                {connectionError && (
                  <p className="error-banner" role="alert">
                    {connectionError}
                  </p>
                )}
                <div className="consent-actions">
                  <button
                    type="button"
                    className="secondary"
                    disabled={connecting}
                    onClick={() => setConsentPhase(null)}
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    disabled={connecting}
                    onClick={() => {
                      if (overrideSources.length) {
                        setConsentPhase("environment_override");
                        return;
                      }
                      void connectApproved();
                    }}
                  >
                    {connecting
                      ? "Connecting..."
                      : connectionError
                        ? "Retry connection"
                        : "Allow and start importing"}
                  </button>
                </div>
              </>
            ) : (
              <>
                <p className="eyebrow">
                  Second confirmation · Environment variable override
                </p>
                <h2 id="override-consent-title">
                  Confirm reading these custom paths?
                </h2>
                <p>
                  The directories below come from environment variables, not the
                  app's default save location. Confirm that the paths really
                  belong to the matching AI app.
                </p>
                <ConsentSourceList sources={overrideSources} />
                {connectionError && (
                  <p className="error-banner" role="alert">
                    {connectionError}
                  </p>
                )}
                <div className="consent-actions">
                  <button
                    type="button"
                    className="secondary"
                    disabled={connecting}
                    onClick={() => setConsentPhase("general")}
                  >
                    Back
                  </button>
                  <button
                    type="button"
                    disabled={connecting}
                    onClick={() => void connectApproved()}
                  >
                    {connecting
                      ? "Connecting..."
                      : connectionError
                        ? "Retry connection"
                        : "Confirm paths and authorize import"}
                  </button>
                </div>
              </>
            )}
          </section>
        </div>
      ) : null}
    </section>
  );
}

function ConsentSourceList({ sources }: { sources: ImportState["sources"] }) {
  return (
    <ul className="consent-source-list">
      {sources.map((source) => (
        <li key={source.source_id}>
          <strong>{source.display_name}</strong>
          <span>{source.candidate_path}</span>
        </li>
      ))}
    </ul>
  );
}

function sourceName(sources: ImportState["sources"], sourceId: string): string {
  return (
    sources.find((source) => source.source_id === sourceId)?.display_name ??
    sourceId
  );
}

function failureLabel(reason: string): string {
  const labels: Record<string, string> = {
    grant_not_available: "Grant record not created",
    environment_override_confirmation_required: "Custom paths not confirmed",
    source_not_available: "Data directory unavailable right now",
  };
  return labels[reason] ?? reason;
}

function isActive(job: { status: string }) {
  return ["pending", "running"].includes(job.status);
}

function isSyncJob(job: ImportState["jobs"][number]) {
  return job.operation === "sync";
}

function foundSessionsLabel(
  source: ImportState["sources"][number],
): string | null {
  // Only print a number the Core actually measured. An absent or null
  // session_count means "not measured", not zero, so it must never turn into a
  // number on screen. An installed host with nothing readable yet still owes
  // the person an honest sentence rather than a bare zero, which reads as a
  // broken scan.
  const count = source.session_count;
  if (count === null || count === undefined) return null;
  return count > 0
    ? `Found ${count} conversations`
    : "Installed, no conversations found yet";
}

function capabilityLabel(source: ImportState["sources"][number]) {
  if (source.sync_completeness === "one_time_snapshot")
    return "One-time snapshot; does not follow changes in the source app";
  // Deliberately no interval here: the poll period is a server setting
  // (live_sync_interval_seconds, default 60s) that the browser never sees, and
  // the number previously written here said 5 minutes — five times the truth.
  const following =
    source.sync_completeness === "full_dynamic"
      ? "Full dynamic sync"
      : "Additions only sync";
  return source.capabilities.live_sync === "implemented"
    ? following
    : "Manual import only";
}

function sourceOrder(
  left: ImportState["sources"][number],
  right: ImportState["sources"][number],
) {
  return (
    Number(right.authorized) - Number(left.authorized) ||
    Number(right.available) - Number(left.available) ||
    left.display_name.localeCompare(right.display_name)
  );
}

function stateLabel(
  source: ImportState["sources"][number],
  job: ImportState["jobs"][number] | undefined,
) {
  if (job && isSyncJob(job)) {
    const labels: Record<ImportState["jobs"][number]["status"], string> = {
      pending: "Waiting to check additions",
      running: "Checking for additions",
      paused: "Additions sync paused",
      interrupted: "Additions sync interrupted",
      failed: "Additions sync failed",
      completed:
        job.errors > 0
          ? "Additions sync has issues"
          : "Additions check completed",
      cancelled: "Additions sync cancelled",
    };
    return labels[job.status];
  }
  if (job?.status === "pending") return "Waiting to import";
  if (job?.status === "running") return "Importing";
  if (job?.status === "completed" && job.errors > 0) return "Import has issues";
  if (job?.status === "failed") return "Needs attention";
  if (job?.status === "paused" || job?.status === "interrupted")
    return "Progress kept";
  if (job?.status === "cancelled") return "Import cancelled";
  if (
    source.authorized &&
    job?.status === "completed" &&
    source.imported_messages === 0 &&
    job.stored === 0 &&
    job.skipped
  )
    return "No importable conversations were recognized";
  if (source.authorized) return "Connected";
  if (source.available) return "Can connect";
  return "Source location not found";
}

function stateTone(
  source: ImportState["sources"][number],
  job: ImportState["jobs"][number] | undefined,
) {
  if (
    job?.status === "failed" ||
    (job?.status === "completed" && job.errors > 0) ||
    (job?.status === "completed" &&
      !isSyncJob(job) &&
      source.imported_messages === 0 &&
      job.stored === 0 &&
      job.skipped)
  )
    return "warning";
  if (job && isActive(job)) return "active";
  if (source.authorized) return "success";
  return "neutral";
}

/** The edit an install would make, shown before it is made.
 *
 * Recall is installed by writing into another program's own settings file. The
 * user is the one who owns that file, so they see the lines that would change
 * — not a summary of them — and approve that specific edit.
 */
function InstallPreviewSheet({
  preview,
  busy,
  onCancel,
  onApprove,
}: {
  preview: RecallInstallPreview;
  busy: boolean;
  onCancel: () => void;
  onApprove: () => void;
}) {
  return (
    <div className="consent-backdrop">
      <section
        className="consent-sheet install-preview-sheet"
        role="dialog"
        aria-modal="true"
        aria-labelledby="install-preview-title"
        onKeyDown={(event) => {
          if (event.key === "Escape" && !busy) onCancel();
        }}
      >
        <p className="eyebrow">Confirm before installing</p>
        <h2 id="install-preview-title">
          This will change {preview.hostLabel}'s configuration
        </h2>
        <p>
          Installing recall writes into {preview.hostLabel} own configuration
          file. These are the lines that would change, and nothing is written
          until you confirm; you can disconnect and restore at any time.
        </p>
        <p className="consent-exclusion">
          Confirming also allows {preview.hostLabel} to use memories at the
          default scope (all tools and imported sources), so there is no second
          authorization to click; to narrow the scope, click "Adjust scope" in
          the app row after connecting.
        </p>
        <InstallFileList preview={preview} hostLabel={preview.hostLabel} />
        <div className="consent-actions">
          <button
            type="button"
            className="secondary"
            disabled={busy}
            onClick={onCancel}
          >
            Cancel
          </button>
          <button type="button" disabled={busy} onClick={onApprove}>
            {busy ? "Connecting..." : "Connect and allow memory use"}
          </button>
        </div>
      </section>
    </div>
  );
}

/** The lines one install would change, for one host.
 *
 * Shared by the single-host sheet and the bulk sheet: the same edit has to be
 * readable in both, because approving several installs at once is only the
 * same decision repeated, not a different one.
 */
function InstallFileList({
  preview,
  hostLabel,
}: {
  preview: RecallInstallPreview;
  hostLabel: string;
}) {
  const changed = preview.files.filter((file) => file.hunk !== null);
  if (!changed.length)
    return <p className="consent-exclusion">Nothing needs to change.</p>;
  return (
    <ul className="install-preview-files">
      {changed.map((file) => (
        <li key={file.path}>
          <div className="install-preview-file-head">
            <code>{file.path}</code>
            <span
              className={
                file.owner === "host"
                  ? "badge badge-warning"
                  : "badge badge-neutral"
              }
            >
              {file.owner === "host"
                ? file.existed
                  ? `${hostLabel}'s existing file · merged write`
                  : `${hostLabel}'s configuration · new`
                : file.existed
                  ? "MemHub file · replaced"
                  : "MemHub file · new"}
            </span>
          </div>
          {/* Lines wrap rather than scroll sideways: the path above already
              names this block, and a scroll region no keyboard can reach would
              hide the end of a long line. */}
          <pre className="install-preview-diff">
            <DiffLines tone="context" lines={file.hunk?.contextBefore} />
            <DiffLines tone="removed" lines={file.hunk?.removed} />
            <DiffLines tone="added" lines={file.hunk?.added} />
            <DiffLines tone="context" lines={file.hunk?.contextAfter} />
          </pre>
        </li>
      ))}
    </ul>
  );
}

/** Every app at once, in one confirmation.
 *
 * Connecting is per-app work: each one means a write into that program's own
 * settings file plus a grant, and doing it app by app meant the same two
 * approvals over and over. This shows what each install would write — the same
 * lines as the single-app sheet — and one button that does all of it.
 */
function BulkConnectSheet({
  steps,
  busy,
  errors,
  done,
  onCancel,
  onApprove,
}: {
  steps: BulkStep[];
  busy: boolean;
  errors: Record<string, string>;
  done: string[] | null;
  onCancel: () => void;
  onApprove: () => void;
}) {
  const pending = steps.filter((step) => !step.error);
  const skipped = steps.length - pending.length;
  const unfinished = steps.flatMap((step) => {
    const reason = step.error ?? errors[step.hostId];
    return reason ? [`${step.label} (${reason})`] : [];
  });
  return (
    <div className="consent-backdrop">
      <section
        className="consent-sheet install-preview-sheet"
        role="dialog"
        aria-modal="true"
        aria-busy={busy}
        aria-labelledby="bulk-connect-title"
        onKeyDown={(event) => {
          if (event.key === "Escape" && !busy) onCancel();
        }}
      >
        <p className="eyebrow">First confirmation</p>
        <h2 id="bulk-connect-title">Connect {pending.length} apps</h2>
        <p>
          Each app gets a read connection installed, and memory use is allowed
          with every tool and scope that can be verified right now. Below is
          what would be written, and nothing is written until you confirm; you
          can disconnect them one by one afterwards.
        </p>
        <ul className="bulk-host-list">
          {steps.map((step) => {
            const changed =
              step.install?.files.filter((file) => file.hunk !== null).length ??
              0;
            return (
              <li key={step.hostId}>
                <div className="bulk-host-head">
                  <strong>{step.label}</strong>
                  <span>
                    {step.error
                      ? `Skipped: ${step.error}`
                      : step.install
                        ? "Install the connection and allow memory use"
                        : "Installed, memory use allowed"}
                  </span>
                </div>
                {step.install ? (
                  <details>
                    <summary>Would change {changed} files</summary>
                    <InstallFileList
                      preview={step.install}
                      hostLabel={step.label}
                    />
                  </details>
                ) : null}
                {errors[step.hostId] && (
                  <p className="recall-link-detail" role="alert">
                    {errors[step.hostId]}. You can retry it alone or try again
                    later.
                  </p>
                )}
              </li>
            );
          })}
        </ul>
        {skipped > 0 && (
          <p className="consent-exclusion">
            {skipped} apps could not be prepared and were skipped; the other
            apps are unaffected.
          </p>
        )}
        {done && (
          <p role="status">
            {unfinished.length
              ? `${done.length} apps connected, ${unfinished.length} unfinished: ${unfinished.join(", ")}. You can retry each one below.`
              : `${done.length} apps connected.`}
          </p>
        )}
        <div className="consent-actions">
          <button
            type="button"
            className="secondary"
            disabled={busy}
            onClick={onCancel}
          >
            {done ? "Close" : "Cancel"}
          </button>
          {done ? null : (
            <button
              type="button"
              disabled={busy || pending.length === 0}
              onClick={onApprove}
            >
              {busy
                ? "Connecting..."
                : `Confirm connecting ${pending.length} apps`}
            </button>
          )}
        </div>
      </section>
    </div>
  );
}

/** The same review, for the edit that takes us back out.
 *
 * Disconnecting was the one operation that rewrote another program's
 * configuration with nothing shown first. It is a smaller edit than installing
 * but not a safer one: the host's file has been live since, and what comes back
 * is whatever was recorded at install time.
 */
function UninstallPreviewSheet({
  preview,
  busy,
  onCancel,
  onApprove,
}: {
  preview: RecallUninstallPreview;
  busy: boolean;
  onCancel: () => void;
  onApprove: () => void;
}) {
  const changed = preview.files.filter(
    (file) => file.hunk !== null || file.deleted,
  );
  return (
    <div className="consent-backdrop">
      <section
        className="consent-sheet install-preview-sheet"
        role="dialog"
        aria-modal="true"
        aria-labelledby="uninstall-preview-title"
        onKeyDown={(event) => {
          if (event.key === "Escape" && !busy) onCancel();
        }}
      >
        <p className="eyebrow">Confirm before disconnecting</p>
        <h2 id="uninstall-preview-title">
          This will restore {preview.hostLabel}'s configuration
        </h2>
        <p>
          Disconnecting undoes what the install wrote and restores how the file
          looked before. These are the lines that would change, and nothing is
          written until you confirm; you can install again afterwards.
        </p>
        {changed.length === 0 ? (
          <p className="consent-exclusion">Nothing needs to change.</p>
        ) : (
          <ul className="install-preview-files">
            {changed.map((file) => (
              <li key={file.path}>
                <div className="install-preview-file-head">
                  <code>{file.path}</code>
                  <span
                    className={
                      file.owner === "host"
                        ? "badge badge-warning"
                        : "badge badge-neutral"
                    }
                  >
                    {file.deleted
                      ? file.owner === "host"
                        ? `${preview.hostLabel}'s configuration · deleted (it did not exist before the install)`
                        : "MemHub file · deleted"
                      : file.owner === "host"
                        ? `${preview.hostLabel}'s existing file · restored`
                        : "MemHub file · restored"}
                  </span>
                </div>
                {file.deleted ? (
                  <p className="consent-exclusion">
                    The whole file is removed, with no empty directory left
                    behind.
                  </p>
                ) : (
                  <pre className="install-preview-diff">
                    <DiffLines
                      tone="context"
                      lines={file.hunk?.contextBefore}
                    />
                    <DiffLines tone="removed" lines={file.hunk?.removed} />
                    <DiffLines tone="added" lines={file.hunk?.added} />
                    <DiffLines tone="context" lines={file.hunk?.contextAfter} />
                  </pre>
                )}
              </li>
            ))}
          </ul>
        )}
        <div className="consent-actions">
          <button
            type="button"
            className="secondary"
            disabled={busy}
            onClick={onCancel}
          >
            Cancel
          </button>
          <button type="button" disabled={busy} onClick={onApprove}>
            {busy ? "Restoring..." : "Confirm disconnect"}
          </button>
        </div>
      </section>
    </div>
  );
}

/** One run of diff lines sharing a tone.
 *
 * Consecutive lines of the same kind are one element rather than one per line:
 * a run of added lines then reads as a single band, and there is no list to
 * key — configuration lines repeat, so their own text cannot identify them and
 * their position is the only thing left, which is not an identity at all.
 */
function DiffLines({
  tone,
  lines,
}: {
  tone: "context" | "removed" | "added";
  lines: string[] | undefined;
}) {
  if (!lines?.length) return null;
  const sign = tone === "removed" ? "-" : tone === "added" ? "+" : " ";
  return (
    <span className={`diff-${tone}`}>
      {lines.map((line) => `${sign} ${line}`).join("\n")}
    </span>
  );
}
