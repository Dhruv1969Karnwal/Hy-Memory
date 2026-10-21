import { Download, Flag, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  getDetailedLogging,
  getMemoryProcessing,
  setDetailedLogging,
} from "./api";
import {
  type BackgroundSnapshot,
  readBackgroundHealth,
} from "./background-health-read";
import { clockStamp } from "./clock-stamp";
import { countLabel } from "./count-label";
import { exportDiagnosticsFile } from "./diagnostics-export";
import type { RecallHostStatus } from "./global";
import { refreshRecallHosts } from "./host-state";
import {
  blockReasonLabel,
  retryingBlockLabel,
  splitBlockedStages,
  taskKindLabel,
} from "./memory-processing-labels";
import { subscribePoll } from "./poll-scheduler";
import { R6Button, R6Switch } from "./R6Controls";
import { LINK_WORDING } from "./recall-host-wording";
import type { useImportSources } from "./use-import-sources";

type ImportState = ReturnType<typeof useImportSources>;

type Reading<T> = {
  value: T | null;
  pending: boolean;
  confirmedAt: string | null;
  failedAt: string | null;
};

function useDiagnosticRead<T>(load: () => Promise<T>, pollMs?: number) {
  const [reading, setReading] = useState<Reading<T>>({
    value: null,
    pending: true,
    confirmedAt: null,
    failedAt: null,
  });
  const alive = useRef(false);
  const version = useRef(0);
  const pending = useRef(false);
  const read = useCallback(async () => {
    const request = ++version.current;
    pending.current = true;
    setReading((previous) => ({ ...previous, pending: true }));
    try {
      const value = await load();
      if (!alive.current || request !== version.current) return;
      setReading({
        value,
        pending: false,
        confirmedAt: new Date().toISOString(),
        failedAt: null,
      });
    } catch {
      if (!alive.current || request !== version.current) return;
      setReading((previous) => ({
        ...previous,
        pending: false,
        failedAt: new Date().toISOString(),
      }));
    } finally {
      if (alive.current && request === version.current) pending.current = false;
    }
  }, [load]);

  useEffect(() => {
    alive.current = true;
    void read();
    const stop =
      pollMs === undefined
        ? () => {}
        : subscribePoll({
            key: "health",
            intervalMs: pollMs,
            run: () => {
              if (!pending.current) return read();
              return undefined;
            },
          });
    return () => {
      alive.current = false;
      ++version.current;
      stop();
    };
  }, [read, pollMs]);
  return { ...reading, read };
}

async function readHosts() {
  if (!window.oneMemory) return "no-bridge" as const;
  return (await refreshRecallHosts()).filter((host) => host.present);
}

function ReadTime({ label, value }: { label: string; value: string }) {
  return (
    <p className="health-stamp">
      {label}
      <time dateTime={value}>{clockStamp(value)}</time>
    </p>
  );
}

function ReadReceipt({
  pending,
  confirmedAt,
  failedAt,
}: Pick<Reading<unknown>, "pending" | "confirmedAt" | "failedAt">) {
  return (
    <>
      {failedAt ? (
        <p role="status">
          Temporarily unavailable
          {confirmedAt
            ? " · Data is stale; below is the last successful read."
            : "."}
        </p>
      ) : null}
      {pending ? (
        <p role="status">
          {confirmedAt
            ? "Reloading status; below is the last successful read..."
            : "Reading status..."}
        </p>
      ) : null}
      {confirmedAt ? (
        <ReadTime label="Last successful read" value={confirmedAt} />
      ) : null}
      {failedAt ? <ReadTime label="Failed read at" value={failedAt} /> : null}
    </>
  );
}

/** Everything needed to answer "is it working" in one place.
 *
 * All four readings already existed, on three different screens, and none of
 * them is conclusive alone: the service can be up while the pipeline is
 * stalled, the pipeline can be running while every host reads nothing. The
 * cost of them being apart is that nobody thinks to put them together.
 *
 * Aggregation only — no endpoint is added, and every word here comes from
 * the table the originating screen already uses.
 */
export function HealthPanel({ imports }: { imports: ImportState }) {
  const [detailedLogging, setDetailedLoggingState] = useState(false);
  const [detailedLoggingPending, setDetailedLoggingPending] = useState(false);
  const backgroundRead = useDiagnosticRead(readBackgroundHealth);
  const processingRead = useDiagnosticRead(getMemoryProcessing, 5_000);
  const hostsRead = useDiagnosticRead(readHosts);
  const processing = processingRead.value;
  const hosts = hostsRead.value;
  const [refreshing, setRefreshing] = useState(false);
  const [diagnosticAction, setDiagnosticAction] = useState<
    "incident" | "export" | null
  >(null);
  const [diagnosticMessage, setDiagnosticMessage] = useState("");
  const [sourceFailedAt, setSourceFailedAt] = useState<string | null>(null);
  const refreshFlight = useRef(false);
  const lifecycle = useRef(0);

  useEffect(() => {
    void getDetailedLogging()
      .then(setDetailedLoggingState)
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    ++lifecycle.current;
    return () => {
      ++lifecycle.current;
    };
  }, []);

  async function refresh() {
    if (refreshFlight.current) return;
    refreshFlight.current = true;
    setRefreshing(true);
    const session = lifecycle.current;
    const sourceRead = async () => {
      try {
        await imports.refresh();
        if (session === lifecycle.current) setSourceFailedAt(null);
      } catch {
        if (session === lifecycle.current)
          setSourceFailedAt(new Date().toISOString());
      }
    };
    await Promise.all([
      backgroundRead.read(),
      processingRead.read(),
      hostsRead.read(),
      sourceRead(),
    ]);
    if (session !== lifecycle.current) return;
    refreshFlight.current = false;
    setRefreshing(false);
  }

  async function markIncident() {
    if (!window.oneMemory?.markDiagnosticIncident) {
      setDiagnosticMessage("Record the issue time in the desktop app.");
      return;
    }
    setDiagnosticAction("incident");
    setDiagnosticMessage("");
    try {
      const description = window.prompt(
        "Briefly describe the issue you just hit (optional)",
        "",
      );
      if (description === null) {
        setDiagnosticAction(null);
        return;
      }
      const receipt =
        await window.oneMemory.markDiagnosticIncident(description);
      await window.oneMemory.copyText?.(receipt.incident_id);
      setDiagnosticMessage(`Incident ID copied: ${receipt.incident_id}`);
    } catch {
      setDiagnosticMessage("The issue time was not recorded; try again.");
    } finally {
      setDiagnosticAction(null);
    }
  }

  async function exportDiagnostics() {
    if (!window.oneMemory?.exportDiagnostics) {
      setDiagnosticMessage("Export diagnostics in the desktop app.");
      return;
    }
    setDiagnosticAction("export");
    setDiagnosticMessage("");
    try {
      const result = await exportDiagnosticsFile();
      setDiagnosticMessage(result.message);
    } finally {
      setDiagnosticAction(null);
    }
  }

  // The shared snapshot distinguishes read failures from unrelated action errors.
  const sourceUnavailable =
    imports.sourceSnapshot?.stale ??
    (Boolean(imports.error) || sourceFailedAt !== null);
  const sourceConfirmedAt = imports.sourceSnapshot?.confirmedAt;
  const sourceReceipt = (
    <>
      {imports.loading ? (
        <p role="status">Reading local service and source status...</p>
      ) : sourceUnavailable ? (
        <p role="status">
          Temporarily unavailable
          {sourceConfirmedAt || imports.health || imports.sources.length
            ? " · Data is stale; below is the last successful read."
            : "."}
        </p>
      ) : null}
      {sourceConfirmedAt ? (
        <ReadTime label="Last successful read" value={sourceConfirmedAt} />
      ) : null}
      {sourceUnavailable && sourceFailedAt ? (
        <ReadTime label="Failed read at" value={sourceFailedAt} />
      ) : null}
    </>
  );

  // Retries are summed rather than listed. This panel is where a developer
  // looks, and even here the pipeline's internal guard names are not what a
  // person reads: the count says how much is in flight, and the code that
  // failed is on the task row in Core.
  const retrying = processing
    ? splitBlockedStages(processing.blocked_stages).retrying
    : 0;

  return (
    <section className="view health-view" aria-labelledby="health-title">
      <header className="view-header">
        <div>
          <h1 id="health-title">Diagnostics</h1>
        </div>
      </header>

      <div className="health-actions health-actions-primary">
        <R6Button
          variant="primary"
          pending={diagnosticAction === "export"}
          onClick={() => void exportDiagnostics()}
        >
          <Download size={16} aria-hidden="true" />
          Export logs (send to developers for debugging)
        </R6Button>
      </div>

      <section className="health-block" aria-label="Local service">
        <h2>Local service</h2>
        {sourceReceipt}
        {imports.health ? (
          <p>
            {`${sourceUnavailable || imports.loading ? "Last read" : "Running"} · Version ${imports.health.version} · Instance ${imports.health.instance_id}`}
          </p>
        ) : !sourceUnavailable && !imports.loading ? (
          <p>Not connected to the local Core yet.</p>
        ) : null}
      </section>

      <section className="health-block" aria-label="Background services">
        <h2>Background services</h2>
        {backgroundRead.value === "no-bridge" ? (
          <p>
            Background services cannot be checked in this environment; view them
            in a desktop app that supports this feature.
          </p>
        ) : (
          <>
            <ReadReceipt {...backgroundRead} />
            {backgroundRead.value ? (
              <>
                <p>
                  {backgroundRead.pending || backgroundRead.failedAt
                    ? "Last read: "
                    : ""}
                  {backgroundVerdict(backgroundRead.value.state)}
                </p>
                <p>
                  Core:{runtimeStateLabel(backgroundRead.value.core)} · Host
                  bridge: {runtimeStateLabel(backgroundRead.value.broker)}
                </p>
              </>
            ) : null}
          </>
        )}
      </section>

      <section className="health-block" aria-label="Per-source totals">
        <h2>Per-source totals</h2>
        {sourceReceipt}
        {imports.sources.length ? (
          <dl>
            {imports.sources.map((source) => (
              <div key={source.source_id}>
                <dt>{source.display_name}</dt>
                <dd>
                  {`${countLabel(source.imported_conversations, "conversation")} · ${countLabel(source.imported_messages, "message")}`}
                </dd>
              </div>
            ))}
          </dl>
        ) : !sourceUnavailable && !imports.loading ? (
          <p>No sources connected yet.</p>
        ) : null}
      </section>

      <section className="health-block" aria-label="Memory processing">
        <h2>Memory processing</h2>
        <ReadReceipt {...processingRead} />
        {processing ? (
          <>
            <p>
              {`${processing.running} running · ${processing.pending} queued · ${processing.completed} completed · ${processing.failed} failed`}
            </p>
            {processing.metrics ? (
              <dl aria-label="Import and organizing details">
                <div>
                  <dt>Source text and full conversation turns</dt>
                  <dd>{`${countLabel(processing.metrics.raw_evidence, "original")} · ${countLabel(processing.metrics.complete_turns, "complete turn")}`}</dd>
                </div>
                <div>
                  <dt>HY organizing</dt>
                  <dd>{`${countLabel(processing.metrics.hy_calls, "call")} · ${processing.metrics.completed} done · ${countLabel(processing.metrics.unknown, "uncertain result")} · ${processing.metrics.failed} failed · ${processing.metrics.cancelled} cancelled`}</dd>
                </div>
                <div>
                  <dt>Output</dt>
                  <dd>{`${countLabel(processing.metrics.governed_memories, "memory")}`}</dd>
                </div>
                <div>
                  <dt>WorkBuddy memory lookups</dt>
                  <dd>{`${countLabel(processing.metrics.workbuddy.injected, "injection")} · ${countLabel(processing.metrics.workbuddy.timed_out, "timeout")} · ${processing.metrics.workbuddy.unavailable} unavailable`}</dd>
                </div>
              </dl>
            ) : null}
            {processing.blocked_stages.length ? (
              <ul>
                {splitBlockedStages(processing.blocked_stages).named.map(
                  (block) => (
                    <li key={`${block.kind}:${block.reason}`}>
                      {`${taskKindLabel(block.kind)} stuck: ${blockReasonLabel(block.reason)} (${countLabel(block.count, "item")})`}
                    </li>
                  ),
                )}
                {retrying ? <li>{retryingBlockLabel(retrying)}</li> : null}
              </ul>
            ) : !processingRead.failedAt && !processingRead.pending ? (
              <p>Nothing needs attention right now.</p>
            ) : null}
          </>
        ) : null}
      </section>

      <section className="health-block" aria-label="AI connections">
        <h2>AI connections</h2>
        {/* Host state lives in the desktop shell. In a browser it is not
            false, it is unavailable, and reporting it as not connected would send
            the user to repair links that may well be fine. */}
        {hosts === "no-bridge" || !window.oneMemory ? (
          <p>Not readable in the browser; view it in the desktop app.</p>
        ) : (
          <>
            <ReadReceipt {...hostsRead} />
            {hosts?.length ? (
              <dl>
                {hosts.map((host) => (
                  <div key={host.hostId}>
                    <dt>{host.label}</dt>
                    <dd>
                      {hostsRead.failedAt || hostsRead.pending
                        ? "Last read: "
                        : ""}
                      {hostVerdict(host)}
                    </dd>
                  </div>
                ))}
              </dl>
            ) : hosts && !hostsRead.failedAt && !hostsRead.pending ? (
              <p>No connectable AI tools were detected on this machine.</p>
            ) : null}
          </>
        )}
      </section>
      <section className="health-block" aria-label="Detailed logs">
        <div className="health-setting-row">
          <div>
            <h2>Detailed logs (for debugging)</h2>
            <p>
              Off by default. Turn it on only when a developer asks you to help
              debug.
            </p>
          </div>
          <R6Switch
            label="Detailed logs (for debugging)"
            checked={detailedLogging}
            pending={detailedLoggingPending}
            onCheckedChange={(enabled) => {
              setDetailedLoggingPending(true);
              void setDetailedLogging(enabled)
                .then(setDetailedLoggingState)
                .catch(() =>
                  setDiagnosticMessage(
                    "Failed to save the verbose logging setting; try again.",
                  ),
                )
                .finally(() => setDetailedLoggingPending(false));
            }}
          />
        </div>
      </section>
      <div className="health-actions">
        <R6Button
          variant="primary"
          pending={refreshing}
          onClick={() => void refresh()}
        >
          <RefreshCw size={16} aria-hidden="true" />
          Reload status
        </R6Button>
        <R6Button
          variant="quiet"
          pending={diagnosticAction === "incident"}
          onClick={() => void markIncident()}
        >
          <Flag size={16} aria-hidden="true" />
          Record issue time
        </R6Button>
      </div>
      {diagnosticMessage ? (
        <p className="health-diagnostic-message" role="status">
          {diagnosticMessage}
        </p>
      ) : null}
    </section>
  );
}

/** A host with no self-check on record has not been shown to work, and the
 * install flow's own wording for that is not connected. */
function hostVerdict(host: RecallHostStatus): string {
  if (!host.installed) return LINK_WORDING.unreachable;
  const state = host.selfCheck?.state;
  return (state ? LINK_WORDING[state] : undefined) ?? LINK_WORDING.unreachable;
}

function backgroundVerdict(state: BackgroundSnapshot["state"]): string {
  return {
    healthy: "Background services available",
    stopped: "Background services stopped",
    degraded: "Background services partially unavailable",
    untrusted: "Background service identity failed verification",
    unknown: "Background status not confirmed yet",
  }[state];
}

function runtimeStateLabel(
  state: BackgroundSnapshot["core"] | BackgroundSnapshot["broker"],
): string {
  return {
    running: "Running",
    stopped: "Stopped",
    unhealthy: "Cannot connect",
    untrusted: "Identity failed verification",
    unknown: "Not confirmed yet",
    not_checked: "Not checked",
  }[state];
}
