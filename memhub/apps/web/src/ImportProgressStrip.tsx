import type {
  ImportIssue,
  ImportJob,
  SourceResponse,
} from "@one-memory/contracts";
import { LoaderCircle, Pause, Play, RefreshCw, Settings2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import {
  type MemoryArrival,
  memoryArrivalTitle,
  resolveMemoryArrival,
} from "./memory-arrival-state";
import {
  blockReasonLabel,
  importIssueLabel,
  taskKindLabel,
} from "./memory-processing-labels";
import {
  type SourceJobActionsProps,
  settledAction,
  validReceipt,
} from "./SourceJobActions";
import "./import-progress-strip.css";
import { groupedCountLabel } from "./count-label";

const RUNNING = new Set(["running"]);
const PAUSABLE = new Set(["pending", "running"]);
const RESUMABLE = new Set(["paused", "interrupted"]);

type JobAction = "pause" | "resume";
type JobFailure = {
  action: JobAction;
  failures: { job: ImportJob; message: string }[];
};

/** What a job's stored status means, in the reader's own language.
 *
 * The receipts used to print `running` and `completed` between two job ids.
 * A reader looking at their own import does not need the pipeline's enum; they
 * need to know whether it finished. */
const JOB_STATUS_LABELS: Record<string, string> = {
  pending: "Waiting",
  running: "In progress",
  completed: "Completed",
  failed: "Failed",
  paused: "Paused",
  interrupted: "Interrupted",
  cancelled: "Stopped",
};

export function jobStatusLabel(status: string): string {
  return JOB_STATUS_LABELS[status] ?? status;
}

export interface ImportProgressStripProps {
  jobs: ImportJob[];
  sources: SourceResponse[];
  backfilling: boolean;
  onOpenSources(): void;
  sourceSnapshot?: { confirmedAt: string | null; stale: boolean };
  onRefresh?(): Promise<unknown>;
  arrival?: MemoryArrival;
  onConfigureModel?(): void;
  /** Pauses or continues one job and resolves with its receipt. */
  onJobAction?: SourceJobActionsProps["onAction"];
  issues?: readonly ImportIssue[];
  /** Settings keeps the status line and hides these two controls. */
  showSourceLink?: boolean;
  showProgressDetails?: boolean;
  userFacing?: boolean;
}

/** Named sources and confirmed counts, without claiming cached jobs are live. */
export function ImportProgressStrip({
  jobs,
  sources,
  backfilling,
  onOpenSources,
  sourceSnapshot,
  onRefresh,
  arrival,
  onConfigureModel,
  onJobAction,
  issues,
  showSourceLink = true,
  showProgressDetails = true,
  userFacing = false,
}: ImportProgressStripProps) {
  const [refreshing, setRefreshing] = useState(false);
  const [refreshError, setRefreshError] = useState<{
    epoch: number;
    message: string;
  } | null>(null);
  const flight = useRef(false);
  const [jobFlight, setJobFlight] = useState<JobAction | null>(null);
  const [jobFailure, setJobFailure] = useState<JobFailure | null>(null);
  const jobsInFlight = useRef(false);
  const mounted = useRef(false);
  const stale = sourceSnapshot?.stale === true;
  const confirmedAt = sourceSnapshot?.confirmedAt;
  const hasConfirmedTime =
    typeof confirmedAt === "string" && Number.isFinite(Date.parse(confirmedAt));
  const snapshotKey = JSON.stringify([confirmedAt ?? null, stale]);
  const snapshot = useRef({ key: snapshotKey, epoch: 0 });
  if (snapshot.current.key !== snapshotKey) {
    snapshot.current = { key: snapshotKey, epoch: snapshot.current.epoch + 1 };
  }
  const visibleError =
    stale && refreshError?.epoch === snapshot.current.epoch
      ? refreshError.message
      : null;

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  async function refresh() {
    if (flight.current || !onRefresh) return;
    const epoch = snapshot.current.epoch;
    flight.current = true;
    setRefreshing(true);
    setRefreshError(null);
    try {
      // The controller certifies source snapshots. Callback resolution is not proof.
      await onRefresh();
    } catch (reason) {
      if (mounted.current && snapshot.current.epoch === epoch) {
        setRefreshError({
          epoch,
          message: `Verification failed; status still pending verification.${
            reason instanceof Error && reason.message.trim()
              ? reason.message
              : "Try again later."
          }`,
        });
      }
    } finally {
      flight.current = false;
      if (mounted.current) setRefreshing(false);
    }
  }

  /** One job at a time, each checked against its own receipt. */
  async function changeJobs(action: JobAction, targets: ImportJob[]) {
    if (jobsInFlight.current || !onJobAction || !targets.length) return;
    jobsInFlight.current = true;
    setJobFlight(action);
    setJobFailure(null);
    const failures: JobFailure["failures"] = [];
    for (const target of targets) {
      try {
        const result = await onJobAction(target.job_id, action);
        if (!validReceipt(result, target))
          throw new Error("No valid receipt received for this job.");
        if (!settledAction(action, result))
          throw new Error("The backend has not confirmed this operation yet.");
      } catch (reason) {
        failures.push({
          job: target,
          message:
            reason instanceof Error && reason.message.trim()
              ? reason.message
              : "The operation result could not be confirmed.",
        });
      }
    }
    jobsInFlight.current = false;
    if (!mounted.current) return;
    setJobFlight(null);
    setJobFailure(failures.length ? { action, failures } : null);
  }

  const nameOf = (sourceId: string) =>
    sources.find((item) => item.source_id === sourceId)?.display_name ??
    sourceId;

  const latest = latestBySource(jobs);
  const running = latest.filter((item) => RUNNING.has(item.status));
  const runningHistory = running.filter((item) => !isSyncJob(item));
  const runningSync = running.filter(isSyncJob);
  const pending = latest.filter((item) => item.status === "pending");
  const failed = latest.filter((item) => item.status === "failed");
  const completedWithErrors = latest.filter(
    (item) => item.status === "completed" && item.errors > 0,
  );
  const paused = latest.filter((item) =>
    ["paused", "interrupted", "cancelled"].includes(item.status),
  );
  const history = latest.filter((item) => !isSyncJob(item));
  const pausable = history.filter((item) => PAUSABLE.has(item.status));
  const resumable = history.filter((item) => RESUMABLE.has(item.status));
  const jobAction: JobAction | null = pausable.length
    ? "pause"
    : resumable.length
      ? "resume"
      : null;
  const importPaused =
    !stale &&
    !pausable.length &&
    history.some((item) => item.status === "paused");
  const retryTargets = (jobFailure?.failures ?? []).flatMap(({ job: failed }) =>
    history.filter(
      (item) =>
        item.job_id === failed.job_id &&
        (jobFailure?.action === "pause" ? PAUSABLE : RESUMABLE).has(
          item.status,
        ),
    ),
  );
  const currentArrival = arrival
    ? resolveMemoryArrival({ arrival, processing: null, sources, loaded: true })
    : null;
  const needsAction =
    currentArrival &&
    [
      "model-required",
      "failed",
      "stopped",
      "idle",
      "checking",
      "unavailable",
    ].includes(currentArrival.phase);
  const generating = !stale && currentArrival?.phase === "processing";
  const importing = !stale && running.length > 0;
  const progressing =
    !stale &&
    !needsAction &&
    (importing ||
      backfilling ||
      generating ||
      currentArrival?.phase === "importing");

  if (userFacing) {
    const firstImport = !stale && runningHistory.length > 0;
    const actionable =
      currentArrival !== null &&
      ["model-required", "failed", "stopped", "unavailable"].includes(
        currentArrival.phase,
      );
    if (
      !firstImport &&
      !actionable &&
      !failed.length &&
      !completedWithErrors.length
    ) {
      return null;
    }
    const imported = sources.reduce(
      (sum, source) => sum + source.imported_conversations,
      0,
    );
    const message = firstImport
      ? `Importing your conversation history, ${groupedCountLabel(imported, "conversation")} imported so far. Memories will appear gradually; you can keep using the app.`
      : currentArrival?.phase === "model-required"
        ? "The model is not set up yet; new conversations will not be organized into memories for now."
        : "Some content failed to process; check Settings for details.";
    return (
      <div
        className="import-strip"
        role="status"
        aria-live="polite"
        data-feedback="compact"
      >
        <div className="import-strip-headline">
          <div className="import-strip-heading">
            {firstImport && (
              <span className="import-strip-pulse" aria-hidden="true" />
            )}
            <strong>{message}</strong>
          </div>
          {!firstImport && onConfigureModel && (
            <button
              type="button"
              className="secondary"
              onClick={onConfigureModel}
            >
              <Settings2 size={15} aria-hidden="true" />
              Go to settings
            </button>
          )}
        </div>
      </div>
    );
  }

  if (
    !stale &&
    !running.length &&
    !pending.length &&
    !failed.length &&
    !paused.length &&
    !completedWithErrors.length &&
    !backfilling &&
    (!currentArrival || currentArrival.phase === "idle")
  )
    return null;

  const knownSources = sources.filter(
    (source) =>
      source.authorized ||
      source.imported_messages > 0 ||
      latest.some((item) => item.source_id === source.source_id),
  );
  const confirmedMessages = knownSources.reduce(
    (sum, source) => sum + source.imported_messages,
    0,
  );
  const confirmedConversations = knownSources.reduce(
    (sum, source) => sum + source.imported_conversations,
    0,
  );
  const savedInLatestJobs = latest.reduce((sum, item) => sum + item.stored, 0);
  const canShowCounts =
    (!stale || hasConfirmedTime) &&
    (knownSources.length > 0 || latest.length > 0);
  const updating = stale ? refreshing : currentArrival?.refreshing === true;
  const title = stale
    ? updating
      ? "Saved content is kept; verifying progress"
      : "Progress needs updating"
    : currentArrival && needsAction
      ? currentArrival.phase === "model-required"
        ? "Waiting to resume"
        : memoryArrivalTitle(currentArrival)
      : generating
        ? "Memories are taking shape"
        : importing || currentArrival?.phase === "importing"
          ? running.length > 0 && running.every(isSyncJob)
            ? "Checking for new conversations"
            : "Importing conversations"
          : backfilling
            ? "Preparing to organize"
            : "Waiting to resume";

  return (
    <div
      className="import-strip"
      role="status"
      aria-live="polite"
      aria-busy={refreshing}
      data-stale={stale}
      data-feedback="compact"
    >
      <div className="import-strip-headline">
        <div className="import-strip-heading">
          {progressing && (
            <span className="import-strip-pulse" aria-hidden="true" />
          )}
          <strong>{title}</strong>
          {canShowCounts && (
            <span className="import-strip-counts">
              {confirmedMessages > 0
                ? `Saved ${groupedCountLabel(confirmedMessages, "original")} · ${groupedCountLabel(confirmedConversations, "conversation")}`
                : `${groupedCountLabel(savedInLatestJobs, "original")} saved this run`}
              {currentArrival?.memoryCount !== undefined &&
              Number.isFinite(currentArrival.memoryCount) &&
              currentArrival.memoryCount >= 0
                ? ` · ${groupedCountLabel(currentArrival.memoryCount, "memory")}`
                : ""}
            </span>
          )}
        </div>
        <span className="import-strip-support">
          {stale ||
          currentArrival?.phase === "unavailable" ||
          currentArrival?.phase === "checking"
            ? "Saved content is kept."
            : currentArrival?.phase === "model-required"
              ? "Check the organizing model; saved originals are kept on this device."
              : importPaused
                ? "Import paused; saved content is kept; continue anytime."
                : generating
                  ? "Recent conversations are being organized first; more memories will appear soon"
                  : currentArrival?.phase === "failed" ||
                      currentArrival?.phase === "stopped"
                    ? importing
                      ? "Originals are still importing; memory organizing is pending; saved content is kept."
                      : "Saved content is kept; organizing can continue."
                    : importing || currentArrival?.phase === "importing"
                      ? "Saving originals; import progress does not reflect memory generation progress."
                      : backfilling
                        ? "Preparing authorized conversations; no running generation job confirmed yet."
                        : "Saved content is kept."}
        </span>
      </div>

      {showProgressDetails && (
        <details className="memory-feedback-details import-strip-details">
          <summary>View progress and details</summary>
          <div className="import-strip-body">
            {stale && (
              <div className="import-strip-snapshot">
                <strong>Source status pending verification</strong>
                {hasConfirmedTime ? (
                  <>
                    <span className="import-strip-confirmed-time">
                      Last confirmed source status:&#32;
                      <time dateTime={confirmedAt}>
                        {new Date(confirmedAt as string).toLocaleString(
                          "zh-CN",
                          {
                            hour12: false,
                          },
                        )}
                      </time>
                    </span>
                    {knownSources.length > 0 && (
                      <span>
                        Last confirmed sources:&#32;
                        {knownSources
                          .map(
                            (item) =>
                              `${item.display_name} ${groupedCountLabel(item.imported_messages, "original")} · ${groupedCountLabel(item.imported_conversations, "conversation")}`,
                          )
                          .join("; ")}
                      </span>
                    )}
                    {latest.length > 0 && (
                      <span>
                        Job receipts:&#32;
                        {latest
                          .map(
                            (item) =>
                              `${nameOf(item.source_id)} ${isSyncJob(item) ? "New-item sync" : "History import"}, scanned ${groupedCountLabel(item.discovered, "record")}, ${jobSummary(item, isSyncJob(item))}`,
                          )
                          .join("; ")}
                      </span>
                    )}
                  </>
                ) : (
                  <span>
                    No confirmed source status yet; refresh to verify.
                  </span>
                )}
                {refreshing && <span>Checking source status...</span>}
                {visibleError && (
                  <span className="import-strip-refresh-error">
                    {visibleError}
                  </span>
                )}
              </div>
            )}

            {!stale &&
              [runningHistory, runningSync].map((group, index) => {
                if (!group.length) return null;
                const sync = index === 1;
                const totals = group.reduce(
                  (sum, item) => ({
                    discovered: sum.discovered + item.discovered,
                    stored: sum.stored + item.stored,
                    merged: sum.merged + item.merged,
                    skipped: sum.skipped + item.skipped,
                    errors: sum.errors + item.errors,
                  }),
                  {
                    discovered: 0,
                    stored: 0,
                    merged: 0,
                    skipped: 0,
                    errors: 0,
                  },
                );
                return (
                  <span
                    className="import-strip-operation"
                    key={sync ? "sync" : "history"}
                  >
                    <span>
                      {sync ? "Checking " : "Importing "}
                      {group.map((item) => nameOf(item.source_id)).join(", ")}
                      {sync ? " for new conversations" : " history"}
                      {totals.discovered > 0
                        ? `, scanned ${groupedCountLabel(totals.discovered, "record")}`
                        : ""}
                      .
                    </span>
                    <span>{jobSummary(totals, sync)}</span>
                  </span>
                );
              })}

            {!stale && !importing && backfilling && (
              <span>
                Preparing organizing jobs for authorized conversations.
              </span>
            )}

            {!stale &&
              pending.map((item) => (
                <span key={item.job_id}>
                  {nameOf(item.source_id)}
                  {isSyncJob(item)
                    ? " new-item sync waiting to check"
                    : " history waiting to import"}
                  {" · "}
                  {jobSummary(item, isSyncJob(item))}
                </span>
              ))}

            {generating && (currentArrival?.backlogPending ?? 0) > 0 && (
              <span>
                Large backlog; organizing in background batches to preserve
                quality
              </span>
            )}

            {failed.map((item) => (
              <span className="import-strip-failed" key={item.source_id}>
                {stale ? "Last job receipt: " : ""}
                {nameOf(item.source_id)}
                {isSyncJob(item)
                  ? " new-item sync failed"
                  : " failed to read through history"}
                .{!stale && jobSummary(item, isSyncJob(item))}
              </span>
            ))}

            {paused.map((item) => (
              <span className="import-strip-failed" key={item.source_id}>
                {stale ? "Last job receipt: " : ""}
                {nameOf(item.source_id)}
                {item.status === "cancelled"
                  ? " stopped, "
                  : isSyncJob(item)
                    ? item.status === "paused"
                      ? " new-item sync paused, "
                      : " new-item sync interrupted, "
                    : " history import not finished,"}
                {stale
                  ? "Go to Sources to verify."
                  : isSyncJob(item)
                    ? "New-item checks can continue."
                    : "Import can continue."}
                {!stale && jobSummary(item, isSyncJob(item))}
              </span>
            ))}

            {completedWithErrors.map((item) => (
              <span className="import-strip-failed" key={item.source_id}>
                {stale ? "Last job receipt: " : ""}
                {nameOf(item.source_id)}
                {isSyncJob(item)
                  ? " this new-item check "
                  : " this history check "}
                finished, but read issues remain
                {stale ? "." : `; ${jobSummary(item, isSyncJob(item))}.`}
              </span>
            ))}

            {!stale && knownSources.length > 0 && (
              <span className="import-strip-total">
                Total saved:&#32;
                {knownSources
                  .map(
                    (item) =>
                      `${item.display_name} ${groupedCountLabel(item.imported_messages, "original")} · ${groupedCountLabel(item.imported_conversations, "conversation")}`,
                  )
                  .join("; ")}
              </span>
            )}

            {(issues?.length ?? 0) > 0 && (
              <ul className="import-strip-receipts">
                {(issues ?? []).map((issue) => (
                  <li key={`${issue.job_id}:${issue.issue_id}`}>
                    {issue.kind === "error" ? "read issue" : "skipped record"}
                    {" · "}
                    {importIssueLabel(issue.reason)}
                  </li>
                ))}
              </ul>
            )}

            {(currentArrival?.blockedStages?.length ?? 0) > 0 && (
              <ul className="import-strip-receipts">
                {(currentArrival?.blockedStages ?? []).map((block) => (
                  <li key={`${block.kind}:${block.reason}`}>
                    {taskKindLabel(block.kind)} ·{" "}
                    {blockReasonLabel(block.reason)} ·{" "}
                    {groupedCountLabel(block.count, "item")}
                  </li>
                ))}
              </ul>
            )}

            {jobs.length > 0 && (
              <ul
                className="import-strip-receipts"
                aria-label="Full job receipts"
              >
                {jobs.map((item) => (
                  <li key={item.job_id}>
                    {nameOf(item.source_id)} ·{" "}
                    {isSyncJob(item) ? "New-item sync" : "History import"} ·{" "}
                    {jobStatusLabel(item.status)} ·{" "}
                    {jobSummary(item, isSyncJob(item))}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </details>
      )}

      {!stale && !jobFlight && jobFailure && (
        <p
          className="import-strip-refresh-error import-strip-job-error"
          role="alert"
        >
          {jobFailure.failures
            .map(
              ({ job: failed, message }) =>
                `${nameOf(failed.source_id)} could not ${jobFailure.action === "pause" ? "pause" : "resume"}: ${message}`,
            )
            .join("; ")}
        </p>
      )}
      <div className="import-strip-actions">
        {!stale && onJobAction && (jobFlight || jobAction) && (
          <button
            type="button"
            className="quiet import-strip-job"
            disabled={jobFlight !== null}
            onClick={() =>
              jobAction &&
              void changeJobs(
                jobAction,
                jobAction === "pause" ? pausable : resumable,
              )
            }
          >
            {jobFlight ? (
              <LoaderCircle size={16} aria-hidden="true" />
            ) : jobAction === "pause" ? (
              <Pause size={16} aria-hidden="true" />
            ) : (
              <Play size={16} aria-hidden="true" />
            )}
            {jobFlight === "pause"
              ? "Pausing..."
              : jobFlight === "resume"
                ? "Resuming..."
                : jobAction === "pause"
                  ? "Pause all"
                  : "Continue all"}
          </button>
        )}
        {!stale && !jobFlight && jobFailure && retryTargets.length > 0 && (
          <button
            type="button"
            className="quiet import-strip-job"
            onClick={() => void changeJobs(jobFailure.action, retryTargets)}
          >
            <RefreshCw size={16} aria-hidden="true" />
            {jobFailure.action === "pause" ? "Retry pause" : "Retry resume"}
          </button>
        )}
        {!stale &&
          currentArrival?.phase === "model-required" &&
          onConfigureModel && (
            <button type="button" className="quiet" onClick={onConfigureModel}>
              <Settings2 size={16} aria-hidden="true" />
              Set up model
            </button>
          )}
        {!stale &&
          currentArrival &&
          ["failed", "stopped"].includes(currentArrival.phase) &&
          onConfigureModel && (
            <button type="button" className="quiet" onClick={onConfigureModel}>
              <Settings2 size={16} aria-hidden="true" />
              Continue organizing
            </button>
          )}
        {stale && onRefresh && (
          <button
            type="button"
            className="quiet import-strip-refresh"
            title="Refresh and verify"
            disabled={refreshing}
            onClick={() => void refresh()}
          >
            <RefreshCw size={16} aria-hidden="true" />
            Refresh and verify
          </button>
        )}
        {showSourceLink && (
          <button type="button" className="quiet" onClick={onOpenSources}>
            View sources
          </button>
        )}
      </div>
    </div>
  );
}

/** A late update to an old resumed job does not make it the newest source job. */
function latestBySource(jobs: ImportJob[]): ImportJob[] {
  const newest = new Map<string, ImportJob>();
  for (const item of jobs) {
    const held = newest.get(item.source_id);
    if (
      !held ||
      timestamp(item.created_at) > timestamp(held.created_at) ||
      (timestamp(item.created_at) === timestamp(held.created_at) &&
        timestamp(item.updated_at) >= timestamp(held.updated_at))
    )
      newest.set(item.source_id, item);
  }
  return [...newest.values()];
}

function timestamp(value: string) {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : Number.NEGATIVE_INFINITY;
}

function isSyncJob(job: ImportJob) {
  return job.operation === "sync";
}

/** What a job or a group of jobs actually saved, in the reader's own language.
 *
 * A sync that found nothing new and a history import that saved nothing are
 * different sentences, so the caller says which one it is. */
function jobSummary(
  totals: {
    stored: number;
    merged: number;
    skipped: number;
    errors: number;
  },
  sync = false,
) {
  return `${sync ? `${groupedCountLabel(totals.stored, "new original")} this run` : `${groupedCountLabel(totals.stored, "original")} saved`}${totals.merged > 0 ? ` · ${totals.merged.toLocaleString("zh-CN")} already existed` : ""}${totals.skipped > 0 ? ` · ${totals.skipped.toLocaleString("zh-CN")} skipped` : ""}${totals.errors > 0 ? ` · ${groupedCountLabel(totals.errors, "issue")}` : ""}`;
}
