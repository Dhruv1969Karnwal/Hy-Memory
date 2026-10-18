import type { ImportJob, SourceResponse } from "@one-memory/contracts";
import { Pause, Play } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { connectionHostId } from "./connection-hosts";
import { R6ColdStartFrame } from "./R6ColdStartFrame";
import { R6Button } from "./R6Controls";
import { R6ImportProgress } from "./R6ImportProgress";
import "./first-import-r6.css";

export function importHostEligible(
  sources: readonly SourceResponse[],
  sourceIds: readonly string[],
) {
  const ids = new Set(sourceIds);
  return sources.filter(
    (source) =>
      ids.has(source.source_id) &&
      !source.source_id.startsWith("deepseek-harness") &&
      source.source_id !== "openclaw" &&
      (connectionHostId(source.source_id) !== null ||
        (source.capabilities.installer === "implemented" &&
          source.capabilities.recall === "implemented")),
  );
}

export function defaultHostSourceIds(
  sources: readonly SourceResponse[],
  sourceIds: readonly string[],
) {
  const eligible = importHostEligible(sources, sourceIds).map(
    (source) => source.source_id,
  );
  const known = new Set(eligible);
  for (const id of sourceIds) {
    if (known.has(id) || id.startsWith("deepseek-harness") || id === "openclaw")
      continue;
    if (connectionHostId(id) !== null) {
      eligible.push(id);
      known.add(id);
    }
  }
  return eligible;
}

export type ImportConnectionDraft = {
  hostSourceIds: readonly string[];
};
export interface FirstImportReviewProps {
  sourceIds: readonly string[];
  sources: SourceResponse[];
  /** Only this accepted history batch; sync jobs are never actionable here. */
  jobs: ImportJob[];
  acceptedJobIds?: readonly string[];
  failedSources?: Record<string, string>;
  sourceSnapshot?: { confirmedAt: string | null; stale: boolean };
  busy?: boolean;
  error?: string | null;
  connectionNote?: string;
  recovery?: { destination: "memory" | "settings" | "sources"; onSkip(): void };
  onChangeJob?(
    jobId: string,
    action: "pause" | "resume" | "cancel",
  ): Promise<ImportJob>;
  connectionDraft?: ImportConnectionDraft;
  onConnectionDraftChange?(draft: ImportConnectionDraft): void;
  /** True only when the parent has connected step3's explicit execution/receipts. */
  connectionExecutionAvailable?: boolean;
  hostConnectionAvailable?: boolean;
  /** Parent routes settings to step3. This component never completes onboarding. */
  onContinue(destination: "memory" | "settings" | "sources"): void;
}

export function FirstImportReview({
  sourceIds,
  sources,
  jobs,
  acceptedJobIds,
  failedSources = {},
  sourceSnapshot,
  busy = false,
  error,
  connectionNote,
  hostConnectionAvailable = true,
  recovery,
  onChangeJob,
  connectionDraft,
  onConnectionDraftChange,
  onContinue,
}: FirstImportReviewProps) {
  const stale = sourceSnapshot?.stale === true;
  const ids = [...new Set(sourceIds)];
  // Without explicit IDs, capture the initial batch once; later sync/history jobs
  // must not become targets just because their source matches.
  const initialJobs = useRef(
    jobs
      .filter(
        (job) => job.operation !== "sync" && sourceIds.includes(job.source_id),
      )
      .map((job) => job.job_id),
  );
  const batchIds = acceptedJobIds ?? initialJobs.current;
  const [receipts, setReceipts] = useState<Record<string, ImportJob>>({});
  const batch = jobs
    .filter(
      (job) =>
        batchIds.includes(job.job_id) &&
        ids.includes(job.source_id) &&
        job.operation !== "sync",
    )
    .map((job) => {
      const saved = receipts[job.job_id];
      return saved &&
        sameJob(saved, job) &&
        Date.parse(saved.updated_at) >= Date.parse(job.updated_at)
        ? saved
        : job;
    });
  const historyWindowLimited = batch.some(
    (job) => typeof job.history_after === "string",
  );
  const progressStale = stale && batch.length === 0;
  const [changing, setChanging] = useState(false);
  const [actionMessage, setActionMessage] = useState("");
  const batchFlight = useRef(false),
    stopBatch = useRef(false),
    mounted = useRef(true);
  const live = useRef({ batch, sources, stale, busy });
  live.current = { batch, sources, stale, busy };
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      stopBatch.current = true;
    };
  }, []);
  const canPause = batch.some((job) =>
    ["running", "pending"].includes(job.status),
  );
  const canResume = batch.some((job) =>
    ["paused", "interrupted", "failed"].includes(job.status),
  );
  const allCompleted =
    batch.length > 0 &&
    ids.every((id) => batch.some((job) => job.source_id === id)) &&
    batch.every((job) => job.status === "completed") &&
    !Object.keys(failedSources).length;
  const allPaused =
    batch.length > 0 && batch.every((job) => job.status === "paused");
  const importing =
    !stale &&
    batch.some(
      (job) =>
        ["pending", "running"].includes(job.status) &&
        sources.some(
          (source) => source.source_id === job.source_id && source.authorized,
        ),
    );
  const displayImporting =
    importing ||
    (!stale &&
      !allCompleted &&
      !allPaused &&
      batch.length === 0 &&
      ids.some((id) =>
        sources.some((source) => source.source_id === id && source.authorized),
      ));
  const title = "Your conversations are coming in; memory starts here";
  const hostEligible = importHostEligible(sources, ids);
  const draftReady = !!connectionDraft && !!onConnectionDraftChange;

  function leave(destination: "memory" | "settings" | "sources") {
    stopBatch.current = true;
    onContinue(destination);
  }
  async function changeBatch(action: "pause" | "resume") {
    if (
      batchFlight.current ||
      !onChangeJob ||
      busy ||
      (stale && action === "resume")
    )
      return;
    const targets = batch
      .filter((job) =>
        action === "pause"
          ? ["running", "pending"].includes(job.status)
          : ["paused", "interrupted", "failed"].includes(job.status),
      )
      .map((job) => ({
        job,
        sourceKey: sourceKey(
          sources.find((source) => source.source_id === job.source_id),
        ),
      }));
    if (!targets.length) return;
    batchFlight.current = true;
    stopBatch.current = false;
    setChanging(true);
    setActionMessage("");
    let confirmed = 0,
      failed = 0;
    try {
      for (const target of targets) {
        const current = live.current;
        const job = current.batch.find((job) => sameJob(job, target.job));
        const source = current.sources.find(
          (source) => source.source_id === target.job.source_id,
        );
        if (
          stopBatch.current ||
          !mounted.current ||
          (current.stale && action === "resume") ||
          current.busy
        )
          break;
        if (
          !job ||
          !source?.authorized ||
          sourceKey(source) !== target.sourceKey
        ) {
          failed++;
          continue;
        }
        if (
          action === "resume" &&
          !["paused", "interrupted", "failed"].includes(job.status)
        )
          continue;
        if (action === "pause" && !["pending", "running"].includes(job.status))
          continue;
        try {
          const saved = await onChangeJob(job.job_id, action);
          if (!mounted.current) return;
          const sourceNow = live.current.sources.find(
            (source) => source.source_id === job.source_id,
          );
          if (
            (live.current.stale && action === "resume") ||
            !sourceNow?.authorized ||
            sourceKey(sourceNow) !== target.sourceKey ||
            !live.current.batch.some((value) => sameJob(value, job))
          ) {
            stopBatch.current = true;
            failed++;
            continue;
          }
          if (
            !sameJob(saved, job) ||
            Date.parse(saved.updated_at) < Date.parse(job.updated_at) ||
            !Number.isFinite(Date.parse(saved.updated_at)) ||
            !(action === "pause"
              ? ["paused", "completed", "cancelled", "failed"].includes(
                  saved.status,
                )
              : ["pending", "running", "completed"].includes(saved.status))
          )
            throw Error("invalid_receipt");
          setReceipts((previous) => ({ ...previous, [saved.job_id]: saved }));
          confirmed++;
        } catch {
          failed++;
        }
      }
      if (mounted.current)
        setActionMessage(
          `${stopBatch.current ? "Later actions stopped; operations already sent are still checked against receipts. " : ""}Confirmed ${confirmed}, ${failed} unconfirmed.${failed ? "Confirmed items are kept. Check the unconfirmed tasks and retry." : ""}`,
        );
    } finally {
      batchFlight.current = false;
      if (mounted.current) setChanging(false);
    }
  }

  return (
    <R6ColdStartFrame
      step={2}
      title={title}
      eyebrow="Import in progress - connect your apps while you wait"
      footer={
        <>
          <div className="first-import-r6-footer-note">
            <p>
              {allCompleted
                ? "Conversations are saved; memory still needs to be generated."
                : displayImporting
                  ? "The import continues in the background. No need to wait."
                  : stale
                    ? "Re-checking sources; saved counts still follow task receipts."
                    : allPaused
                      ? "This history import is paused. Saved conversations are still here."
                      : "The import is waiting to run. Saved conversations are still here."}
            </p>
            {(error || actionMessage) && (
              <p
                role={
                  error?.includes("app connection") ||
                  error?.includes("successful settings are kept")
                    ? "status"
                    : error
                      ? "alert"
                      : "status"
                }
                className={
                  error?.includes("app connection") ||
                  error?.includes("successful settings are kept")
                    ? "first-import-r6-partial-success"
                    : undefined
                }
              >
                {error || actionMessage}
              </p>
            )}
          </div>
          <div className="first-import-r6-footer-actions">
            {changing ? (
              <R6Button
                onClick={() => {
                  stopBatch.current = true;
                }}
              >
                Stop later actions
              </R6Button>
            ) : onChangeJob && (canPause || canResume) ? (
              <R6Button
                disabled={busy || (stale && !canPause)}
                onClick={() => void changeBatch(canPause ? "pause" : "resume")}
              >
                {canPause ? (
                  <Pause size={16} aria-hidden="true" />
                ) : (
                  <Play size={16} aria-hidden="true" />
                )}
                {canPause ? "Pause import" : "Resume import"}
              </R6Button>
            ) : null}
            {recovery ? (
              <>
                <R6Button
                  disabled={busy}
                  onClick={() => {
                    stopBatch.current = true;
                    recovery.onSkip();
                  }}
                >
                  {recovery.destination === "sources"
                    ? "Go to apps anyway"
                    : recovery.destination === "settings"
                      ? "Go to settings anyway"
                      : "Go to workspace anyway"}
                </R6Button>
                <R6Button
                  variant="primary"
                  disabled={busy}
                  onClick={() => leave(recovery.destination)}
                >
                  Retry save and continue
                </R6Button>
              </>
            ) : (
              <R6Button
                variant="primary"
                disabled={busy}
                onClick={() => leave("settings")}
              >
                Configure a model to build memory
              </R6Button>
            )}
          </div>
        </>
      }
    >
      <div className="first-import-r6-layout">
        <section
          className="first-import-r6-main"
          aria-label="Import and source text"
        >
          <p className="first-import-r6-intro">
            {allCompleted
              ? "This import is finished. Saved conversations do not need to be imported again."
              : "Conversations are saved on this machine first. You can set up the follow-up connections at the same time."}
          </p>
          <p className="first-import-r6-subintro">
            Importing does not grant app read access by itself. Memory
            generation starts only after a model is configured.
          </p>
          {historyWindowLimited && (
            <p className="first-import-r6-history-window" role="status">
              The conversation data set is large. To control usage and keep app
              timelines aligned, this run imports only the last 3 days of
              conversations for all apps.
            </p>
          )}
          <R6ImportProgress
            sourceIds={ids}
            sources={sources}
            jobs={batch}
            stale={progressStale}
          />
          <section className="first-import-r6-feed" aria-label="Import notes">
            <h2>Importing your conversations</h2>
            <p className="first-import-r6-feed-status">
              The background keeps importing; you can safely move to the next
              step
            </p>
            {Object.keys(failedSources).length > 0 && (
              <section
                className="first-import-r6-failures"
                aria-label="Imports that never started"
              >
                {Object.keys(failedSources).map((id) => (
                  <p key={id}>
                    {sources.find((source) => source.source_id === id)
                      ?.display_name ?? id}{" "}
                    failed to start importing; other accepted tasks are
                    unaffected.
                  </p>
                ))}
                <R6Button
                  variant="quiet"
                  disabled={busy}
                  onClick={() => leave("sources")}
                >
                  View apps not imported
                </R6Button>
              </section>
            )}
          </section>
        </section>
        <aside className="first-import-r6-options" aria-label="Connect apps">
          <p className="first-import-r6-eyebrow">Keep memory growing</p>
          <h2>Connect your apps</h2>
          <p className="first-import-r6-options-intro">
            Applies to the supported apps selected in this run.
          </p>
          <SelectionTile
            title={
              hostConnectionAvailable
                ? "Connect apps in one click"
                : "Test instance does not connect apps"
            }
            body={
              hostConnectionAvailable
                ? "Installs a memory connection so the selected apps can use relevant memories from MemHub."
                : "A separate test data directory is in use, so local app connection config is not modified."
            }
            checked={
              hostConnectionAvailable &&
              !!hostEligible.length &&
              hostEligible.every((source) =>
                connectionDraft?.hostSourceIds.includes(source.source_id),
              )
            }
            disabled={
              !draftReady ||
              busy ||
              !hostEligible.length ||
              !hostConnectionAvailable
            }
            onChange={(checked) =>
              onConnectionDraftChange?.({
                hostSourceIds: checked
                  ? hostEligible.map((source) => source.source_id)
                  : [],
              })
            }
          />
          {connectionNote && (
            <p className="first-import-r6-options-note">{connectionNote}</p>
          )}
          <p className="first-import-r6-options-note">
            {draftReady
              ? "After you press the bottom button, the background connects each app separately; one app failing does not affect the others."
              : "Apps cannot be connected right now; no connection is installed."}
          </p>
        </aside>
      </div>
    </R6ColdStartFrame>
  );
}

function SelectionTile({
  title,
  body,
  checked,
  disabled,
  onChange,
}: {
  title: string;
  body: string;
  checked: boolean;
  disabled: boolean;
  onChange(checked: boolean): void;
}) {
  return (
    <label
      className="first-import-r6-selection"
      data-checked={checked}
      aria-disabled={disabled}
    >
      <strong>{title}</strong>
      <input
        type="checkbox"
        aria-label={title}
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
      />
      <span>{body}</span>
    </label>
  );
}

function sourceKey(source: SourceResponse | undefined) {
  return JSON.stringify(
    source
      ? [
          source.source_id,
          source.authorized,
          source.grant_revision,
          source.root_id,
          source.root_label,
          source.candidate_path,
          source.candidate_root_id,
          source.candidate_revision,
        ]
      : null,
  );
}
function sameJob(a: ImportJob, b: ImportJob) {
  return (
    a.job_id === b.job_id &&
    a.source_id === b.source_id &&
    a.user_id === b.user_id &&
    a.created_at === b.created_at &&
    (a.operation ?? "history") === (b.operation ?? "history")
  );
}
