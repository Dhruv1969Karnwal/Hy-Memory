import type { ImportJob, SourceResponse } from "@one-memory/contracts";
import { countLabel } from "./count-label";

interface SourceCardProps {
  source: SourceResponse;
  job?: ImportJob;
  busyKey: string | null;
  onAuthorize(source: SourceResponse, confirmedOverride?: boolean): void;
  onDetect(sourceId: string): void;
  onImport(sourceId: string): void;
  onChangeJob(jobId: string, action: "pause" | "resume" | "cancel"): void;
  onRevoke(sourceId: string): void;
}

export function SourceCard({
  source,
  job,
  busyKey,
  onAuthorize,
  onDetect,
  onImport,
  onChangeJob,
  onRevoke,
}: SourceCardProps) {
  const isBusy = busyKey?.endsWith(source.source_id) ?? false;
  const authorize = () => {
    if (source.candidate_kind !== "environment_override") {
      onAuthorize(source, false);
      return;
    }
    const confirmed = window.confirm(
      `This source is set by an environment variable. Allow MemHub to read the AI conversations in this directory?\n\n${source.candidate_path}`,
    );
    if (confirmed) onAuthorize(source, true);
  };
  return (
    <article
      className="source-card"
      aria-labelledby={`${source.source_id}-title`}
    >
      <div className="source-heading">
        <div>
          <h3 id={`${source.source_id}-title`}>{source.display_name}</h3>
          <p>{source.storage_kinds.join(" · ")}</p>
        </div>
        <span className="status-label">{statusLabel(source, job)}</span>
      </div>
      <p className="source-capability">
        {source.capabilities.verification === "real_replayed"
          ? "Verified with real sample replay"
          : "Simulated samples only"}
        {" · "}
        {source.capabilities.live_sync === "implemented"
          ? "Continuous incremental sync"
          : "Manual import only"}
        {" · "}
        {/* The plan requires every source to say which kind of following it
            offers: a source we can only read forward from will keep serving a
            memory whose original sentence the user has since deleted. Calling
            that "live sync" would promise a copy that tracks the source. */}
        {source.sync_completeness === "one_time_snapshot"
          ? "One-time snapshot; does not follow changes in the source app"
          : source.sync_completeness === "full_dynamic"
            ? "Full dynamic sync"
            : "Add-only sync; deletions at the source are not followed"}
        {" · "}
        {/* Whether *this* machine has recall wired into the host is read from
            its install journal and shown in the recall section; here we can
            only honestly say an installer exists for it. */}
        {source.capabilities.recall === "implemented"
          ? "In-app recall supported"
          : "In-app recall not supported yet"}
      </p>

      {job ? <ImportCounts job={job} /> : null}

      {sessionSummary(source)}

      <div className="source-actions">
        {!source.authorized ? (
          <button type="button" disabled={isBusy} onClick={authorize}>
            {source.candidate_kind === "environment_override"
              ? "Confirm and authorize custom directory"
              : "Authorize this source"}
          </button>
        ) : (
          <>
            {source.detected === null ? (
              <button
                type="button"
                disabled={isBusy}
                onClick={() => onDetect(source.source_id)}
              >
                Detect history
              </button>
            ) : null}
            {source.detected === false ? (
              <button
                type="button"
                disabled={isBusy}
                onClick={() => onDetect(source.source_id)}
              >
                Re-check
              </button>
            ) : null}
            {source.detected === true && canStart(job) ? (
              <button
                type="button"
                disabled={isBusy}
                onClick={() => onImport(source.source_id)}
              >
                {job?.status === "completed"
                  ? source.sync_completeness === "one_time_snapshot"
                    ? "Re-read snapshot"
                    : "Check for new content"
                  : "Start import"}
              </button>
            ) : null}
            {job?.status === "running" || job?.status === "pending" ? (
              <>
                <button
                  type="button"
                  className="secondary"
                  onClick={() => onChangeJob(job.job_id, "pause")}
                >
                  Pause
                </button>
                <button
                  type="button"
                  className="secondary"
                  onClick={() => onChangeJob(job.job_id, "cancel")}
                >
                  Cancel
                </button>
              </>
            ) : null}
            {job && ["paused", "interrupted", "failed"].includes(job.status) ? (
              <>
                <button
                  type="button"
                  onClick={() => onChangeJob(job.job_id, "resume")}
                >
                  Continue
                </button>
                <button
                  type="button"
                  className="secondary"
                  onClick={() => onChangeJob(job.job_id, "cancel")}
                >
                  Cancel
                </button>
              </>
            ) : null}
            <button
              type="button"
              className="quiet"
              disabled={isBusy}
              onClick={() => onRevoke(source.source_id)}
            >
              Disconnect
            </button>
          </>
        )}
      </div>
    </article>
  );
}

function ImportCounts({ job }: { job: ImportJob }) {
  return (
    <dl className="import-counts" aria-live="polite">
      <div>
        <dt>Discovered</dt>
        <dd>{job.discovered}</dd>
      </div>
      <div>
        <dt>New</dt>
        <dd>{job.stored}</dd>
      </div>
      <div>
        <dt>Already present</dt>
        <dd>{job.merged}</dd>
      </div>
      <div>
        <dt>Skipped</dt>
        <dd>{job.skipped}</dd>
      </div>
      <div>
        <dt>Errors</dt>
        <dd>{job.errors}</dd>
      </div>
    </dl>
  );
}

function sessionSummary(source: SourceResponse) {
  // Only print a number the core actually measured. An absent or null
  // session_count means "not measured", not zero, so it must never turn into
  // a number on screen. A detected host with nothing readable yet still owes
  // the person an honest sentence rather than a bare zero.
  if (source.detected !== true) return null;
  const count = source.session_count;
  if (count === null || count === undefined) return null;
  return (
    <p className="session-count">
      {count > 0
        ? `Found ${countLabel(count, "conversation")}`
        : "Installed, no conversations found yet"}
    </p>
  );
}

function canStart(job: ImportJob | undefined): boolean {
  return !job || ["completed", "cancelled"].includes(job.status);
}

/** What the person is being asked to authorize, in the words they care about.
 *
 * A count the core did not send is not a count of zero, so an absent or null
 * `session_count` yields no line at all rather than "Found 0 conversations".
 */
function sessionCountLabel(source: SourceResponse): string | null {
  const count = source.session_count;
  if (count === undefined || count === null) return null;
  if (count <= 0) return "Installed, no conversations found yet";
  return `Found ${countLabel(count, "conversation")}`;
}

function statusLabel(
  source: SourceResponse,
  job: ImportJob | undefined,
): string {
  if (!source.authorized) return "Not authorized";
  if (source.detected === null) return "Waiting for detection";
  if (!source.detected) return "No records found";
  if (!job) return "Ready to import";
  const labels: Record<ImportJob["status"], string> = {
    pending: "Waiting to start",
    running: "Importing in the background",
    paused: "Paused",
    interrupted: "Interrupted at last exit",
    completed: "Import complete",
    failed: "Import failed",
    cancelled: "Canceled",
  };
  return labels[job.status];
}
