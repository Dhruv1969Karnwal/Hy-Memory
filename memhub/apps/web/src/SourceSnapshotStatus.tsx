import { RefreshCw } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import "./source-snapshot-status.css";

export function SourceSnapshotStatus({
  snapshot,
  onRefresh,
}: {
  snapshot?: { confirmedAt: string | null; stale: boolean };
  onRefresh(): Promise<unknown>;
}) {
  const [pending, setPending] = useState(false);
  const [failed, setFailed] = useState(false);
  const flight = useRef(false);
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  const snapshotIdentity = JSON.stringify([
    snapshot?.confirmedAt ?? null,
    snapshot?.stale ?? null,
  ]);
  const previousSnapshot = useRef(snapshotIdentity);
  useEffect(() => {
    if (previousSnapshot.current !== snapshotIdentity) {
      previousSnapshot.current = snapshotIdentity;
      setFailed(false);
    }
  }, [snapshotIdentity]);

  async function refresh() {
    if (flight.current) return;
    flight.current = true;
    setPending(true);
    setFailed(false);
    try {
      await onRefresh();
    } catch {
      if (alive.current) setFailed(true);
    } finally {
      flight.current = false;
      if (alive.current) setPending(false);
    }
  }

  if (!snapshot) return null;
  const confirmed =
    snapshot.confirmedAt && Number.isFinite(Date.parse(snapshot.confirmedAt))
      ? snapshot.confirmedAt
      : null;
  return (
    <div className="source-snapshot-status" data-stale={snapshot.stale}>
      <div className="source-snapshot-copy" role="status" aria-live="polite">
        {snapshot.stale && <strong>Status pending verification</strong>}
        {confirmed ? (
          <span>
            {snapshot.stale ? "Last confirmed " : "Status confirmed at "}
            <time
              dateTime={confirmed}
              title={new Date(confirmed).toLocaleString()}
            >
              {new Date(confirmed).toLocaleString()}
            </time>
          </span>
        ) : (
          <span>No confirmed status yet</span>
        )}
        {failed && (
          <span role="alert">
            {snapshot.stale
              ? "No new status yet; the last receipt was kept."
              : "The refresh failed; the last confirmed status is still shown."}
          </span>
        )}
      </div>
      <button
        type="button"
        className="quiet source-snapshot-refresh"
        title={pending ? "Checking source status" : "Refresh source status"}
        aria-label="Refresh source status"
        aria-busy={pending}
        disabled={pending}
        onClick={() => void refresh()}
      >
        <RefreshCw size={16} aria-hidden="true" />
      </button>
    </div>
  );
}
