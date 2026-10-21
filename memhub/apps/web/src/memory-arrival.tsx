import { FileText, Link2, LoaderCircle, Settings2 } from "lucide-react";
import {
  type MemoryArrival,
  memoryArrivalTitle,
  resolveMemoryArrival,
} from "./memory-arrival-state";
import { blockReasonLabel, taskKindLabel } from "./memory-processing-labels";
import { useMediaPreference } from "./use-media-preference";
import "./memory-arrival.css";
import "./import-progress-strip.css";
import { groupedCountLabel } from "./count-label";

export type { MemoryArrival, MemoryArrivalPhase } from "./memory-arrival-state";

export function MemoryArrivalPanel({
  arrival: providedArrival,
  hasMemories = false,
  suspended = false,
  onViewHistory,
  onConfigureModel,
  onConnect,
  showDetails = true,
}: {
  arrival: MemoryArrival;
  hasMemories?: boolean;
  suspended?: boolean;
  onViewHistory?(): void;
  onConfigureModel(): void;
  onConnect(): void;
  showDetails?: boolean;
}) {
  const reduceMotion = useMediaPreference("(prefers-reduced-motion: reduce)");
  const arrival = resolveMemoryArrival({
    arrival: providedArrival,
    processing: null,
    sources: [],
    loaded: true,
  });
  if (hasMemories && arrival.phase === "idle" && !arrival.importing)
    return null;
  const active =
    arrival.phase === "processing" || arrival.phase === "importing";
  const hasOriginals = arrival.importedMessages > 0;
  const title =
    hasMemories && arrival.phase === "idle"
      ? "Source text still importing"
      : memoryArrivalTitle(arrival);
  const description = arrivalDescription(arrival);
  return (
    <section
      className="memory-arrival memory-arrival-neutral"
      data-phase={arrival.phase}
      data-compact={String(hasMemories)}
      aria-label="Memory organizing status"
    >
      <div className="memory-arrival-heading" role="status" aria-live="polite">
        {active && (
          <LoaderCircle
            size={16}
            aria-hidden="true"
            className="memory-arrival-spinner"
            data-moving={String(!reduceMotion && !suspended)}
          />
        )}
        <strong>{title}</strong>
      </div>
      {(hasOriginals || arrival.importing) && (
        <p className="memory-arrival-counts">
          Saved {groupedCountLabel(arrival.importedMessages, "message")}
          {" · "}
          {groupedCountLabel(arrival.importedConversations, "conversation")}
        </p>
      )}
      {arrival.importing &&
        arrival.phase !== "importing" &&
        arrival.phase !== "idle" && (
          <p className="memory-arrival-secondary">
            Source text still importing
          </p>
        )}
      {(!hasMemories ||
        active ||
        ["failed", "unavailable", "stopped", "checking"].includes(
          arrival.phase,
        )) && <p className="memory-arrival-description">{description}</p>}
      {showDetails &&
        ((arrival.blockedStages?.length ?? 0) > 0 ||
          (arrival.phase === "processing" &&
            (arrival.backlogPending ?? 0) > 0)) && (
          <details className="memory-feedback-details">
            <summary>View progress and details</summary>
            {arrival.phase === "processing" &&
              (arrival.backlogPending ?? 0) > 0 && (
                <p>
                  There is a large backlog; batches are being organized in the
                  background to protect quality
                </p>
              )}
            <ul>
              {arrival.blockedStages?.map((block) => (
                <li key={`${block.kind}:${block.reason}`}>
                  {taskKindLabel(block.kind)} · {blockReasonLabel(block.reason)}{" "}
                  · {groupedCountLabel(block.count, "item")}
                </li>
              ))}
            </ul>
          </details>
        )}
      <div className="memory-arrival-actions">
        {arrival.phase === "model-required" && (
          <button type="button" onClick={onConfigureModel}>
            <Settings2 size={15} aria-hidden="true" />
            Set up memory organizing
          </button>
        )}
        {["failed", "unavailable", "stopped"].includes(arrival.phase) && (
          <button
            type="button"
            className="secondary"
            onClick={onConfigureModel}
          >
            <Settings2 size={15} aria-hidden="true" />
            View organizing status
          </button>
        )}
        {hasOriginals && onViewHistory && (
          <button type="button" className="secondary" onClick={onViewHistory}>
            <FileText size={15} aria-hidden="true" />
            View saved conversations
          </button>
        )}
        {!hasOriginals && !arrival.importing && arrival.phase === "idle" && (
          <button type="button" onClick={onConnect}>
            <Link2 size={15} aria-hidden="true" />
            Connect an app
          </button>
        )}
      </div>
    </section>
  );
}

function arrivalDescription(arrival: MemoryArrival): string {
  const retained =
    arrival.importedMessages > 0
      ? "Saved source text is kept on this device. "
      : "";
  switch (arrival.phase) {
    case "model-required":
      return `${retained}Check the organizing model, then continue.`;
    case "processing":
      return "Recent conversations are organized first; more memories will appear as they arrive";
    case "importing":
      return "Saving source text; import progress does not reflect memory generation progress.";
    case "failed":
      return `${retained}Check the failure reason first, then decide whether to retry.`;
    case "unavailable":
      return "The most recent confirmed information is kept; current progress needs re-checking.";
    case "checking":
      return arrival.refreshing
        ? "Checking local records; unconfirmed progress is not counted."
        : "Saved content kept; progress needs refreshing.";
    case "stopped":
      return `${retained}Organizing is stopped; view the status to continue.`;
    default:
      return arrival.importedMessages > 0
        ? "No organizing task is running right now."
        : "Connect an app or import conversation files, and the source text will first be saved on this device.";
  }
}
