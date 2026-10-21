import type { MemoryDetail } from "@one-memory/contracts";
import { Check, Globe2, LockKeyhole, Settings2 } from "lucide-react";
import { useId, useRef, useState } from "react";
import { apiErrorCode } from "./api";
import { isDefiniteRefusal } from "./write-outcome";

type Mode = "limited" | "shared" | "disabled";
const limits = (scopes: string[]) =>
  scopes.filter((scope) => !["all-apps", "private", "*"].includes(scope));
const modeOf = (scopes: string[]): Mode =>
  scopes.includes("all-apps")
    ? "shared"
    : scopes.includes("private")
      ? "disabled"
      : "limited";

export function memoryProvenanceLabel(detail: MemoryDetail): string {
  const evidence = detail.evidence.filter((item) => item.available);
  if (
    evidence.length &&
    evidence.every((item) => item.source_id === "memory-content-export")
  ) {
    const authors = new Set(
      evidence.map((item) => item.written_by ?? "unknown"),
    );
    const author =
      authors.size !== 1
        ? "Multiple authors"
        : authors.has("person")
          ? "User original text"
          : authors.has("agent")
            ? "Agent original text"
            : "Author unknown";
    return `Imported existing memory - ${author}`;
  }
  return detail.memory.origin === "manual"
    ? "Added manually"
    : detail.memory.origin === "answered"
      ? "Generated from Q&A"
      : detail.memory.origin === "agent"
        ? "Compiled from AI replies"
        : "Compiled from the original conversation";
}

export function MemoryRecallScope({
  detail,
  busy,
  onSave,
}: {
  detail: MemoryDetail;
  busy: boolean;
  onSave(scopes: string[]): Promise<void>;
}) {
  const id = useId();
  const editButton = useRef<HTMLButtonElement>(null);
  const [editing, setEditing] = useState(false);
  const current = modeOf(detail.memory.scopes);
  const [mode, setMode] = useState<Mode>(current);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const previous =
    [
      detail.memory,
      ...[...detail.history].sort((a, b) => b.revision - a.revision),
    ]
      .map((item) => limits(item.scopes))
      .find((items) => items.length) ?? [];
  const disabled = busy || saving;
  const editable = detail.memory.status === "confirmed";
  const summary =
    current === "shared"
      ? "All connected tools"
      : current === "disabled"
        ? "External memory recall is stopped"
        : "Existing limited scope";
  const close = () => {
    setEditing(false);
    setMode(current);
    setError("");
    queueMicrotask(() => editButton.current?.focus());
  };
  const save = async () => {
    setSaving(true);
    setError("");
    try {
      await onSave(
        mode === "disabled"
          ? ["private"]
          : mode === "shared"
            ? ["all-apps", ...previous]
            : previous,
      );
      close();
    } catch (reason) {
      // The trace (in `MemorySpace.saveScopes`) logs a definite refusal as
      // `failed` and everything else as `unknown`; the sentence the user reads
      // has to match, or the log and the screen disagree about the same write.
      setError(
        apiErrorCode(reason) === "memory_revision_conflict"
          ? "The memory has been updated. Reopen this memory before adjusting the scope."
          : isDefiniteRefusal(reason)
            ? "The scope change was not submitted; the current scope stays unchanged. Fix it and try again."
            : "The save result is unconfirmed. Try again.",
      );
    } finally {
      setSaving(false);
    }
  };
  return (
    <section
      className="memory-recall-scope"
      aria-label="External memory recall scope"
    >
      <div className="recall-scope-heading">
        <span>
          <LockKeyhole size={15} aria-hidden="true" />
          External memory recall
        </span>
        {editable && !editing && (
          <button
            type="button"
            className="quiet map-tool"
            title="Adjust recall scope"
            aria-label="Adjust recall scope"
            ref={editButton}
            disabled={disabled}
            onClick={() => {
              setMode(current);
              setEditing(true);
            }}
          >
            <Settings2 size={16} aria-hidden="true" />
          </button>
        )}
      </div>
      <p className="recall-scope-summary">{summary}</p>
      {editing && (
        <>
          <fieldset disabled={disabled}>
            <legend className="sr-only">Choose the recall scope</legend>
            {(
              [
                ...(previous.length
                  ? [["limited", "Existing limited scope"]]
                  : []),
                ["shared", "All connected tools"],
                ["disabled", "Stop external memory recall"],
              ] as [Mode, string][]
            ).map(([value, label]) => (
              <label key={value} className="recall-scope-option">
                <input
                  type="radio"
                  name={id}
                  value={value}
                  checked={mode === value}
                  onChange={() => {
                    setMode(value);
                    setError("");
                  }}
                />
                {value === "shared" ? (
                  <Globe2 size={16} aria-hidden="true" />
                ) : (
                  <LockKeyhole size={16} aria-hidden="true" />
                )}
                <span>{label}</span>
              </label>
            ))}
          </fieldset>
          <div className="recall-scope-actions">
            <button
              type="button"
              disabled={disabled || mode === current}
              onClick={() => void save()}
            >
              <Check size={15} aria-hidden="true" />
              {saving ? "Saving..." : "Save recall scope"}
            </button>
            <button
              type="button"
              className="quiet"
              aria-label="Cancel scope change"
              disabled={disabled}
              onClick={close}
            >
              Cancel
            </button>
          </div>
        </>
      )}
      {error && (
        <p className="notice danger" role="alert">
          {error}
        </p>
      )}
      <p className="recall-scope-boundary">
        Raw-text retrieval still follows source authorization; content already
        brought out cannot be withdrawn.
      </p>
    </section>
  );
}
