import { useEffect, useRef } from "react";

export function MemoryDeleteDialog({
  count,
  onAnswer,
}: {
  count: number;
  onAnswer(value: boolean): void;
}) {
  const panel = useRef<HTMLElement>(null);
  useEffect(() => {
    const previous = document.activeElement;
    panel.current?.querySelector<HTMLButtonElement>("button")?.focus();
    return () => {
      if (previous instanceof HTMLElement && previous.isConnected)
        previous.focus();
    };
  }, []);
  return (
    <div className="consent-backdrop">
      <section
        className="consent-sheet memory-delete-sheet"
        role="dialog"
        aria-modal="true"
        aria-labelledby="memory-delete-title"
        ref={panel}
        onKeyDown={(event) => {
          if (event.key === "Escape") onAnswer(false);
          if (event.key !== "Tab") return;
          const buttons =
            panel.current?.querySelectorAll<HTMLButtonElement>("button");
          const first = buttons?.[0],
            last = buttons?.[1];
          if (event.shiftKey && document.activeElement === first) {
            event.preventDefault();
            last?.focus();
          }
          if (!event.shiftKey && document.activeElement === last) {
            event.preventDefault();
            first?.focus();
          }
        }}
      >
        <p className="eyebrow">Memory management</p>
        <h2 id="memory-delete-title">
          Delete{count === 1 ? " this memory" : ` these ${count} memories`}?
        </h2>
        <div className="delete-impact">
          <h3>Will be deleted</h3>
          <p>
            The selected memories, plus their search vectors and graph
            relations.
          </p>
          <h3>Will be kept</h3>
          <p>
            Original evidence still referenced by other memories. Conversations
            in the source app are not modified.
          </p>
        </div>
        <p className="delete-alternative">
          If only the content is wrong, cancel and choose "Correct" instead.
        </p>
        <div className="consent-actions">
          <button
            type="button"
            className="secondary"
            onClick={() => onAnswer(false)}
          >
            Cancel
          </button>
          <button
            type="button"
            className="danger"
            onClick={() => onAnswer(true)}
          >
            Confirm Delete
          </button>
        </div>
      </section>
    </div>
  );
}
