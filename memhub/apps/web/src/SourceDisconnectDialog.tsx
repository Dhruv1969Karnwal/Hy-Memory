import { useEffect, useRef } from "react";

export function SourceDisconnectDialog({
  name,
  onCancel,
  onConfirm,
}: {
  name: string;
  onCancel(): void;
  onConfirm(): void;
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
        ref={panel}
        className="consent-sheet"
        role="dialog"
        aria-modal="true"
        aria-labelledby="disconnect-source-title"
        onKeyDown={(event) => {
          if (event.key === "Escape") onCancel();
          if (event.key !== "Tab") return;
          const controls = panel.current?.querySelectorAll("button");
          const first = controls?.[0],
            last = controls?.[1];
          if (event.shiftKey && document.activeElement === first) {
            event.preventDefault();
            last?.focus();
          } else if (!event.shiftKey && document.activeElement === last) {
            event.preventDefault();
            first?.focus();
          }
        }}
      >
        <h2 id="disconnect-source-title">Revoke authorization for {name}?</h2>
        <p>
          Stops sync and external reads of related memories. Imported source
          text and revision history stay on this machine; conversations in the
          source app are not modified.
        </p>
        <div className="consent-actions">
          <button type="button" className="secondary" onClick={onCancel}>
            Cancel
          </button>
          <button type="button" onClick={onConfirm}>
            Revoke authorization
          </button>
        </div>
      </section>
    </div>
  );
}
