import type {
  GraphWidthDecision,
  GraphWidthStatus,
} from "@one-memory/contracts";
import { useEffect, useRef, useState } from "react";
import { decideGraphWidth, getGraphWidth } from "./api";
import { countLabel } from "./count-label";
import { traceUiAction } from "./ui-trace";
import { isDefiniteRefusal, writeTraceStatus } from "./write-outcome";

/** Ask before the widths cost anything, and default to changing nothing.
 *
 * Without this the app installs, opens, looks healthy, and quietly fails
 * every extraction with `graph_width_mismatch:...` — the failure is in the
 * task record and nowhere the user looks. The three options are the ones the
 * core already implements; the only judgement encoded here is which one is
 * preselected, and that has to be the one that touches nothing.
 */
export function GraphWidthPrompt({
  onResolved = () => undefined,
}: {
  onResolved?(): void;
}) {
  const [status, setStatus] = useState<GraphWidthStatus | null>(null);
  const [decision, setDecision] = useState<GraphWidthDecision>("keep_existing");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [settled, setSettled] = useState(false);
  const [exportedPath, setExportedPath] = useState<string | null>(null);
  const panel = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const result = await getGraphWidth();
        if (!cancelled) setStatus(result);
      } catch {
        // A core that is not up yet is not a mismatch. Staying silent is the
        // right answer here; extraction has not started either.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (settled || !status?.mismatch) return;
    const element = panel.current;
    if (!element) return;
    const before = document.activeElement;
    element.querySelector<HTMLInputElement>("input:checked")?.focus();
    return () => {
      if (before instanceof HTMLElement && before.isConnected) before.focus();
    };
  }, [settled, status]);

  if (settled || !status?.mismatch) return null;
  const { stored_width, expected_width } = status.mismatch;

  async function confirm() {
    setBusy(true);
    setError(null);
    const trace = traceUiAction("memory", "confirm");
    try {
      const receipt = await decideGraphWidth(decision, trace.traceId);
      if (receipt.exported_path) setExportedPath(receipt.exported_path);
      // The decision reached Core: that request succeeded whether or not the
      // graph still reports a mismatch.
      trace.done("succeeded");
      // Exporting is preparation, not a decision: the core still reports the
      // mismatch, and closing here would strand the user one step short of
      // the choice they exported in order to make.
      if (receipt.mismatch) return;
      setSettled(true);
      onResolved();
    } catch (reason) {
      const refused = isDefiniteRefusal(reason);
      trace.done(writeTraceStatus(reason), {
        reason: refused
          ? "graph_width_decision_failed"
          : "graph_width_decision_result_unknown",
      });
      // A refusal is a fact; a timeout is not. Only the refusal branch may
      // present the raw code as "failed", and the unknown branch says so.
      setError(
        refused
          ? reason instanceof Error
            ? reason.message
            : String(reason)
          : "The width choice is not confirmed yet and the existing memory library was not changed. Retry.",
      );
    } finally {
      setBusy(false);
    }
  }

  const choices: { value: GraphWidthDecision; label: string }[] = [
    {
      value: "keep_existing",
      label: `Keep the existing memory library (default): stay on the original embedding configuration and leave the ${countLabel(status.memory_count, "memory")} and the graph untouched`,
    },
    {
      value: "rebuild_with_new",
      label: `Switch to the built-in model: the existing graph is archived rather than deleted, and the relations of ${countLabel(status.memory_count, "memory")} have to be rebuilt`,
    },
    {
      value: "export_first",
      label:
        "Export first and decide later: nothing changes now, come back and choose after the export",
    },
  ];

  return (
    <div
      className="graph-width-prompt"
      role="dialog"
      aria-modal="true"
      aria-label="The existing memory library does not match the built-in model"
      ref={panel}
      onKeyDown={(event) => {
        if (event.key !== "Tab" || !panel.current) return;
        const controls = [
          ...panel.current.querySelectorAll<HTMLElement>(
            "button:not(:disabled), input:not(:disabled)",
          ),
        ];
        const first = controls[0];
        const last = controls.at(-1);
        if (!first || !last) return;
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }}
    >
      <h2>The existing memory library does not match the built-in model</h2>
      <p>
        {`Your memory library was built at ${countLabel(stored_width, "dimension")} and the built-in model produces ${expected_width}. They do not connect, so extraction keeps failing until one is chosen.`}
      </p>
      <ul>
        {choices.map((choice) => (
          <li key={choice.value}>
            <label>
              <input
                type="radio"
                name="graph-width-decision"
                value={choice.value}
                aria-label={choice.label}
                checked={decision === choice.value}
                onChange={() => setDecision(choice.value)}
              />
              <span>{choice.label}</span>
            </label>
          </li>
        ))}
      </ul>
      {exportedPath ? (
        <p className="notice">
          Exported to <code>{exportedPath}</code>. The original library was not
          touched, so you can choose again now.
        </p>
      ) : null}
      {error ? <p className="notice">{error}</p> : null}
      <button type="button" disabled={busy} onClick={() => void confirm()}>
        Do that
      </button>
    </div>
  );
}
