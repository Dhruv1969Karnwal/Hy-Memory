import type {
  Citation,
  GovernedMemory,
  Matter,
  MatterClaim,
  MatterContext,
  MatterEvidence,
  MatterRun,
  MemoryDetail,
  MemoryGraphNode,
  MemoryGraphResponse,
} from "@one-memory/contracts";
import {
  ArrowUp,
  BookOpen,
  Check,
  ChevronRight,
  Copy,
  FileText,
  History,
  Link2,
  Network,
  PanelRightOpen,
  Pencil,
  RefreshCw,
  Search,
  ShieldCheck,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  addMatterNote,
  discoverMatters,
  getMatterContext,
  getMatterEvidence,
  getMatterHistory,
  getMatterRuns,
  getMatters,
  getMemories,
  getMemory,
  getMemoryGraph,
  patchMatter,
  refreshMatter,
} from "./api";
import { MatterMemoryPanel } from "./MatterMemoryPanel";
import { MemoryGraph3D } from "./MemoryGraph3D";
import { projectMatterMap } from "./matter-map";
import { memoryWriteErrorText } from "./model-error-text";
import { traceUiAction } from "./ui-trace";
import { isDefiniteRefusal } from "./write-outcome";
import "./matters.css";

const kinds: Record<MatterClaim["kind"], string> = {
  goal: "Goal",
  decision: "Confirmed decisions",
  constraint: "Constraints and boundaries",
  proposal: "Proposal",
  progress: "Progress log",
  open_question: "Still to decide",
  experience: "Lessons and pitfalls",
};
const order: MatterClaim["kind"][] = [
  "goal",
  "decision",
  "constraint",
  "open_question",
  "progress",
  "experience",
  "proposal",
];
const authors: Record<string, string> = {
  human_statement: "User statement",
  agent_report: "Agent self-report",
  mixed: "Cross sources",
  unknown: "Author unconfirmed",
};
const sources: Record<string, string> = {
  cursor: "Cursor",
  codex: "Codex",
  workbuddy: "WorkBuddy",
  "claude-code": "Claude Code",
  "one-memory-manual": "Added manually",
};
const runErrors: Record<string, string> = {
  matter_model_budget_exhausted:
    "The model project budget is exhausted; add budget or switch models.",
  matter_model_http_401:
    "Model authentication failed; check the model configuration.",
  matter_model_http_402:
    "The model service asks you to check the balance or payment status; the original words are preserved.",
  hy_memory_model_payment_required:
    "The memory model service asks you to check the balance or payment status; the original words are preserved.",
  matter_sources_changed_during_generation:
    "The sources changed while organizing was running; organize again.",
  matter_model_output_truncated:
    "Model output exceeded the budget; the organized result was not saved.",
  matter_collection_incomplete:
    "The read scope exceeded this run's budget; HY extraction has not started.",
  matter_import_paused: "Source import is paused; the file was not updated.",
  matter_import_failed: "Source import failed; the file was not updated.",
  matter_import_incomplete:
    "The source scan had unfinished items or read errors; the file was not updated.",
  matter_import_source_unavailable:
    "An authorized source is unavailable; the file was not updated.",
  matter_import_source_revoked:
    "The source authorization was revoked; the file was not updated.",
  matter_synthesis_coverage_incomplete:
    "Key memories were not fully retained; the file failed verification.",
  matter_organization_coverage_incomplete:
    "Matter assignment left out key memories; the file failed verification.",
  hy_memory_input_exceeds_window:
    "A single message exceeded the HY processing window; extraction was not complete.",
  hy_memory_model_empty_response:
    "The memory model returned empty results repeatedly; this run's extraction did not complete.",
  hy_memory_model_output_truncated:
    "The memory model output was incomplete; this run's extraction did not complete.",
  hy_memory_model_request_failed:
    "The memory model request failed; this run's extraction did not complete.",
};
const time = (value: string) =>
  new Date(value).toLocaleString("zh-CN", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

type MatterRead<T> = { version: string | null; value: T };

export function MatterWorkspace({ onOpenSources }: { onOpenSources(): void }) {
  const [matters, setMatters] = useState<Matter[]>([]);
  const [runs, setRuns] = useState<MatterRun[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [mattersReadFailed, setMattersReadFailed] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [search, setSearch] = useState("");
  const [tab, setTab] = useState<"map" | "wiki" | "history">("map");
  const [mapData, setMapData] = useState<{
    memories: GovernedMemory[];
    graph: MemoryGraphResponse | null;
  }>({ memories: [], graph: null });
  const [mapError, setMapError] = useState("");
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [memoryRead, setMemoryRead] = useState<MatterRead<MemoryDetail> | null>(
    null,
  );
  const [memoryLoading, setMemoryLoading] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const memoryRequest = useRef(0);
  const [historyRead, setHistoryRead] = useState<MatterRead<Matter[]> | null>(
    null,
  );
  const [evidenceRead, setEvidenceRead] =
    useState<MatterRead<MatterEvidence> | null>(null);
  const [citation, setCitation] = useState<Citation | null>(null);
  const [sourceLoading, setSourceLoading] = useState(false);
  const [note, setNote] = useState("");
  const [notice, setNotice] = useState("");
  const [edit, setEdit] = useState(false);
  const [title, setTitle] = useState("");
  const [boundary, setBoundary] = useState("");
  const [context, setContext] = useState<MatterContext | null>(null);
  const [copied, setCopied] = useState(false);
  const evidenceRequest = useRef(0);
  const contextRequest = useRef(0);
  const mapRequest = useRef(0);
  const contextDialog = useRef<HTMLElement>(null);
  const loadSequence = useRef(0);
  const matter = matters.find((m) => m.matter_id === selected);
  const contentVersion = matter
    ? `${matter.matter_id}:${matter.version}:${matter.availability}`
    : null;
  const currentContentVersion = useRef(contentVersion);
  currentContentVersion.current = contentVersion;
  const memoryDetail =
    memoryRead?.version === contentVersion ? memoryRead.value : null;
  const history =
    historyRead?.version === contentVersion ? historyRead.value : [];
  const evidence =
    evidenceRead?.version === contentVersion ? evidenceRead.value : null;

  useEffect(() => {
    if (!context) return;
    const before = document.activeElement;
    const panel = contextDialog.current;
    panel?.querySelector<HTMLTextAreaElement>("textarea")?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setContext(null);
      }
      if (event.key !== "Tab" || !panel) return;
      const controls = [
        ...panel.querySelectorAll<HTMLElement>(
          "button:not(:disabled), textarea",
        ),
      ];
      const first = controls[0];
      const last = controls.at(-1);
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      if (before instanceof HTMLElement) before.focus();
    };
  }, [context]);

  const load = useCallback(async () => {
    const request = ++loadSequence.current;
    try {
      const [items, jobs] = await Promise.allSettled([
        getMatters().then((items) => {
          if (request !== loadSequence.current) return;
          setMatters((current) =>
            JSON.stringify(current) === JSON.stringify(items) ? current : items,
          );
          setSelected((current) =>
            items.some((m) => m.matter_id === current)
              ? current
              : (items[0]?.matter_id ?? null),
          );
          setMattersReadFailed(false);
          setLoading(false);
        }),
        getMatterRuns().then((jobs) => {
          if (request === loadSequence.current) setRuns(jobs);
        }),
      ]);
      if (request !== loadSequence.current) return;
      setMattersReadFailed(items.status === "rejected");
      setError(
        items.status === "rejected"
          ? "Could not read the matter files. Check the local service and retry."
          : jobs.status === "rejected"
            ? "Organization progress is unavailable right now; the matter files were refreshed."
            : "",
      );
    } catch {
      if (request === loadSequence.current)
        setError(
          "Could not read the matter files. Check the local service and retry.",
        );
    } finally {
      if (request === loadSequence.current) setLoading(false);
    }
  }, []);
  useEffect(() => {
    void load();
  }, [load]);
  const loadMap = useCallback(async () => {
    const request = ++mapRequest.current;
    try {
      const [memories, graph] = await Promise.all([
        getMemories(),
        getMemoryGraph(),
      ]);
      if (request !== mapRequest.current) return;
      const next = { memories, graph };
      setMapData((current) =>
        JSON.stringify(current) === JSON.stringify(next) ? current : next,
      );
      setMapError("");
    } catch {
      if (request === mapRequest.current)
        setMapError(
          "Base memories are unavailable right now; you can retry the read.",
        );
    }
  }, []);
  useEffect(() => {
    if (matters.length) void loadMap();
    else setMapData({ memories: [], graph: null });
    return () => {
      mapRequest.current += 1;
    };
  }, [loadMap, matters]);
  const running = runs.find((r) => r.status === "running");
  const hasRunning = Boolean(running);
  useEffect(() => {
    let cancelled = false;
    let timer: number;
    const refresh = async () => {
      if (!document.hidden) await load();
      if (!cancelled)
        timer = window.setTimeout(refresh, hasRunning ? 2500 : 5000);
    };
    const visible = () => {
      if (!document.hidden) void load();
    };
    timer = window.setTimeout(refresh, hasRunning ? 2500 : 5000);
    document.addEventListener("visibilitychange", visible);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", visible);
    };
  }, [hasRunning, load]);
  const projection = useMemo(
    () => projectMatterMap(matter, mapData.memories, mapData.graph),
    [matter, mapData],
  );
  useEffect(() => {
    if (
      selectedNodeId &&
      !projection.graph.nodes.some((node) => node.node_id === selectedNodeId)
    ) {
      memoryRequest.current += 1;
      setSelectedNodeId(null);
      setMemoryRead(null);
      setMemoryLoading(false);
    }
  }, [projection, selectedNodeId]);
  const seenContentVersion = useRef(contentVersion);
  useEffect(() => {
    if (seenContentVersion.current === contentVersion) return;
    seenContentVersion.current = contentVersion;
    // Every derived panel belongs to the same version, not just its source excerpt.
    evidenceRequest.current += 1;
    memoryRequest.current += 1;
    contextRequest.current += 1;
    setCitation(null);
    setEvidenceRead(null);
    setSourceLoading(false);
    setMemoryRead(null);
    setMemoryLoading(false);
    setContext(null);
    setCopied(false);
    setHistoryRead(null);
    setEdit(false);
    setTitle("");
    setBoundary("");
  }, [contentVersion]);
  const previousSelected = useRef(selected);
  useEffect(() => {
    if (previousSelected.current !== selected) {
      previousSelected.current = selected;
      evidenceRequest.current += 1;
      setEvidenceRead(null);
      setCitation(null);
      setEdit(false);
      setContext(null);
      setTab("map");
      memoryRequest.current += 1;
      setSelectedNodeId(null);
      setMemoryRead(null);
      setMemoryLoading(false);
      setNote("");
      setNotice("");
    }
  }, [selected]);
  useEffect(() => {
    if (
      tab !== "history" ||
      !matter ||
      !["ready", "needs_update"].includes(matter.availability)
    )
      return;
    let cancelled = false;
    setHistoryRead(null);
    getMatterHistory(matter.matter_id)
      .then((items) => {
        if (!cancelled && contentVersion === currentContentVersion.current)
          setHistoryRead({ version: contentVersion, value: items });
      })
      .catch(() => {
        if (!cancelled) setError("Failed to read the version history.");
      });
    return () => {
      cancelled = true;
    };
  }, [tab, matter, contentVersion]);

  async function openContext() {
    if (matter?.availability !== "ready") return;
    const request = ++contextRequest.current;
    const result = await getMatterContext(matter.matter_id);
    if (
      request !== contextRequest.current ||
      contentVersion !== currentContentVersion.current
    )
      return;
    if (
      result.matter_id !== matter.matter_id ||
      result.version !== matter.version ||
      result.availability !== "ready"
    ) {
      throw new Error("matter_context_changed");
    }
    setContext(result);
    setCopied(false);
  }

  async function action(
    operation: (traceId?: string) => Promise<unknown>,
    mode: "read" | "write" = "write",
  ) {
    setBusy(true);
    setError("");
    setNotice("");
    const trace = mode === "write" ? traceUiAction("matters", "submit") : null;
    try {
      await operation(trace?.traceId);
      await load();
      trace?.done("succeeded");
    } catch (reason) {
      setError(
        "The action did not complete. The material or version may have changed; refresh and retry.",
      );
      trace?.done(isDefiniteRefusal(reason) ? "failed" : "unknown", {
        reason: isDefiniteRefusal(reason)
          ? "matter_write_failed"
          : "matter_write_result_unknown",
      });
    } finally {
      setBusy(false);
    }
  }
  async function openEvidence(cite: Citation, focusMemory = true) {
    if (!matter) return;
    const request = ++evidenceRequest.current;
    setCitation(cite);
    setEvidenceRead(null);
    setSourceLoading(true);
    setInspectorOpen(true);
    memoryRequest.current += 1;
    setMemoryLoading(false);
    setMemoryRead(null);
    const referenced = matter.claims.find((claim) =>
      claim.citations.some((item) => item.evidence_id === cite.evidence_id),
    )?.memory_refs[0];
    const node = projection.graph.nodes.find(
      (item) => item.memory_id === referenced?.memory_id,
    );
    if (node && focusMemory) setSelectedNodeId(node.node_id);
    try {
      const result = await getMatterEvidence(
        matter.matter_id,
        cite.evidence_id,
      );
      if (
        request === evidenceRequest.current &&
        contentVersion === currentContentVersion.current
      )
        setEvidenceRead({ version: contentVersion, value: result });
    } catch {
      if (request === evidenceRequest.current)
        setError("The original evidence changed or is unavailable.");
    } finally {
      if (request === evidenceRequest.current) setSourceLoading(false);
    }
  }
  async function selectMemory(memoryId: string) {
    const node = projection.graph.nodes.find(
      (item) => item.memory_id === memoryId,
    );
    const expected = projection.memories.find(
      (memory) => memory.memory_id === memoryId,
    );
    if (!node || !expected) return;
    setSelectedNodeId(node.node_id);
    evidenceRequest.current += 1;
    setCitation(null);
    setEvidenceRead(null);
    setSourceLoading(false);
    const request = ++memoryRequest.current;
    setMemoryLoading(true);
    setInspectorOpen(true);
    setMemoryRead(null);
    try {
      const detail = await getMemory(memoryId);
      if (
        request !== memoryRequest.current ||
        contentVersion !== currentContentVersion.current
      )
        return;
      if (
        detail.memory.memory_id !== memoryId ||
        detail.memory.revision !== expected.revision ||
        detail.memory.status !== expected.status
      ) {
        throw new Error("matter_memory_changed");
      }
      setMemoryRead({ version: contentVersion, value: detail });
    } catch {
      if (request === memoryRequest.current)
        setMapError("This base memory has changed or cannot be read.");
    } finally {
      if (request === memoryRequest.current) setMemoryLoading(false);
    }
  }
  function selectNode(node: MemoryGraphNode) {
    setSelectedNodeId(node.node_id);
    if (node.memory_id) void selectMemory(node.memory_id);
    else if (node.source_id) {
      const cite = matter?.claims
        .flatMap((claim) => claim.citations)
        .find((item) => item.source_id === node.source_id);
      if (cite) void openEvidence(cite, false);
    }
  }
  const filtered = matters.filter((m) =>
    `${m.title} ${m.tags.join(" ")}`
      .toLowerCase()
      .includes(search.toLowerCase()),
  );

  return (
    <section className={`matter-workspace matter-view-${tab}`}>
      <header className="matter-topbar">
        <div>
          <p className="matter-brand">MemHub</p>
          <h1>Workspace</h1>
        </div>
        <div className="matter-top-actions">
          <span className="local-state">
            <ShieldCheck size={14} /> Local library
          </span>
          <button type="button" className="secondary" onClick={onOpenSources}>
            <Link2 size={15} /> Sources and connections
          </button>
          <button
            type="button"
            disabled={busy || !!running}
            onClick={() => void action((traceId) => discoverMatters(traceId))}
          >
            <RefreshCw size={15} /> Organize history
          </button>
        </div>
      </header>
      {error && (
        <div className="matter-error" role="alert">
          {error}
          <button type="button" className="quiet" onClick={() => void load()}>
            Retry
          </button>
        </div>
      )}
      {notice && (
        <p className="matter-notice" role="status">
          {notice}
        </p>
      )}
      {running && (
        <section className="matter-progress" aria-live="polite">
          <div>
            <strong>{running.phase}</strong>
            <span>
              Base memories {running.completed_batches} /{" "}
              {running.total_batches || "..."} batches -{" "}
              {Math.round(running.elapsed_seconds)} seconds
            </span>
          </div>
          <progress
            max={running.total_batches || 1}
            value={running.completed_batches}
          />
          {running.observations.map((item) => (
            <p key={item}>{item}</p>
          ))}
        </section>
      )}
      {!running && runs[0]?.status === "failed" && (
        <div className="matter-error" role="status">
          Last organization did not complete -{" "}
          {memoryWriteErrorText(runs[0].error) ??
            runErrors[runs[0].error ?? ""] ??
            "The model result failed validation; retry."}
        </div>
      )}
      {!running && runs[0] && (
        <div className="matter-run-receipt" role="status">
          <span>
            {runs[0].coverage_complete
              ? "Complete read this run"
              : "Incomplete read this run"}
          </span>
          <span>Read {runs[0].selected_records} records</span>
          <span>HY input {runs[0].hy_input_records ?? 0} records</span>
          {runs[0].omitted_records > 0 && (
            <span>{runs[0].omitted_records} records not processed</span>
          )}
          {runs[0].status === "completed" && (
            <span>File pending verification</span>
          )}
          {runs[0].failed_phase && <span>Failed: {runs[0].failed_phase}</span>}
        </div>
      )}
      <div className="matter-columns">
        <aside className="matter-sidebar" aria-label="Matter list">
          <div className="matter-sidebar-head">
            <span>My matters</span>
            <span>
              {loading
                ? "Verifying"
                : mattersReadFailed
                  ? "Pending verification"
                  : matters.length}
            </span>
          </div>
          <label className="matter-search">
            <Search size={14} />
            <input
              aria-label="Search matters"
              placeholder="Search matters"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </label>
          {loading && (
            <div
              className="matter-skeleton"
              role="status"
              aria-label="Loading matters"
            />
          )}
          {filtered.map((item) => (
            <button
              type="button"
              key={item.matter_id}
              className={`matter-item ${selected === item.matter_id ? "selected" : ""}`}
              aria-pressed={selected === item.matter_id}
              onClick={() => setSelected(item.matter_id)}
            >
              <BookOpen size={17} />
              <span>
                <strong>{item.title}</strong>
                <small>
                  {item.claims.length} memory points - {item.source_count}{" "}
                  sources
                </small>
                <em>
                  {item.scope_confirmed
                    ? "Scope confirmed"
                    : "Needs calibration"}
                </em>
              </span>
            </button>
          ))}
          {!!search && !filtered.length && (
            <p className="matter-muted">
              {mattersReadFailed
                ? "Matter read failed; the pending verification state is kept."
                : "No matching matters"}
            </p>
          )}
          <footer className="matter-sidebar-footer">
            {runs[0] ? (
              <>
                <span>{runs[0].selected_records} messages joined this run</span>
                <span>
                  {runs[0].omitted_records.toLocaleString()} messages outside
                  this run
                </span>
              </>
            ) : (
              <span>
                {loading
                  ? "Loading matters"
                  : mattersReadFailed
                    ? "Matter read failed; the current state is pending verification"
                    : "Only generated files are shown"}
              </span>
            )}
          </footer>
        </aside>
        <article className="matter-document">
          {!matter && !loading && !error && (
            <div className="matter-empty">
              <BookOpen size={32} />
              <h2>No matters organized yet</h2>
              <button
                type="button"
                disabled={busy || !!running}
                onClick={() =>
                  void action((traceId) => discoverMatters(traceId))
                }
              >
                Organize from authorized history
              </button>
            </div>
          )}
          {matter && (
            <>
              <div className="matter-breadcrumb">
                My matters <ChevronRight size={13} />{" "}
                <span>File v{matter.version}</span>
              </div>
              <header className="matter-heading">
                <div>
                  <h2>{matter.title}</h2>
                  <p>{matter.purpose}</p>
                </div>
                <button
                  type="button"
                  className="quiet icon-button"
                  title="Edit matter scope"
                  aria-label="Edit matter scope"
                  disabled={
                    busy ||
                    !["ready", "needs_update"].includes(matter.availability)
                  }
                  onClick={() => {
                    setEdit(true);
                    setTitle(matter.title);
                    setBoundary(matter.boundary);
                  }}
                >
                  <Pencil size={16} />
                </button>
              </header>
              <div className="matter-metadata">
                <span>Content pending verification</span>
                <span>
                  {matter.source_engine === "hy-memory"
                    ? "HY Memory extraction"
                    : "Legacy exploration result - not through HY"}
                </span>
                <span>
                  <ShieldCheck size={13} /> {matter.evidence_count} evidence
                  items
                </span>
                <span>Updated {time(matter.updated_at)}</span>
                <span>{matter.model}</span>
              </div>
              <label className="matter-agent-access" hidden={tab === "map"}>
                <input
                  type="checkbox"
                  checked={matter.agent_access}
                  disabled={
                    busy ||
                    matter.availability !== "ready" ||
                    matter.source_engine !== "hy-memory"
                  }
                  onChange={(e) =>
                    void action((traceId) =>
                      patchMatter(
                        matter.matter_id,
                        {
                          expected_version: matter.version,
                          agent_access: e.target.checked,
                        },
                        traceId,
                      ),
                    )
                  }
                />
                Allow connected agents to read this matter
              </label>
              {matter.availability !== "ready" && (
                <div className="matter-update-warning">
                  {matter.availability === "needs_update"
                    ? "New material awaits organization; the old conclusions are not given to agents yet."
                    : "The sources or base memories have changed; the old file is hidden."}
                  <button
                    type="button"
                    className="quiet"
                    disabled={busy || !!running}
                    onClick={() =>
                      void action((traceId) =>
                        refreshMatter(matter.matter_id, traceId),
                      )
                    }
                  >
                    <RefreshCw size={14} /> Update file
                  </button>
                </div>
              )}
              {!matter.scope_confirmed && matter.availability === "ready" && (
                <div className="matter-calibration">
                  <div>
                    <strong>
                      Is this what you have been working on recently?
                    </strong>
                    <p>{matter.boundary}</p>
                  </div>
                  <button
                    type="button"
                    className="secondary"
                    disabled={busy}
                    onClick={() =>
                      void action((traceId) =>
                        patchMatter(
                          matter.matter_id,
                          {
                            expected_version: matter.version,
                            scope_confirmed: true,
                          },
                          traceId,
                        ),
                      )
                    }
                  >
                    <Check size={14} /> Confirm matter scope
                  </button>
                </div>
              )}
              {edit && (
                <form
                  className="matter-edit"
                  onSubmit={(e) => {
                    e.preventDefault();
                    void action(async (traceId) => {
                      await patchMatter(
                        matter.matter_id,
                        {
                          expected_version: matter.version,
                          title,
                          boundary,
                        },
                        traceId,
                      );
                      setEdit(false);
                    });
                  }}
                >
                  <label>
                    Matter name
                    <input
                      value={title}
                      maxLength={100}
                      minLength={2}
                      onChange={(e) => setTitle(e.target.value)}
                      required
                    />
                  </label>
                  <label>
                    Matter scope
                    <textarea
                      value={boundary}
                      maxLength={800}
                      onChange={(e) => setBoundary(e.target.value)}
                      required
                    />
                  </label>
                  <div>
                    <button type="submit" disabled={busy}>
                      Save
                    </button>
                    <button
                      type="button"
                      className="quiet"
                      onClick={() => setEdit(false)}
                    >
                      Cancel
                    </button>
                  </div>
                </form>
              )}
              <div className="matter-tabs">
                <div role="tablist" aria-label="File views">
                  <button
                    type="button"
                    role="tab"
                    aria-selected={tab === "map"}
                    onClick={() => setTab("map")}
                  >
                    <Network size={14} /> Memory map
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={tab === "wiki"}
                    onClick={() => setTab("wiki")}
                  >
                    <FileText size={14} /> Knowledge file
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={tab === "history"}
                    onClick={() => setTab("history")}
                  >
                    <History size={14} /> Version history
                  </button>
                </div>
                <button
                  type="button"
                  className="quiet"
                  disabled={busy || matter.availability !== "ready"}
                  onClick={() => void action(openContext, "read")}
                >
                  <Copy size={14} /> Task context
                </button>
                <button
                  type="button"
                  className="quiet icon-button"
                  title="Base memories and source text"
                  aria-label="Open memory inspector"
                  onClick={() => setInspectorOpen(true)}
                >
                  <PanelRightOpen size={16} />
                </button>
              </div>
              <div className="matter-map-stage" hidden={tab !== "map"}>
                {mapError && (
                  <p className="matter-error" role="status">
                    {mapError}{" "}
                    <button
                      type="button"
                      className="quiet"
                      onClick={() => void loadMap()}
                    >
                      Retry read
                    </button>
                  </p>
                )}
                {projection.graph.nodes.length ? (
                  <>
                    <MemoryGraph3D
                      graph={projection.graph}
                      selectedNodeId={selectedNodeId}
                      onSelectNode={selectNode}
                    />
                    <fieldset
                      className="memory-map-legend matter-map-legend"
                      aria-label="Graph legend"
                    >
                      <span data-kind="memory">Base memories</span>
                      <span data-kind="source">Sources</span>
                      <span data-kind="schema">Schema</span>
                      <span data-kind="topic">Topics</span>
                      <span>
                        {projection.memories.length} memories -{" "}
                        {projection.graph.edges.length} links
                      </span>
                    </fieldset>
                  </>
                ) : (
                  <p className="matter-map-empty">
                    {matter.source_engine === "legacy_direct"
                      ? "Legacy exploration results have no HY base memories; see them in the knowledge file."
                      : "No base memory links to show yet."}
                  </p>
                )}
              </div>
              {tab === "wiki" ? (
                <div className="matter-wiki">
                  {order.map((kind) => {
                    const claims = matter.claims.filter((c) => c.kind === kind);
                    return claims.length ? (
                      <section className="matter-claim-section" key={kind}>
                        <h3>
                          <span className={`kind-dot ${kind}`} />
                          {kinds[kind]}
                          <small>{claims.length}</small>
                        </h3>
                        {claims.map((claim, index) => (
                          <div
                            className={`matter-claim ${claim.state ?? "current"}`}
                            key={`${kind}-${claim.statement}`}
                          >
                            <p>{claim.statement}</p>
                            {claim.change_note && (
                              <p className="claim-change">
                                {claim.change_note}
                              </p>
                            )}
                            <div className="claim-meta">
                              <span>{authors[claim.basis]}</span>
                              {claim.state && claim.state !== "current" && (
                                <span>
                                  {
                                    {
                                      historical: "History entry",
                                      superseded: "Superseded",
                                      unresolved: "Pending verification",
                                    }[claim.state]
                                  }
                                </span>
                              )}
                              {claim.scope && <span>{claim.scope}</span>}
                              {claim.citations.map((cite, i) => (
                                <button
                                  type="button"
                                  className="citation-link"
                                  key={`${cite.evidence_id}-${cite.quote}`}
                                  aria-label={`View quote ${sources[cite.source_id] ?? cite.source_id} ${index + 1}-${i + 1}`}
                                  onClick={() => void openEvidence(cite)}
                                >
                                  <Link2 size={12} />
                                  {sources[cite.source_id] ?? cite.source_id}
                                </button>
                              ))}
                            </div>
                          </div>
                        ))}
                      </section>
                    ) : null;
                  })}
                  {!!matter.uncertainties.length && (
                    <section className="matter-uncertainties">
                      <h3>Information gaps and items pending verification</h3>
                      <ul>
                        {matter.uncertainties.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    </section>
                  )}
                </div>
              ) : tab === "history" ? (
                <ol className="matter-history">
                  {history.map((item) => (
                    <li key={item.version}>
                      <span>v{item.version}</span>
                      <div>
                        <strong>{item.title}</strong>
                        <p>
                          {time(item.updated_at)} · {item.claims.length} memory
                          points -{" "}
                          {item.scope_confirmed
                            ? "Scope confirmed"
                            : "Not calibrated"}
                        </p>
                      </div>
                    </li>
                  ))}
                </ol>
              ) : null}
              <form
                className="matter-composer"
                onSubmit={(e) => {
                  e.preventDefault();
                  if (!note.trim()) return;
                  void action(async (traceId) => {
                    await addMatterNote(
                      matter.matter_id,
                      note.trim(),
                      matter.version,
                      traceId,
                    );
                    setNote("");
                    setNotice(
                      "The note was saved as the user's own words. Context calls resume after the file is updated.",
                    );
                  });
                }}
              >
                <label htmlFor="matter-note">Additions or corrections</label>
                <div>
                  <textarea
                    id="matter-note"
                    placeholder="Anything else about this matter..."
                    value={note}
                    maxLength={20000}
                    onChange={(e) => setNote(e.target.value)}
                  />
                  <button
                    type="submit"
                    className="icon-button"
                    title="Save note"
                    aria-label="Save note"
                    disabled={
                      busy ||
                      !note.trim() ||
                      !["ready", "needs_update"].includes(matter.availability)
                    }
                  >
                    <ArrowUp size={18} />
                  </button>
                </div>
              </form>
            </>
          )}
        </article>
        {inspectorOpen && (
          <button
            type="button"
            className="matter-inspector-backdrop"
            aria-label="Collapse memory inspector"
            onClick={() => setInspectorOpen(false)}
          />
        )}
        <aside
          className={`matter-evidence ${inspectorOpen ? "is-open" : ""}`}
          aria-label="Original evidence"
          onKeyDown={(event) => {
            if (event.key === "Escape") setInspectorOpen(false);
          }}
        >
          <header className="matter-inspector-heading">
            <h3>Memories and source text</h3>
            <button
              type="button"
              className="quiet icon-button"
              title="Close inspector"
              aria-label="Close memory inspector"
              onClick={() => setInspectorOpen(false)}
            >
              <X size={16} />
            </button>
          </header>
          <MatterMemoryPanel
            memories={projection.memories}
            selectedMemoryId={
              projection.graph.nodes.find(
                (node) => node.node_id === selectedNodeId,
              )?.memory_id ?? null
            }
            detail={memoryDetail}
            loading={memoryLoading}
            onSelect={(id) => void selectMemory(id)}
          />
          <header>
            <h3>Original evidence</h3>
            {citation && (
              <button
                type="button"
                className="quiet icon-button"
                title="Close quote"
                aria-label="Close quote"
                onClick={() => {
                  evidenceRequest.current += 1;
                  setCitation(null);
                  setEvidenceRead(null);
                }}
              >
                <X size={15} />
              </button>
            )}
          </header>
          {sourceLoading ? (
            <div
              className="matter-skeleton"
              role="status"
              aria-label="Loading source text"
            />
          ) : evidence ? (
            <>
              <div className="evidence-source">
                <span className="source-monogram">
                  {(sources[evidence.source_id] ?? evidence.source_id).slice(
                    0,
                    1,
                  )}
                </span>
                <div>
                  <strong>
                    {sources[evidence.source_id] ?? evidence.source_id}
                  </strong>
                  <small>
                    {evidence.written_by === "person"
                      ? "User's own words"
                      : "Agent message"}
                  </small>
                </div>
              </div>
              <p className="evidence-time">
                {evidence.created_at
                  ? time(evidence.created_at)
                  : "Message time unknown"}
              </p>
              <p className="evidence-time">
                Observed at {time(evidence.observed_at)}
              </p>
              <div className="evidence-verbatim">
                {highlight(evidence.text, citation?.quote ?? "")}
              </div>
              <p className="evidence-id">Evidence {evidence.evidence_id}</p>
            </>
          ) : !memoryDetail && !memoryLoading ? (
            <div className="evidence-empty">
              <Link2 size={25} />
              <p>No quote selected</p>
            </div>
          ) : null}
        </aside>
      </div>
      {context &&
        matter?.availability === "ready" &&
        context.matter_id === matter.matter_id &&
        context.version === matter.version && (
          <div className="matter-modal-backdrop">
            <section
              className="matter-context-modal"
              ref={contextDialog}
              role="dialog"
              aria-modal="true"
              aria-labelledby="context-title"
            >
              <header>
                <h2 id="context-title">Task context - v{context.version}</h2>
                <button
                  type="button"
                  className="quiet icon-button"
                  title="Close context"
                  aria-label="Close context"
                  onClick={() => setContext(null)}
                >
                  <X size={18} />
                </button>
              </header>
              <textarea
                aria-label="Task context content"
                readOnly
                value={context.text}
              />
              <footer>
                <span>
                  Historical reference - sources and verification status kept
                </span>
                <button
                  type="button"
                  onClick={async () => {
                    try {
                      await navigator.clipboard.writeText(context.text);
                      setCopied(true);
                    } catch {
                      setError(
                        "Copy failed; you can select the text in the context box.",
                      );
                    }
                  }}
                >
                  <Copy size={14} /> {copied ? "Copied" : "Copy context"}
                </button>
              </footer>
            </section>
          </div>
        )}
    </section>
  );
}

function highlight(text: string, quote: string) {
  const index = quote ? text.indexOf(quote) : -1;
  if (index < 0) return text;
  return (
    <>
      {text.slice(0, index)}
      <mark>{quote}</mark>
      {text.slice(index + quote.length)}
    </>
  );
}
