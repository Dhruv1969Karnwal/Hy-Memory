import type {
  AskMemoryResponse,
  ConversationTranscript,
  GovernedMemory,
  MemoryDetail,
  MemoryEvidenceExcerpt,
  MemoryGraphEdge,
  MemoryGraphNode,
  MemoryGraphResponse,
  MemoryProcessingSummary,
  SourceResponse,
} from "@one-memory/contracts";
import { Check, PencilLine, RotateCcw, Trash2 } from "lucide-react";
import {
  type FormEvent,
  type RefObject,
  useId,
  useMemo,
  useState,
} from "react";
import { getConversation } from "./api";
import { countLabel } from "./count-label";
import { HandoffBar } from "./HandoffBar";
import { MemoryGraph3D } from "./MemoryGraph3D";
import { MemoryRecallScope, memoryProvenanceLabel } from "./MemoryRecallScope";
import { extractionStateLabel } from "./memory-processing-labels";
import {
  type Finding,
  findingsOf,
  mattersOf,
  memoriesInMatter,
} from "./workbench-findings";
import { singletonTags, tagCounts } from "./workbench-tags";

/** The memory tab: three columns over one graph.
 *
 * Left is what the memories are about, centre is how they relate and where
 * questions get asked, right is the words they were taken from. The previous
 * layout put a graph on a near-black field with a strip of memory buttons
 * under it and no way to reach the original conversation, so a memory was
 * something you could read but never check. */
export function Workbench({
  memories,
  graph,
  sources,
  processing,
  loaded,
  detail,
  draft,
  scopes,
  busy,
  answer,
  streaming,
  onDraft,
  onScopes,
  onOpenMemory,
  onCloseDetail,
  onSave,
  onSaveScopes,
  onAction,
  onSyncIndex,
  onAsk,
  onSaveAnswer,
  onBuildRelations,
  onConnect,
  onConfigureModel,
  requestIds,
}: {
  memories: GovernedMemory[];
  graph: MemoryGraphResponse | null;
  sources: SourceResponse[];
  processing: MemoryProcessingSummary | null;
  loaded: boolean;
  detail: MemoryDetail | null;
  detailLoad?: {
    memoryId: string;
    status: "loading" | "failed";
    error?: string;
  } | null;
  overlayBlocked?: boolean;
  draft: string;
  scopes: string;
  busy: boolean;
  answer: AskMemoryResponse | null;
  streaming: { text: string; memories: GovernedMemory[] } | null;
  onDraft(next: string): void;
  onScopes(next: string): void;
  onOpenMemory(memoryId: string): void;
  onCloseDetail(): void;
  onSave(event: FormEvent<HTMLFormElement>): void;
  onSaveScopes?(scopes: string[]): Promise<void>;
  onAction(action: "undo" | "delete"): void;
  onSyncIndex(): void;
  onAsk(question: string): void;
  onSaveAnswer(draft: SavedAnswer): void;
  onBuildRelations(): void;
  onConnect(): void;
  onConfigureModel(): void;
  /** Cleared whenever the draft changes, so a retried correction never
   * reuses the request id prepared for text the user has since replaced. */
  requestIds: RefObject<Map<string, string>>;
}) {
  const [matter, setMatter] = useState("");
  const [selectedNode, setSelectedNode] = useState<MemoryGraphNode | null>(
    null,
  );
  const [highlighted, setHighlighted] = useState<string[] | null>(null);
  const [question, setQuestion] = useState("");
  const [openEvidence, setOpenEvidence] = useState<string | null>(null);
  const [transcript, setTranscript] = useState<ConversationTranscript | null>(
    null,
  );
  const [transcriptError, setTranscriptError] = useState<string | null>(null);
  // Which matter the answer on screen is a synthesis of, if any. Held here
  // rather than asked of the answer, because the answer cannot tell: it is
  // the same endpoint either way, and the difference is what was asked.
  const [pageOf, setPageOf] = useState<string | null>(null);

  const matters = useMemo(
    () => mattersOf(memories, sources),
    [memories, sources],
  );
  const inMatter = useMemo(
    () => memoriesInMatter(memories, matter),
    [memories, matter],
  );
  const findings = useMemo(
    () => findingsOf({ memories: inMatter, processing, now: new Date() }),
    [inMatter, processing],
  );
  // Counted across the whole library, not this matter. A tag used once here
  // and once elsewhere is a tag used twice; scoping this to the matter would
  // report a split that is not there.
  const onceOnly = useMemo(() => new Set(singletonTags(memories)), [memories]);
  const allTags = useMemo(() => tagCounts(memories), [memories]);
  const shown = useMemo(() => {
    // Not `graph === null`. Before extraction has run the graph request
    // succeeds and comes back with nothing in it, which is the state a new
    // install is actually in — the null case only happens when the request
    // fails. Keying the skeleton on null meant the one screen it existed for
    // never saw it.
    const drawn = graph ? narrow(graph, inMatter, matter) : null;
    return drawn?.nodes.length ? drawn : skeleton(sources, graph);
  }, [graph, inMatter, matter, sources]);
  const evidence = useMemo(
    () => orderEvidence(detail, highlighted, memories),
    [detail, highlighted, memories],
  );

  function pickFinding(finding: Finding) {
    setHighlighted(finding.memoryIds.length ? finding.memoryIds : null);
    setTranscript(null);
    setOpenEvidence(null);
    if (finding.memoryIds.length) onOpenMemory(finding.memoryIds[0]);
  }

  async function openTranscript(evidenceId: string) {
    setTranscriptError(null);
    try {
      setTranscript(await getConversation(evidenceId));
    } catch {
      setTranscript(null);
      setTranscriptError(
        "The source text for this conversation is no longer in the library.",
      );
    }
  }

  const nameOf = (sourceId: string | null) =>
    sources.find((item) => item.source_id === sourceId)?.display_name ??
    sourceId ??
    "Unknown source";

  return (
    <div className="workbench">
      <Findings findings={findings} onPick={pickFinding} />

      <div className="workbench-grid">
        {/* Topics, not matters. What this column lists is the tags two or more
            memories share. A real matter — the project or task a memory
            belongs to — is a workspace scope, and its plaintext name lives on
            the evidence rows in Core; nothing here can recover it from the
            hash, so calling this column "matter" named something it was not. */}
        <nav className="wb-matters" aria-label="Topics">
          <p className="wb-column-label">Topics</p>
          {allTags.length ? (
            // Says the two numbers the column actually shows: how many
            // topics took shape, and how many tags did not. Reporting one
            // total meant reading "22 topic tags" above six rows.
            <p className="wb-tag-tally">
              {onceOnly.size === 0
                ? `Topics formed: ${allTags.length - onceOnly.size}`
                : `Topics formed: ${allTags.length - onceOnly.size}, appeared once: ${onceOnly.size}`}
            </p>
          ) : null}
          {matters.map((item) => (
            <div key={item.scope || "all"}>
              <button
                type="button"
                className={item.scope === matter ? "wb-matter on" : "wb-matter"}
                aria-current={item.scope === matter ? "true" : undefined}
                onClick={() => {
                  setMatter(item.scope);
                  setSelectedNode(null);
                  setHighlighted(null);
                  setTranscript(null);
                }}
              >
                {item.label}
                <small>
                  {countLabel(item.memoryCount, "memory")}
                  {item.sourceCount
                    ? ` · ${countLabel(item.sourceCount, "source")}`
                    : ""}
                </small>
              </button>
              {/* The graph is a canvas: clicking a circle is a mouse-only
                  gesture, and it is the only way in to a memory. This list is
                  the same door for a keyboard, a screen reader, and anyone
                  who would rather read than aim. */}
              {item.scope === matter && inMatter.length ? (
                <ul className="wb-matter-memories">
                  {inMatter.map((memory) => (
                    <li key={memory.memory_id}>
                      <button
                        type="button"
                        className={
                          detail?.memory.memory_id === memory.memory_id
                            ? "wb-memory on"
                            : "wb-memory"
                        }
                        data-status={memory.status}
                        aria-current={
                          detail?.memory.memory_id === memory.memory_id
                            ? "true"
                            : undefined
                        }
                        onClick={() => {
                          setSelectedNode(null);
                          setHighlighted([memory.memory_id]);
                          setTranscript(null);
                          setOpenEvidence(null);
                          onOpenMemory(memory.memory_id);
                        }}
                      >
                        {memory.content}
                      </button>
                      {/* Outside the button on purpose: the content is
                          clamped to two lines, and a tag row inside it gets
                          clamped away with the overflow. */}
                      {memory.tags.length ? (
                        <p className="wb-tags">
                          {memory.tags.map((tag) => (
                            <span
                              key={tag}
                              className="wb-tag"
                              data-once={onceOnly.has(tag) ? "true" : undefined}
                              title={
                                onceOnly.has(tag)
                                  ? "This topic appears on this memory only. Either it happened once, or one topic was split into two names."
                                  : undefined
                              }
                            >
                              {tag}
                            </span>
                          ))}
                        </p>
                      ) : null}
                    </li>
                  ))}
                </ul>
              ) : null}
            </div>
          ))}
        </nav>

        <div className="wb-stage">
          <div className="wb-graph" data-testid="workbench-graph">
            <p className="wb-graph-title">
              {matters.find((item) => item.scope === matter)?.label ?? "All"}
              {shown.nodes.length ? " relationship graph" : ""}
            </p>
            <div className="wb-graph-legend">
              <span data-kind="memory">Memory</span>
              <span data-kind="schema">HY summary</span>
              <span data-kind="topic">Topic</span>
              <span data-kind="source">Source app</span>
            </div>
            {/* Laid out by the row, not by offsets. Both were absolutely
                positioned with a right edge each, and measuring them found
                eight pixels of overlap — the relations button is a hundred
                and fifty wide until it says Summarizing..., at which point it is
                not. A guessed offset cannot hold against a label that
                changes. */}
            <div className="wb-graph-actions">
              {inMatter.length ? (
                <button
                  type="button"
                  className="quiet"
                  disabled={busy}
                  onClick={() => {
                    const label =
                      matters.find((item) => item.scope === matter)?.label ??
                      "All";
                    setPageOf(matter);
                    onAsk(
                      `Write up "${label}": where it stands now, what is already decided, and what is still open.`,
                    );
                  }}
                >
                  Read a summary of this matter
                </button>
              ) : null}
              <button
                type="button"
                className="quiet"
                disabled={busy || !graph?.nodes.length}
                onClick={onBuildRelations}
              >
                {busy ? "Summarizing..." : "Build relations with HY Memory"}
              </button>
            </div>
            {shown.nodes.length ? (
              <MemoryGraph3D
                graph={shown}
                selectedNodeId={
                  selectedNode?.node_id ??
                  (detail ? `memory:${detail.memory.memory_id}` : null)
                }
                onSelectNode={(node) => {
                  setSelectedNode(node);
                  setTranscript(null);
                  setOpenEvidence(null);
                  if (node.memory_id) {
                    setHighlighted([node.memory_id]);
                    onOpenMemory(node.memory_id);
                  } else {
                    setHighlighted(null);
                  }
                }}
              />
            ) : (
              <EmptyStage loaded={loaded} onConnect={onConnect} />
            )}
          </div>
          {graph?.nodes.length ? (
            <p className="wb-receipt">
              Drawn{" "}
              {shown.nodes.filter((node) => node.node_type === "memory").length}{" "}
              / {countLabel(graph.total_memories, "memory")} in total
              <span>{graphReceipt(shown)}</span>
            </p>
          ) : null}

          <NodeDetail
            node={selectedNode}
            graph={shown}
            detail={detail}
            draft={draft}
            scopes={scopes}
            busy={busy}
            matterLabel={
              matters.find((item) => item.scope === matter)?.label ?? "All"
            }
            onDraft={onDraft}
            onScopes={onScopes}
            onSave={onSave}
            onSaveScopes={onSaveScopes}
            onAction={onAction}
            onSyncIndex={onSyncIndex}
            requestIds={requestIds}
            onClose={() => {
              setSelectedNode(null);
              onCloseDetail();
            }}
          />

          <form
            className="wb-ask"
            onSubmit={(event) => {
              event.preventDefault();
              if (!question.trim()) return;
              setPageOf(null);
              onAsk(question.trim());
              setQuestion("");
            }}
          >
            <div className="wb-chips">
              {chipsFor(inMatter).map((chip) => (
                <button
                  key={chip}
                  type="button"
                  className="wb-chip"
                  onClick={() => {
                    setPageOf(null);
                    onAsk(chip);
                  }}
                >
                  {chip}
                </button>
              ))}
            </div>
            <div className="wb-ask-row">
              <input
                aria-label="Ask about this matter"
                placeholder="Ask about this matter; answers use confirmed memories only"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
              />
              <button type="submit" disabled={!question.trim()}>
                Ask
              </button>
            </div>
            {answer || streaming ? (
              <Answer
                onSaveAnswer={onSaveAnswer}
                asPage={pageOf !== null}
                answer={answer}
                streaming={streaming}
                onOpenMemory={onOpenMemory}
              />
            ) : null}
          </form>

          <HandoffBar
            memories={inMatter.filter(
              (memory) => memory.memory_id === detail?.memory.memory_id,
            )}
            matterLabel={
              matters.find((item) => item.scope === matter)?.label ?? "All"
            }
          />
        </div>

        <aside className="wb-evidence" aria-label="Original conversation">
          <p className="wb-column-label">Original conversation</p>
          <p className="wb-evidence-note">
            {detail
              ? `This memory comes from the ${countLabel(evidence.length, "quote")} below, ordered by time`
              : "Click any circle in the graph and the quotes behind it show up here"}
          </p>
          {evidence.map((item) => (
            <article
              key={item.evidence_id}
              className="wb-evidence-card"
              data-open={openEvidence === item.evidence_id ? "true" : undefined}
            >
              <button
                type="button"
                className="wb-evidence-head"
                onClick={() => {
                  const next =
                    openEvidence === item.evidence_id ? null : item.evidence_id;
                  setOpenEvidence(next);
                  setTranscript(null);
                  setTranscriptError(null);
                }}
              >
                <span className="wb-evidence-meta">
                  <b>{nameOf(item.source_id)}</b>
                  <span>{when(item.created_at)}</span>
                </span>
                {item.extraction_state &&
                item.extraction_state !== "extracted" ? (
                  <small className="wb-evidence-state">
                    {extractionStateLabel(item.extraction_state)}
                  </small>
                ) : null}
                <span className="wb-evidence-excerpt">
                  {item.available
                    ? item.verbatim
                    : "The source quote can no longer be read"}
                </span>
              </button>
              {openEvidence === item.evidence_id ? (
                <div className="wb-evidence-body">
                  <button
                    type="button"
                    className="secondary"
                    onClick={() => void openTranscript(item.evidence_id)}
                  >
                    View full conversation
                  </button>
                  {transcriptError ? (
                    <p className="wb-transcript-error">{transcriptError}</p>
                  ) : null}
                  {transcript ? (
                    <ol className="wb-transcript">
                      {transcript.messages.map((message) => (
                        <li key={message.evidence_id}>
                          <span className="wb-transcript-role">
                            {message.written_by === "user"
                              ? "You"
                              : "Assistant"}
                          </span>
                          {message.verbatim}
                        </li>
                      ))}
                    </ol>
                  ) : null}
                </div>
              ) : null}
            </article>
          ))}
          {!detail && !memories.length && loaded ? (
            <p className="wb-evidence-note">
              No memories yet. Imported quotes can be searched directly under
              "Past conversations".
            </p>
          ) : null}
        </aside>
      </div>

      {processing?.blocked_stages.length ? (
        <p className="wb-floor">
          Some conversations are queued waiting on a condition. The source text
          is already stored and searchable any time.
          <button type="button" className="quiet" onClick={onConfigureModel}>
            See what is missing
          </button>
        </p>
      ) : null}
    </div>
  );
}

function Findings({
  findings,
  onPick,
}: {
  findings: Finding[];
  onPick(finding: Finding): void;
}) {
  if (!findings.length) return null;
  return (
    <section className="wb-findings" aria-label="What I noticed for you">
      <div className="wb-findings-head">
        <b>What I noticed for you</b>
        <span>
          All of it is computed from local records. Click one to highlight the
          memories behind it in the graph.
        </span>
      </div>
      {findings.map((finding) => (
        <div key={finding.id} className="wb-finding">
          <span className={`wb-dot k-${finding.kind}`} aria-hidden="true" />
          <div>
            <span className="wb-finding-label">{finding.label}</span>
            {finding.text}
            {finding.memoryIds.length ? (
              <button
                type="button"
                className="quiet wb-finding-link"
                onClick={() => onPick(finding)}
              >
                Show in graph
              </button>
            ) : null}
          </div>
        </div>
      ))}
    </section>
  );
}

/** The centre card: what the selected circle is, and — for a memory — the
 * only place its wording can be changed or the memory dropped. */
export function NodeDetail({
  node,
  graph,
  detail,
  draft,
  busy,
  matterLabel,
  onDraft,
  onSave,
  onSaveScopes,
  onAction,
  onSyncIndex,
  requestIds,
  onClose,
  compact = false,
}: {
  node: MemoryGraphNode | null;
  graph: MemoryGraphResponse | null;
  detail: MemoryDetail | null;
  draft: string;
  scopes: string;
  busy: boolean;
  matterLabel: string;
  onDraft(next: string): void;
  onScopes(next: string): void;
  onSave(event: FormEvent<HTMLFormElement>): void;
  onSaveScopes?(scopes: string[]): Promise<void>;
  onAction(action: "undo" | "delete"): void;
  onSyncIndex(): void;
  requestIds: RefObject<Map<string, string>>;
  onClose(): void;
  compact?: boolean;
}) {
  const [editing, setEditing] = useState(!compact);
  const formId = useId();
  if (detail) {
    const memory = detail.memory;
    return (
      <article className="wb-detail" aria-label="This memory">
        <div className="wb-detail-head">
          <span className={`wb-status s-${memory.status}`}>
            {statusLabel(memory.status)}
          </span>
          <span className="wb-detail-rev">revision {memory.revision}</span>
          <button type="button" className="quiet" onClick={onClose}>
            Collapse
          </button>
        </div>
        {detail.index_sync?.status === "pending" ? (
          // Said plainly, because the consequence is not in the word "index":
          // the memory here reads correctly and search does not, and nobody
          // would guess that from a sync badge.
          <p className="notice" role="status">
            Search still shows the wording before your edit.
            {detail.index_sync.last_error_code ===
            "hy_memory_update_rejected" ? (
              // HY has no such memory to re-embed, so retrying cannot end
              // differently. Offering the button anyway would have the user
              // pressing it until they gave up.
              "HY Memory has no such memory, so a retry cannot make up for it."
            ) : (
              <button
                type="button"
                className="quiet"
                disabled={busy}
                onClick={onSyncIndex}
              >
                Retry sync
              </button>
            )}
          </p>
        ) : null}
        {compact && !editing && (
          <p className="map-memory-reading">{memory.content}</p>
        )}
        {compact && !editing && (
          <p className="map-origin">{memoryProvenanceLabel(detail)}</p>
        )}
        {onSaveScopes && (
          <MemoryRecallScope
            key={`${memory.memory_id}:${memory.revision}`}
            detail={detail}
            busy={busy}
            onSave={onSaveScopes}
          />
        )}
        {editing && memory.status === "confirmed" ? (
          <form
            id={formId}
            onSubmit={onSave}
            aria-label="Correct the memory content"
          >
            <label>
              Memory content
              <textarea
                name="memory-content"
                aria-label="Memory content"
                autoComplete="off"
                required
                rows={3}
                value={draft}
                onChange={(event) => {
                  requestIds.current?.clear();
                  onDraft(event.target.value);
                }}
              />
            </label>
          </form>
        ) : null}
        <div className="wb-detail-actions">
          {!editing && memory.status === "confirmed" && (
            <button
              type="button"
              className="secondary"
              onClick={() => setEditing(true)}
            >
              <PencilLine size={14} aria-hidden="true" />
              Correct
            </button>
          )}
          {editing && memory.status === "confirmed" ? (
            <button type="submit" form={formId} disabled={busy}>
              <Check size={14} aria-hidden="true" />
              Save correction
            </button>
          ) : null}
          {compact && editing && (
            <button
              type="button"
              className="secondary"
              disabled={busy}
              onClick={() => {
                onDraft(memory.content);
                requestIds.current?.clear();
                setEditing(false);
              }}
            >
              Cancel editing
            </button>
          )}
          {detail.history.length > 1 && memory.status !== "deleted" ? (
            <button
              type="button"
              className="secondary"
              disabled={busy}
              onClick={() => onAction("undo")}
              title="Undo last change"
              aria-label="Undo last change"
            >
              <RotateCcw size={15} aria-hidden="true" />
            </button>
          ) : null}
          {memory.status !== "deleted" ? (
            <button
              type="button"
              className="danger"
              disabled={busy}
              onClick={() => onAction("delete")}
            >
              <Trash2 size={14} aria-hidden="true" />
              Delete
            </button>
          ) : null}
        </div>
      </article>
    );
  }

  if (node) {
    const trace = graph ? traceFrom(graph, node.node_id, 2) : [];
    const direct = trace.filter((step) => step.hop === 1);
    return (
      <article className="wb-detail" aria-label="This node">
        <div className="wb-detail-head">
          <span className="wb-status">{nodeTypeLabel(node.node_type)}</span>
          <strong>{node.label}</strong>
        </div>
        <p className="wb-detail-note">
          This is not a memory, it is what memories hang off. Direct relations:{" "}
          {direct.length},{trace.length - direct.length} more reachable in two
          hops.
        </p>
        <ul className="wb-trace">
          {trace.slice(0, 8).map((step) => (
            <li key={`${step.edge.edge_id}:${step.hop}`}>
              <span className="wb-trace-hop">
                {step.hop === 1 ? "Direct" : "Two hops"}
              </span>
              {relationLabel(step.edge.relation)} · {step.node.label}
              <span className="wb-trace-origin">
                {edgeOriginLabel(step.edge.origin)}
              </span>
            </li>
          ))}
        </ul>
      </article>
    );
  }

  return (
    <article className="wb-detail wb-detail-idle" aria-label="This matter">
      <div className="wb-detail-head">
        <strong>{matterLabel}</strong>
      </div>
      <p className="wb-detail-note">
        The circles are memories and the lines mean they were mentioned in the
        same conversation. Select one and this turns into its provenance and how
        to change it. Switch matter on the left and the graph follows.
      </p>
    </article>
  );
}

/** What gets filed back when an answer is worth keeping. */
export type SavedAnswer = { content: string; evidence_ids: string[] };

/** Past this, an answer is a page and not a memory.
 *
 * The reason for a ceiling at all is what recall does with what it finds:
 * it hands memories to another AI as the person's own words, and a thousand
 * characters of model prose arriving that way is the failure the llm-wiki
 * write-ups describe — a lossy summary that outlives the thing it summarised.
 * A long answer is not refused, it is sent to the matter page, which is
 * rebuilt from the memories every time and never stands in for one. */
const ANSWER_AS_MEMORY_LIMIT = 600;

export function Answer({
  answer,
  streaming,
  onOpenMemory,
  onSaveAnswer,
  asPage,
  suspended = false,
}: {
  answer: AskMemoryResponse | null;
  streaming: { text: string; memories: GovernedMemory[] } | null;
  onOpenMemory(memoryId: string): void;
  onSaveAnswer(draft: SavedAnswer): void;
  asPage: boolean;
  suspended?: boolean;
}) {
  const shownMemories = (answer ?? streaming)?.memories ?? [];
  // Only a finished answer, and only one that stands on something. Filing
  // back what a stream is halfway through writing would keep a sentence that
  // was never true, and an answer with no memories under it has nothing to
  // trace to — it would be the one kind of memory that cannot be checked.
  const groundedAnswer =
    !asPage && answer && shownMemories.length > 0 && answer.mode !== "empty"
      ? answer
      : null;
  const tooLong =
    groundedAnswer !== null &&
    groundedAnswer.answer.length > ANSWER_AS_MEMORY_LIMIT;
  return (
    <div className="wb-answer" aria-live="polite">
      <p>
        {answer ? answer.answer : streaming?.text || "Retrieving memories..."}
      </p>
      <p className="wb-answer-source">
        {answer
          ? `${answer.mode === "generated" ? "Model synthesis" : answer.mode === "extractive" ? "Source excerpt" : "No match"} · based on ${countLabel(shownMemories.length, "confirmed memory")}`
          : `Recalled ${countLabel(shownMemories.length, "memory")}`}
      </p>
      {groundedAnswer && !tooLong ? (
        <button
          type="button"
          className="quiet"
          onClick={() =>
            onSaveAnswer({
              content: groundedAnswer.answer,
              evidence_ids: [
                ...new Set(shownMemories.flatMap((item) => item.evidence_ids)),
              ],
            })
          }
        >
          Save as memory
        </button>
      ) : null}
      {tooLong ? (
        <p className="wb-answer-source">
          This answer is too long for the current save limit, so it cannot be
          saved as a memory yet.
        </p>
      ) : null}
      {asPage ? (
        // Not offered as a memory, and said out loud why. A synthesis is
        // rebuilt from the memories under it every time it is read, which is
        // the only reason it is safe to let a model write it: a stored
        // summary drifts from what it summarised and nothing notices, and a
        // rebuilt one cannot.
        <p className="wb-answer-source">
          This is a summary, not a memory. It is regenerated every time, so it
          cannot drift away from the memories below.
        </p>
      ) : null}
      <ol>
        {shownMemories.map((memory) => (
          <li key={memory.memory_id}>
            <button
              type="button"
              className="quiet"
              onClick={() => onOpenMemory(memory.memory_id)}
            >
              {memory.content}
            </button>
          </li>
        ))}
      </ol>
      {answer && answer.mode !== "empty" && !streaming && (
        <HandoffBar
          memories={shownMemories}
          suspended={suspended}
          matterLabel="This answer"
          answerText={answer.answer}
        />
      )}
    </div>
  );
}

function EmptyStage({
  loaded,
  onConnect,
}: {
  loaded: boolean;
  onConnect(): void;
}) {
  if (!loaded) {
    return (
      <div className="wb-graph-empty">
        <p>Reading memories...</p>
        <small>
          It appears by itself once the local Core starts; no refresh needed.
        </small>
      </div>
    );
  }
  return (
    <div className="wb-graph-empty">
      <p>There is nothing to connect here yet.</p>
      <small>
        After you authorize a source, the source and topic skeleton appears
        first and memories attach to it as they arrive.
      </small>
      <button type="button" onClick={onConnect}>
        Authorize a source
      </button>
    </div>
  );
}

/** Keep only what belongs to the chosen matter.
 *
 * Nodes that are not memories stay: a source or a topic is what the memories
 * in this matter hang off, and dropping them would leave the circles floating
 * with nothing to be attached to. */
function narrow(
  graph: MemoryGraphResponse,
  inMatter: GovernedMemory[],
  matter: string,
): MemoryGraphResponse {
  if (!matter) return graph;
  const keep = new Set(inMatter.map((item) => item.memory_id));
  const nodes = graph.nodes.filter(
    (node) => node.node_type !== "memory" || keep.has(node.memory_id ?? ""),
  );
  const ids = new Set(nodes.map((node) => node.node_id));
  return {
    ...graph,
    nodes,
    edges: graph.edges.filter(
      (edge) => ids.has(edge.source_node_id) && ids.has(edge.target_node_id),
    ),
  };
}

/** A graph before there are any memories.
 *
 * Drawn from sources that actually have conversations imported, so the first
 * thing a new user sees has a shape rather than a sentence explaining why
 * there is no shape. Every circle here is a real, countable import. */
function skeleton(
  sources: SourceResponse[],
  graph: MemoryGraphResponse | null,
): MemoryGraphResponse {
  const withHistory = sources.filter(
    (item) => item.authorized && item.imported_conversations > 0,
  );
  return {
    total_memories: graph?.total_memories ?? 0,
    projected_memories: 0,
    hy_graph_available: graph?.hy_graph_available ?? false,
    nodes: withHistory.map((item) => ({
      node_id: `source:${item.source_id}`,
      node_type: "source" as const,
      label: `${item.display_name} · ${countLabel(item.imported_conversations, "conversation")}`,
      memory_id: null,
      source_id: item.source_id,
      kind: null,
      status: null,
      confidence: null,
    })),
    edges: [],
  };
}

/** Newest first, and filtered to the highlighted memories when a finding put
 * some on screen. */
function orderEvidence(
  detail: MemoryDetail | null,
  highlighted: string[] | null,
  memories: GovernedMemory[],
): MemoryEvidenceExcerpt[] {
  if (!detail) return [];
  const relevant =
    highlighted && !highlighted.includes(detail.memory.memory_id)
      ? evidenceOf(highlighted, memories)
      : null;
  const items = detail.evidence.filter(
    (item) => !relevant || relevant.has(item.evidence_id),
  );
  return [...items].sort((left, right) => {
    const a = left.created_at ? Date.parse(left.created_at) : 0;
    const b = right.created_at ? Date.parse(right.created_at) : 0;
    return b - a;
  });
}

function evidenceOf(
  memoryIds: string[],
  memories: GovernedMemory[],
): Set<string> {
  const wanted = new Set(memoryIds);
  const out = new Set<string>();
  for (const memory of memories) {
    if (!wanted.has(memory.memory_id)) continue;
    for (const id of memory.evidence_ids) out.add(id);
  }
  return out;
}

/** Questions built from what is actually in this matter, so a chip never
 * offers to answer something there is no record of. */
function chipsFor(memories: GovernedMemory[]): string[] {
  if (!memories.length) return [];
  const chips = ["What changed here recently"];
  if (memories.length >= 3) chips.push("Summarize these into a few points");
  return chips;
}

function when(iso: string | null): string {
  if (!iso) return "Unknown time";
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) return "Unknown time";
  return `${at.getMonth() + 1}/${at.getDate()} ${String(at.getHours()).padStart(2, "0")}:${String(at.getMinutes()).padStart(2, "0")}`;
}

function statusLabel(status: GovernedMemory["status"]): string {
  if (status === "confirmed") return "Available";
  if (status === "superseded") return "Superseded";
  return "Deleted";
}

function nodeTypeLabel(type: MemoryGraphNode["node_type"]): string {
  if (type === "source") return "Source app";
  if (type === "topic") return "Topic";
  if (type === "schema") return "HY summary";
  return "Memory";
}

function graphReceipt(graph: MemoryGraphResponse): string {
  const counts = {
    provenance: 0,
    same_conversation: 0,
    shared_tag: 0,
    hy_relation: 0,
  };
  for (const edge of graph.edges) counts[edge.origin] += 1;
  const parts = [
    `Quote source ${countLabel(counts.provenance, "line")}`,
    `Same conversation ${countLabel(counts.same_conversation, "line")}`,
  ];
  // Only mentioned once there is one. A young library has no shared tag yet,
  // and a line reading 0 would be one more thing to read past.
  if (counts.shared_tag > 0)
    parts.push(`Same topic ${countLabel(counts.shared_tag, "line")}`);
  if (counts.hy_relation > 0) {
    parts.push(`HY Memory summary ${countLabel(counts.hy_relation, "line")}`);
    return parts.join(" · ");
  }
  parts.push(
    graph.hy_graph_available
      ? "HY Memory has not summarized any relations yet; run System 2 with the button above"
      : "No usable model is configured, so HY Memory summary relations cannot be read",
  );
  return parts.join(" · ");
}
function edgeOriginLabel(origin: MemoryGraphEdge["origin"]): string {
  if (origin === "hy_relation") return "HY Memory summary";
  if (origin === "same_conversation") return "Same conversation";
  // Named before the fallthrough. The chain ends in Quote source, so a missing
  // case here is not an error — it is a line that says the wrong thing.
  if (origin === "shared_tag") return "Same topic";
  return "Quote source";
}

/** Walk out from one node so each drawn line can be checked by hand. */
function relationLabel(relation: string): string {
  const labels: Record<string, string> = {
    evidence_source: "From",
    same_conversation: "Same conversation",
    shared_tag: "Same topic",
    derived_from: "Derived from",
    tagged_with: "In topic",
    related_to: "Related",
  };
  return labels[relation] ?? relation.replaceAll("_", " ");
}

/** Two numbers rather than one total.
 *
 * 675 waiting reads as "this is stuck". "3 new, 672 historical" says what is
 * actually true: the ones from this morning are minutes away, and the rest is
 * the import catching up. Same total, entirely different feeling (R-02). */
function traceFrom(
  graph: MemoryGraphResponse,
  nodeId: string,
  hops: number,
): { edge: MemoryGraphEdge; node: MemoryGraphNode; hop: number }[] {
  const byId = new Map(graph.nodes.map((node) => [node.node_id, node]));
  const seen = new Set([nodeId]);
  const trace: { edge: MemoryGraphEdge; node: MemoryGraphNode; hop: number }[] =
    [];
  let frontier = [nodeId];
  for (let hop = 1; hop <= hops; hop += 1) {
    const next: string[] = [];
    for (const current of frontier) {
      for (const edge of graph.edges) {
        const otherId =
          edge.source_node_id === current
            ? edge.target_node_id
            : edge.target_node_id === current
              ? edge.source_node_id
              : null;
        if (otherId === null || seen.has(otherId)) continue;
        const node = byId.get(otherId);
        if (!node) continue;
        seen.add(otherId);
        next.push(otherId);
        trace.push({ edge, node, hop });
      }
    }
    frontier = next;
    if (frontier.length === 0) break;
  }
  return trace;
}
