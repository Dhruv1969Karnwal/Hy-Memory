import type {
  ConversationSearchHit,
  ConversationTranscript,
  MemoryGraphNode,
  MemoryLibraryPage,
  RecentConversationItem,
} from "@one-memory/contracts";
import {
  ArrowUp,
  ChevronLeft,
  ChevronRight,
  FileText,
  Network,
  RefreshCw,
  Search,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  getConversation,
  getRecentConversations,
  searchConversations,
} from "./api";
import { HandoffBar } from "./HandoffBar";
import { MemoryEvidencePanel } from "./MemoryEvidencePanel";
import { MemoryGraph3D } from "./MemoryGraph3D";
import { MemoryArrivalPanel } from "./memory-arrival";
import {
  type MemoryArrival,
  resolveMemoryArrival,
} from "./memory-arrival-state";
import {
  type MapAxis,
  mapGroups,
  projectMap,
  projectPagedSourceMap,
} from "./memory-map-model";
import { extractionStateLabel } from "./memory-processing-labels";
import { OptionMenu } from "./OptionMenu";
import { subscribePoll } from "./poll-scheduler";
import { R6Button } from "./R6Controls";
import { SelectedMemoryDetail } from "./SelectedMemoryDetail";
import { SourceImportReceipt } from "./SourceImportReceipt";
import { sourceNameOr } from "./source-label";
import { useMediaPreference } from "./use-media-preference";
import { Answer, NodeDetail, type Workbench } from "./Workbench";

type Props = Parameters<typeof Workbench>[0] & {
  simplified?: boolean;
  hidden?: boolean;
  id?: string;
  arrival?: MemoryArrival;
  onViewHistory?(): void;
  onMutationComplete?(id: string): void;
  focusDialogue?: boolean;
  onFocusDialogueConsumed?(): void;
  libraryView?: {
    query: string;
    source: string;
    page: MemoryLibraryPage | null;
    loading: boolean;
    failed: boolean;
    stale: boolean;
    onQuery(value: string): void;
    onSource(value: string): void;
    resetPage(): void;
    onPage(offset: number): void;
    onRetry(): void;
  };
};

/** A search hit republished as a dialogue row.
 *
 * Both panes list the same shape, so a hit has to arrive carrying what a
 * recent conversation carries, including who wrote it: the author used to be
 * filled in here as the user because the endpoint never sent one, which made
 * every hit look like something the person themselves had said.
 */
export function hitToDialogueItem(
  item: ConversationSearchHit,
): RecentConversationItem {
  return {
    ...item,
    role: item.role ?? "",
    written_by: item.written_by ?? "",
    imported_at: item.created_at ?? "",
    time_basis: item.created_at ? "conversation" : "import",
    truncated: false,
  };
}

function authorLabel(writtenBy: string) {
  if (writtenBy === "person") return "You";
  if (writtenBy === "agent") return "Assistant";
  if (writtenBy === "tool") return "Tool";
  return "Unknown";
}

function dayStamp(value: string | null | undefined) {
  if (!value || !Number.isFinite(Date.parse(value))) return "Time unknown";
  const date = new Date(value);
  const now = new Date();
  if (
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate()
  )
    return "Today";
  return date.toLocaleDateString("zh-CN");
}

/**
 * D96: the memory page no longer offers the raw-dialogue tab. The code stays
 * (the full mode's conversation search and a memory's own "Original conversations N"
 * still read dialogues); flipping this brings the tab back.
 */
export const DIALOGUE_PANE_VISIBLE = false;

/** "10/2", with the year only when it is not this year. */
function monthDay(value: string): string | null {
  if (!Number.isFinite(Date.parse(value))) return null;
  const date = new Date(value);
  const day = `${date.getMonth() + 1}/${date.getDate()}`;
  return date.getFullYear() === new Date().getFullYear()
    ? day
    : `${date.getFullYear()}-${day}`;
}

/** D96: when the cited conversation happened, not when the memory was generated. */
function sourceStamp(
  time: { at: string; basis: "conversation" | "import" } | undefined,
  fallback: string | null | undefined,
): string {
  const day = time ? monthDay(time.at) : null;
  if (time && day)
    return time.basis === "conversation"
      ? `Conversation on ${day}`
      : `Imported on ${day}`;
  return dayStamp(fallback);
}

export function MemoryMap(props: Props) {
  const {
    memories,
    graph: baseGraph,
    sources,
    detail,
    loaded,
    onOpenMemory,
    onCloseDetail,
    onAsk,
    busy,
    hidden = false,
    id,
  } = props;
  const library = props.libraryView;
  const serverPaged = library !== undefined;
  const graph = useMemo(
    () =>
      serverPaged
        ? projectPagedSourceMap(baseGraph, library?.page ?? null, sources)
        : baseGraph,
    [baseGraph, serverPaged, library?.page, sources],
  );
  const refreshing = !!library?.page && library.loading;
  const overlaysSuspended = hidden || !!props.overlayBlocked;
  const [axis, setAxis] = useState<MapAxis>("source");
  const [groupKey, setGroupKey] = useState("");
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [groupNode, setGroupNode] = useState<MemoryGraphNode | null>(null);
  const [question, setQuestion] = useState("");
  const [evidenceId, setEvidenceId] = useState<string | null>(null);
  const [transcript, setTranscript] = useState<ConversationTranscript | null>(
    null,
  );
  const [transcriptError, setTranscriptError] = useState("");
  const [transcriptLoading, setTranscriptLoading] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [filterNotice, setFilterNotice] = useState("");
  const [arrivedIds, setArrivedIds] = useState<string[]>([]);
  const [libraryPane, setLibraryPane] = useState<"memories" | "dialogues">(
    "memories",
  );
  const [dialoguePage, setDialoguePage] = useState<
    RecentConversationItem[] | null
  >(null);
  const [dialogueFailed, setDialogueFailed] = useState(false);
  const [selectedDialogueId, setSelectedDialogueId] = useState<string | null>(
    null,
  );
  const [dialogueFilter, setDialogueFilter] = useState("");
  const [dialogueTranscript, setDialogueTranscript] =
    useState<ConversationTranscript | null>(null);
  const [dialogueTranscriptError, setDialogueTranscriptError] = useState("");
  const consumeDialogueFocus = useRef(props.onFocusDialogueConsumed);
  consumeDialogueFocus.current = props.onFocusDialogueConsumed;
  const seenMemoryIds = useRef<Set<string> | null>(null);
  const compact = useMediaPreference("(max-width: 1100px)");
  const composing = useRef(false);
  const generation = useRef(0);
  const displayedVersion = useRef("");
  const pane = useRef<HTMLElement>(null);
  const memoryPaneTabs = useRef<Record<string, HTMLButtonElement | null>>({});
  const openedDefault = useRef(false);
  const scrollSelection = useRef(false);
  useEffect(() => {
    if (!props.focusDialogue) return;
    if (DIALOGUE_PANE_VISIBLE) setLibraryPane("dialogues");
    consumeDialogueFocus.current?.();
  }, [props.focusDialogue]);
  useEffect(() => {
    if (libraryPane !== "dialogues" || hidden) return;
    let stopped = false;
    let busy = false;
    const sourceId = library?.source || undefined;
    async function load() {
      if (busy || document.hidden) return;
      busy = true;
      try {
        const query = dialogueFilter.trim();
        const page = query
          ? await searchConversations(query, 100)
          : await getRecentConversations({ limit: 30, sourceId });
        if (stopped) return;
        const items = query
          ? "hits" in page
            ? page.hits
                .filter((item) => !sourceId || item.source_id === sourceId)
                .map(hitToDialogueItem)
            : []
          : "items" in page
            ? page.items
            : [];
        setDialoguePage(items);
        setDialogueFailed(false);
        setSelectedDialogueId((current) =>
          current && items.some((item) => item.evidence_id === current)
            ? current
            : (items[0]?.evidence_id ?? null),
        );
      } catch {
        if (!stopped) setDialogueFailed(true);
      } finally {
        busy = false;
      }
    }
    void load();
    const stop = subscribePoll({
      // This surface is the dialogue list (`getRecentConversations`), not the
      // governed-memory library that `memory-library` keys elsewhere.
      key: "conversations",
      intervalMs: 10_000,
      run: load,
    });
    return () => {
      stopped = true;
      stop();
    };
  }, [libraryPane, hidden, library?.source, dialogueFilter]);
  useEffect(() => {
    if (libraryPane !== "dialogues" || !selectedDialogueId || hidden) return;
    let stopped = false;
    setDialogueTranscript(null);
    setDialogueTranscriptError("");
    void getConversation(selectedDialogueId).then(
      (value) => {
        if (!stopped) setDialogueTranscript(value);
      },
      () => {
        if (!stopped)
          setDialogueTranscriptError(
            "The source text cannot be read right now",
          );
      },
    );
    return () => {
      stopped = true;
    };
  }, [libraryPane, selectedDialogueId, hidden]);
  const controlledSource = library
    ? library.source
      ? `source:${library.source}`
      : ""
    : groupKey;
  const active = useMemo(
    () => memories.filter((m) => m.status !== "deleted"),
    [memories],
  );
  const arrival = resolveMemoryArrival({
    arrival: props.arrival,
    processing: props.processing,
    sources,
    loaded,
  });
  useEffect(() => {
    if (!loaded || library) return;
    const ids = active.map((memory) => memory.memory_id);
    if (!seenMemoryIds.current) {
      seenMemoryIds.current = new Set(ids);
      return;
    }
    const added = ids.filter((id) => !seenMemoryIds.current?.has(id));
    for (const id of added) seenMemoryIds.current.add(id);
    const currentIds = new Set(ids);
    setArrivedIds((previous) => {
      const retained = previous.filter((id) => currentIds.has(id));
      return !added.length && retained.length === previous.length
        ? previous
        : [...retained, ...added];
    });
  }, [active, loaded, library]);
  const arrived = new Set(arrivedIds);
  const arrivalCount = active.filter((memory) =>
    arrived.has(memory.memory_id),
  ).length;
  const groups = useMemo(
    () => mapGroups(active, graph, axis),
    [active, graph, axis],
  );
  const group = groups.find((g) => g.key === groupKey);
  const sourceGroups = useMemo(
    () => mapGroups(active, graph, "source"),
    [active, graph],
  );
  function sourceLabel(memoryId: string): string {
    if (library) {
      const ids = library.page?.source_ids_by_memory?.[memoryId];
      return ids?.length
        ? ids
            .map((id) =>
              sourceNameOr(sources, id, "Source pending verification"),
            )
            .join(", ")
        : "Source pending verification";
    }
    return (
      sourceGroups
        .filter((entry) => entry.memoryIds.includes(memoryId))
        .map((entry) => entry.label)
        .join(", ") || "Source pending verification"
    );
  }
  const filtered = useMemo(
    () =>
      active.filter(
        (m) =>
          (serverPaged
            ? axis === "source" ||
              !groupKey ||
              group?.memoryIds.includes(m.memory_id)
            : !group || group.memoryIds.includes(m.memory_id)) &&
          (serverPaged ||
            !search ||
            `${m.content} ${m.tags.join(" ")}`
              .toLocaleLowerCase()
              .includes(search.toLocaleLowerCase())),
      ),
    [active, group, search, serverPaged, axis, groupKey],
  );
  const projection = useMemo(() => {
    const result = projectMap(
      graph,
      active,
      axis,
      filtered.map((m) => m.memory_id),
    );
    if (active.length || serverPaged) return result;
    return {
      ...result,
      nodes: sources
        .filter(
          (s) =>
            s.authorized &&
            s.imported_conversations > 0 &&
            (!groupKey || groupKey === `source:${s.source_id}`) &&
            (!search ||
              `${s.display_name} ${s.source_id}`
                .toLocaleLowerCase()
                .includes(search.toLocaleLowerCase())),
        )
        .map((s) => ({
          node_id: `source:${s.source_id}`,
          node_type: "source" as const,
          source_id: s.source_id,
          memory_id: null,
          label: `${s.display_name} · ${s.imported_conversations} conversations`,
          status: null,
          kind: null,
          confidence: null,
        })),
    };
  }, [graph, axis, filtered, active, groupKey, search, sources, serverPaged]);
  const visibleGroupNode = groupNode
    ? (projection.nodes.find(
        (node) =>
          node.node_id === groupNode.node_id &&
          (node.node_type !== "source" ||
            sources.some(
              (source) =>
                source.source_id === node.source_id && source.authorized,
            )),
      ) ?? null)
    : null;
  const selectedSource =
    visibleGroupNode?.node_type === "source"
      ? sources.find(
          (source) =>
            source.source_id === visibleGroupNode.source_id &&
            source.authorized,
        )
      : undefined;
  useEffect(() => {
    // Source revocation also clears a suspended preview, without closing shared memory detail.
    if (groupNode && !visibleGroupNode) {
      setGroupNode(null);
      setInspectorOpen(false);
    }
  }, [groupNode, visibleGroupNode]);
  const requestedId = selectedId ?? detail?.memory.memory_id ?? null;
  const activeId = active.some((memory) => memory.memory_id === requestedId)
    ? requestedId
    : null;
  const visibleDetail =
    detail &&
    (!activeId || activeId === detail.memory.memory_id) &&
    filtered.some((m) => m.memory_id === detail.memory.memory_id)
      ? detail
      : null;
  const lastDetail = useRef(visibleDetail);
  if (visibleDetail) lastDetail.current = visibleDetail;
  const inspectorDetail =
    visibleDetail ??
    (activeId && lastDetail.current ? lastDetail.current : null);
  const version = visibleDetail
    ? `${visibleDetail.memory.memory_id}:${visibleDetail.memory.revision}:${visibleDetail.memory.status}:${visibleDetail.evidence.map((item) => `${item.evidence_id}:${item.available}`).join(",")}`
    : "";
  const pick = useCallback(
    (memoryId: string, scroll = true) => {
      if (!active.some((memory) => memory.memory_id === memoryId)) return;
      scrollSelection.current = scroll;
      if (!filtered.some((memory) => memory.memory_id === memoryId)) {
        setGroupKey("");
        setSearch("");
        setFilterNotice("Filter cleared to locate the related memory.");
      }
      generation.current++;
      setGroupNode(null);
      setSelectedId(memoryId);
      setTranscript(null);
      setEvidenceId(null);
      setTranscriptError("");
      setTranscriptLoading(false);
      setInspectorOpen(true);
      onOpenMemory(memoryId);
    },
    [active, filtered, onOpenMemory],
  );
  useEffect(() => {
    if (displayedVersion.current === version) return;
    displayedVersion.current = version;
    generation.current++;
    setTranscript(null);
    setEvidenceId(null);
    setTranscriptError("");
    setTranscriptLoading(false);
  }, [version]);
  useEffect(() => {
    if (!props.simplified || openedDefault.current || overlaysSuspended) return;
    if (library?.loading && filtered.length === 0) return;
    if (selectedId || detail) return;
    const newest = filtered[0];
    if (!newest) return;
    openedDefault.current = true;
    pick(newest.memory_id, false);
  }, [
    props.simplified,
    overlaysSuspended,
    library?.loading,
    filtered,
    selectedId,
    detail,
    pick,
  ]);
  useEffect(() => {
    if (!activeId || !scrollSelection.current) return;
    scrollSelection.current = false;
    const container = pane.current;
    const item = Array.from(
      pane.current?.querySelectorAll<HTMLElement>("[data-memory-id]") ?? [],
    ).find((node) => node.dataset.memoryId === activeId);
    if (!container || !item) return;
    const containerBounds = container.getBoundingClientRect();
    const itemBounds = item.getBoundingClientRect();
    if (itemBounds.top < containerBounds.top) {
      container.scrollTop -= containerBounds.top - itemBounds.top;
    } else if (itemBounds.bottom > containerBounds.bottom) {
      container.scrollTop += itemBounds.bottom - containerBounds.bottom;
    }
  }, [activeId]);

  const clear = useCallback(() => {
    generation.current++;
    setSelectedId(null);
    setGroupNode(null);
    setTranscript(null);
    setEvidenceId(null);
    setTranscriptError("");
    setTranscriptLoading(false);
    setInspectorOpen(false);
    onCloseDetail();
  }, [onCloseDetail]);
  useEffect(() => {
    // A background map does not own the shared inspector in another mode.
    if (overlaysSuspended || library?.loading) return;
    if (
      requestedId &&
      !active.some((memory) => memory.memory_id === requestedId)
    ) {
      clear();
      return;
    }
    if (!activeId || filtered.some((memory) => memory.memory_id === activeId))
      return;
    clear();
    setFilterNotice(
      "The selected memory is not in the current filter, so the selection was cleared.",
    );
  }, [
    activeId,
    requestedId,
    active,
    filtered,
    clear,
    overlaysSuspended,
    library?.loading,
  ]);
  function closeInspector() {
    if (!compact) {
      clear();
      return;
    }
    generation.current++;
    setTranscript(null);
    setTranscriptError("");
    setTranscriptLoading(false);
    setInspectorOpen(false);
  }
  function chooseGroup(key: string) {
    setGroupNode(null);
    if (library) {
      clear();
      library.onSource(key.startsWith("source:") ? key.slice(7) : "");
      library.resetPage();
      return;
    }
    setGroupKey(key);
  }
  function pickNode(node: MemoryGraphNode) {
    if (node.memory_id) {
      pick(node.memory_id);
      return;
    }
    clear();
    setGroupNode(node);
    setInspectorOpen(true);
    if (node.node_type === "source" && node.source_id) {
      setAxis("source");
      setGroupKey(`source:${node.source_id}`);
      if (library) {
        library.onSource(node.source_id);
        library.resetPage();
      }
    }
    if (node.node_type === "topic") {
      if (props.simplified) return;
      setAxis("topic");
      setGroupKey(`topic:${node.label}`);
    }
  }
  async function openConversation(id: string) {
    if (transcriptLoading) return;
    const turn = ++generation.current;
    setTranscriptError("");
    setTranscript(null);
    setTranscriptLoading(true);
    try {
      const value = await getConversation(id);
      if (turn === generation.current) setTranscript(value);
    } catch {
      if (turn === generation.current)
        setTranscriptError(
          "This conversation cannot be read right now; the current memory was kept.",
        );
    } finally {
      if (turn === generation.current) setTranscriptLoading(false);
    }
  }
  const evidence = [...(visibleDetail?.evidence ?? [])].sort(
    (a, b) =>
      (Date.parse(b.created_at ?? "") || 0) -
      (Date.parse(a.created_at ?? "") || 0),
  );
  const label =
    group?.label ??
    sources.find(
      (source) =>
        source.authorized && groupKey === `source:${source.source_id}`,
    )?.display_name ??
    "All memories";
  if (
    props.simplified &&
    (!library ||
      (!library.loading &&
        !library.failed &&
        !library.stale &&
        !library.query &&
        !library.source &&
        library.page?.total === 0)) &&
    loaded &&
    !active.length
  ) {
    const running = !!props.processing?.running;
    const queued = !!props.processing?.pending;
    return (
      <section className="selected-memory-empty" hidden={hidden}>
        <div>
          <h2>
            {running
              ? "Started processing the selected conversation"
              : queued
                ? "Processing request accepted"
                : "No memories generated yet"}
          </h2>
          <p>
            {running || queued
              ? "The model connection checks out. The first memories will appear in the list as they are generated."
              : arrival.importedConversations > 0
                ? `Imported ${arrival.importedConversations.toLocaleString()} conversations; they will appear here once organized into memories.`
                : arrival.importedMessages > 0
                  ? "History is saved on this device. Memories will appear here when ready."
                  : "Connect an app and import history to start settling memories."}
            {(running || queued) && (
              <small>
                0 memories generated so far. An accepted request is not a
                finished one; original conversations, model tasks, and memory
                counts are tallied separately.
              </small>
            )}
          </p>
          <div>
            <R6Button onClick={props.onConnect}>Back to app</R6Button>
          </div>
        </div>
        {running && (
          <span className="selected-memory-loading" aria-hidden="true" />
        )}
      </section>
    );
  }
  return (
    <section
      id={id}
      className="core-memory-map"
      aria-label="Memory workbench"
      hidden={hidden}
      inert={hidden}
      onKeyDown={(e) => {
        if (overlaysSuspended || e.key !== "Escape" || e.defaultPrevented)
          return;
        if (
          (e.target as HTMLElement).closest(
            '[role="dialog"], input, textarea, select',
          )
        )
          return;
        clear();
      }}
    >
      <nav className="map-library" aria-label="Sources and topics" ref={pane}>
        {props.simplified && DIALOGUE_PANE_VISIBLE ? (
          <div
            className="map-library-pane"
            role="tablist"
            aria-label="Memories or original conversations"
            onKeyDown={(event) => {
              const order = ["memories", "dialogues"] as const;
              const current = order.indexOf(libraryPane);
              const target =
                event.key === "Home"
                  ? order[0]
                  : event.key === "End"
                    ? order[order.length - 1]
                    : event.key === "ArrowRight"
                      ? order[(current + 1) % order.length]
                      : event.key === "ArrowLeft"
                        ? order[(current - 1 + order.length) % order.length]
                        : null;
              if (!target) return;
              event.preventDefault();
              setLibraryPane(target);
              memoryPaneTabs.current[target]?.focus();
            }}
          >
            {(
              [
                ["memories", "Memories"],
                ["dialogues", "Original conversations"],
              ] as const
            ).map(([value, title]) => (
              <button
                type="button"
                role="tab"
                key={value}
                ref={(node) => {
                  memoryPaneTabs.current[value] = node;
                }}
                aria-selected={libraryPane === value}
                aria-controls={`memory-pane-${value}`}
                onClick={() => setLibraryPane(value)}
              >
                {title}
              </button>
            ))}
          </div>
        ) : null}
        {props.simplified ? (
          <div className="selected-source-filter">
            <label htmlFor="selected-source-filter">Source</label>
            <OptionMenu
              id="selected-source-filter"
              className="om-option-menu--plain"
              value={controlledSource}
              options={[
                { value: "", label: "All apps" },
                ...(library?.source &&
                !sources.some(
                  (source) =>
                    source.source_id === library.source && source.authorized,
                )
                  ? [
                      {
                        value: `source:${library.source}`,
                        label: "Selected source pending verification",
                      },
                    ]
                  : []),
                ...(library
                  ? sources
                      .filter((source) => source.authorized)
                      .map((source) => ({
                        value: `source:${source.source_id}`,
                        label: source.display_name,
                      }))
                  : groups.map((group) => ({
                      value: group.key,
                      label: group.label,
                    }))),
              ]}
              onChange={chooseGroup}
            />
          </div>
        ) : (
          <fieldset className="map-axis" aria-label="Map organization">
            {(
              [
                ["source", "Source"],
                ["topic", "Topics"],
              ] as const
            ).map(([value, title]) => (
              <button
                type="button"
                key={value}
                aria-pressed={axis === value}
                onClick={() => {
                  setGroupNode(null);
                  setAxis(value);
                  setGroupKey("");
                }}
              >
                {title}
              </button>
            ))}
          </fieldset>
        )}
        <div className="map-search">
          <Search size={16} aria-hidden="true" />
          <input
            aria-label="Filter map memories"
            name="map-memory-search"
            autoComplete="off"
            placeholder={
              libraryPane === "dialogues"
                ? "Search original conversations"
                : props.simplified
                  ? "Search memories"
                  : "Filter memories..."
            }
            value={
              libraryPane === "dialogues"
                ? dialogueFilter
                : (library?.query ?? search)
            }
            maxLength={library ? 512 : undefined}
            onKeyDown={(e) => {
              // The refresh button next to this list re-reads the page;
              // Enter has to do the same thing, not sit there doing nothing.
              if (e.key !== "Enter" || e.nativeEvent.isComposing) return;
              e.preventDefault();
              library?.onRetry();
            }}
            onChange={(e) => {
              if (libraryPane === "dialogues") {
                setDialogueFilter(e.target.value);
                return;
              }
              setFilterNotice("");
              setSearch(e.target.value);
              if (library) {
                clear();
                library.onQuery(e.target.value);
                library.resetPage();
              }
            }}
          />
        </div>
        {library && libraryPane === "memories" && (
          <div className="map-list-heading" aria-live="polite">
            <span>
              {library.stale
                ? library.page && library.page.total > 0
                  ? `Total ${library.page.total} memories · checking sources...`
                  : "Checking sources..."
                : library.failed
                  ? "Read failed"
                  : library.loading && library.page?.total === 0
                    ? "Reading..."
                    : library.page
                      ? `Total ${library.page.total} memories${refreshing ? " · checking..." : ""}`
                      : "Reading..."}
            </span>
            {!library.stale && (
              <button
                type="button"
                className="quiet map-tool"
                aria-label="Refresh map memories"
                title="Refresh map memories"
                disabled={library.loading || hidden}
                onClick={library.onRetry}
              >
                <RefreshCw size={16} aria-hidden="true" />
              </button>
            )}
          </div>
        )}
        {!props.simplified && (
          <div className="map-groups">
            <button
              type="button"
              className={!groupKey ? "selected" : ""}
              aria-pressed={!groupKey}
              onClick={() => chooseGroup("")}
            >
              All memories <span>{active.length}</span>
            </button>
            {groups.map((g) => (
              <button
                type="button"
                key={g.key}
                className={g.key === groupKey ? "selected" : ""}
                aria-pressed={g.key === groupKey}
                onClick={() => chooseGroup(g.key)}
                title={g.label}
              >
                <span>{g.label}</span>
                <span>{g.memoryIds.length}</span>
              </button>
            ))}
          </div>
        )}
        {!props.simplified && (
          <div className="map-list-heading">
            <span>{label}</span>
            <span>{filtered.length} memories</span>
          </div>
        )}
        {arrivalCount > 0 && (
          <div className="map-arrival-notice">
            <span role="status">New memories: {arrivalCount}</span>
            <button
              type="button"
              title="Dismiss new memories notice"
              aria-label="Dismiss new memories notice"
              onClick={() => setArrivedIds([])}
            >
              <X size={15} aria-hidden="true" />
            </button>
          </div>
        )}
        {filterNotice && (
          <p className="map-filter-notice" role="status">
            {filterNotice}
          </p>
        )}
        {libraryPane === "dialogues" ? (
          <>
            <div className="map-list-heading" aria-live="polite">
              <span>
                {dialogueFailed
                  ? "Read failed"
                  : dialoguePage
                    ? `Last ${dialoguePage.length}`
                    : "Reading..."}
              </span>
            </div>
            <ul
              className="map-memory-list"
              id="memory-pane-dialogues"
              aria-label="Original conversations"
            >
              {(dialoguePage ?? []).map((item) => (
                <li key={item.evidence_id}>
                  <button
                    type="button"
                    className="map-memory-item"
                    aria-pressed={selectedDialogueId === item.evidence_id}
                    onClick={() => setSelectedDialogueId(item.evidence_id)}
                  >
                    <span>
                      {item.excerpt}
                      {item.truncated ? "…" : ""}
                    </span>
                    <small>
                      {sourceNameOr(
                        sources,
                        item.source_id,
                        "Source pending verification",
                      )}{" "}
                      · {dayStamp(item.imported_at)} ·{" "}
                      {authorLabel(item.written_by)}
                    </small>
                  </button>
                </li>
              ))}
            </ul>
            {dialogueFailed ? (
              <p role="alert">
                Original conversations cannot be read right now
              </p>
            ) : null}
            {dialoguePage && !dialoguePage.length ? (
              <p className="map-empty">
                {dialogueFilter.trim()
                  ? "No matching original conversations"
                  : "No original conversations imported yet"}
              </p>
            ) : null}
          </>
        ) : null}
        <ul
          className="map-memory-list"
          aria-label="Map memory list"
          id="memory-pane-memories"
          hidden={libraryPane === "dialogues"}
        >
          {filtered.map((m) => (
            <li key={m.memory_id} data-memory-id={m.memory_id}>
              <button
                type="button"
                className="map-memory-item"
                aria-pressed={activeId === m.memory_id}
                onClick={() => pick(m.memory_id)}
              >
                <span className={`map-state ${m.status}`} aria-hidden="true" />
                <span>{m.content}</span>
                <small>
                  {props.simplified ? (
                    <>
                      {`${sourceLabel(m.memory_id)} · ${sourceStamp(
                        library?.page?.source_time_by_memory?.[m.memory_id],
                        m.updated_at,
                      )}`}
                      {/* D99: the conversation date alone would hide that this
                          one was made today, so the generated day stays too. */}
                      <span className="map-memory-made">
                        Settled {dayStamp(m.updated_at)}
                      </span>
                    </>
                  ) : (
                    <>Available · {m.evidence_ids.length} source segments</>
                  )}
                </small>
              </button>
            </li>
          ))}
        </ul>
        {libraryPane === "memories" && library?.failed && (
          <div role="alert">
            <p>Could not read memories. Try again.</p>
            <R6Button onClick={library.onRetry}>Reload map memories</R6Button>
          </div>
        )}
        {libraryPane === "memories" &&
          library?.page &&
          !library.loading &&
          !library.failed &&
          !library.stale &&
          !filtered.length && (
            <p className="map-empty">
              {library.query || library.source
                ? "No matching memories"
                : groupKey
                  ? "No topic memories match on this page"
                  : "No memories generated yet"}
            </p>
          )}
        {libraryPane === "memories" &&
          library?.page &&
          !library.loading &&
          !library.failed &&
          !library.stale && (
            <nav aria-label="Map memory pager" className="map-list-heading">
              <button
                type="button"
                className="quiet map-tool"
                aria-label="Previous page of memories"
                title="Previous page of memories"
                disabled={library.page.offset === 0}
                onClick={() => {
                  clear();
                  if (axis === "topic") setGroupKey("");
                  if (library.page)
                    library.onPage(library.page.offset - library.page.limit);
                }}
              >
                <ChevronLeft size={16} aria-hidden="true" />
              </button>
              <span>
                Total {library.page.total.toLocaleString()} memories, page{" "}
                {library.page.total === 0
                  ? 0
                  : Math.floor(library.page.offset / library.page.limit) +
                    1}{" "}
                / {Math.ceil(library.page.total / library.page.limit)}
              </span>
              <button
                type="button"
                className="quiet map-tool"
                aria-label="Next page of memories"
                title="Next page of memories"
                disabled={
                  library.page.offset + library.page.items.length >=
                  library.page.total
                }
                onClick={() => {
                  clear();
                  if (axis === "topic") setGroupKey("");
                  if (library.page)
                    library.onPage(library.page.offset + library.page.limit);
                }}
              >
                <ChevronRight size={16} aria-hidden="true" />
              </button>
            </nav>
          )}
        {!library && loaded && !filtered.length && (search || groupKey) && (
          <p className="map-empty">No matching memories</p>
        )}
        {(!props.simplified || (!library && !active.length)) && (
          <MemoryArrivalPanel
            showDetails={false}
            arrival={arrival}
            hasMemories={active.length > 0}
            suspended={overlaysSuspended}
            onViewHistory={props.onViewHistory}
            onConfigureModel={props.onConfigureModel}
            onConnect={props.onConnect}
          />
        )}
      </nav>
      <section className="map-stage" aria-label="Memory relation graph">
        <header className="map-stage-heading">
          <div>
            <span className="map-kicker" hidden={props.simplified}>
              {axis === "topic" ? "Topic view (not matters)" : "Source view"}
            </span>
            <h2>
              {libraryPane === "dialogues"
                ? "Original conversations"
                : props.simplified
                  ? "Memory links"
                  : label}
            </h2>
            {props.simplified && (
              <p className="selected-graph-subtitle">
                {libraryPane === "dialogues"
                  ? "Just-imported raw words, not settled memories yet"
                  : projection.nodes.length
                    ? "Source attribution"
                    : "The view appears once memories have settled"}
              </p>
            )}
          </div>
          {compact && (activeId || selectedSource) && !inspectorOpen && (
            <button
              type="button"
              className="quiet map-tool"
              aria-label={
                selectedSource
                  ? "View source text of selected source"
                  : "View source text of selected memory"
              }
              title="View source text"
              onClick={() => setInspectorOpen(true)}
            >
              <FileText size={17} />
            </button>
          )}
          {!props.simplified && (
            <button
              type="button"
              className="quiet map-tool"
              title="Rebuild existing relations"
              aria-label="Rebuild existing relations"
              disabled={busy || !graph?.nodes.length}
              onClick={props.onBuildRelations}
            >
              <Network size={17} />
            </button>
          )}
        </header>
        <div className="map-canvas" data-tour-id="memory-map-canvas">
          {libraryPane === "dialogues" ? (
            <div className="map-dialogue-reading">
              {dialogueTranscriptError ? (
                <p role="alert">{dialogueTranscriptError}</p>
              ) : dialogueTranscript ? (
                <ol className="map-transcript">
                  {dialogueTranscript.messages.map((message) => (
                    <li key={message.evidence_id}>
                      <small>{authorLabel(message.written_by)}</small>
                      <p>{message.verbatim}</p>
                    </li>
                  ))}
                </ol>
              ) : selectedDialogueId ? (
                <p>Reading source text...</p>
              ) : (
                <p>Select an original conversation</p>
              )}
            </div>
          ) : projection.nodes.length && (!props.simplified || !hidden) ? (
            <MemoryGraph3D
              graph={projection}
              selectedNodeId={
                visibleGroupNode?.node_id ??
                (activeId ? `memory:${activeId}` : null)
              }
              onSelectNode={pickNode}
            />
          ) : (
            <div className="map-empty-stage">
              <Network size={32} strokeWidth={1} />
              <p>
                {library?.stale
                  ? "The view appears once memories have settled"
                  : library?.failed
                    ? "Memories cannot be read right now"
                    : library?.page &&
                        (library.query || library.source || groupKey)
                      ? "No memories match the current filter"
                      : !loaded
                        ? "Reading memories..."
                        : active.length
                          ? "No memories match the current filter"
                          : arrival.importedMessages > 0
                            ? "Saved conversations have not produced memories yet"
                            : "No memories to show yet"}
              </p>
            </div>
          )}
        </div>
        <footer className="map-meta">
          {/* The legend explains the graph. Keeping it while the stage shows
              something else left it dangling under a view it cannot describe. */}
          {!props.simplified && libraryPane === "memories" && (
            <div className="map-legend">
              <span data-kind="memory">Memory</span>
              <span data-kind="topic">Topics</span>
              <span data-kind="source">Source</span>
              <span data-kind="schema">Summary</span>
            </div>
          )}
          {props.simplified ? (
            projection.nodes.length ? (
              <span>Lines show source attribution</span>
            ) : null
          ) : (
            <span>{filtered.length} memories · currently loaded</span>
          )}
        </footer>
        {!props.simplified && (
          <form
            className="map-question"
            onSubmit={(e) => {
              e.preventDefault();
              if (!composing.current && question.trim() && !props.streaming) {
                onAsk(question.trim());
                setQuestion("");
              }
            }}
          >
            <input
              aria-label="Ask your memories"
              name="map-question"
              autoComplete="off"
              placeholder="Ask across all authorized available memories..."
              value={question}
              onCompositionStart={() => {
                composing.current = true;
              }}
              onCompositionEnd={() => {
                composing.current = false;
              }}
              onKeyDown={(event) => {
                if (
                  event.key === "Enter" &&
                  (event.nativeEvent.isComposing || composing.current)
                )
                  event.preventDefault();
              }}
              onChange={(e) => setQuestion(e.target.value)}
            />
            <button
              type="submit"
              title="Send question"
              aria-label="Send question"
              disabled={!question.trim() || !!props.streaming}
            >
              <ArrowUp size={17} />
            </button>
          </form>
        )}
        <div hidden={overlaysSuspended}>
          {!props.simplified && (props.answer || props.streaming) && (
            <Answer
              answer={props.answer}
              streaming={props.streaming}
              onOpenMemory={pick}
              onSaveAnswer={props.onSaveAnswer}
              asPage={false}
              suspended={overlaysSuspended}
            />
          )}
          {!props.simplified && (
            <HandoffBar
              suspended={overlaysSuspended}
              memories={filtered.filter(
                (memory) => memory.memory_id === activeId,
              )}
              matterLabel={label}
            />
          )}
        </div>
      </section>
      {/* Keep inspector drafts mounted; suspension releases its modal effects. */}
      <MemoryEvidencePanel
        open={inspectorOpen && !!(activeId || visibleGroupNode)}
        suspended={overlaysSuspended}
        onClose={closeInspector}
      >
        <header className="map-inspector-heading">
          <h2>
            {inspectorDetail
              ? "Memory details"
              : selectedSource
                ? "Source text"
                : visibleGroupNode
                  ? "Linked node"
                  : "Memory details"}
          </h2>
          {(activeId || visibleGroupNode) && (
            <button
              type="button"
              className="quiet map-tool"
              aria-label="Close memory details"
              title="Close details"
              onClick={closeInspector}
            >
              <X size={17} />
            </button>
          )}
        </header>
        {selectedSource ? (
          inspectorOpen &&
          !overlaysSuspended && (
            <SourceImportReceipt
              key={selectedSource.source_id}
              source={selectedSource}
            />
          )
        ) : props.simplified && inspectorDetail ? (
          <div
            aria-busy={
              refreshing ||
              inspectorDetail.memory.memory_id !== activeId ||
              undefined
            }
            onClickCapture={(event) => {
              // React portal events follow this owner too, including editor saves.
              if (
                refreshing &&
                (event.target as Element).closest("button, summary")
              ) {
                event.preventDefault();
                event.stopPropagation();
              }
            }}
            onSubmitCapture={(event) => {
              if (refreshing) {
                event.preventDefault();
                event.stopPropagation();
              }
            }}
          >
            {/* The pane used to keep the previously read memory on screen
                until the new one arrived, which read as "nothing happened"
                for two or three seconds. A read is not a value: show that a
                read is happening instead. */}
            {activeId && inspectorDetail.memory.memory_id !== activeId ? (
              <div className="map-detail-loading">
                <div className="map-detail-skeleton" aria-hidden="true">
                  <i />
                  <i />
                  <i />
                  <i />
                </div>
                <p role="status">Reading this memory...</p>
              </div>
            ) : (
              <SelectedMemoryDetail
                detail={inspectorDetail}
                sources={sources}
                conversationAt={
                  library?.page?.source_time_by_memory?.[
                    inspectorDetail.memory.memory_id
                  ] ?? null
                }
                suspended={overlaysSuspended}
                onSaveScopes={props.onSaveScopes}
                onChanged={(id) => props.onMutationComplete?.(id)}
                hideActions
              />
            )}
          </div>
        ) : inspectorDetail || visibleGroupNode ? (
          <NodeDetail
            key={
              inspectorDetail?.memory.memory_id ||
              visibleGroupNode?.node_id ||
              "empty"
            }
            compact
            node={visibleGroupNode}
            graph={projection}
            detail={inspectorDetail}
            draft={props.draft}
            scopes={props.scopes}
            busy={busy}
            matterLabel={label}
            onDraft={props.onDraft}
            onScopes={props.onScopes}
            onSave={props.onSave}
            onSaveScopes={props.onSaveScopes}
            onAction={props.onAction}
            onSyncIndex={props.onSyncIndex}
            requestIds={props.requestIds}
            onClose={closeInspector}
          />
        ) : (
          <div
            className={`map-inspector-empty${activeId ? " map-inspector-loading" : ""}`}
          >
            {activeId ? (
              <div className="map-detail-skeleton" aria-hidden="true">
                <i />
                <i />
                <i />
                <i />
              </div>
            ) : (
              <FileText size={28} strokeWidth={1} />
            )}
            {props.detailLoad?.memoryId === activeId &&
            props.detailLoad.status === "failed" ? (
              <>
                <p role="alert">
                  The source text for this memory cannot be read right now; your
                  selection was kept.
                </p>
                <button
                  type="button"
                  className="secondary"
                  onClick={() => activeId && onOpenMemory(activeId)}
                >
                  Retry reading source text
                </button>
              </>
            ) : props.simplified && !activeId && filtered.length ? (
              // Nothing is selected, but this conversation has already
              // settled: naming what came out of it beats a placeholder that
              // asks for the settle that already happened. Only the
              // single-column surface needs it — the three-column one already
              // lists the same memories in the pane beside this one.
              <>
                <p>{`${filtered.length} memories settled`}</p>
                <ul className="map-settled-list" aria-label="Settled memories">
                  {filtered.map((m) => (
                    <li key={m.memory_id}>{m.content}</li>
                  ))}
                </ul>
              </>
            ) : (
              <p>
                {activeId
                  ? "Reading memory details..."
                  : "View details once memories have settled"}
              </p>
            )}
          </div>
        )}
        {!props.simplified && evidence.length > 0 && (
          <h3 className="map-section-title">
            Original conversations <span>{evidence.length}</span>
          </h3>
        )}
        {!props.simplified &&
          evidence.map((item) => (
            <article
              key={item.evidence_id}
              className="map-evidence"
              data-evidence-id={item.evidence_id}
            >
              <div className="map-evidence-meta">
                <strong>
                  {sourceNameOr(sources, item.source_id, "Unknown source")}
                </strong>
                <span>{author(item.written_by)}</span>
              </div>
              <time>
                {item.created_at && Number.isFinite(Date.parse(item.created_at))
                  ? new Date(item.created_at).toLocaleString("zh-CN", {
                      year: "numeric",
                      month: "2-digit",
                      day: "2-digit",
                      hour: "2-digit",
                      minute: "2-digit",
                    })
                  : "Time unknown"}
              </time>
              <button
                type="button"
                className="map-quote"
                disabled={!item.available}
                aria-expanded={evidenceId === item.evidence_id}
                aria-controls={`evidence-body-${item.evidence_id}`}
                onClick={() => {
                  generation.current++;
                  setTranscript(null);
                  setTranscriptError("");
                  setTranscriptLoading(false);
                  setEvidenceId(
                    evidenceId === item.evidence_id ? null : item.evidence_id,
                  );
                }}
              >
                {item.available
                  ? item.verbatim
                  : "The source text is no longer readable"}
              </button>
              {item.extraction_state &&
                item.extraction_state !== "extracted" && (
                  <small>{extractionStateLabel(item.extraction_state)}</small>
                )}
              {item.available && evidenceId === item.evidence_id && (
                <div
                  className="map-evidence-expanded"
                  id={`evidence-body-${item.evidence_id}`}
                >
                  <p className="map-section-title">Related memories</p>
                  {active
                    .filter((m) => m.evidence_ids.includes(item.evidence_id))
                    .map((m) => (
                      <button
                        className="quiet map-related"
                        type="button"
                        key={m.memory_id}
                        aria-label={`Locate related memory: ${m.content}`}
                        onClick={() => pick(m.memory_id)}
                      >
                        {m.content}
                        <ChevronRight size={14} />
                      </button>
                    ))}
                  <button
                    type="button"
                    className="quiet"
                    disabled={transcriptLoading}
                    onClick={() => void openConversation(item.evidence_id)}
                  >
                    {transcriptLoading
                      ? "Reading the full conversation..."
                      : transcriptError
                        ? "Retry reading the full conversation"
                        : "See full conversation"}
                  </button>
                  {transcriptError && <p role="alert">{transcriptError}</p>}
                  {transcript && (
                    <ol className="map-transcript">
                      {transcript.messages.map((m) => (
                        <li key={m.evidence_id}>
                          <small>{author(m.written_by)}</small>
                          <p>{m.verbatim}</p>
                        </li>
                      ))}
                    </ol>
                  )}
                </div>
              )}
            </article>
          ))}
        {!props.simplified && !!visibleDetail?.history.length && (
          <details className="map-history">
            <summary>
              Edit history · {visibleDetail.history.length} versions
            </summary>
            {visibleDetail.history.map((h) => (
              <article key={h.revision}>
                <small>Revision {h.revision}</small>
                <p>{h.content}</p>
              </article>
            ))}
          </details>
        )}
      </MemoryEvidencePanel>
    </section>
  );
}

function author(who: string | null) {
  return who === "person" || who === "user"
    ? "You"
    : who === "agent" || who === "assistant"
      ? "Agent"
      : who === "tool"
        ? "Tool"
        : "Author pending verification";
}
