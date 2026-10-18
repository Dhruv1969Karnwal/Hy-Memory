// @vitest-environment jsdom
import type {
  GovernedMemory,
  MemoryGraphResponse,
  MemoryLibraryPage,
  SourceResponse,
} from "@one-memory/contracts";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import * as api from "./api";
import { MemorySpace } from "./MemorySpace";

vi.mock("./api", () => ({
  getMemories: vi.fn(),
  getMemoryLibrary: vi.fn(),
  getMemoryGraph: vi.fn(),
  getMemoryProcessing: vi.fn(),
  getModelProfiles: vi.fn(),
  getMemory: vi.fn(),
  getConversation: vi.fn(),
  correctMemory: vi.fn(),
  mutateMemory: vi.fn(),
  apiErrorCode: () => null,
}));
vi.mock("./GraphWidthPrompt", () => ({ GraphWidthPrompt: () => null }));
vi.mock("./MemoryNoteComposer", () => ({ MemoryNoteComposer: () => null }));
vi.mock("./MemoryGraph3D", () => ({
  MemoryGraph3D: ({
    graph,
    onSelectNode,
  }: {
    graph: MemoryGraphResponse;
    onSelectNode?(node: MemoryGraphResponse["nodes"][number]): void;
  }) => (
    <div data-testid="graph-placeholder" data-graph={JSON.stringify(graph)}>
      {graph.nodes
        .filter((node) => node.node_type === "source")
        .map((node) => (
          <button
            key={node.node_id}
            type="button"
            aria-label={`Select ${node.label} in the graph`}
            onClick={() => onSelectNode?.(node)}
          >
            {node.label}
          </button>
        ))}
    </div>
  ),
}));

const sources = ["cursor", "codex"].map(
  (id) =>
    ({
      source_id: id,
      display_name: id,
      authorized: true,
      grant_revision: 1,
      imported_messages: 501,
      imported_conversations: 1,
    }) as SourceResponse,
);
const corpus: GovernedMemory[] = Array.from({ length: 501 }, (_, index) => ({
  memory_id: `m${index + 1}`,
  user_id: "local",
  content:
    index === 500 ? "Unique target of item 501" : `Saved item ${index + 1}`,
  kind: "chat",
  status: "confirmed",
  origin: "extracted",
  revision: 1,
  scopes: ["all-apps"],
  tags: [],
  evidence_ids: [`e${index + 1}`],
  created_at: "2026-09-19T00:00:00Z",
  updated_at: "2026-09-19T00:00:00Z",
  deleted_at: null,
}));
const sourceOf = (memory: GovernedMemory) =>
  memory.memory_id === "m501" ? "codex" : "cursor";
function graph(items: GovernedMemory[]): MemoryGraphResponse {
  return {
    total_memories: 501,
    projected_memories: items.length,
    hy_graph_available: false,
    nodes: [
      ...sources.map((source) => ({
        node_id: `source:${source.source_id}`,
        node_type: "source" as const,
        source_id: source.source_id,
        memory_id: null,
        label: source.display_name,
        kind: null,
        status: null,
        confidence: null,
      })),
      ...items.map((memory) => ({
        node_id: `memory:${memory.memory_id}`,
        node_type: "memory" as const,
        memory_id: memory.memory_id,
        source_id: null,
        label: memory.content,
        kind: memory.kind,
        status: memory.status,
        confidence: null,
      })),
    ],
    edges: items.map((memory) => ({
      edge_id: `edge:${memory.memory_id}`,
      source_node_id: `source:${sourceOf(memory)}`,
      target_node_id: `memory:${memory.memory_id}`,
      relation: "source",
      origin: "provenance" as const,
      weight: null,
    })),
  };
}
beforeEach(() => {
  vi.resetAllMocks();
  // Match current Core route defaults: /memories is limit=100 unless requested.
  vi.mocked(api.getMemories).mockImplementation(async (query = "") =>
    corpus.filter((item) => item.content.includes(query)).slice(0, 100),
  );
  vi.mocked(api.getMemoryLibrary).mockImplementation(async (query = {}) => {
    const items = corpus.filter(
      (item) =>
        (!query.query || item.content.includes(query.query)) &&
        (!query.active_only || item.status === "confirmed") &&
        (!query.status || item.status === query.status) &&
        (!query.source || sourceOf(item) === query.source),
    );
    return {
      items: items.slice(
        query.offset ?? 0,
        (query.offset ?? 0) + (query.limit ?? 100),
      ),
      total: items.length,
      offset: query.offset ?? 0,
      limit: query.limit ?? 100,
      source_ids_by_memory: Object.fromEntries(
        items
          .slice(query.offset ?? 0, (query.offset ?? 0) + (query.limit ?? 100))
          .map((item) => [item.memory_id, [sourceOf(item)]]),
      ),
    };
  });
  vi.mocked(api.getMemoryGraph).mockResolvedValue(graph(corpus.slice(0, 100)));
  vi.mocked(api.getMemoryProcessing).mockResolvedValue(null as never);
  vi.mocked(api.getModelProfiles).mockResolvedValue([]);
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});
function processingBlocked(reason: string, count = 1) {
  return {
    pending: count,
    running: 0,
    fresh_pending: 0,
    backlog_pending: 0,
    recent_completions: [],
    waiting_for_model: 0,
    completed: 0,
    failed: 0,
    blocked_stages: [{ kind: "extract" as const, reason, count }],
    outside_window_pending: 0,
    extraction_window_preset: "3m" as const,
    manual_extract_batches_planned: 0,
    manual_extract_batches_current: 0,
    active_queue: 0,
    in_flight_count: 0,
    seconds_per_completion: null,
  };
}
async function mount() {
  const view = render(<MemorySpace simplified sources={sources} />);
  await screen.findByText("501 settled · 100 shown");
  return view;
}
function visibleMapItems() {
  return within(
    screen.getByRole("list", { name: "Map memory list" }),
  ).queryAllByRole("listitem");
}
function projectedGraph(): MemoryGraphResponse {
  return JSON.parse(
    screen.getByTestId("graph-placeholder").dataset.graph ?? "{}",
  );
}
async function chooseSource(label: string) {
  fireEvent.click(screen.getByLabelText("Source"));
  fireEvent.click(await screen.findByRole("option", { name: label }));
}

it("default map search finds an existing 501st memory rather than declaring no matches from the first page", async () => {
  await mount();
  expect(
    within(screen.getByRole("list", { name: "Map memory list" })).getAllByRole(
      "listitem",
    ),
  ).toHaveLength(100);
  fireEvent.change(screen.getByLabelText("Filter map memories"), {
    target: { value: "Unique target of item 501" },
  });
  fireEvent.keyDown(screen.getByLabelText("Filter map memories"), {
    key: "Enter",
  });
  await waitFor(() =>
    expect(
      within(screen.getByRole("list", { name: "Map memory list" })).queryByText(
        "Unique target of item 501",
      ),
    ).not.toBeNull(),
  );
  expect(screen.queryByText("No matching memories")).toBeNull();
  const row = within(screen.getByRole("list", { name: "Map memory list" }))
    .getByText("Unique target of item 501")
    .closest("li");
  expect(row?.querySelector("small")?.textContent).toContain("codex");
  expect(row?.textContent).not.toContain("Source pending verification");
  const projected: MemoryGraphResponse = JSON.parse(
    screen.getByTestId("graph-placeholder").dataset.graph ?? "{}",
  );
  expect(projected.nodes.some((node) => node.node_id === "source:codex")).toBe(
    true,
  );
  expect(
    projected.edges.some(
      (edge) =>
        edge.origin === "provenance" &&
        [edge.source_node_id, edge.target_node_id].includes("memory:m501") &&
        [edge.source_node_id, edge.target_node_id].includes("source:codex"),
    ),
  ).toBe(true);
});

it("shows all evidence sources even while filtering one source or viewing topic labels", async () => {
  vi.mocked(api.getMemoryLibrary).mockResolvedValue({
    items: [{ ...corpus[500], tags: ["Planning topic"] }],
    total: 1,
    offset: 0,
    limit: 100,
    source_ids_by_memory: { m501: ["cursor", "codex"] },
  });
  render(<MemorySpace simplified sources={sources} />);
  await screen.findByText("1 settled · 1 shown");
  await chooseSource("cursor");
  await act(async () => {});
  const row = () =>
    within(screen.getByRole("list", { name: "Map memory list" }))
      .getByText("Unique target of item 501")
      .closest("li");
  expect(row()?.querySelector("small")?.textContent).toContain("cursor, codex");
  expect(row()?.querySelector("small")?.textContent).not.toContain(
    "Planning topic",
  );
  expect(screen.queryByText("More filters")).toBeNull();
  expect(screen.queryByRole("button", { name: "View by topic" })).toBeNull();
});

it("old page receipts without source metadata do not guess provenance from graph or selected filter", async () => {
  vi.mocked(api.getMemoryLibrary).mockResolvedValue({
    items: [corpus[0]],
    total: 1,
    offset: 0,
    limit: 100,
  });
  render(<MemorySpace simplified sources={sources} />);
  await screen.findByText("1 settled · 1 shown");
  await chooseSource("cursor");
  await act(async () => {});
  const row = within(screen.getByRole("list", { name: "Map memory list" }))
    .getByText(corpus[0].content)
    .closest("li");
  expect(row?.querySelector("small")?.textContent).toContain(
    "Source pending verification",
  );
});

it("rejects a source map for another page instead of displaying incorrect provenance", async () => {
  vi.mocked(api.getMemoryLibrary).mockResolvedValue({
    items: [corpus[0]],
    total: 1,
    offset: 0,
    limit: 100,
    source_ids_by_memory: { m501: ["codex"] },
  });
  render(<MemorySpace simplified sources={sources} />);
  await screen.findByText("Could not read memories. Try again.");
  expect(
    within(
      screen.getByRole("list", { name: "Map memory list" }),
    ).queryAllByRole("listitem"),
  ).toHaveLength(0);
});

it("OBS-13: the settled-count headline names the filter it is counting", async () => {
  await mount();
  expect(screen.getByText("501 settled · 100 shown")).toBeTruthy();
  fireEvent.change(screen.getByLabelText("Filter map memories"), {
    target: { value: "Unique target of item 501" },
  });
  await waitFor(() =>
    expect(
      screen.getByText(
        "1 settled (keyword: Unique target of item 501) · 1 shown",
      ),
    ).toBeTruthy(),
  );
  fireEvent.change(screen.getByLabelText("Filter map memories"), {
    target: { value: "" },
  });
  await chooseSource("codex");
  await waitFor(() =>
    expect(
      screen.getByText("1 settled (source: codex) · 1 shown"),
    ).toBeTruthy(),
  );
});

it("source selection includes an authorized source whose memories are outside the graph/page prefix", async () => {
  await mount();
  const select = screen.getByLabelText("Source");
  // The graph already contains its source node, but no first-page provenance edge.
  fireEvent.click(select);
  expect(
    within(screen.getByRole("listbox")).queryByRole("option", {
      name: "codex",
    }),
  ).not.toBeNull();
  fireEvent.click(screen.getByRole("option", { name: "codex" }));
  await waitFor(() =>
    expect(
      within(screen.getByRole("list", { name: "Map memory list" })).queryByText(
        "Unique target of item 501",
      ),
    ).not.toBeNull(),
  );
});

it("positive control: management search reaches the 501st record through the same library API", async () => {
  await mount();
  fireEvent(window, new Event("one-memory:manage"));
  await screen.findByText("501 total");
  fireEvent.change(screen.getByLabelText("Search the memory library"), {
    target: { value: "Unique target of item 501" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Search" }));
  await screen.findByRole("checkbox", {
    name: "Select: Unique target of item 501",
  });
  expect(screen.getByText("1 total")).toBeTruthy();
  expect(api.getMemoryLibrary).toHaveBeenLastCalledWith(
    {
      query: "Unique target of item 501",
      status: "confirmed",
      offset: 0,
      limit: 100,
    },
    expect.any(AbortSignal),
  );
});

it("late map poll cannot restore revoked-source body after authority changes", async () => {
  const view = await mount();
  vi.useFakeTimers();
  // Re-enter to establish the map poll under this controlled clock.
  view.rerender(<MemorySpace simplified sources={sources} active={false} />);
  view.rerender(<MemorySpace simplified sources={sources} />);
  await act(async () => {});
  let resolve!: (
    items: Awaited<ReturnType<typeof api.getMemoryLibrary>>,
  ) => void;
  vi.mocked(api.getMemoryLibrary).mockReturnValueOnce(
    new Promise((done) => {
      resolve = done;
    }),
  );
  await act(async () => {
    vi.advanceTimersByTime(10000);
  });
  const calls = vi.mocked(api.getMemoryLibrary).mock.calls.length;
  expect(calls).toBeGreaterThan(1);
  vi.mocked(api.getMemories).mockResolvedValue([]);
  vi.mocked(api.getMemoryGraph).mockResolvedValue(graph([]));
  view.rerender(
    <MemorySpace
      simplified
      sources={sources.map((source) => ({
        ...source,
        authorized: false,
        grant_revision: 2,
      }))}
      sourceSnapshotStale
    />,
  );
  vi.mocked(api.getMemory).mockClear();
  await act(async () =>
    resolve({ items: corpus.slice(0, 100), total: 501, offset: 0, limit: 100 }),
  );
  expect(
    within(screen.getByRole("list", { name: "Map memory list" })).queryByText(
      "Saved item 1",
    ),
  ).toBeNull();
  expect(api.getMemory).not.toHaveBeenCalled();
});

it("pages through confirmed records without dropping any or scanning the entire library", async () => {
  const mixed = corpus.map((item, index) => ({
    ...item,
    status: "confirmed" as const,
    updated_at: new Date(
      Date.UTC(2026, 8, 19, 0, 0, 501 - index),
    ).toISOString(),
  }));
  vi.mocked(api.getMemoryLibrary).mockImplementation(async (query = {}) => {
    const matches = mixed.filter((item) =>
      query.active_only
        ? item.status === "confirmed"
        : item.status === query.status,
    );
    return {
      items: matches.slice(query.offset ?? 0, (query.offset ?? 0) + 100),
      total: matches.length,
      offset: query.offset ?? 0,
      limit: 100,
    };
  });
  await mount();
  const list = () => screen.getByRole("list", { name: "Map memory list" });
  const ids = () =>
    Array.from(list().querySelectorAll("[data-memory-id]")).map((row) =>
      row.getAttribute("data-memory-id"),
    );
  const seen = new Set(ids());
  expect(api.getMemoryLibrary).toHaveBeenCalledTimes(1);
  for (let page = 1; page <= 5; page++) {
    fireEvent.click(
      screen.getByRole("button", { name: "Next page of memories" }),
    );
    await waitFor(() => expect(ids()[0]).toBe(`m${page * 100 + 1}`));
    for (const id of ids()) {
      expect(seen.has(id)).toBe(false);
      seen.add(id);
    }
  }
  expect(seen.size).toBe(501);
  expect(
    screen.getByRole("button", { name: "Next page of memories" }),
  ).toHaveProperty("disabled", true);
  expect(api.getMemoryLibrary).toHaveBeenCalledTimes(6);
  expect(
    vi
      .mocked(api.getMemoryLibrary)
      .mock.calls.every(
        ([query]) =>
          query?.limit === 100 &&
          query.active_only === true &&
          query.status === undefined,
      ),
  ).toBe(true);
  fireEvent.click(
    screen.getByRole("button", { name: "Previous page of memories" }),
  );
  await waitFor(() => expect(ids()[0]).toBe("m401"));
});

it("distinguishes failed search from empty, then retries the same query", async () => {
  await mount();
  const actual = vi.mocked(api.getMemoryLibrary).getMockImplementation();
  vi.mocked(api.getMemoryLibrary).mockRejectedValue(new Error("offline"));
  fireEvent.change(screen.getByLabelText("Filter map memories"), {
    target: { value: "target" },
  });
  await screen.findByText("Could not read memories. Try again.");
  expect(screen.queryByText("No matching memories")).toBeNull();
  expect(screen.queryByText("No memories have been generated yet")).toBeNull();
  expect(
    within(
      screen.getByRole("list", { name: "Map memory list" }),
    ).queryAllByRole("listitem"),
  ).toHaveLength(0);
  if (!actual) throw new Error("Missing fixture implementation");
  vi.mocked(api.getMemoryLibrary).mockImplementation(actual);
  fireEvent.click(screen.getByRole("button", { name: "Reload map memories" }));
  await within(
    screen.getByRole("list", { name: "Map memory list" }),
  ).findByText("Unique target of item 501");
  fireEvent.change(screen.getByLabelText("Filter map memories"), {
    target: { value: "Nothing matches this" },
  });
  await screen.findByText("No matching memories");
  expect(screen.queryByText("No memories have been generated yet")).toBeNull();
  expect(
    screen.getByText("0 settled (keyword: Nothing matches this) · 0 shown"),
  ).toBeTruthy();
});

it("aborts an older query after a source change and never replaces the newer result", async () => {
  await mount();
  const actual = vi.mocked(api.getMemoryLibrary).getMockImplementation();
  const resolve: Array<(value: MemoryLibraryPage) => void> = [];
  vi.mocked(api.getMemoryLibrary).mockImplementation(
    () =>
      new Promise((done) => {
        resolve.push(done);
      }),
  );
  fireEvent.change(screen.getByLabelText("Filter map memories"), {
    target: { value: "Old search" },
  });
  await waitFor(() => expect(resolve).toHaveLength(1));
  const signals = vi
    .mocked(api.getMemoryLibrary)
    .mock.calls.slice(-1)
    .map((call) => call[1]);
  if (!actual) throw new Error("Missing fixture implementation");
  vi.mocked(api.getMemoryLibrary).mockImplementation(actual);
  await chooseSource("codex");
  fireEvent.change(screen.getByLabelText("Filter map memories"), {
    target: { value: "target" },
  });
  await within(
    screen.getByRole("list", { name: "Map memory list" }),
  ).findByText("Unique target of item 501");
  expect(signals.every((signal) => signal?.aborted)).toBe(true);
  await act(async () => {
    resolve[0]({ items: [corpus[0]], total: 1, offset: 0, limit: 100 });
  });
  expect(
    within(screen.getByRole("list", { name: "Map memory list" })).queryByText(
      "Saved item 1",
    ),
  ).toBeNull();
  expect(vi.mocked(api.getMemoryLibrary).mock.calls.at(-1)?.[0]).toMatchObject({
    query: "target",
    source: "codex",
    offset: 0,
  });
});

it("hidden map aborts page reads and does not accept late body or poll until reactivated", async () => {
  const view = await mount();
  vi.useFakeTimers();
  const resolve: Array<(value: MemoryLibraryPage) => void> = [];
  vi.mocked(api.getMemoryLibrary).mockImplementation(
    () =>
      new Promise((done) => {
        resolve.push(done);
      }),
  );
  fireEvent.change(screen.getByLabelText("Filter map memories"), {
    target: { value: "Late" },
  });
  await act(async () => {});
  const signals = vi
    .mocked(api.getMemoryLibrary)
    .mock.calls.slice(-1)
    .map((call) => call[1]);
  view.rerender(<MemorySpace simplified sources={sources} active={false} />);
  expect(signals.every((signal) => signal?.aborted)).toBe(true);
  const reads = vi.mocked(api.getMemoryLibrary).mock.calls.length;
  await act(async () => {
    resolve[0]({
      items: [{ ...corpus[0], content: "Late body" }],
      total: 1,
      offset: 0,
      limit: 100,
    });
    vi.advanceTimersByTime(20000);
  });
  expect(screen.queryByText("Late body")).toBeNull();
  expect(api.getMemoryLibrary).toHaveBeenCalledTimes(reads);
  vi.mocked(api.getMemoryLibrary).mockResolvedValue({
    items: [],
    total: 0,
    offset: 0,
    limit: 100,
  });
  view.rerender(<MemorySpace simplified sources={sources} />);
  await act(async () => {});
  expect(vi.mocked(api.getMemoryLibrary).mock.calls.length).toBeGreaterThan(
    reads,
  );
});

it("simplified map stays on source grouping and does not offer extra filters", async () => {
  render(<MemorySpace simplified sources={sources} />);
  await screen.findByText("501 settled · 100 shown");
  expect(screen.getByLabelText("Source")).toBeTruthy();
  expect(screen.queryByText("More filters")).toBeNull();
  expect(screen.queryByRole("button", { name: "View by topic" })).toBeNull();
  expect(screen.queryByLabelText("Topic on this page")).toBeNull();
  expect(screen.getByText("Source attribution")).toBeTruthy();
});

it("keeps the populated map while a search is in flight and does not treat it as a missing model", async () => {
  vi.mocked(api.getMemoryProcessing).mockResolvedValue(
    processingBlocked("secure_storage_unavailable"),
  );
  await mount();
  expect(
    screen.queryByRole("status", { name: "Memories are not being generated" }),
  ).toBeNull();
  let resolve!: (page: MemoryLibraryPage) => void;
  vi.mocked(api.getMemoryLibrary).mockImplementation(
    () =>
      new Promise((done) => {
        resolve = done;
      }),
  );
  fireEvent.change(screen.getByLabelText("Filter map memories"), {
    target: { value: "onememory" },
  });
  await act(async () => {});
  expect(screen.queryByText("Loading...")).toBeNull();
  expect(screen.getByText("501 settled · 100 shown")).toBeTruthy();
  expect(
    screen.queryByRole("status", { name: "Memories are not being generated" }),
  ).toBeNull();
  expect(visibleMapItems()).toHaveLength(100);
  expect(projectedGraph().nodes.length).toBeGreaterThan(0);
  await act(async () =>
    resolve({
      items: [],
      total: 0,
      offset: 0,
      limit: 100,
      source_ids_by_memory: {},
    }),
  );
  await screen.findByText("No matching memories");
  expect(
    screen.queryByRole("status", { name: "Memories are not being generated" }),
  ).toBeNull();
  expect(screen.queryByText("No memories have been generated yet")).toBeNull();
});

it("clicking a source node does not empty the map or raise a model prompt while the source page loads", async () => {
  vi.mocked(api.getMemoryProcessing).mockResolvedValue(
    processingBlocked("secure_storage_unavailable", 2),
  );
  await mount();
  vi.mocked(api.getMemoryLibrary).mockImplementation(
    () => new Promise(() => {}),
  );
  fireEvent.click(
    screen.getByRole("button", { name: "Select cursor in the graph" }),
  );
  await act(async () => {});
  expect(screen.getByLabelText("Source")).toHaveProperty(
    "value",
    "source:cursor",
  );
  expect(screen.queryByText("Loading...")).toBeNull();
  expect(
    screen.queryByRole("status", { name: "Memories are not being generated" }),
  ).toBeNull();
  expect(visibleMapItems().length).toBeGreaterThan(0);
  expect(projectedGraph().nodes.length).toBeGreaterThan(0);
  expect(screen.queryByText("No memories have been generated yet")).toBeNull();
});

it("resets server offsets on source/search changes and keeps management pagination independent", async () => {
  await mount();
  fireEvent.click(
    screen.getByRole("button", { name: "Next page of memories" }),
  );
  await waitFor(() =>
    expect(vi.mocked(api.getMemoryLibrary).mock.lastCall?.[0]).toMatchObject({
      active_only: true,
      offset: 100,
    }),
  );
  await screen.findByRole("button", { name: "Previous page of memories" });
  await chooseSource("codex");
  await within(
    screen.getByRole("list", { name: "Map memory list" }),
  ).findByText("Unique target of item 501");
  expect(vi.mocked(api.getMemoryLibrary).mock.lastCall?.[0]).toMatchObject({
    source: "codex",
    offset: 0,
  });
  fireEvent(window, new Event("one-memory:manage"));
  await screen.findByText("501 total");
  expect(vi.mocked(api.getMemoryLibrary).mock.lastCall?.[0]).toEqual({
    query: "",
    status: "confirmed",
    offset: 0,
    limit: 100,
  });
  fireEvent.click(screen.getByRole("button", { name: "Next page" }));
  await waitFor(() =>
    expect(vi.mocked(api.getMemoryLibrary).mock.lastCall?.[0]?.offset).toBe(
      100,
    ),
  );
  fireEvent(window, new Event("one-memory:manage"));
  await within(
    screen.getByRole("list", { name: "Map memory list" }),
  ).findByText("Unique target of item 501");
  expect(vi.mocked(api.getMemoryLibrary).mock.lastCall?.[0]).toMatchObject({
    source: "codex",
    active_only: true,
    offset: 0,
  });
});

it("explicit refresh keeps the search scope and replaces the page with the latest result", async () => {
  await mount();
  fireEvent.change(screen.getByLabelText("Filter map memories"), {
    target: { value: "target" },
  });
  const list = screen.getByRole("list", { name: "Map memory list" });
  await within(list).findByText("Unique target of item 501");
  vi.mocked(api.getMemoryLibrary).mockResolvedValueOnce({
    items: [{ ...corpus[500], content: "target updated", revision: 2 }],
    total: 1,
    offset: 0,
    limit: 100,
  });
  fireEvent.click(screen.getByRole("button", { name: "Refresh map memories" }));
  await within(list).findByText("target updated");
  expect(within(list).queryByText("Unique target of item 501")).toBeNull();
  expect(vi.mocked(api.getMemoryLibrary).mock.lastCall?.[0]).toEqual({
    query: "target",
    active_only: true,
    offset: 0,
    limit: 100,
  });
});

it("rejects a non-active or wrong-offset response instead of displaying it as a valid page", async () => {
  await mount();
  vi.mocked(api.getMemoryLibrary).mockResolvedValueOnce({
    items: [{ ...corpus[0], status: "deleted" }],
    total: 1,
    offset: 0,
    limit: 100,
  });
  fireEvent.click(screen.getByRole("button", { name: "Refresh map memories" }));
  await screen.findByText("Could not read memories. Try again.");
  expect(
    within(screen.getByRole("list", { name: "Map memory list" })).queryByText(
      "Saved item 1",
    ),
  ).toBeNull();
  vi.mocked(api.getMemoryLibrary).mockResolvedValueOnce({
    items: [corpus[0]],
    total: 501,
    offset: 100,
    limit: 100,
  });
  fireEvent.click(screen.getByRole("button", { name: "Reload map memories" }));
  await screen.findByText("Could not read memories. Try again.");
  expect(
    within(
      screen.getByRole("list", { name: "Map memory list" }),
    ).queryAllByRole("listitem"),
  ).toHaveLength(0);
});

it("keeps the graph and list mounted during the ten-second refresh", async () => {
  vi.useFakeTimers();
  const view = render(<MemorySpace simplified sources={sources} />);
  await act(async () => {});
  expect(screen.getByText("501 settled · 100 shown")).toBeTruthy();
  const graphOwner = screen.getByTestId("graph-placeholder");
  const reads = vi.mocked(api.getMemoryLibrary).mock.calls.length;
  let done!: (page: MemoryLibraryPage) => void;
  vi.mocked(api.getMemoryLibrary).mockReturnValueOnce(
    new Promise((resolve) => {
      done = resolve;
    }),
  );
  await act(async () => {
    vi.advanceTimersByTime(10000);
  });
  expect(api.getMemoryLibrary).toHaveBeenCalledTimes(reads + 1);
  expect(screen.getByTestId("graph-placeholder")).toBe(graphOwner);
  expect(visibleMapItems()).toHaveLength(100);
  expect(screen.queryByText("Loading...")).toBeNull();
  await act(async () =>
    done({ items: corpus.slice(0, 100), total: 501, offset: 0, limit: 100 }),
  );
  expect(screen.getByTestId("graph-placeholder")).toBe(graphOwner);
  expect(visibleMapItems()).toHaveLength(100);
  view.unmount();
});
