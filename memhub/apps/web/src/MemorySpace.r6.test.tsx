// @vitest-environment jsdom
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { DIALOGUE_PANE_VISIBLE } from "./MemoryMap";
import { MemorySpace } from "./MemorySpace";

const { reads } = vi.hoisted(() => ({
  reads: {
    memories: vi.fn(),
    graph: vi.fn(),
    processing: vi.fn(),
    profiles: vi.fn(),
    library: vi.fn(),
    mapLibrary: vi.fn(),
    recent: vi.fn(),
    conversation: vi.fn(),
  },
}));
vi.mock("./api", async (importOriginal) => ({
  ...(await importOriginal<object>()),
  getMemories: reads.memories,
  getMemoryGraph: reads.graph,
  getMemoryProcessing: reads.processing,
  getModelProfiles: reads.profiles,
  getMemoryLibrary: (query: { active_only?: boolean }, signal?: AbortSignal) =>
    query.active_only === true
      ? reads.mapLibrary(query, signal)
      : reads.library(query, signal),
  getRecentConversations: reads.recent,
  getConversation: reads.conversation,
}));
vi.mock("./GraphWidthPrompt", () => ({ GraphWidthPrompt: () => null }));
vi.mock("./MemoryNoteComposer", () => ({
  MemoryNoteComposer: ({
    onPendingChange,
  }: {
    onPendingChange?(note: {
      evidenceId: string;
      text: string;
      state: string;
    }): void;
  }) => (
    <button
      type="button"
      onClick={() =>
        onPendingChange?.({
          evidenceId: "note-1",
          text: "Project weekly meetings are on Wednesday mornings.",
          state: "waiting_for_model",
        })
      }
    >
      Add a memory
    </button>
  ),
}));
vi.mock("./MemoryGraph3D", () => ({
  MemoryGraph3D: () => <div data-testid="graph-owner" />,
}));
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.clearAllMocks();
});
function setup() {
  const items = [
    {
      memory_id: "m1",
      content: "Saved memory",
      status: "confirmed",
      kind: "chat",
      origin: "extracted",
      tags: [],
      scopes: ["all-apps"],
      evidence_ids: ["e1"],
      revision: 1,
    },
  ];
  reads.memories.mockResolvedValue(items);
  reads.graph.mockResolvedValue({
    nodes: [
      {
        node_id: "memory:m1",
        node_type: "memory",
        memory_id: "m1",
        label: "Saved memory",
      },
    ],
    edges: [],
    total_memories: 1,
    projected_memories: 1,
  });
  reads.processing.mockResolvedValue({
    running: 0,
    pending: 0,
    blocked_stages: [],
    completed: 1,
  });
  reads.profiles.mockResolvedValue([]);
  reads.library.mockResolvedValue({
    items: [],
    total: 0,
    offset: 0,
    limit: 100,
  });
  reads.mapLibrary.mockResolvedValue({
    items,
    total: items.length,
    offset: 0,
    limit: 100,
  });
  reads.recent.mockResolvedValue({
    items: [
      {
        evidence_id: "e-live",
        conversation_id: "c1",
        source_id: "cursor",
        role: "user",
        written_by: "person",
        created_at: null,
        imported_at: "2026-09-23T06:47:00Z",
        excerpt: "Quote just synced",
        truncated: false,
      },
    ],
    total: 1,
    offset: 0,
    limit: 30,
  });
  reads.conversation.mockResolvedValue({
    conversation_id: "c1",
    source_id: "cursor",
    messages: [
      {
        evidence_id: "e-live",
        role: "user",
        written_by: "person",
        created_at: null,
        verbatim: "Full text of the quote just synced",
      },
    ],
  });
}

it("keeps a saved note visible while it waits for model processing", async () => {
  setup();
  render(<MemorySpace simplified onConfigureModel={vi.fn()} />);
  fireEvent.click(screen.getByRole("button", { name: "Add a memory" }));
  expect(
    screen.getByText(
      /Note saved, waiting to be processed: Project weekly meetings are on Wednesday mornings./,
    ),
  ).toBeTruthy();
  expect(
    screen.getByRole("button", { name: "Configure the processing model" }),
  ).toBeTruthy();
  expect(screen.queryByText("No memories have been generated yet")).toBeNull();
});
// D96: the memory page no longer has the raw-dialogue tab; runs again if it comes back.
it.skipIf(!DIALOGUE_PANE_VISIBLE)(
  "lists imported dialogue from the memory sidebar and opens the original",
  async () => {
    setup();
    render(<MemorySpace simplified />);
    await screen.findByText("Saved memory");
    fireEvent.click(screen.getByRole("tab", { name: "Raw dialogue" }));
    expect(await screen.findByText("Quote just synced")).toBeTruthy();
    expect(
      await screen.findByText("Full text of the quote just synced"),
    ).toBeTruthy();
    expect(
      screen.getByText("Freshly imported dialogue, not a settled memory yet"),
    ).toBeTruthy();
    expect(screen.queryByTestId("graph-owner")).toBeNull();
    fireEvent.click(screen.getByRole("tab", { name: "Memory" }));
    expect(await screen.findByTestId("graph-owner")).toBeTruthy();
  },
);
it("hides a recoverable graph read failure on the three-column surface", async () => {
  setup();
  reads.graph.mockRejectedValue(new Error("graph offline"));
  render(<MemorySpace simplified />);
  // Two copies now: the library list, and the detail pane naming what the
  // conversation settled (OBS-14). Waiting on one copy used to be enough.
  await screen.findAllByText("Saved memory");
  expect(screen.queryByText(/Failed to load the memory graph/)).toBeNull();
  expect(screen.queryByRole("alert")).toBeNull();
});
it("shows the three-column surface without four modes or a management button", async () => {
  setup();
  render(<MemorySpace simplified />);
  await screen.findByTestId("graph-owner");
  expect(reads.mapLibrary).toHaveBeenCalledWith(
    { query: "", active_only: true, offset: 0, limit: 100 },
    expect.any(AbortSignal),
  );
  expect(reads.library).not.toHaveBeenCalled();
  expect(
    screen.queryByRole("tablist", { name: "Memory space mode" }),
  ).toBeNull();
  expect(screen.queryByLabelText("Ask memories")).toBeNull();
  expect(screen.queryByRole("button", { name: "Manage memories" })).toBeNull();
  fireEvent(window, new Event("one-memory:manage"));
  expect(
    screen.getByRole("navigation", { name: "Memory governance status" }),
  ).toBeTruthy();
  await waitFor(() =>
    expect(reads.library).toHaveBeenCalledWith(
      { query: "", status: "confirmed", offset: 0, limit: 100 },
      expect.any(AbortSignal),
    ),
  );
  fireEvent(window, new Event("one-memory:manage"));
  expect(await screen.findByTestId("graph-owner")).toBeTruthy();
});
it("stops all page polling and disposes graph while inactive, then resumes", async () => {
  setup();
  vi.useFakeTimers();
  const view = render(<MemorySpace simplified />);
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
  expect(reads.memories).toHaveBeenCalled();
  expect(screen.getByTestId("graph-owner")).toBeTruthy();
  view.rerender(<MemorySpace simplified active={false} />);
  const counts = Object.values(reads).map((fn) => fn.mock.calls.length);
  await act(async () => {
    vi.advanceTimersByTime(30000);
    await Promise.resolve();
  });
  expect(Object.values(reads).map((fn) => fn.mock.calls.length)).toEqual(
    counts,
  );
  expect(screen.queryByTestId("graph-owner")).toBeNull();
  view.rerender(<MemorySpace simplified />);
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
  expect(reads.memories.mock.calls.length).toBeGreaterThan(counts[0]);
});
