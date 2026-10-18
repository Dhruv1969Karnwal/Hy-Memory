// @vitest-environment jsdom

import type { GovernedMemory, MemoryDetail } from "@one-memory/contracts";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { MemorySpace } from "./MemorySpace";
import { pollNow } from "./poll-scheduler";

// Text queries include hidden retained views by default; assertions target the visible owner.
const visibleText = { selector: ":not([hidden] *)" };

vi.mock("./MemoryGraph3D", () => ({
  MemoryGraph3D: ({ selectedNodeId }: { selectedNodeId: string | null }) => (
    <div data-testid="memory-graph" data-selected={selectedNodeId} />
  ),
}));

const memory: GovernedMemory = {
  memory_id: "memory-1",
  user_id: "local",
  kind: "chat",
  status: "confirmed",
  content: "Initial memory",
  scopes: ["all-apps"],
  evidence_ids: ["evidence-1"],
  origin: "extracted",
  tags: [],
  revision: 1,
  created_at: "2026-09-01T00:00:00Z",
  updated_at: "2026-09-01T00:00:00Z",
  deleted_at: null,
};

const detail: MemoryDetail = {
  memory,
  evidence: [
    {
      evidence_id: "evidence-1",
      available: true,
      source_id: "cursor",
      role: "user",
      written_by: "person",
      created_at: null,
      verbatim: "Original words",
    },
  ],
  history: [
    {
      memory_id: "memory-1",
      revision: 1,
      action: "create",
      status: "confirmed",
      content: "Initial memory",
      scopes: ["all-apps"],
      created_at: "2026-09-01T00:00:00Z",
    },
  ],
  index_sync: null,
};

const {
  apiErrorCode,
  getMemories,
  getMemory,
  getMemoryGraph,
  getMemoryProcessing,
  askMemoriesStream,
  mutateMemory,
  correctMemory,
  setMemoryScopes,
  syncMemoryIndex,
  setExtractionWindow,
  manualExtract,
  retryUnknownMemoryProcessing,
  getGraphWidth,
  decideGraphWidth,
  getModelProfiles,
  createMemory,
  digestMemoryGraph,
  searchConversations,
  getConversation,
  getConversationBrowser,
  getConversationMemories,
  getMemoryNotes,
  saveMemoryNote,
  processMemoryNote,
} = vi.hoisted(() => ({
  apiErrorCode: vi.fn(),
  getMemories: vi.fn(),
  getMemory: vi.fn(),
  getMemoryGraph: vi.fn(),
  getMemoryProcessing: vi.fn(),
  askMemoriesStream: vi.fn(),
  mutateMemory: vi.fn(),
  correctMemory: vi.fn(),
  setMemoryScopes: vi.fn(),
  syncMemoryIndex: vi.fn(),
  setExtractionWindow: vi.fn(),
  manualExtract: vi.fn(),
  retryUnknownMemoryProcessing: vi.fn(),
  getGraphWidth: vi.fn(),
  decideGraphWidth: vi.fn(),
  getModelProfiles: vi.fn(),
  createMemory: vi.fn(),
  digestMemoryGraph: vi.fn(),
  searchConversations: vi.fn(),
  getConversation: vi.fn(),
  getConversationBrowser: vi.fn(),
  getConversationMemories: vi.fn(),
  getMemoryNotes: vi.fn(),
  saveMemoryNote: vi.fn(),
  processMemoryNote: vi.fn(),
}));

const LIBRARY_REFRESH_MS = 10_000;

vi.mock("./api", () => ({
  apiErrorCode,
  getMemories,
  getMemory,
  getMemoryGraph,
  getMemoryProcessing,
  askMemoriesStream,
  mutateMemory,
  correctMemory,
  setMemoryScopes,
  syncMemoryIndex,
  setExtractionWindow,
  manualExtract,
  retryUnknownMemoryProcessing,
  getGraphWidth,
  decideGraphWidth,
  getModelProfiles,
  createMemory,
  digestMemoryGraph,
  searchConversations,
  getConversation,
  getConversationBrowser,
  getConversationMemories,
  getMemoryNotes,
  saveMemoryNote,
  processMemoryNote,
}));

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

describe("MemorySpace", () => {
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  beforeEach(() => {
    vi.resetAllMocks();
    apiErrorCode.mockReturnValue(null);
    getMemories.mockResolvedValue([memory]);
    getMemory.mockResolvedValue(detail);
    getMemoryGraph.mockResolvedValue({
      total_memories: 1,
      projected_memories: 1,
      nodes: [
        {
          node_id: "memory:memory-1",
          node_type: "memory",
          label: "Initial memory",
          memory_id: "memory-1",
          source_id: null,
          kind: "chat",
          status: "confirmed",
        },
      ],
      edges: [],
    });
    getMemoryProcessing.mockResolvedValue({
      pending: 0,
      running: 0,
      fresh_pending: 0,
      backlog_pending: 0,
      recent_completions: [],
      waiting_for_model: 0,
      completed: 1,
      failed: 0,
      unknown: 0,
      unknown_reasons: {},
      blocked_stages: [],
      outside_window_pending: 0,
      extraction_window_preset: "3m",
      manual_extract_batches_planned: 0,
      manual_extract_batches_current: 0,
      active_queue: 0,
      in_flight_count: 0,
      seconds_per_completion: null,
    });
    retryUnknownMemoryProcessing.mockResolvedValue({ requeued: 0 });
    mutateMemory.mockResolvedValue({
      ...memory,
      status: "confirmed",
      revision: 2,
    });
    correctMemory.mockResolvedValue({
      ...memory,
      content: "Corrected",
      revision: 2,
    });
    setMemoryScopes.mockResolvedValue({
      ...memory,
      scopes: ["private"],
      revision: 2,
    });
    syncMemoryIndex.mockResolvedValue({
      memory_id: "memory-1",
      status: "synced",
      attempts: 2,
      last_error_code: null,
      updated_at: "2026-09-01T00:00:00Z",
    });
    getGraphWidth.mockResolvedValue({ mismatch: null, memory_count: 1 });
    decideGraphWidth.mockResolvedValue({
      archived_path: null,
      mismatch: null,
    });
    getModelProfiles.mockResolvedValue([
      { profile_id: "profile-1", is_active: true },
    ]);
    getMemoryNotes.mockResolvedValue([]);
    searchConversations.mockResolvedValue({
      hits: [
        {
          evidence_id: "history-1",
          conversation_id: "history-conversation",
          source_id: "cursor",
          workspace: null,
          project: null,
          created_at: null,
          excerpt: "History excerpt",
        },
      ],
      empty_reason: null,
    });
    getConversation.mockResolvedValue({
      conversation_id: "history-conversation",
      source_id: "cursor",
      messages: [
        {
          evidence_id: "history-1",
          role: "user",
          written_by: "person",
          created_at: null,
          verbatim: "History original",
        },
      ],
    });
    getConversationBrowser.mockResolvedValue({
      items: [],
      total: 0,
      offset: 0,
      limit: 50,
    });
    getConversationMemories.mockResolvedValue({ items: [] });
    setExtractionWindow.mockResolvedValue({ window_preset: "1y" });
    manualExtract.mockResolvedValue({
      promoted: 25,
      batches_planned: 3,
      batches_current: 1,
    });
    askMemoriesStream.mockImplementation(async function* () {
      yield { type: "recall", memories: [memory] };
      yield { type: "delta", text: "Grounded " };
      yield { type: "delta", text: "answer" };
      yield {
        type: "final",
        answer: "Grounded answer",
        memories: [memory],
        mode: "generated",
      };
    });
  });

  describe("S1.3 teaching state preservation", () => {
    function expectNoTeachingWrites() {
      for (const write of [
        mutateMemory,
        correctMemory,
        setMemoryScopes,
        syncMemoryIndex,
        setExtractionWindow,
        manualExtract,
        decideGraphWidth,
        createMemory,
        digestMemoryGraph,
        saveMemoryNote,
        processMemoryNote,
      ])
        expect(write).not.toHaveBeenCalled();
    }

    it("navigates the memory mode tabs with the arrow keys", async () => {
      const view = render(<MemorySpace />);
      void view;
      const mapTab = screen.getByRole("tab", { name: "Workbench" });
      const libraryTab = screen.getByRole("tab", { name: "Library" });
      // WAI-ARIA tabs pattern: one tabstop only, the selected one.
      expect(mapTab.tabIndex).toBe(0);
      expect(libraryTab.tabIndex).toBe(-1);
      mapTab.focus();
      await userEvent.keyboard("{ArrowRight}");
      expect(libraryTab.getAttribute("aria-selected")).toBe("true");
      expect(libraryTab.tabIndex).toBe(0);
      expect(document.activeElement).toBe(libraryTab);
      await userEvent.keyboard("{ArrowLeft}");
      expect(mapTab.getAttribute("aria-selected")).toBe("true");
      expect(document.activeElement).toBe(mapTab);
      // Home/End jump to the ends of the tablist.
      await userEvent.keyboard("{End}");
      const historyTab = screen.getByRole("tab", { name: "History" });
      expect(historyTab.getAttribute("aria-selected")).toBe("true");
      expect(document.activeElement).toBe(historyTab);
      await userEvent.keyboard("{Home}");
      expect(mapTab.getAttribute("aria-selected")).toBe("true");
      // Every tab points at a panel that exists.
      for (const label of ["Workbench", "Library", "Ask memories", "History"]) {
        const tab = screen.getByRole("tab", { name: label });
        const controls = tab.getAttribute("aria-controls");
        expect(controls).toBeTruthy();
        expect(view.container.querySelector(`#${controls}`)).not.toBeNull();
      }
    });

    it("keeps the map DOM, selection, filters and question across every teaching view and route", async () => {
      const view = render(<MemorySpace />);
      const map = await screen.findByRole("region", {
        name: "Memory workbench",
      });
      await userEvent.type(
        within(map).getByLabelText("Filter map memories"),
        "Initial",
      );
      await userEvent.click(
        await within(map).findByRole("button", { name: /Initial memory/ }),
      );
      await within(map).findByText("Original words");
      await userEvent.type(
        within(map).getByLabelText("Ask your memories"),
        "Unsent map question",
      );
      await userEvent.click(
        within(map).getByRole("button", { name: "Correct" }),
      );
      fireEvent.change(within(map).getByLabelText("Memory content"), {
        target: { value: "Unsent correction" },
      });
      const canvas = screen.getByTestId("memory-graph");
      // The teaching prop is gone; the invariant to keep is that switching
      // between the real tabs must not rebuild the map or drop its state.
      for (const [label, mode] of [
        ["Workbench", "map"],
        ["Library", "library"],
        ["Ask memories", "ask"],
        ["History", "history"],
        ["Workbench", "map"],
      ] as const) {
        await userEvent.click(screen.getByRole("tab", { name: label }));
        expect(view.container.querySelector(".core-memory-map")).toBe(map);
        expect(screen.getByTestId("memory-graph")).toBe(canvas);
        expect(map.hidden).toBe(mode !== "map");
        expect(
          view.container
            .querySelector(`[data-tour-id="memory-${mode}-tab"]`)
            ?.getAttribute("aria-selected"),
        ).toBe("true");
      }
      view.rerender(<MemorySpace active={false} />);
      expect(screen.queryByRole("region", { name: "Memory space" })).toBeNull();
      expect(map.hidden).toBe(true);
      view.rerender(<MemorySpace />);
      expect(
        screen
          .getByRole("tab", { name: "Workbench" })
          .getAttribute("aria-selected"),
      ).toBe("true");
      expect(screen.getByTestId("memory-graph")).toBe(canvas);
      expect(canvas.dataset.selected).toBe("memory:memory-1");
      expect(
        (within(map).getByLabelText("Filter map memories") as HTMLInputElement)
          .value,
      ).toBe("Initial");
      expect(
        (within(map).getByLabelText("Ask your memories") as HTMLInputElement)
          .value,
      ).toBe("Unsent map question");
      expect(
        (within(map).getByLabelText("Memory content") as HTMLTextAreaElement)
          .value,
      ).toBe("Unsent correction");
      expect(getMemory).toHaveBeenCalledTimes(1);
      expect(askMemoriesStream).not.toHaveBeenCalled();
      expect(searchConversations).not.toHaveBeenCalled();
      for (const label of ["Library", "Ask memories", "History", "Workbench"]) {
        await userEvent.click(screen.getByRole("tab", { name: label }));
        expect(screen.getByTestId("memory-graph")).toBe(canvas);
      }
      expect(canvas.dataset.selected).toBe("memory:memory-1");
      expect(
        (within(map).getByLabelText("Ask your memories") as HTMLInputElement)
          .value,
      ).toBe("Unsent map question");
      expectNoTeachingWrites();
    });

    it.each(["confirmed", "superseded", "deleted"] as const)(
      "restores a %s library detail, draft, status, selection and query",
      async (status) => {
        const target = { ...memory, status };
        getMemories.mockResolvedValue([target]);
        getMemory.mockResolvedValue({ ...detail, memory: target });
        const view = render(<MemorySpace />);
        await userEvent.click(screen.getByRole("tab", { name: "Library" }));
        await userEvent.click(
          screen.getByRole("button", {
            name:
              status === "confirmed"
                ? /Available/
                : status === "superseded"
                  ? /Superseded/
                  : /Deleted/,
          }),
        );
        const query = screen.getByLabelText(
          "Search the memory library",
        ) as HTMLInputElement;
        await userEvent.type(query, "Unsubmitted library query");
        const checkbox = (await screen.findByRole("checkbox", {
          name: "Select: Initial memory",
        })) as HTMLInputElement;
        await userEvent.click(checkbox);
        await userEvent.click(
          screen.getByRole("button", { name: /^Initial memory/ }),
        );
        const inspector = await screen.findByRole("complementary", {
          name: "Memory detail",
        });
        if (status === "confirmed")
          fireEvent.change(within(inspector).getByLabelText("Memory content"), {
            target: { value: "Library correction draft" },
          });
        if (status === "confirmed") {
          await userEvent.click(
            within(inspector).getByRole("button", {
              name: "Adjust recall scope",
            }),
          );
          await userEvent.click(
            within(inspector).getByRole("radio", {
              name: "Stop external memory recall",
            }),
          );
        }
        // Tabs are the only way to change mode now, and a tab click clears the
        // answer and closes the detail by design. The invariant still worth
        // pinning is that suspending and resuming the surface keeps the
        // library detail, draft, selection and query.
        view.rerender(<MemorySpace active={false} />);
        view.rerender(<MemorySpace />);
        expect(
          screen
            .getByRole("tab", { name: "Library" })
            .getAttribute("aria-selected"),
        ).toBe("true");
        expect(screen.getByLabelText("Search the memory library")).toBe(query);
        expect(query.value).toBe("Unsubmitted library query");
        expect(
          screen.getByRole("checkbox", { name: "Select: Initial memory" }),
        ).toBe(checkbox);
        expect(checkbox.checked).toBe(true);
        expect(
          screen.getByRole("complementary", { name: "Memory detail" }),
        ).toBe(inspector);
        expect(within(inspector).getByText("Original words")).toBeTruthy();
        if (status === "confirmed")
          expect(
            (
              within(inspector).getByLabelText(
                "Memory content",
              ) as HTMLTextAreaElement
            ).value,
          ).toBe("Library correction draft");
        if (status === "confirmed")
          expect(
            (
              within(inspector).getByRole("radio", {
                name: "Stop external memory recall",
              }) as HTMLInputElement
            ).checked,
          ).toBe(true);
        expect(getMemory).toHaveBeenCalledTimes(1);
        expect(getMemories).toHaveBeenCalledTimes(1);
        expectNoTeachingWrites();
      },
    );

    it("keeps the real history owner and results without repeating searches on tour or ordinary tabs", async () => {
      const view = render(<MemorySpace />);
      expect(searchConversations).not.toHaveBeenCalled();
      await userEvent.click(screen.getByRole("tab", { name: "History" }));
      const input = screen.getByLabelText(
        "Search conversation history",
      ) as HTMLInputElement;
      await userEvent.type(input, "History query");
      await userEvent.click(screen.getByRole("button", { name: "Search" }));
      await userEvent.click(
        await screen.findByRole("button", { name: "View source text" }),
      );
      const original = await screen.findByText("History original");
      await userEvent.click(screen.getByRole("tab", { name: "Workbench" }));
      await userEvent.click(screen.getByRole("tab", { name: "History" }));
      expect(screen.getByLabelText("Search conversation history")).toBe(input);
      for (const [label, mode] of [
        ["Workbench", "map"],
        ["Ask memories", "ask"],
        ["Library", "library"],
        ["History", "history"],
      ] as const) {
        await userEvent.click(screen.getByRole("tab", { name: label }));
        expect(
          screen
            .getByRole("tab", { selected: true })
            .getAttribute("data-tour-id"),
        ).toBe(`memory-${mode}-tab`);
      }
      view.rerender(<MemorySpace active={false} />);
      view.rerender(<MemorySpace />);
      expect(
        screen
          .getByRole("tab", { name: "History" })
          .getAttribute("aria-selected"),
      ).toBe("true");
      expect(screen.getByLabelText("Search conversation history")).toBe(input);
      expect(input.value).toBe("History query");
      expect(screen.getByText("History original")).toBe(original);
      expect(searchConversations).toHaveBeenCalledExactlyOnceWith(
        "History query",
      );
      expect(getConversation).toHaveBeenCalledExactlyOnceWith("history-1");
      expectNoTeachingWrites();
    });

    it("preserves an existing answer and scope without asking again", async () => {
      const view = render(<MemorySpace />);
      await userEvent.click(screen.getByRole("tab", { name: "Ask memories" }));
      await userEvent.selectOptions(
        screen.getByLabelText("Ask memory scope"),
        "*",
      );
      await userEvent.type(
        screen.getByLabelText("Ask the confirmed memories"),
        "Question",
      );
      await userEvent.click(
        screen.getByRole("button", { name: "Answer from memories" }),
      );
      const answer = await screen.findByRole("heading", {
        name: "Grounded answer",
      });
      // A tab click intentionally clears the answer, so the preserved state to
      // assert is across a suspend/resume of the same tab, not across tabs.
      view.rerender(<MemorySpace active={false} />);
      view.rerender(<MemorySpace />);
      expect(screen.getByRole("heading", { name: "Grounded answer" })).toBe(
        answer,
      );
      expect(
        (screen.getByLabelText("Ask memory scope") as HTMLSelectElement).value,
      ).toBe("*");
      expect(
        (
          screen.getByLabelText(
            "Ask the confirmed memories",
          ) as HTMLInputElement
        ).value,
      ).toBe("Question");
      // The trace id is the 4th argument: the SSE path uses raw `fetch`, so it
      // must carry the header itself for Core's kernel legs to join the
      // renderer's ui.action/ui.result.
      expect(askMemoriesStream).toHaveBeenCalledExactlyOnceWith(
        "Question",
        "*",
        undefined,
        expect.any(String),
      );
      expectNoTeachingWrites();
    });

    it("does not pass inactive events through to business controls", async () => {
      const view = render(<MemorySpace />);
      await userEvent.click(
        await screen.findByRole("button", { name: /Initial memory/ }),
      );
      await userEvent.click(screen.getByRole("button", { name: "Correct" }));
      const save = screen.getByRole("button", { name: "Save correction" });
      const correction = screen
        .getByRole("textbox", { name: "Memory content" })
        .closest("form");
      if (!correction) throw new Error("Correction form missing");
      // Take the surface out of active use: nothing on it may still write.
      view.rerender(<MemorySpace active={false} />);
      fireEvent.click(save);
      fireEvent.submit(correction);
      expectNoTeachingWrites();
      const queryForm = view.container.querySelector(
        '[data-tour-id="memory-query-bar"]',
      );
      if (!queryForm) throw new Error("Query form missing");
      fireEvent.submit(queryForm);
      expectNoTeachingWrites();
      view.rerender(<MemorySpace />);
      fireEvent.submit(correction);
      await waitFor(() => expect(correctMemory).toHaveBeenCalledTimes(1));
    });

    it("shows a late blocking dialog without changing the user's mode", async () => {
      const width = deferred<{
        mismatch: {
          stored_width: number;
          expected_width: number;
          graph_path: string;
        };
        memory_count: number;
      }>();
      getGraphWidth.mockReturnValue(width.promise);
      const view = render(<MemorySpace />);
      await userEvent.click(screen.getByRole("tab", { name: "Ask memories" }));
      await act(async () =>
        width.resolve({
          mismatch: {
            stored_width: 768,
            expected_width: 512,
            graph_path: "/tmp/synthetic-graph",
          },
          memory_count: 1,
        }),
      );
      const dialog = await screen.findByRole("dialog", {
        name: "The existing memory library does not match the built-in model",
      });
      expectNoTeachingWrites();
      await userEvent.click(
        within(dialog).getByRole("button", { name: "Do that" }),
      );
      expect(decideGraphWidth).toHaveBeenCalledExactlyOnceWith(
        "keep_existing",
        expect.any(String),
      );
      view.rerender(<MemorySpace />);
      // Dismissing the dialog must not move the user off the tab they chose.
      expect(
        screen
          .getByRole("tab", { name: "Ask memories" })
          .getAttribute("aria-selected"),
      ).toBe("true");
    });

    it("retains an unsaved composer draft while its inactive modal is suspended", async () => {
      const view = render(<MemorySpace />);
      await userEvent.click(screen.getByRole("button", { name: "Add memory" }));
      const draft = screen.getByLabelText(
        "Note this down",
      ) as HTMLTextAreaElement;
      await userEvent.type(draft, "Unsent note");
      view.rerender(<MemorySpace active={false} />);
      expect(screen.queryByRole("dialog")).toBeNull();
      const event = new KeyboardEvent("keydown", {
        key: "Escape",
        bubbles: true,
        cancelable: true,
      });
      document.dispatchEvent(event);
      expect(event.defaultPrevented).toBe(false);
      view.rerender(<MemorySpace />);
      expect(screen.getByLabelText("Note this down")).toBe(draft);
      expect(draft.value).toBe("Unsent note");
      expectNoTeachingWrites();
    });

    it("keeps its tab anchors while loading and empty states are shown", async () => {
      const items = deferred<GovernedMemory[]>();
      getMemories.mockReturnValue(items.promise);
      getMemoryGraph.mockResolvedValue({
        nodes: [],
        edges: [],
        total_memories: 0,
      });
      const view = render(<MemorySpace />);
      expect(screen.getByText("Reading memories...")).toBeTruthy();
      expect(
        view.container.querySelector('[data-tour-id="memory-map-canvas"]')
          ?.className,
      ).toBe("map-canvas");
      await userEvent.click(screen.getByRole("tab", { name: "Library" }));
      expect(
        view.container.querySelector('[data-tour-id="memory-status-tabs"]')
          ?.tagName,
      ).toBe("NAV");
      expect(screen.getByText("Loading the memory library...")).toBeTruthy();
      await userEvent.click(screen.getByRole("tab", { name: "Ask memories" }));
      expect(
        view.container.querySelector('[data-tour-id="memory-query-bar"]')
          ?.tagName,
      ).toBe("FORM");
      await act(async () => items.resolve([]));
      view.rerender(<MemorySpace />);
      // The arrival status belongs to the workbench surface, not the ask tab.
      await userEvent.click(screen.getByRole("tab", { name: "Workbench" }));
      expect(
        screen.getByRole("region", { name: "Memory organizing status" }),
      ).toBeTruthy();
      expect(screen.getByText("No conversations saved yet")).toBeTruthy();
      expect(screen.queryByTestId("memory-graph")).toBeNull();
      expectNoTeachingWrites();
    });
  });

  describe("Persistent prompt when no model is configured (RL-04)", () => {
    function blockedOnModel(reason: string) {
      getMemoryProcessing.mockResolvedValue({
        pending: 0,
        running: 0,
        fresh_pending: 0,
        backlog_pending: 0,
        recent_completions: [],
        waiting_for_model: 7,
        completed: 0,
        failed: 0,
        blocked_stages: [{ kind: "extract", reason, count: 7 }],
        outside_window_pending: 0,
        extraction_window_preset: "3m",
        manual_extract_batches_planned: 0,
        manual_extract_batches_current: 0,
        active_queue: 0,
        in_flight_count: 0,
        seconds_per_completion: null,
      });
    }

    it("keeps the prompt resident with no usable model and links straight to settings", async () => {
      const onConfigureModel = vi.fn();
      blockedOnModel("model_profile_not_active");

      render(
        <MemorySpace
          onConnect={() => {}}
          onConfigureModel={onConfigureModel}
        />,
      );

      const prompt = await screen.findByRole("status", {
        name: "Memories are not being generated",
      });
      expect(prompt.textContent).toContain(
        "No generation model is enabled yet",
      );

      // Resident: switching to another mode must not take it away.
      await userEvent.click(screen.getByRole("tab", { name: "Library" }));
      expect(
        screen.getByRole("status", {
          name: "Memories are not being generated",
        }),
      ).toBeTruthy();

      await userEvent.click(
        screen.getByRole("button", { name: "Configure the generation model" }),
      );
      expect(onConfigureModel).toHaveBeenCalledTimes(1);
    });

    it("uses the existing reason mapping rather than a hard-coded sentence", async () => {
      blockedOnModel("model_api_key_missing");

      render(<MemorySpace onConnect={() => {}} onConfigureModel={() => {}} />);

      const prompt = await screen.findByRole("status", {
        name: "Memories are not being generated",
      });
      expect(prompt.textContent).toContain("Generation model API key missing");
      expect(prompt.textContent).not.toContain(
        "No generation model is enabled yet",
      );
    });

    it("drops the prompt once a model is configured", async () => {
      // The default mock has empty blocked_stages and completed 1, i.e. memories are coming out normally.
      render(<MemorySpace onConnect={() => {}} onConfigureModel={() => {}} />);

      await waitFor(() => {
        expect(screen.getByText("Initial memory", visibleText)).toBeTruthy();
      });
      expect(
        screen.queryByRole("status", {
          name: "Memories are not being generated",
        }),
      ).toBeNull();
    });

    it("does not pose as a model problem when the block reason is unrelated to the model", async () => {
      blockedOnModel("hy_memory_unavailable");

      render(<MemorySpace onConnect={() => {}} onConfigureModel={() => {}} />);

      await waitFor(() => {
        expect(screen.getByText("Initial memory", visibleText)).toBeTruthy();
      });
      expect(
        screen.queryByRole("status", {
          name: "Memories are not being generated",
        }),
      ).toBeNull();
    });
  });

  describe("Library multi-select and bulk actions (RL-27)", () => {
    const rows = [
      { ...memory, memory_id: "memory-1", content: "First", revision: 1 },
      { ...memory, memory_id: "memory-2", content: "Second", revision: 3 },
      { ...memory, memory_id: "memory-3", content: "Third", revision: 2 },
    ];

    async function openLibrary() {
      getMemories.mockResolvedValue(rows);
      mutateMemory.mockImplementation(async (id: string) => {
        const target = rows.find((item) => item.memory_id === id);
        if (!target) throw new Error("Unexpected batch target");
        return { ...target, status: "deleted", revision: target.revision + 1 };
      });
      render(<MemorySpace onConnect={() => {}} />);
      await userEvent.click(screen.getByRole("tab", { name: "Library" }));
      await waitFor(() => {
        expect(
          screen.getByRole("checkbox", { name: "Select: Second" }),
        ).toBeTruthy();
      });
    }

    async function confirmDeletion() {
      await userEvent.click(
        screen.getByRole("button", { name: "Confirm Delete" }),
      );
    }

    async function bulkDelete(count: number) {
      await userEvent.click(
        screen.getByRole("button", { name: `Delete ${count} selected` }),
      );
      await confirmDeletion();
    }

    async function select(...contents: string[]) {
      for (const content of contents) {
        await userEvent.click(
          screen.getByRole("checkbox", { name: `Select: ${content}` }),
        );
      }
    }

    it("sends one bulk delete request per item, each with its own revision", async () => {
      vi.spyOn(window, "confirm").mockReturnValue(true);
      await openLibrary();
      await select("First", "Second");

      await userEvent.click(
        screen.getByRole("button", { name: "Delete 2 selected" }),
      );
      await userEvent.click(
        screen.getByRole("button", { name: "Confirm Delete" }),
      );

      await waitFor(() => {
        expect(mutateMemory).toHaveBeenCalledTimes(2);
      });
      const calls = mutateMemory.mock.calls.map((call) => [
        call[0],
        call[1],
        call[2],
      ]);
      expect(calls).toContainEqual(["memory-1", "delete", 1]);
      expect(calls).toContainEqual(["memory-2", "delete", 3]);
      expect(calls.map((call) => call[0])).not.toContain("memory-3");
    });

    it("confirms a bulk delete first and sends nothing when it is cancelled", async () => {
      vi.spyOn(window, "confirm").mockReturnValue(false);
      await openLibrary();
      await select("First");

      await userEvent.click(
        screen.getByRole("button", { name: "Delete 1 selected" }),
      );
      await userEvent.click(screen.getByRole("button", { name: "Cancel" }));

      expect(mutateMemory).not.toHaveBeenCalled();
    });

    it("offers no bulk action when nothing is selected", async () => {
      await openLibrary();

      expect(
        screen.queryByRole("button", { name: /Delete \d+ selected/ }),
      ).toBeNull();
    });

    it("has no pending-confirmation or rejected filter in the library, and no confirm or reject action", async () => {
      await openLibrary();

      const tabs = screen.getByRole("navigation", {
        name: "Memory governance status",
      });
      expect(
        within(tabs)
          .getAllByRole("button")
          .map((tab) => tab.textContent?.replace(/\s*\d*$/, "")),
      ).toEqual(["Available", "Superseded", "Deleted"]);

      await select("First");
      const bulk = screen.getByRole("group", { name: "Bulk actions" });
      expect(
        within(bulk)
          .getAllByRole("button")
          .map((button) => button.textContent),
      ).toEqual(["Delete 1 selected", "Clear selection"]);

      getMemory.mockResolvedValue({ ...detail, memory: rows[0] });
      await userEvent.click(screen.getByRole("button", { name: /^First/ }));
      const inspector = await screen.findByRole("complementary", {
        name: "Memory detail",
      });
      await within(inspector).findByText("Original words");
      expect(
        within(inspector).getByRole("button", { name: "Delete" }),
      ).toBeTruthy();
      expect(
        within(inspector).queryByRole("button", { name: /Confirm|Reject/ }),
      ).toBeNull();
    });

    it("S1.3: changing status clears hidden targets from the next batch", async () => {
      await openLibrary();
      await select("First");
      await userEvent.click(screen.getByRole("button", { name: /Superseded/ }));
      expect(screen.queryByRole("group", { name: "Bulk actions" })).toBeNull();
      await userEvent.click(screen.getByRole("button", { name: /Available/ }));
      await select("Second");
      await bulkDelete(1);
      await waitFor(() => expect(mutateMemory).toHaveBeenCalledTimes(1));
      expect(mutateMemory.mock.calls[0].slice(0, 3)).toEqual([
        "memory-2",
        "delete",
        3,
      ]);
    });

    it("S1.3: changing the search clears selection before new results arrive", async () => {
      await openLibrary();
      await select("First", "Second");
      await userEvent.type(
        screen.getByLabelText("Search the memory library"),
        "Third",
      );
      expect(screen.queryByRole("group", { name: "Bulk actions" })).toBeNull();
      expect(mutateMemory).not.toHaveBeenCalled();
    });

    it("S1.3: removed memories cannot leave a phantom selected count", async () => {
      await openLibrary();
      await select("First", "Second");
      getMemories.mockResolvedValue([rows[2]]);
      // The library surface is refreshed through the shared scheduler now;
      // one call runs exactly one tick for it.
      await act(async () => pollNow("memory-library"));
      expect(screen.queryByRole("group", { name: "Bulk actions" })).toBeNull();
      getMemories.mockResolvedValue(rows);
      await act(async () => pollNow("memory-library"));
      expect(
        (
          screen.getByRole("checkbox", {
            name: "Select: First",
          }) as HTMLInputElement
        ).checked,
      ).toBe(false);
    });

    it("S1.3: same-tick submits send one batch and lock its selection", async () => {
      await openLibrary();
      await select("First");
      const pending = deferred<GovernedMemory>();
      mutateMemory.mockReturnValueOnce(pending.promise);
      const remove = screen.getByRole("button", { name: "Delete 1 selected" });
      try {
        act(() => {
          remove.click();
          remove.click();
        });
        await confirmDeletion();
        expect(mutateMemory).toHaveBeenCalledTimes(1);
        expect(
          (
            screen.getByRole("checkbox", {
              name: "Select: Second",
            }) as HTMLInputElement
          ).disabled,
        ).toBe(true);
        expect(
          (
            screen.getByRole("button", {
              name: "Clear selection",
            }) as HTMLButtonElement
          ).disabled,
        ).toBe(true);
        expect(
          (
            screen.getByLabelText(
              "Search the memory library",
            ) as HTMLInputElement
          ).disabled,
        ).toBe(true);
      } finally {
        await act(async () =>
          pending.resolve({
            ...rows[0],
            status: "deleted",
            revision: 2,
          }),
        );
      }
    });

    it("S1.3: delete owns the batch lock before confirmation and cancellation releases it", async () => {
      await openLibrary();
      await select("First");
      const remove = screen.getByRole("button", { name: "Delete 1 selected" });
      act(() => {
        remove.click();
        remove.click();
      });
      try {
        expect(
          (
            screen.getByRole("button", {
              name: "Delete 1 selected",
            }) as HTMLButtonElement
          ).disabled,
        ).toBe(true);
        expect(mutateMemory).not.toHaveBeenCalled();
      } finally {
        await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
      }
      await bulkDelete(1);
      await waitFor(() => expect(mutateMemory).toHaveBeenCalledTimes(1));
    });

    it.each(["memories", "graph"])(
      "S1.3: a failed final %s refresh releases busy without retrying saved items",
      async (panel) => {
        await openLibrary();
        await select("First", "Second");
        getMemories.mockResolvedValue([
          { ...rows[0], status: "deleted", revision: 2 },
          { ...rows[1], status: "deleted", revision: 4 },
          rows[2],
        ]);
        const reader = panel === "memories" ? getMemories : getMemoryGraph;
        reader.mockRejectedValueOnce(new Error("Final refresh unavailable"));
        await bulkDelete(2);
        await waitFor(() => expect(mutateMemory).toHaveBeenCalledTimes(2));
        await userEvent.click(
          screen.getByRole("checkbox", { name: "Select: Third" }),
        );
        await waitFor(() =>
          expect(
            (
              screen.getByRole("button", {
                name: "Delete 1 selected",
              }) as HTMLButtonElement
            ).disabled,
          ).toBe(false),
        );
        expect(
          screen.getByRole("status", { name: "Bulk action result" })
            .textContent,
        ).toContain("Bulk deletion: 2 succeeded");
        if (panel === "graph") {
          await waitFor(
            () => expect(getMemoryGraph.mock.calls.length).toBeGreaterThan(1),
            { timeout: 3500 },
          );
          expect(
            screen.queryByText(/Failed to load the memory graph/),
          ).toBeNull();
        } else {
          expect(
            (await screen.findByRole("alert", {}, { timeout: 3500 }))
              .textContent,
          ).toContain("the list refresh failed");
        }
        expect(
          screen.queryByRole("checkbox", { name: "Select: First" }),
        ).toBeNull();
        expect(
          screen.queryByRole("checkbox", { name: "Select: Second" }),
        ).toBeNull();
        expect(mutateMemory).toHaveBeenCalledTimes(2);
      },
    );

    it("S1.3: per-item retry keeps the failed intent and never includes successful or newly selected items", async () => {
      await openLibrary();
      apiErrorCode.mockImplementation((reason: Error) => reason.message);
      mutateMemory
        .mockResolvedValueOnce({
          ...rows[0],
          status: "deleted",
          revision: 2,
        })
        .mockRejectedValueOnce(new Error("local_store_unavailable"));
      getMemories.mockResolvedValue([
        { ...rows[0], status: "deleted", revision: 2 },
        rows[1],
        rows[2],
      ]);
      await select("First", "Second");
      await bulkDelete(2);
      const failure = await screen.findByRole("group", {
        name: "Action failed: Second",
      });
      expect(failure.textContent).toContain("The local memory store is busy");
      expect(
        screen.getByRole("status", { name: "Bulk action result" }).textContent,
      ).toContain("Bulk deletion: 1 succeeded, 1 not completed");
      expect(
        (
          screen.getByRole("checkbox", {
            name: "Select: Second",
          }) as HTMLInputElement
        ).checked,
      ).toBe(true);
      const failedCall = mutateMemory.mock.calls[1];
      await select("Third");
      await userEvent.click(
        within(failure).getByRole("button", { name: "Retry deletion" }),
      );
      await confirmDeletion();
      await waitFor(() => expect(mutateMemory).toHaveBeenCalledTimes(3));
      // A retry is its own traced interaction (fresh trace id); the intent that
      // must be preserved is the idempotent request id and CAS revision.
      expect(mutateMemory.mock.calls[2].slice(0, 4)).toEqual(
        failedCall.slice(0, 4),
      );
      expect(mutateMemory.mock.calls[2][4]).toEqual(expect.any(String));
      expect(
        screen.queryByRole("group", { name: "Action failed: Second" }),
      ).toBeNull();
      expect(
        (
          screen.getByRole("checkbox", {
            name: "Select: Third",
          }) as HTMLInputElement
        ).checked,
      ).toBe(true);
    });

    it.each(["memories", "graph"])(
      "S1.3: partial failure survives a %s refresh failure and only failed targets can be retried",
      async (panel) => {
        await openLibrary();
        await select("First", "Second");
        mutateMemory
          .mockResolvedValueOnce({
            ...rows[0],
            status: "deleted",
            revision: 2,
          })
          .mockRejectedValueOnce(new Error("Connection lost"));
        const reader = panel === "graph" ? getMemoryGraph : getMemories;
        reader.mockRejectedValueOnce(new Error("Refresh failed"));
        await bulkDelete(2);
        const retry = await screen.findByRole("button", {
          name: "Retry 1 failed",
        });
        await waitFor(() =>
          expect((retry as HTMLButtonElement).disabled).toBe(false),
        );
        expect(
          screen.getByRole("group", { name: "Action failed: Second" }),
        ).toBeTruthy();
        if (panel === "graph") {
          await waitFor(
            () => expect(getMemoryGraph.mock.calls.length).toBeGreaterThan(1),
            { timeout: 3500 },
          );
          expect(
            screen.queryByText(/Failed to load the memory graph/),
          ).toBeNull();
        } else {
          expect(
            (await screen.findByRole("alert", {}, { timeout: 3500 }))
              .textContent,
          ).toContain("the list refresh failed");
        }
        const failedCall = mutateMemory.mock.calls[1];
        await userEvent.click(retry);
        await confirmDeletion();
        await waitFor(() => expect(mutateMemory).toHaveBeenCalledTimes(3));
        expect(mutateMemory.mock.calls[2].slice(0, 4)).toEqual(
          failedCall.slice(0, 4),
        );
      },
    );

    it("S1.3: retry uses the original revision even if refresh returns a newer version", async () => {
      await openLibrary();
      await select("Second");
      mutateMemory.mockRejectedValueOnce(new Error("Response lost"));
      getMemories.mockResolvedValue([
        rows[0],
        { ...rows[1], revision: 4 },
        rows[2],
      ]);
      await bulkDelete(1);
      const retry = await screen.findByRole("button", {
        name: "Retry deletion",
      });
      const failedCall = mutateMemory.mock.calls[0];
      await userEvent.click(retry);
      await confirmDeletion();
      await waitFor(() => expect(mutateMemory).toHaveBeenCalledTimes(2));
      expect(mutateMemory.mock.calls[1].slice(0, 4)).toEqual(
        failedCall.slice(0, 4),
      );
      expect(mutateMemory.mock.calls[1][2]).toBe(3);
    });

    it("S1.3: editing a detail draft cannot replace a failed request identity", async () => {
      await openLibrary();
      getMemory.mockResolvedValue({ ...detail, memory: rows[1] });
      await select("Second");
      mutateMemory.mockRejectedValueOnce(new Error("Response lost"));
      await bulkDelete(1);
      const retry = await screen.findByRole("button", {
        name: "Retry deletion",
      });
      const failedCall = mutateMemory.mock.calls[0];
      await userEvent.click(screen.getByRole("button", { name: /^Second/ }));
      await userEvent.type(
        await screen.findByRole("textbox", { name: "Memory content" }),
        " draft",
      );
      await userEvent.click(retry);
      await confirmDeletion();
      await waitFor(() => expect(mutateMemory).toHaveBeenCalledTimes(2));
      expect(mutateMemory.mock.calls[1].slice(0, 4)).toEqual(
        failedCall.slice(0, 4),
      );
      expect(correctMemory).not.toHaveBeenCalled();
    });

    it("S1.3: an idempotent receipt cannot roll back a newer visible revision when refresh fails", async () => {
      await openLibrary();
      await select("Second");
      mutateMemory.mockRejectedValueOnce(new Error("Response lost"));
      getMemories.mockResolvedValue([
        rows[0],
        { ...rows[1], content: "Second updated", revision: 5 },
        rows[2],
      ]);
      await bulkDelete(1);
      const retry = await screen.findByRole("button", {
        name: "Retry deletion",
      });
      mutateMemory.mockResolvedValueOnce({
        ...rows[1],
        status: "deleted",
        revision: 4,
      });
      getMemoryGraph.mockRejectedValueOnce(new Error("Graph refresh failed"));
      await userEvent.click(retry);
      await confirmDeletion();
      await waitFor(() => expect(mutateMemory).toHaveBeenCalledTimes(2));
      expect(screen.getByText("Second updated", visibleText)).toBeTruthy();
      expect(screen.queryByText("Second", visibleText)).toBeNull();
      expect(mutateMemory.mock.calls[1].slice(0, 4)).toEqual(
        mutateMemory.mock.calls[0].slice(0, 4),
      );
    });

    it("S1.3: an unknown write outcome asks for explicit retry instead of claiming an automatic retry", async () => {
      await openLibrary();
      await select("Second");
      mutateMemory.mockRejectedValueOnce(new Error("Response lost"));
      await bulkDelete(1);
      const failure = await screen.findByRole("group", {
        name: "Action failed: Second",
      });
      expect(failure.textContent).toContain(
        "No result received for the operation",
      );
      expect(failure.textContent).not.toContain("being retried");
      expect(mutateMemory).toHaveBeenCalledTimes(1);
    });

    it("S1.3: revocation removes a failed row and its retry without discarding other failures", async () => {
      await openLibrary();
      await select("First", "Second");
      mutateMemory.mockRejectedValue(new Error("Response lost"));
      await bulkDelete(2);
      await screen.findByRole("button", { name: "Retry 2 failed" });
      getMemories.mockResolvedValue([rows[1], rows[2]]);
      await act(async () => pollNow("memory-library"));
      expect(
        screen.queryByRole("group", { name: "Action failed: First" }),
      ).toBeNull();
      expect(screen.queryByText("First")).toBeNull();
      await userEvent.click(
        screen.getByRole("button", { name: "Retry 1 failed" }),
      );
      await confirmDeletion();
      await waitFor(() => expect(mutateMemory).toHaveBeenCalledTimes(3));
      expect(mutateMemory.mock.calls[2][0]).toBe("memory-2");
    });

    it("S1.3: an older poll cannot restore a successfully processed item", async () => {
      const pollResult = deferred<GovernedMemory[]>();
      const secondMutation = deferred<GovernedMemory>();
      try {
        await openLibrary();
        await select("First", "Second");
        getMemories.mockReturnValueOnce(pollResult.promise);
        act(() => void pollNow("memory-library"));
        mutateMemory
          .mockResolvedValueOnce({
            ...rows[0],
            status: "deleted",
            revision: 2,
          })
          .mockReturnValueOnce(secondMutation.promise);
        await bulkDelete(2);
        await waitFor(() => expect(mutateMemory).toHaveBeenCalledTimes(2));
        await act(async () => pollResult.resolve(rows));
        expect(
          screen.queryByRole("checkbox", { name: "Select: First" }),
        ).toBeNull();
      } finally {
        await act(async () => {
          pollResult.resolve(rows);
          secondMutation.resolve({
            ...rows[1],
            status: "deleted",
            revision: 4,
          });
        });
      }
    });

    it("S1.3: a failing item keeps its selection while another item is committing", async () => {
      await openLibrary();
      await select("First", "Second", "Third");
      const lastMutation = deferred<GovernedMemory>();
      mutateMemory
        .mockRejectedValueOnce(new Error("Connection lost"))
        .mockResolvedValueOnce({
          ...rows[1],
          status: "deleted",
          revision: 4,
        })
        .mockReturnValueOnce(lastMutation.promise);
      try {
        await bulkDelete(3);
        await waitFor(() => expect(mutateMemory).toHaveBeenCalledTimes(3));
        expect(
          screen.getByRole("group", { name: "Action failed: First" }),
        ).toBeTruthy();
        expect(
          (
            screen.getByRole("checkbox", {
              name: "Select: First",
            }) as HTMLInputElement
          ).checked,
        ).toBe(true);
      } finally {
        await act(async () =>
          lastMutation.resolve({
            ...rows[2],
            status: "deleted",
            revision: 3,
          }),
        );
      }
    });

    it("S1.3: completing an ask cannot release a batch waiting on its mutation", async () => {
      await openLibrary();
      const stream = deferred<void>();
      const pending = deferred<GovernedMemory>();
      askMemoriesStream.mockImplementation(async function* () {
        await stream.promise;
        yield {
          type: "final",
          answer: "Answer",
          memories: [],
          mode: "generated",
        };
      });
      await userEvent.click(screen.getByRole("tab", { name: "Workbench" }));
      await userEvent.type(
        screen.getByLabelText("Ask your memories"),
        "Question",
      );
      await userEvent.click(
        screen.getByRole("button", { name: "Send question" }),
      );
      await userEvent.click(screen.getByRole("tab", { name: "Library" }));
      await select("First");
      mutateMemory.mockReturnValueOnce(pending.promise);
      try {
        await bulkDelete(1);
        await act(async () => stream.resolve());
        expect(
          (
            screen.getByRole("button", {
              name: "Delete 1 selected",
            }) as HTMLButtonElement
          ).disabled,
        ).toBe(true);
      } finally {
        await act(async () => {
          stream.resolve();
          pending.resolve({
            ...rows[0],
            status: "deleted",
            revision: 2,
          });
        });
      }
    });

    it("S1.3: retrying a failed deletion still requires confirmation", async () => {
      await openLibrary();
      await select("Second");
      mutateMemory.mockRejectedValueOnce(new Error("Response lost"));
      await userEvent.click(
        screen.getByRole("button", { name: "Delete 1 selected" }),
      );
      await userEvent.click(
        screen.getByRole("button", { name: "Confirm Delete" }),
      );
      const retry = await screen.findByRole("button", {
        name: "Retry deletion",
      });
      await userEvent.click(retry);
      expect(mutateMemory).toHaveBeenCalledTimes(1);
      await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
      expect(
        screen.getByRole("group", { name: "Action failed: Second" }),
      ).toBeTruthy();
      await userEvent.click(retry);
      await userEvent.click(
        screen.getByRole("button", { name: "Confirm Delete" }),
      );
      await waitFor(() => expect(mutateMemory).toHaveBeenCalledTimes(2));
      expect(mutateMemory.mock.calls[1].slice(0, 4)).toEqual(
        mutateMemory.mock.calls[0].slice(0, 4),
      );
    });

    it.each([false, true])(
      "S1.3: the first saved item invalidates an old stream before later items or refresh finish (recalled=%s)",
      async (recalled) => {
        await openLibrary();
        const stream = deferred<void>();
        const secondMutation = deferred<GovernedMemory>();
        const refresh = deferred<GovernedMemory[]>();
        askMemoriesStream.mockImplementation(async function* () {
          if (recalled) {
            yield { type: "recall", memories: [rows[0]] };
            yield { type: "delta", text: "Old batch answer" };
          }
          await stream.promise;
          yield {
            type: "final",
            answer: "Late old batch answer",
            memories: [rows[0]],
            mode: "generated",
          };
        });
        await userEvent.click(screen.getByRole("tab", { name: "Workbench" }));
        await userEvent.type(
          screen.getByLabelText("Ask your memories"),
          "Question",
        );
        await userEvent.click(
          screen.getByRole("button", { name: "Send question" }),
        );
        await waitFor(() => expect(askMemoriesStream).toHaveBeenCalledTimes(1));
        await userEvent.click(screen.getByRole("tab", { name: "Library" }));
        await select("First", "Second");
        mutateMemory
          .mockResolvedValueOnce({
            ...rows[0],
            status: "deleted",
            revision: 2,
          })
          .mockReturnValueOnce(secondMutation.promise);
        getMemories.mockReturnValueOnce(refresh.promise);
        try {
          await bulkDelete(2);
          await waitFor(() => expect(mutateMemory).toHaveBeenCalledTimes(2));
          await userEvent.click(screen.getByRole("tab", { name: "Workbench" }));
          expect(screen.queryByText("Old batch answer")).toBeNull();
          await act(async () => stream.resolve());
          expect(screen.queryByText("Late old batch answer")).toBeNull();
        } finally {
          await act(async () => {
            stream.resolve();
            secondMutation.resolve({
              ...rows[1],
              status: "deleted",
              revision: 4,
            });
            refresh.resolve(rows);
          });
        }
      },
    );
  });

  it("keeps asking until the core is up, rather than once", async () => {
    // The real sequence on every launch: the window is there before the core
    // is. This used to be tried once, so a library of 483 memories stayed
    // blank for the whole session.
    getMemories
      .mockRejectedValueOnce(new Error("core not up yet"))
      .mockResolvedValue([{ ...memory, status: "confirmed" }]);

    render(<MemorySpace onConnect={() => {}} />);
    await userEvent.click(screen.getByRole("tab", { name: "Library" }));

    await waitFor(() => {
      expect(
        screen.getByText("Not connected to the local Core yet, retrying..."),
      ).toBeTruthy();
    });
    // No refresh, no navigation: the memories arrive on their own.
    await waitFor(
      () => {
        expect(screen.getByText("Initial memory", visibleText)).toBeTruthy();
      },
      { timeout: 4_000 },
    );
    expect(getMemories.mock.calls.length).toBeGreaterThan(1);
    expect(
      screen.queryByText("Not connected to the local Core yet, retrying..."),
    ).toBeNull();
  });

  it("does not claim you have no memories while it is still reading", async () => {
    // The lie that made this invisible: an unread library and an empty one
    // said the same thing, so the failure looked like a fact about the user.
    getMemories.mockRejectedValue(new Error("core not up yet"));
    getMemoryGraph.mockRejectedValue(new Error("core not up yet"));

    render(<MemorySpace onConnect={() => {}} />);

    await waitFor(() => {
      expect(screen.getByText("Reading memories...")).toBeTruthy();
    });
    expect(
      screen.queryByText("No memories have passed evidence validation yet."),
    ).toBeNull();
  });

  it("shows the library when only the graph is unavailable", async () => {
    // They were fetched as one, so a failing graph threw the memories away.
    getMemories.mockResolvedValue([{ ...memory, status: "confirmed" }]);
    getMemoryGraph.mockRejectedValue(new Error("graph unavailable"));

    render(<MemorySpace onConnect={() => {}} />);
    await userEvent.click(screen.getByRole("tab", { name: "Library" }));

    await waitFor(() => {
      expect(screen.getByText("Initial memory", visibleText)).toBeTruthy();
    });
  });

  it("shows newly available memory without waiting for a slow graph response", async () => {
    const stalled = deferred<Awaited<ReturnType<typeof getMemoryGraph>>>();
    getMemories.mockResolvedValue([{ ...memory, status: "confirmed" }]);
    getMemoryGraph.mockReturnValue(stalled.promise);
    render(<MemorySpace />);
    await userEvent.click(screen.getByRole("tab", { name: "Library" }));
    await waitFor(() =>
      expect(screen.getByText("Initial memory", visibleText)).toBeTruthy(),
    );
    await act(async () =>
      stalled.resolve({
        nodes: [],
        edges: [],
        total_memories: 1,
        projected_memories: 0,
      }),
    );
  });

  it("retries a failed graph read without showing it to the user", async () => {
    vi.useFakeTimers();
    getMemoryGraph.mockRejectedValueOnce(new Error("graph offline"));
    render(<MemorySpace />);
    await act(async () => {});
    expect(screen.queryByText(/Failed to load the memory graph/)).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(getMemories).toHaveBeenCalledTimes(1);
    const pending = deferred<Awaited<ReturnType<typeof getMemoryGraph>>>();
    getMemoryGraph.mockReturnValueOnce(pending.promise);
    // One library refresh cycle.
    await act(async () => vi.advanceTimersByTimeAsync(LIBRARY_REFRESH_MS));
    expect(getMemoryGraph).toHaveBeenCalledTimes(2);
    expect(getMemories).toHaveBeenCalledTimes(2);
    expect(screen.queryByText(/Failed to load the memory graph/)).toBeNull();
    await act(async () =>
      pending.resolve({
        nodes: [],
        edges: [],
        total_memories: 1,
        projected_memories: 0,
      }),
    );
    expect(screen.queryByRole("alert")).toBeNull();
    expect(mutateMemory).not.toHaveBeenCalled();
    expect(setMemoryScopes).not.toHaveBeenCalled();
  });

  it.each(["success", "failure"])(
    "ignores an old deferred graph %s after a newer poll",
    async (outcome) => {
      vi.useFakeTimers();
      render(<MemorySpace />);
      await act(async () => {});
      fireEvent.click(screen.getByRole("tab", { name: "Library" }));
      fireEvent.change(screen.getByLabelText("Search the memory library"), {
        target: { value: "Initial" },
      });
      let resolve!: (value: unknown) => void;
      let reject!: (reason: Error) => void;
      getMemoryGraph.mockReturnValueOnce(
        new Promise((done, fail) => {
          resolve = done;
          reject = fail;
        }),
      );
      const form = screen
        .getByLabelText("Search the memory library")
        .closest("form");
      expect(form).not.toBeNull();
      if (form) fireEvent.submit(form);
      await act(async () => {});
      await act(async () => vi.advanceTimersByTimeAsync(1500));
      expect(getMemoryGraph).toHaveBeenCalledTimes(2);
      // Search has finished even though its projection remains pending.
      expect(
        (screen.getByRole("button", { name: "Search" }) as HTMLButtonElement)
          .disabled,
      ).toBe(false);
      if (outcome === "success")
        getMemoryGraph.mockRejectedValueOnce(new Error("newer graph failure"));
      await act(async () => vi.advanceTimersByTimeAsync(LIBRARY_REFRESH_MS));
      expect(getMemoryGraph).toHaveBeenCalledTimes(3);
      expect(screen.queryByText(/Failed to load the memory graph/)).toBeNull();
      expect(screen.queryByRole("alert")).toBeNull();
      await act(async () => {
        if (outcome === "success")
          resolve({
            nodes: [],
            edges: [],
            total_memories: 1,
            projected_memories: 0,
          });
        else reject(new Error("older graph failure"));
      });
      expect(screen.queryByText(/Failed to load the memory graph/)).toBeNull();
      expect(screen.queryByRole("alert")).toBeNull();
      expect(mutateMemory).not.toHaveBeenCalled();
    },
  );

  it("turns a long queue into a wait the reader can plan around", async () => {
    getMemoryProcessing.mockResolvedValue({
      pending: 677,
      fresh_pending: 0,
      backlog_pending: 677,
      recent_completions: [],
      running: 1,
      stranded_running: 0,
      waiting_for_model: 0,
      completed: 51,
      failed: 0,
      blocked_stages: [],
      outside_window_pending: 0,
      extraction_window_preset: "3m",
      manual_extract_batches_planned: 0,
      manual_extract_batches_current: 0,
      active_queue: 0,
      in_flight_count: 0,
      seconds_per_completion: 112,
    });
    render(<MemorySpace />);

    expect(
      await screen.findByText("1 processing · 677 backlog items (~21 hours)"),
    ).toBeTruthy();
  });

  it("shows unknown results once and only retries after confirmation", async () => {
    const confirm = vi
      .spyOn(window, "confirm")
      .mockReturnValueOnce(false)
      .mockReturnValueOnce(true);
    getMemoryProcessing.mockResolvedValue({
      pending: 0,
      fresh_pending: 0,
      backlog_pending: 0,
      recent_completions: [],
      running: 0,
      stranded_running: 0,
      waiting_for_model: 0,
      completed: 43,
      failed: 2,
      unknown: 17,
      unknown_reasons: { model_content_outcome_unknown: 17 },
      blocked_stages: [],
      outside_window_pending: 0,
      extraction_window_preset: "3m",
      manual_extract_batches_planned: 0,
      manual_extract_batches_current: 0,
      active_queue: 0,
      in_flight_count: 0,
      seconds_per_completion: 20,
    });
    retryUnknownMemoryProcessing.mockResolvedValue({ requeued: 17 });
    render(<MemorySpace />);

    expect(
      await screen.findByText(
        "17 conversations have an uncertain processing result",
      ),
    ).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Reprocess" }));
    expect(retryUnknownMemoryProcessing).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Reprocess" }));
    await waitFor(() =>
      expect(retryUnknownMemoryProcessing).toHaveBeenCalledOnce(),
    );
    expect(confirm).toHaveBeenCalledTimes(2);
    confirm.mockRestore();
  });

  it("splits the queue into new and catch-up rather than one total", async () => {
    // 675 as a single number reads as "stuck". Split, it says the three from
    // this morning are minutes away and the rest is the import (R-02).
    getMemoryProcessing.mockResolvedValue({
      pending: 675,
      fresh_pending: 3,
      backlog_pending: 672,
      recent_completions: ["2026-09-08T00:31:00+00:00"],
      running: 1,
      stranded_running: 0,
      waiting_for_model: 0,
      completed: 51,
      failed: 0,
      blocked_stages: [],
      outside_window_pending: 0,
      extraction_window_preset: "3m",
      manual_extract_batches_planned: 0,
      manual_extract_batches_current: 0,
      active_queue: 0,
      in_flight_count: 0,
      seconds_per_completion: 112,
    });
    render(<MemorySpace />);

    expect(
      await screen.findByText(
        "1 processing · 3 new pending · 672 backlog items (~21 hours)",
      ),
    ).toBeTruthy();
  });

  it("omits zero dispatch telemetry when the queue is only running", async () => {
    getMemoryProcessing.mockResolvedValue({
      pending: 2,
      fresh_pending: 2,
      backlog_pending: 0,
      recent_completions: [],
      running: 1,
      stranded_running: 0,
      waiting_for_model: 0,
      completed: 0,
      failed: 0,
      blocked_stages: [],
      outside_window_pending: 0,
      extraction_window_preset: "3m",
      manual_extract_batches_planned: 0,
      manual_extract_batches_current: 0,
      active_queue: 0,
      in_flight_count: 0,
      seconds_per_completion: 60,
    });
    render(<MemorySpace />);
    expect(
      await screen.findByText("1 processing · 2 new pending (~2 min)"),
    ).toBeTruthy();
    expect(screen.queryByText(/0 queued/)).toBeNull();
  });

  it("names stranded running work instead of a healthy in-flight count", async () => {
    getMemoryProcessing.mockResolvedValue({
      pending: 2,
      fresh_pending: 2,
      backlog_pending: 0,
      recent_completions: [],
      running: 2,
      stranded_running: 1,
      waiting_for_model: 0,
      completed: 0,
      failed: 0,
      blocked_stages: [],
      outside_window_pending: 0,
      extraction_window_preset: "3m",
      manual_extract_batches_planned: 0,
      manual_extract_batches_current: 0,
      active_queue: 0,
      in_flight_count: 0,
      seconds_per_completion: 60,
    });
    render(<MemorySpace />);
    expect(
      await screen.findByText(
        "2 processing · 2 new pending · 1 with stalled task status, recovering (~2 min)",
      ),
    ).toBeTruthy();
  });

  it("does not invent a healthy zero when stranded_running is unknown", async () => {
    getMemoryProcessing.mockResolvedValue({
      pending: 2,
      fresh_pending: 2,
      backlog_pending: 0,
      recent_completions: [],
      running: 2,
      waiting_for_model: 0,
      completed: 0,
      failed: 0,
      blocked_stages: [],
      outside_window_pending: 0,
      extraction_window_preset: "3m",
      manual_extract_batches_planned: 0,
      manual_extract_batches_current: 0,
      active_queue: 0,
      in_flight_count: 0,
      seconds_per_completion: 60,
    });
    render(<MemorySpace />);

    expect(
      await screen.findByText(
        "2 processing · 2 new pending · 2 with unknown task status (~2 min)",
      ),
    ).toBeTruthy();
    expect(
      screen.queryByText(/with stalled task status, recovering/),
    ).toBeNull();
  });

  it("names the stuck pipeline stage instead of implying there is nothing", async () => {
    getMemoryProcessing.mockResolvedValue({
      pending: 0,
      running: 0,
      waiting_for_model: 4,
      completed: 0,
      failed: 0,
      blocked_stages: [
        { kind: "extract", reason: "model_profile_not_active", count: 4 },
      ],
      outside_window_pending: 0,
      extraction_window_preset: "3m",
      manual_extract_batches_planned: 0,
      manual_extract_batches_current: 0,
      active_queue: 0,
      in_flight_count: 0,
      seconds_per_completion: null,
    });
    render(<MemorySpace />);

    expect(
      await screen.findByText(
        "Memory extraction: No generation model is enabled yet (4)",
      ),
    ).toBeTruthy();
  });

  it("keeps the pipeline's own retries out of the reader's way", async () => {
    getMemoryProcessing.mockResolvedValue({
      pending: 0,
      running: 1,
      waiting_for_model: 0,
      completed: 3,
      failed: 2,
      blocked_stages: [
        {
          kind: "extract",
          reason: "hy_memory_write_verification_clause_incomplete",
          count: 1,
        },
        {
          kind: "extract",
          reason: "hy_memory_write_verification_failed",
          count: 1,
        },
      ],
      outside_window_pending: 0,
      extraction_window_preset: "3m",
      manual_extract_batches_planned: 0,
      manual_extract_batches_current: 0,
      active_queue: 0,
      in_flight_count: 0,
      seconds_per_completion: null,
    });
    const { container } = render(<MemorySpace />);

    // One calm sentence, and the two stalls summed into it rather than named.
    expect(
      await screen.findByText(
        "2 items: Failed to organize this round; retrying",
        {
          exact: false,
        },
      ),
    ).toBeTruthy();
    // Neither an internal guard name nor the amber "Processing needs attention" heading: the
    // pipeline is handling these itself.
    expect(container.textContent).not.toContain("hy_memory_write_verification");
    expect(screen.queryByText("Processing needs attention")).toBeNull();
    expect(screen.getByText("Processing memories")).toBeTruthy();
  });

  it("shows recalled memories before the answer finishes streaming", async () => {
    const user = userEvent.setup();
    let releaseFinal: () => void = () => undefined;
    const finalReleased = new Promise<void>((resolve) => {
      releaseFinal = resolve;
    });
    askMemoriesStream.mockImplementation(async function* () {
      yield { type: "recall", memories: [memory] };
      yield { type: "delta", text: "Grounded" };
      await finalReleased;
      yield {
        type: "final",
        answer: "Grounded answer",
        memories: [memory],
        mode: "generated",
      };
    });
    render(<MemorySpace />);

    await user.click(await screen.findByRole("tab", { name: "Ask memories" }));
    await user.type(
      screen.getByLabelText("Ask the confirmed memories"),
      "What are my principles?",
    );
    await user.click(
      screen.getByRole("button", { name: "Answer from memories" }),
    );

    expect(
      await screen.findByText("Generating", { exact: false }),
    ).toBeTruthy();
    expect(screen.getByText("Grounded", visibleText)).toBeTruthy();

    releaseFinal();

    expect(
      await screen.findByText("Grounded answer", visibleText),
    ).toBeTruthy();
    expect(
      screen.getByText("Model summary", { ...visibleText, exact: false }),
    ).toBeTruthy();
  });

  it("opens exactly one ui.action/ui.result chain for an answer and hands its id to the stream", async () => {
    // The SSE ask path uses raw `fetch`, so it bypasses the axios interceptor.
    // Without its own trace, `askMemoriesStream` sends no X-One-Memory-Trace-Id
    // and Core logs its kernel legs under a different id — the interaction shows
    // up as a half-chain (ui.action/ui.result with no kernel legs). This pins
    // both ends: one action, one matching result, and the same id on the call.
    const emitted: Array<{
      phase: string;
      status?: string;
      reason?: string;
      trace_id?: string;
    }> = [];
    (window as unknown as { oneMemory: unknown }).oneMemory = {
      logUiTrace: (input: {
        phase: string;
        status?: string;
        reason?: string;
      }) => {
        emitted.push(input);
        return Promise.resolve({ state: "logged" });
      },
    };
    askMemoriesStream.mockImplementation(async function* () {
      yield {
        type: "final",
        answer: "Grounded answer",
        memories: [memory],
        mode: "generated",
      };
    });
    try {
      render(<MemorySpace />);
      await userEvent.type(
        await screen.findByLabelText("Ask your memories"),
        "What are the project requirements?",
      );
      await userEvent.click(
        screen.getByRole("button", { name: "Send question" }),
      );
      await waitFor(() =>
        expect(emitted.filter((e) => e.phase === "result")).toHaveLength(1),
      );
      const actions = emitted.filter((e) => e.phase === "action");
      const results = emitted.filter((e) => e.phase === "result");
      expect(actions).toHaveLength(1);
      expect(results).toHaveLength(1);
      expect(actions[0].trace_id).toBe(results[0].trace_id);
      expect(results[0].status).toBe("succeeded");
      // The very same id went out on the raw-fetch request.
      expect(askMemoriesStream.mock.calls[0][3]).toBe(actions[0].trace_id);
    } finally {
      delete (window as unknown as { oneMemory?: unknown }).oneMemory;
    }
  });

  it("explains a revoked answer without calling it a Core disconnection", async () => {
    askMemoriesStream.mockImplementation(async function* () {
      yield { type: "error", code: "memory_answer_context_changed" };
    });
    render(<MemorySpace />);
    await userEvent.type(
      await screen.findByLabelText("Ask your memories"),
      "Project requirements",
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Send question" }),
    );
    expect(
      await screen.findByText(
        "The memory content or its permissions changed, so the answer stopped. Ask again.",
      ),
    ).toBeTruthy();
    expect(
      screen.queryByText("Not connected to the local Core yet, retrying..."),
    ).toBeNull();
  });

  it("opens original evidence without a confirm or reject step", async () => {
    const user = userEvent.setup();
    render(<MemorySpace />);

    await user.click(
      await screen.findByRole("button", { name: /Initial memory/ }),
    );
    expect(await screen.findByText("Original words", visibleText)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Correct" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Confirm" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Reject" })).toBeNull();
    expect(mutateMemory).not.toHaveBeenCalled();
  });

  it("asks across authorized project memories from the workbench", async () => {
    render(<MemorySpace />);
    await userEvent.type(
      await screen.findByLabelText("Ask your memories"),
      "What are the project requirements?",
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Send question" }),
    );
    await waitFor(() =>
      // The trace id is the 4th argument (the SSE path carries its own header).
      expect(askMemoriesStream).toHaveBeenCalledWith(
        "What are the project requirements?",
        "*",
        undefined,
        expect.any(String),
      ),
    );
  });

  it("hands a saved answer to HY as a note instead of writing the memory itself", async () => {
    askMemoriesStream.mockImplementation(async function* () {
      yield {
        type: "final",
        answer: "Grounded answer",
        memories: [memory],
        mode: "generated",
      };
    });
    render(<MemorySpace />);
    await userEvent.type(
      await screen.findByLabelText("Ask your memories"),
      "What are the project requirements?",
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Send question" }),
    );
    await userEvent.click(
      await screen.findByRole("button", { name: "Save as memory" }),
    );

    await waitFor(() =>
      expect(saveMemoryNote).toHaveBeenCalledWith(
        {
          request_id: expect.any(String),
          text: "Grounded answer",
          scope: "all-apps",
        },
        expect.any(String),
      ),
    );
    expect(createMemory).not.toHaveBeenCalled();
  });

  it("P0-05: local governance invalidates the answer even when the following refresh fails", async () => {
    const confirmed = { ...memory, status: "confirmed" as const };
    getMemories.mockResolvedValue([confirmed]);
    getMemory.mockResolvedValue({ ...detail, memory: confirmed });
    askMemoriesStream.mockImplementation(async function* () {
      yield {
        type: "final",
        answer: "Answer before deletion",
        memories: [confirmed],
        mode: "generated",
      };
    });
    render(<MemorySpace />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Initial memory/ }),
    );
    await screen.findByText("Original words", visibleText);
    await userEvent.type(screen.getByLabelText("Ask your memories"), "Query");
    await userEvent.click(
      screen.getByRole("button", { name: "Send question" }),
    );
    await screen.findByText("Answer before deletion", visibleText);
    getMemories.mockRejectedValue(
      new Error("read failed after successful mutation"),
    );
    mutateMemory.mockResolvedValue({
      ...confirmed,
      status: "deleted",
      revision: 2,
    });
    await userEvent.click(screen.getByRole("button", { name: "Delete" }));
    await userEvent.click(
      screen.getByRole("button", { name: "Confirm Delete" }),
    );
    await waitFor(() => expect(mutateMemory).toHaveBeenCalled());
    expect(screen.queryByText("Answer before deletion")).toBeNull();
    expect(
      screen.queryByRole("button", { name: "Show answer and citations" }),
    ).toBeNull();
  });

  it.each([false, true])(
    "invalidates answers after the recalled memory changes (streaming=%s)",
    async (streaming) => {
      let release: () => void = () => {};
      const waiting = new Promise<void>((resolve) => {
        release = resolve;
      });
      askMemoriesStream.mockImplementation(async function* () {
        yield { type: "recall", memories: [memory] };
        yield { type: "delta", text: "Old answer" };
        if (streaming) await waiting;
        yield {
          type: "final",
          answer: "Old final answer",
          memories: [memory],
          mode: "generated",
        };
      });
      try {
        render(<MemorySpace />);
        await userEvent.click(
          await screen.findByRole("tab", { name: "Ask memories" }),
        );
        await userEvent.type(
          screen.getByLabelText("Ask the confirmed memories"),
          "Query",
        );
        await userEvent.click(
          screen.getByRole("button", { name: "Answer from memories" }),
        );
        expect(
          await screen.findByText(
            streaming ? "Old answer" : "Old final answer",
            visibleText,
          ),
        ).toBeTruthy();
        getMemories.mockResolvedValue([
          { ...memory, revision: 2, status: "deleted" },
        ]);
        await act(async () => pollNow("memory-library"));
        expect(screen.queryByText("Old answer")).toBeNull();
        expect(screen.queryByText("Old final answer")).toBeNull();
        await act(async () => release());
        expect(screen.queryByText("Old final answer")).toBeNull();
      } finally {
        release();
      }
    },
  );

  it("ignores a late detail response after another memory is selected", async () => {
    const second = {
      ...memory,
      memory_id: "memory-2",
      content: "Second memory",
    };
    getMemories.mockResolvedValue([memory, second]);
    let resolveFirst: (value: MemoryDetail) => void = () => {};
    getMemory.mockImplementation((id: string) =>
      id === memory.memory_id
        ? new Promise((resolve) => {
            resolveFirst = resolve;
          })
        : Promise.resolve({
            ...detail,
            memory: second,
            evidence: [{ ...detail.evidence[0], verbatim: "Second original" }],
          }),
    );
    render(<MemorySpace />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Initial memory/ }),
    );
    await userEvent.click(
      screen.getByRole("button", { name: /Second memory/ }),
    );
    expect(
      await screen.findByText("Second original", visibleText),
    ).toBeTruthy();
    await act(async () => resolveFirst(detail));
    await userEvent.click(screen.getByRole("button", { name: "Correct" }));
    expect(
      screen
        .getByRole("textbox", { name: "Memory content" })
        .getAttribute("name"),
    ).toBe("memory-content");
    expect(
      (
        screen.getByRole("textbox", {
          name: "Memory content",
        }) as HTMLTextAreaElement
      ).value,
    ).toBe("Second memory");
    expect(screen.queryByText("Original words")).toBeNull();
  });

  it("saves a correction without editing the evidence field", async () => {
    const user = userEvent.setup();
    render(<MemorySpace />);
    await user.click(
      await screen.findByRole("button", { name: /Initial memory/ }),
    );

    await user.click(await screen.findByRole("button", { name: "Correct" }));
    const content = await screen.findByRole("textbox", {
      name: "Memory content",
    });
    await user.clear(content);
    await user.type(content, "Corrected memory");
    await user.click(screen.getByRole("button", { name: "Save correction" }));

    await waitFor(() =>
      expect(correctMemory).toHaveBeenCalledWith(
        "memory-1",
        1,
        "Corrected memory",
        ["all-apps"],
        expect.any(String),
        expect.any(String),
      ),
    );
    expect(screen.getByText("Original words", visibleText)).toBeTruthy();
  });

  it("saves scope independently of an unsaved wording edit or confirmation", async () => {
    render(<MemorySpace />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Initial memory/ }),
    );
    await userEvent.click(screen.getByRole("button", { name: "Correct" }));
    await userEvent.clear(
      screen.getByRole("textbox", { name: "Memory content" }),
    );
    await userEvent.type(
      screen.getByRole("textbox", { name: "Memory content" }),
      "Unsaved wording",
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Adjust recall scope" }),
    );
    await userEvent.click(
      screen.getByRole("radio", { name: "Stop external memory recall" }),
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Save recall scope" }),
    );
    await waitFor(() =>
      expect(setMemoryScopes).toHaveBeenCalledWith(
        "memory-1",
        1,
        ["private"],
        expect.any(String),
        expect.any(String),
      ),
    );
    expect(correctMemory).not.toHaveBeenCalled();
    expect(mutateMemory).not.toHaveBeenCalled();
    expect(screen.getByText("Original words", visibleText)).toBeTruthy();
  });

  it("a late scope save cannot reopen an earlier selection", async () => {
    const second = {
      ...memory,
      memory_id: "memory-2",
      content: "Second memory",
    };
    getMemories.mockResolvedValue([memory, second]);
    getMemory.mockImplementation((id: string) =>
      Promise.resolve(
        id === "memory-1"
          ? detail
          : {
              ...detail,
              memory: second,
              evidence: [
                { ...detail.evidence[0], verbatim: "Second original" },
              ],
            },
      ),
    );
    let finish: (value: GovernedMemory) => void = () => {};
    setMemoryScopes.mockReturnValue(
      new Promise((resolve) => {
        finish = resolve;
      }),
    );
    render(<MemorySpace />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Initial memory/ }),
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Adjust recall scope" }),
    );
    await userEvent.click(
      screen.getByRole("radio", { name: "Stop external memory recall" }),
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Save recall scope" }),
    );
    await userEvent.click(
      screen.getByRole("button", { name: /Second memory/ }),
    );
    expect(
      await screen.findByText("Second original", visibleText),
    ).toBeTruthy();
    await act(async () =>
      finish({ ...memory, scopes: ["private"], revision: 2 }),
    );
    expect(screen.getByText("Second original", visibleText)).toBeTruthy();
    expect(screen.queryByText("Original words")).toBeNull();
  });

  it.each(["memories", "graph"])(
    "does not call a saved scope unsaved when the following %s refresh fails",
    async (panel) => {
      render(<MemorySpace />);
      await userEvent.click(
        await screen.findByRole("button", { name: /Initial memory/ }),
      );
      await userEvent.click(screen.getByRole("button", { name: "Correct" }));
      fireEvent.change(
        screen.getByRole("textbox", { name: "Memory content" }),
        {
          target: { value: "Unsaved wording" },
        },
      );
      await userEvent.click(
        screen.getByRole("button", { name: "Adjust recall scope" }),
      );
      await userEvent.click(
        screen.getByRole("radio", { name: "Stop external memory recall" }),
      );
      const reader = panel === "graph" ? getMemoryGraph : getMemories;
      reader.mockRejectedValueOnce(new Error("refresh unavailable"));
      await userEvent.click(
        screen.getByRole("button", { name: "Save recall scope" }),
      );
      if (panel === "graph") {
        await waitFor(
          () => expect(getMemoryGraph.mock.calls.length).toBeGreaterThan(1),
          { timeout: 3500 },
        );
        expect(
          screen.queryByText(/Failed to load the memory graph/),
        ).toBeNull();
      } else {
        expect(
          (await screen.findByRole("alert", {}, { timeout: 3500 })).textContent,
        ).toContain(
          "Recall scopes were saved, but the list refresh failed. Reopen the memory page later.",
        );
      }
      expect(
        (
          screen.getByRole("textbox", {
            name: "Memory content",
          }) as HTMLTextAreaElement
        ).value,
      ).toBe("Unsaved wording");
      expect(
        (
          screen.getByRole("button", {
            name: "Save correction",
          }) as HTMLButtonElement
        ).disabled,
      ).toBe(false);
      expect(setMemoryScopes).toHaveBeenCalledTimes(1);
      expect(correctMemory).not.toHaveBeenCalled();
      expect(mutateMemory).not.toHaveBeenCalled();
      expect(screen.getByText("Original words", visibleText)).toBeTruthy();
      expect(
        screen.getByText("External memory recall is stopped", visibleText),
      ).toBeTruthy();
      expect(
        screen.queryByText("The save result is unconfirmed. Try again."),
      ).toBeNull();
    },
  );

  it("Enter on a scope radio cannot submit a separate content correction", async () => {
    render(<MemorySpace />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Initial memory/ }),
    );
    await userEvent.click(screen.getByRole("button", { name: "Correct" }));
    await userEvent.click(
      screen.getByRole("button", { name: "Adjust recall scope" }),
    );
    await userEvent.click(
      screen.getByRole("radio", { name: "Stop external memory recall" }),
    );
    await userEvent.keyboard("{Enter}");
    expect(correctMemory).not.toHaveBeenCalled();
    expect(setMemoryScopes).not.toHaveBeenCalled();
  });

  it("says search is behind, and offers to try again", async () => {
    // A correction that only landed locally leaves the user reading the new
    // text while search matches the old one. Nothing on screen said so, and
    // nothing offered to fix it.
    getMemory.mockResolvedValue({
      ...detail,
      index_sync: {
        memory_id: "memory-1",
        status: "pending",
        attempts: 1,
        last_error_code: "hy_memory_update_failed",
        updated_at: "2026-09-01T00:00:00Z",
      },
    });
    const user = userEvent.setup();
    render(<MemorySpace />);
    await user.click(
      await screen.findByRole("button", { name: /Initial memory/ }),
    );

    expect(
      await screen.findByText(
        /Search still shows the wording before your edit/,
        visibleText,
      ),
    ).toBeTruthy();
    expect(
      screen.getAllByText(/Sync attempted 1 times - Status updated at/).length,
    ).toBeGreaterThan(0);
    await user.click(screen.getByRole("button", { name: "Retry sync" }));

    await waitFor(() =>
      expect(syncMemoryIndex).toHaveBeenCalledWith(
        "memory-1",
        expect.any(String),
      ),
    );
  });

  it("does not offer a retry that cannot succeed", async () => {
    // HY has no such memory to re-embed, so every retry ends the same way.
    // The button would be a loop the user has to decide to stop.
    getMemory.mockResolvedValue({
      ...detail,
      index_sync: {
        memory_id: "memory-1",
        status: "pending",
        attempts: 4,
        last_error_code: "hy_memory_update_rejected",
        updated_at: "2026-09-01T00:00:00Z",
      },
    });
    const user = userEvent.setup();
    render(<MemorySpace />);
    await user.click(
      await screen.findByRole("button", { name: /Initial memory/ }),
    );

    expect(
      screen.getAllByText(/HY Memory has no such memory/).length,
    ).toBeGreaterThan(0);
    expect(screen.queryByRole("button", { name: "Retry sync" })).toBeNull();
  });

  it("does not call a 200 pending index response a successful sync", async () => {
    getMemory.mockResolvedValue({
      ...detail,
      index_sync: {
        memory_id: "memory-1",
        status: "pending",
        attempts: 2,
        last_error_code: "hy_memory_update_failed",
        updated_at: "2026-09-01T00:00:00Z",
      },
    });
    syncMemoryIndex.mockResolvedValueOnce({
      memory_id: "memory-1",
      status: "pending",
      attempts: 3,
      last_error_code: "hy_memory_update_failed",
      updated_at: "2026-09-01T00:01:00Z",
    });
    render(<MemorySpace />);
    await userEvent.click(
      await screen.findByRole("button", { name: /Initial memory/ }),
    );
    await userEvent.click(
      await screen.findByRole("button", { name: "Retry sync" }),
    );
    await waitFor(() =>
      expect(syncMemoryIndex).toHaveBeenCalledWith(
        "memory-1",
        expect.any(String),
      ),
    );
    expect(syncMemoryIndex.mock.calls[0][1]).toMatch(/^[0-9a-f-]{36}$/);
    await waitFor(() =>
      expect(
        screen.getAllByText(/Sync attempted 2 times/).length,
      ).toBeGreaterThan(0),
    );
  });

  it("stays quiet about the index when nobody corrected the memory", async () => {
    // The control: a sync line on every memory would make the user account
    // for a state they never caused.
    const user = userEvent.setup();
    render(<MemorySpace />);
    await user.click(
      await screen.findByRole("button", { name: /Initial memory/ }),
    );

    expect(await screen.findByText("Original words", visibleText)).toBeTruthy();
    expect(
      screen.queryByText(/Search still shows the wording before your edit/),
    ).toBeNull();
    expect(screen.queryByRole("button", { name: "Retry sync" })).toBeNull();
  });

  describe("The three-way choice on a width mismatch (A-33)", () => {
    it("offers the choice as soon as the memory space opens, without waiting for a failed extraction round", async () => {
      getGraphWidth.mockResolvedValue({
        mismatch: {
          stored_width: 768,
          expected_width: 512,
          graph_path: "/tmp/graph.kuzu",
        },
        memory_count: 427,
      });

      render(<MemorySpace />);

      expect(
        await screen.findByRole("dialog", {
          name: "The existing memory library does not match the built-in model",
        }),
      ).toBeTruthy();
    });

    it("does not show this dialog when the widths match", async () => {
      render(<MemorySpace />);

      await waitFor(() => {
        expect(getGraphWidth).toHaveBeenCalled();
      });
      expect(screen.queryByRole("dialog")).toBeNull();
    });
  });

  describe("Extraction window and its three states (M-21)", () => {
    function withDeferred(outside: number, extra: object = {}) {
      getMemoryProcessing.mockResolvedValue({
        pending: 3,
        running: 1,
        fresh_pending: 3,
        backlog_pending: 0,
        recent_completions: [],
        waiting_for_model: 0,
        completed: 51,
        failed: 0,
        blocked_stages: [],
        outside_window_pending: outside,
        extraction_window_preset: "3m",
        manual_extract_batches_planned: 0,
        manual_extract_batches_current: 0,
        active_queue: 0,
        in_flight_count: 0,
        seconds_per_completion: null,
        ...extra,
      });
    }

    it("names the outside-window count separately and says the source text stays searchable", async () => {
      withDeferred(672);

      render(<MemorySpace />);

      expect(
        await screen.findByText(
          /672 outside the window \(source text still searchable\)/,
        ),
      ).toBeTruthy();
    });

    it("shows manual extraction progress as planned batches and the current batch", async () => {
      withDeferred(647, {
        manual_extract_batches_planned: 27,
        manual_extract_batches_current: 2,
        active_queue: 0,
        in_flight_count: 0,
      });

      render(<MemorySpace />);

      expect(
        await screen.findByText(/27 batches planned, currently on batch 2/),
      ).toBeTruthy();
    });

    it("sends the chosen preset to the backend when it changes", async () => {
      const user = userEvent.setup();
      withDeferred(672);

      render(<MemorySpace />);

      const select = await screen.findByLabelText("Automatic extraction range");
      await user.selectOptions(select, "1y");

      await waitFor(() => {
        expect(setExtractionWindow).toHaveBeenCalledWith(
          "1y",
          expect.any(String),
        );
      });
    });

    it("asks for one batch per manual extraction instead of the whole outside window", async () => {
      const user = userEvent.setup();
      withDeferred(672);

      render(<MemorySpace />);

      await user.click(
        await screen.findByRole("button", {
          name: "Manually extract the next batch",
        }),
      );

      await waitFor(() => {
        expect(manualExtract).toHaveBeenCalledTimes(1);
      });
      // No ids means "the next batch": Core picks the batch size, the UI never pushes everything in at once.
      expect(manualExtract).toHaveBeenCalledWith([], expect.any(String));
    });

    it("does not offer a button that cannot extract anything when the window is clear", async () => {
      withDeferred(0);

      render(<MemorySpace />);

      await screen.findByLabelText("Automatic extraction range");
      expect(
        screen.queryByRole("button", {
          name: "Manually extract the next batch",
        }),
      ).toBeNull();
    });

    it("marks a source quote as outside the window and leaves extracted ones unlabelled", async () => {
      const user = userEvent.setup();
      getMemory.mockResolvedValue({
        ...detail,
        evidence: [
          { ...detail.evidence[0], extraction_state: "outside_window" },
          {
            ...detail.evidence[0],
            evidence_id: "evidence-2",
            verbatim: "Extracted words",
            extraction_state: "extracted",
          },
        ],
      });

      render(<MemorySpace />);
      await user.click(
        await screen.findByRole("button", { name: /Initial memory/ }),
      );

      expect(
        await screen.findByText(
          "Not extracted · outside the window",
          visibleText,
        ),
      ).toBeTruthy();
      expect(screen.getByText("Extracted words", visibleText)).toBeTruthy();
      expect(screen.queryByText("Extracted")).toBeNull();
    });
  });

  describe("Where the page lands with no generation model", () => {
    it("still lands on the workbench without an active profile instead of detouring to history", async () => {
      // This used to jump to the history view, arguing that the map, the ask panel and
      // the library are all empty right now. The workbench removes that premise: with no
      // memory at all it still draws the imported sources, lists what is pending and says
      // where extraction is stuck. Detouring hides the one screen that explains this app.
      getModelProfiles.mockResolvedValue([
        { profile_id: "profile-1", is_active: false },
      ]);

      render(<MemorySpace />);

      expect(
        (await screen.findByRole("tab", { name: "Workbench" })).getAttribute(
          "aria-selected",
        ),
      ).toBe("true");
      expect(
        screen.queryByRole("textbox", { name: "Search conversation history" }),
      ).toBeNull();
    });

    it("keeps the history dialogue one click away", async () => {
      const user = userEvent.setup();
      getModelProfiles.mockResolvedValue([
        { profile_id: "profile-1", is_active: false },
      ]);
      render(<MemorySpace />);

      await user.click(await screen.findByRole("tab", { name: "History" }));

      expect(
        screen.getByRole("textbox", { name: "Search conversation history" }),
      ).toBeTruthy();
    });
  });
});
