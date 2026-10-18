// @vitest-environment jsdom
import type {
  GovernedMemory,
  MemoryDetail,
  MemoryLibraryPage,
  MemoryLibraryQuery,
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
import { MemorySpace } from "./MemorySpace";

const api = vi.hoisted(() => ({
  library: vi.fn(),
  mapLibrary: vi.fn(),
  memories: vi.fn(),
  detail: vi.fn(),
  mutate: vi.fn(),
  correct: vi.fn(),
}));
vi.mock("./api", () => ({
  getMemoryLibrary: (query: MemoryLibraryQuery, signal?: AbortSignal) =>
    query.active_only === true
      ? api.mapLibrary(query, signal)
      : api.library(query, signal),
  getMemories: api.memories,
  getMemory: api.detail,
  mutateMemory: api.mutate,
  correctMemory: api.correct,
  apiErrorCode: (reason: { response?: { data?: { detail?: string } } }) =>
    reason?.response?.data?.detail ?? null,
  getMemoryGraph: async () => ({
    nodes: [],
    edges: [],
    total_memories: 1,
    projected_memories: 1,
  }),
  getMemoryProcessing: async () => null,
  getModelProfiles: async () => [],
}));
vi.mock("./GraphWidthPrompt", () => ({ GraphWidthPrompt: () => null }));
vi.mock("./MemoryNoteComposer", () => ({ MemoryNoteComposer: () => null }));
vi.mock("./MemoryMap", () => ({
  MemoryMap: ({
    memories,
    hidden,
  }: {
    memories: GovernedMemory[];
    hidden: boolean;
  }) => (
    <div hidden={hidden} data-testid="map-items">
      {memories.map((m) => m.memory_id).join(",")}
    </div>
  ),
}));
vi.mock("./MemoryRecallScope", () => ({
  MemoryRecallScope: () => null,
  memoryProvenanceLabel: () => "Source quote",
}));
const memory = (
  id: string,
  status: GovernedMemory["status"] = "confirmed",
): GovernedMemory => ({
  memory_id: id,
  user_id: "local",
  content: id,
  kind: "chat",
  status,
  origin: "extracted",
  revision: 3,
  scopes: ["all-apps"],
  tags: [],
  evidence_ids: [],
  created_at: "2026-09-19T00:00:00Z",
  updated_at: "2026-09-19T00:00:00Z",
  deleted_at: null,
});
const page = (
  items = [memory("Library entry")],
  total = 201,
  offset = 0,
): MemoryLibraryPage => ({ items, total, offset, limit: 100 });
function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}
beforeEach(() => {
  vi.resetAllMocks();
  api.memories.mockResolvedValue([memory("map-only")]);
  api.mapLibrary.mockResolvedValue(page([memory("map-only")], 1));
  api.library.mockImplementation(async (q) =>
    page([memory(`Library ${q.offset}`)], 201, q.offset),
  );
  api.detail.mockImplementation(async (id) => ({
    memory: memory(id),
    evidence: [],
    history: [],
    index_sync: null,
  }));
  api.mutate.mockImplementation(async (id, action) => ({
    ...memory(id),
    status: action === "delete" ? "deleted" : "confirmed",
    revision: 4,
  }));
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
});
async function manage(props = {}) {
  const view = render(<MemorySpace simplified {...props} />);
  await waitFor(() =>
    expect(screen.getByTestId("map-items").textContent).toBe("map-only"),
  );
  expect(api.library).not.toHaveBeenCalled();
  expect(api.mapLibrary).toHaveBeenCalledWith(
    { query: "", active_only: true, offset: 0, limit: 100 },
    expect.any(AbortSignal),
  );
  fireEvent(window, new Event("one-memory:manage"));
  await screen.findByText("201 total");
  return view;
}
it("queries server totals and pages without replacing the map collection; submitted search and clear reset offset", async () => {
  await manage();
  expect(api.library).toHaveBeenLastCalledWith(
    { status: "confirmed", query: "", offset: 0, limit: 100 },
    expect.any(AbortSignal),
  );
  fireEvent.click(screen.getByRole("checkbox", { name: "Select: Library 0" }));
  fireEvent.click(screen.getByRole("button", { name: "Next page" }));
  await screen.findByRole("checkbox", { name: "Select: Library 100" });
  expect(screen.queryByRole("group", { name: "Bulk actions" })).toBeNull();
  const input = screen.getByLabelText("Search the memory library");
  const calls = api.library.mock.calls.length;
  fireEvent.change(input, { target: { value: " exact % " } });
  expect(api.library).toHaveBeenCalledTimes(calls);
  fireEvent.click(screen.getByRole("button", { name: "Search" }));
  await waitFor(() =>
    expect(api.library).toHaveBeenLastCalledWith(
      { status: "confirmed", query: " exact % ", offset: 0, limit: 100 },
      expect.any(AbortSignal),
    ),
  );
  fireEvent.click(screen.getByRole("button", { name: "Superseded" }));
  await waitFor(() =>
    expect(api.library.mock.lastCall?.[0]).toMatchObject({
      status: "superseded",
      query: " exact % ",
      offset: 0,
    }),
  );
  fireEvent.click(screen.getByRole("button", { name: "Clear search" }));
  await waitFor(() =>
    expect(api.library.mock.lastCall?.[0]).toMatchObject({
      status: "superseded",
      query: "",
      offset: 0,
    }),
  );
  expect(screen.getByTestId("map-items")).toHaveProperty("hidden", true);
  expect(screen.getByTestId("map-items").textContent).toBe("");
  fireEvent(window, new Event("one-memory:manage"));
  await waitFor(() =>
    expect(screen.getByTestId("map-items").textContent).toBe("map-only"),
  );
  expect(api.mapLibrary.mock.lastCall?.[0]).toEqual({
    query: "",
    active_only: true,
    offset: 0,
    limit: 100,
  });
  expect(api.memories.mock.calls.every((c) => !c[0])).toBe(true);
});
it("keeps off-page detail and draft through page changes and legacy map refresh", async () => {
  vi.useFakeTimers();
  const view = render(<MemorySpace simplified />);
  await act(async () => {});
  fireEvent(window, new Event("one-memory:manage"));
  await act(async () => {});
  fireEvent.click(screen.getByRole("button", { name: /Library 0/ }));
  await act(async () => {});
  fireEvent.change(screen.getByRole("textbox", { name: "Memory content" }), {
    target: { value: "Unsaved draft" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Next page" }));
  await act(async () => {
    vi.advanceTimersByTime(10000);
  });
  expect(
    (
      screen.getByRole("textbox", {
        name: "Memory content",
      }) as HTMLTextAreaElement
    ).value,
  ).toBe("Unsaved draft");
  view.unmount();
});
it("aborts superseded/left-page reads, ignores late success, and distinguishes failure from empty", async () => {
  await manage();
  const late = deferred<MemoryLibraryPage>();
  api.library.mockReturnValueOnce(late.promise);
  fireEvent.click(screen.getByRole("button", { name: "Next page" }));
  const signal = api.library.mock.lastCall?.[1];
  api.library.mockRejectedValueOnce(new Error("offline"));
  fireEvent.click(screen.getByRole("button", { name: "Superseded" }));
  await screen.findByText("Failed to load the memory library. Try again.");
  expect(signal.aborted).toBe(true);
  await act(async () => late.resolve(page([memory("Late private data")])));
  expect(screen.queryByText("Late private data")).toBeNull();
  expect(screen.queryByText("0 total")).toBeNull();
  expect(screen.queryByRole("checkbox")).toBeNull();
  const leave = deferred<MemoryLibraryPage>();
  api.library.mockReturnValueOnce(leave.promise);
  fireEvent.click(
    screen.getByRole("button", { name: "Reload the memory library" }),
  );
  const leaveSignal = api.library.mock.lastCall?.[1];
  fireEvent(window, new Event("one-memory:manage"));
  expect(leaveSignal.aborted).toBe(true);
});
it("clears invalid source pages and pending delete confirmation cannot write after revocation", async () => {
  const source = {
    source_id: "cursor",
    authorized: true,
    grant_revision: 1,
  } as SourceResponse;
  const view = await manage({ sources: [source] });
  fireEvent.click(screen.getByRole("checkbox", { name: "Select: Library 0" }));
  fireEvent.click(screen.getByRole("button", { name: "Delete 1 selected" }));
  view.rerender(
    <MemorySpace
      simplified
      sources={[{ ...source, authorized: false, grant_revision: 2 }]}
      sourceSnapshotStale
    />,
  );
  expect(screen.queryByRole("checkbox")).toBeNull();
  const dialog = screen.queryByRole("dialog");
  if (dialog)
    fireEvent.click(within(dialog).getByRole("button", { name: /Delete/ }));
  await act(async () => {});
  expect(api.mutate).not.toHaveBeenCalled();
});
it("refreshes mutation at the same query/page and recovers when the last page becomes empty", async () => {
  await manage();
  fireEvent.click(screen.getByRole("button", { name: "Next page" }));
  await screen.findByRole("checkbox", { name: "Select: Library 100" });
  fireEvent.click(screen.getByRole("button", { name: "Next page" }));
  await screen.findByRole("checkbox", { name: "Select: Library 200" });
  fireEvent.click(
    screen.getByRole("checkbox", { name: "Select: Library 200" }),
  );
  api.library.mockImplementation(async (q) =>
    page(q.offset >= 200 ? [] : [memory("Remaining memory")], 200, q.offset),
  );
  fireEvent.click(screen.getByRole("button", { name: "Delete 1 selected" }));
  fireEvent.click(
    within(screen.getByRole("dialog")).getByRole("button", { name: /Delete/ }),
  );
  await screen.findByRole("checkbox", { name: "Select: Remaining memory" });
  expect(api.mutate).toHaveBeenCalledWith(
    "Library 200",
    "delete",
    3,
    expect.any(String),
    expect.any(String),
  );
  expect(api.library.mock.calls.slice(-2).map((c) => c[0].offset)).toEqual([
    200, 100,
  ]);
  expect(screen.getByText("200 total")).toBeTruthy();
});

it("failed mutation refresh is not empty or writable, and retries preserve unknown-operation request IDs", async () => {
  await manage();
  fireEvent.click(screen.getByRole("checkbox", { name: "Select: Library 0" }));
  api.mutate.mockRejectedValueOnce(new Error("response lost"));
  api.library.mockRejectedValueOnce(new Error("offline"));
  fireEvent.click(screen.getByRole("button", { name: "Delete 1 selected" }));
  fireEvent.click(
    within(screen.getByRole("dialog")).getByRole("button", { name: /Delete/ }),
  );
  await screen.findByText("Failed to load the memory library. Try again.");
  const request = api.mutate.mock.calls[0];
  expect(request).toEqual([
    "Library 0",
    "delete",
    3,
    expect.any(String),
    expect.any(String),
  ]);
  expect(screen.queryByRole("checkbox")).toBeNull();
  expect(
    screen.queryByRole("button", { name: "Delete 1 selected" }),
  ).toBeNull();
  expect(screen.queryByText("0 total")).toBeNull();
  fireEvent.click(
    screen.getByRole("button", { name: "Reload the memory library" }),
  );
  await screen.findByRole("button", { name: "Retry 1 failed" });
  fireEvent.click(screen.getByRole("button", { name: "Retry 1 failed" }));
  fireEvent.click(
    within(await screen.findByRole("dialog")).getByRole("button", {
      name: /Delete/,
    }),
  );
  await waitFor(() => expect(api.mutate).toHaveBeenCalledTimes(2));
  // Each attempt is its own traced interaction, so the trace id differs; the
  // intent that must be preserved is the idempotent request id and the CAS
  // revision. Compare everything except the fresh trace id.
  expect(api.mutate.mock.calls[1].slice(0, 4)).toEqual(request.slice(0, 4));
  expect(api.mutate.mock.calls[1][4]).toEqual(expect.any(String));
});

it("source revision changes invalidate page/detail immediately and abort stale reads", async () => {
  const source = {
    source_id: "cursor",
    authorized: true,
    grant_revision: 1,
  } as SourceResponse;
  const view = await manage({ sources: [source] });
  fireEvent.click(screen.getByRole("button", { name: /Library 0/ }));
  await screen.findByRole("textbox", { name: "Memory content" });
  const late = deferred<MemoryLibraryPage>();
  api.library.mockReturnValueOnce(late.promise);
  fireEvent.click(screen.getByRole("button", { name: "Next page" }));
  const signal = api.library.mock.lastCall?.[1];
  api.library.mockResolvedValueOnce(page([memory("After grant change")], 1));
  view.rerender(
    <MemorySpace simplified sources={[{ ...source, grant_revision: 2 }]} />,
  );
  expect(signal.aborted).toBe(true);
  expect(screen.queryByRole("textbox", { name: "Memory content" })).toBeNull();
  await screen.findByRole("checkbox", { name: "Select: After grant change" });
  await act(async () =>
    late.resolve(page([memory("Stale source response")], 201, 100)),
  );
  expect(screen.queryByText("Stale source response")).toBeNull();
  expect(api.library.mock.lastCall?.[0].offset).toBe(0);
});

it("stops remaining batch writes when source authority changes during the first request", async () => {
  api.library.mockResolvedValue(page([memory("One"), memory("Two")], 201));
  const view = await manage();
  fireEvent.click(screen.getByRole("checkbox", { name: "Select: One" }));
  fireEvent.click(screen.getByRole("checkbox", { name: "Select: Two" }));
  const write = deferred<GovernedMemory>();
  api.mutate.mockReturnValueOnce(write.promise);
  fireEvent.click(screen.getByRole("button", { name: "Delete 2 selected" }));
  fireEvent.click(
    within(screen.getByRole("dialog")).getByRole("button", { name: /Delete/ }),
  );
  await waitFor(() => expect(api.mutate).toHaveBeenCalledTimes(1));
  view.rerender(<MemorySpace simplified sourceSnapshotStale />);
  await act(async () =>
    write.resolve({ ...memory("One"), status: "deleted", revision: 4 }),
  );
  expect(api.mutate).toHaveBeenCalledTimes(1);
  expect(screen.queryByRole("checkbox")).toBeNull();
});

it("unmount aborts a live query and a late response has no continuation", async () => {
  const view = await manage();
  const late = deferred<MemoryLibraryPage>();
  api.library.mockReturnValueOnce(late.promise);
  fireEvent.click(screen.getByRole("button", { name: "Next page" }));
  const signal = api.library.mock.lastCall?.[1];
  const calls = api.library.mock.calls.length;
  view.unmount();
  expect(signal.aborted).toBe(true);
  await act(async () => late.resolve(page([], 0, 100)));
  expect(api.library).toHaveBeenCalledTimes(calls);
});

it("unmount during a batch stops remaining writes and post-mutation reads", async () => {
  api.library.mockResolvedValue(page([memory("One"), memory("Two")], 201));
  const view = await manage();
  fireEvent.click(screen.getByRole("checkbox", { name: "Select: One" }));
  fireEvent.click(screen.getByRole("checkbox", { name: "Select: Two" }));
  const write = deferred<GovernedMemory>();
  api.mutate.mockReturnValueOnce(write.promise);
  fireEvent.click(screen.getByRole("button", { name: "Delete 2 selected" }));
  fireEvent.click(
    within(screen.getByRole("dialog")).getByRole("button", { name: /Delete/ }),
  );
  await waitFor(() => expect(api.mutate).toHaveBeenCalledTimes(1));
  const reads =
    api.memories.mock.calls.length +
    api.library.mock.calls.length +
    api.mapLibrary.mock.calls.length;
  view.unmount();
  await act(async () =>
    write.resolve({ ...memory("One"), status: "deleted", revision: 4 }),
  );
  expect(api.mutate).toHaveBeenCalledTimes(1);
  expect(
    api.memories.mock.calls.length +
      api.library.mock.calls.length +
      api.mapLibrary.mock.calls.length,
  ).toBe(reads);
});

it("correction keeps submitted filter and page, passes CAS, and refreshes the exact page", async () => {
  await manage();
  fireEvent.change(screen.getByLabelText("Search the memory library"), {
    target: { value: "Old wording" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Search" }));
  await waitFor(() =>
    expect(api.library.mock.lastCall?.[0].query).toBe("Old wording"),
  );
  await screen.findByRole("checkbox", { name: "Select: Library 0" });
  fireEvent.click(screen.getByRole("button", { name: "Next page" }));
  await screen.findByRole("checkbox", { name: "Select: Library 100" });
  fireEvent.click(screen.getByRole("button", { name: /Library 100/ }));
  await screen.findByRole("textbox", { name: "Memory content" });
  fireEvent.change(screen.getByRole("textbox", { name: "Memory content" }), {
    target: { value: "New wording" },
  });
  api.correct.mockResolvedValueOnce({
    ...memory("Library 100"),
    content: "New wording",
    revision: 4,
  });
  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: "Save correction" }),
    ).toHaveProperty("disabled", false),
  );
  fireEvent.click(screen.getByRole("button", { name: "Save correction" }));
  await waitFor(() =>
    expect(api.correct).toHaveBeenCalledWith(
      "Library 100",
      3,
      "New wording",
      ["all-apps"],
      expect.any(String),
      expect.any(String),
    ),
  );
  await waitFor(() =>
    expect(api.library.mock.lastCall?.[0]).toEqual({
      query: "Old wording",
      status: "confirmed",
      offset: 100,
      limit: 100,
    }),
  );
  expect(screen.getByTestId("map-items")).toHaveProperty("hidden", true);
  expect(screen.getByTestId("map-items").textContent).toBe("");
  expect(
    api.mapLibrary.mock.calls.every(
      ([query]) =>
        query.active_only === true &&
        query.query === "" &&
        query.status === undefined,
    ),
  ).toBe(true);
});

it("only successful empty pages show zero and disabled pagination", async () => {
  await manage();
  api.library.mockResolvedValueOnce(page([], 0));
  fireEvent.change(screen.getByLabelText("Search the memory library"), {
    target: { value: "No match" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Search" }));
  await screen.findByText("0 total");
  expect(
    (screen.getByRole("button", { name: "Previous page" }) as HTMLButtonElement)
      .disabled,
  ).toBe(true);
  expect(
    (screen.getByRole("button", { name: "Next page" }) as HTMLButtonElement)
      .disabled,
  ).toBe(true);
  expect(screen.queryByRole("checkbox")).toBeNull();
});

it("active=false aborts management reads and return reloads rather than accepting a late page", async () => {
  const view = await manage();
  const late = deferred<MemoryLibraryPage>();
  api.library.mockReturnValueOnce(late.promise);
  fireEvent.click(screen.getByRole("button", { name: "Next page" }));
  const signal = api.library.mock.lastCall?.[1];
  view.rerender(<MemorySpace simplified active={false} />);
  expect(signal.aborted).toBe(true);
  await act(async () => late.resolve(page([memory("Late page")], 201, 100)));
  api.library.mockResolvedValueOnce(page([memory("Reloaded page")], 201, 100));
  view.rerender(<MemorySpace simplified />);
  await screen.findByRole("checkbox", { name: "Select: Reloaded page" });
  expect(screen.queryByText("Late page")).toBeNull();
});

async function selectedOffPage() {
  vi.useFakeTimers();
  const view = render(<MemorySpace simplified />);
  await act(async () => {});
  fireEvent(window, new Event("one-memory:manage"));
  await act(async () => {});
  fireEvent.click(screen.getByRole("button", { name: /Library 0/ }));
  await act(async () => {});
  fireEvent.change(screen.getByRole("textbox", { name: "Memory content" }), {
    target: { value: "Local unsaved draft" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Next page" }));
  await act(async () => {});
  return view;
}
const advanceDetailCheck = async () => {
  await act(async () => {
    vi.advanceTimersByTime(10000);
  });
};

it.each(["deleted", "403", "404"])(
  "authoritatively hides off-page selected detail on external %s",
  async (kind) => {
    await selectedOffPage();
    const reads = api.detail.mock.calls.length;
    if (kind === "deleted")
      api.detail.mockResolvedValue({
        memory: {
          ...memory("Library 0", "deleted"),
          deleted_at: "2026-09-19T01:00:00Z",
        },
        evidence: [],
        history: [],
      });
    else
      api.detail.mockRejectedValue({
        response: {
          status: Number(kind),
          data: {
            detail:
              kind === "403" ? "memory_source_revoked" : "memory_not_found",
          },
        },
      });
    await advanceDetailCheck();
    expect(api.detail.mock.calls.length).toBeGreaterThan(reads);
    expect(api.detail.mock.lastCall?.[0]).toBe("Library 0");
    expect(
      screen.queryByRole("textbox", { name: "Memory content" }),
    ).toBeNull();
    expect(screen.getByRole("alert").textContent).toContain(
      "can no longer be read",
    );
  },
);

it("external revision preserves draft and original CAS with a visible conflict, without keystroke reads", async () => {
  await selectedOffPage();
  const reads = api.detail.mock.calls.length;
  fireEvent.change(screen.getByRole("textbox", { name: "Memory content" }), {
    target: { value: "Keep writing the draft" },
  });
  expect(api.detail).toHaveBeenCalledTimes(reads);
  api.detail.mockResolvedValue({
    memory: {
      ...memory("Library 0"),
      content: "Newer version from another client",
      revision: 4,
    },
    evidence: [],
    history: [],
  });
  await advanceDetailCheck();
  expect(
    (
      screen.getByRole("textbox", {
        name: "Memory content",
      }) as HTMLTextAreaElement
    ).value,
  ).toBe("Keep writing the draft");
  expect(screen.queryByRole("status", { name: "Detail freshness" })).toBeNull();
  expect(screen.queryByText(/revision 4/)).toBeNull();
  api.correct.mockRejectedValueOnce({
    response: { status: 409, data: { detail: "memory_revision_conflict" } },
  });
  fireEvent.click(screen.getByRole("button", { name: "Save correction" }));
  await act(async () => {});
  expect(api.correct).toHaveBeenCalledWith(
    "Library 0",
    3,
    "Keep writing the draft",
    ["all-apps"],
    expect.any(String),
    expect.any(String),
  );
  expect(screen.getByRole("alert").textContent).toContain(
    "The memory was updated",
  );
});

it("does not show the in-flight verification line on the memory page", async () => {
  await selectedOffPage();
  api.detail.mockReturnValue(new Promise(() => {}));
  await advanceDetailCheck();
  expect(screen.queryByText("Verifying the selected memory...")).toBeNull();
  expect(screen.queryByText(/verification failed/)).toBeNull();
  expect(screen.queryByRole("status", { name: "Detail freshness" })).toBeNull();
});

it("failed authoritative read is not deletion or freshness and disables writes until verified", async () => {
  await selectedOffPage();
  api.detail.mockRejectedValue(new Error("offline"));
  await advanceDetailCheck();
  expect(
    (
      screen.getByRole("textbox", {
        name: "Memory content",
      }) as HTMLTextAreaElement
    ).value,
  ).toBe("Local unsaved draft");
  expect(screen.queryByText(/verification failed/)).toBeNull();
  expect(screen.queryByRole("status", { name: "Detail freshness" })).toBeNull();
  expect(
    (
      screen.getByRole("button", {
        name: "Save correction",
      }) as HTMLButtonElement
    ).disabled,
  ).toBe(true);
  const form = screen
    .getByRole("textbox", { name: "Memory content" })
    .closest("form");
  expect(form).not.toBeNull();
  if (form) fireEvent.submit(form);
  expect(api.correct).not.toHaveBeenCalled();
  api.detail.mockResolvedValue({
    memory: memory("Library 0"),
    evidence: [],
    history: [],
  });
  await advanceDetailCheck();
  expect(
    (
      screen.getByRole("button", {
        name: "Save correction",
      }) as HTMLButtonElement
    ).disabled,
  ).toBe(false);
});

it.each(["selection", "inactive", "revoked", "unmount"])(
  "late authoritative response cannot overwrite after %s",
  async (mode) => {
    const view = await selectedOffPage();
    const pending = deferred<MemoryDetail>();
    const initialReads = api.detail.mock.calls.length;
    api.detail.mockReturnValueOnce(pending.promise);
    await advanceDetailCheck();
    const reads = api.detail.mock.calls.length;
    expect(reads).toBe(initialReads + 1);
    if (mode === "selection") {
      fireEvent.click(screen.getByRole("button", { name: /Library 100/ }));
      await act(async () => {});
    } else if (mode === "inactive")
      view.rerender(<MemorySpace simplified active={false} />);
    else if (mode === "revoked")
      view.rerender(<MemorySpace simplified sourceSnapshotStale />);
    else view.unmount();
    await act(async () =>
      pending.resolve({
        memory: memory("Library 0", "deleted"),
        evidence: [],
        history: [],
        index_sync: null,
      }),
    );
    expect(screen.queryByText(/can no longer be read/)).toBeNull();
    if (mode === "selection")
      expect(
        (
          screen.getByRole("textbox", {
            name: "Memory content",
          }) as HTMLTextAreaElement
        ).value,
      ).toBe("Library 100");
    else {
      await advanceDetailCheck();
      expect(api.detail).toHaveBeenCalledTimes(reads);
    }
  },
);

it("external removal while delete confirmation is open cannot send the old intent", async () => {
  await selectedOffPage();
  fireEvent.click(screen.getByRole("button", { name: "Delete" }));
  api.detail.mockResolvedValue({
    memory: memory("Library 0", "deleted"),
    evidence: [],
    history: [],
  });
  await advanceDetailCheck();
  expect(screen.queryByRole("textbox", { name: "Memory content" })).toBeNull();
  fireEvent.click(
    within(screen.getByRole("dialog")).getByRole("button", { name: /Delete/ }),
  );
  await act(async () => {});
  expect(api.mutate).not.toHaveBeenCalled();
});
