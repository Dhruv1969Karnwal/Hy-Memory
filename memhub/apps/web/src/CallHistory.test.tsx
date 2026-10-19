// @vitest-environment jsdom

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
import { CallHistory } from "./CallHistory";
import type {
  CallEvent,
  CallHistoryPage,
  ResolvedCallReference,
} from "./call-history-client";

const { getHistory, clearEvents, clearAllEvents, getReferences } = vi.hoisted(
  () => ({
    getHistory: vi.fn(),
    clearEvents: vi.fn(),
    clearAllEvents: vi.fn(),
    getReferences: vi.fn(),
  }),
);
const getConversation = vi.hoisted(() => vi.fn());
vi.mock("./api", () => ({
  getConversation,
}));
vi.mock("./call-history-client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./call-history-client")>()),
  getCallHistory: getHistory,
  clearCallEvents: clearEvents,
  clearAllCallEvents: clearAllEvents,
  getCallReferences: getReferences,
}));

function event(id = "event-1", patch: Partial<CallEvent> = {}): CallEvent {
  return {
    event_id: id,
    occurred_at: "2026-09-20T11:03:37.197+08:00",
    host_id: "cursor",
    installation_id: "install-1",
    tool: "om_recall",
    state: "grounded",
    stage: "core_result",
    delivery: "unknown",
    adoption: "unknown",
    host_session_id: null,
    injection_elapsed_ms: null,
    elapsed_ms: 26.8,
    raw_count: 2,
    memory_count: 1,
    references: [
      { kind: "raw", id: "raw-1", revision: 3, source_id: "source-1" },
      { kind: "memory", id: "memory-1", revision: 2, source_id: null },
    ],
    ...patch,
  };
}

function page(
  items = [event()],
  patch: Partial<CallHistoryPage> = {},
): CallHistoryPage {
  return {
    schema_version: 1,
    available: true,
    recording_error: false,
    total: items.length,
    offset: 0,
    limit: 20,
    items,
    retention_days: 30,
    max_events: 10000,
    ...patch,
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

function mount() {
  return render(<CallHistory onBack={vi.fn()} />);
}
function refresh() {
  fireEvent.click(screen.getByRole("button", { name: "Refresh records" }));
}
function row(id: string) {
  return document.querySelector(`[data-event-id="${id}"]`);
}
async function loaded(id = "event-1") {
  await waitFor(() => expect(row(id)).not.toBeNull());
  return row(id) as HTMLElement;
}
async function openClear() {
  fireEvent.click(screen.getByRole("button", { name: "Clear this page" }));
  return screen.findByRole("dialog");
}

const originalShowModal = Object.getOwnPropertyDescriptor(
  HTMLDialogElement.prototype,
  "showModal",
);
beforeEach(() => {
  getHistory.mockReset().mockResolvedValue(page());
  clearEvents.mockReset().mockResolvedValue(undefined);
  clearAllEvents.mockReset().mockResolvedValue(undefined);
  getConversation.mockReset().mockResolvedValue({
    conversation_id: "c1",
    source_id: "source-1",
    messages: [{ verbatim: "current permitted original" }],
  });
  getReferences.mockReset().mockImplementation(async (call: CallEvent) =>
    call.references.map((reference) => ({
      ...reference,
      available: false,
      title: null,
      reason: "unavailable",
    })),
  );
  Object.defineProperty(HTMLDialogElement.prototype, "showModal", {
    configurable: true,
    value: function (this: HTMLDialogElement) {
      this.setAttribute("open", "");
    },
  });
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  if (originalShowModal)
    Object.defineProperty(
      HTMLDialogElement.prototype,
      "showModal",
      originalShowModal,
    );
  else Reflect.deleteProperty(HTMLDialogElement.prototype, "showModal");
});

it("loads a bounded first page and returns to applications", async () => {
  const pending = deferred<CallHistoryPage>();
  getHistory.mockReturnValueOnce(pending.promise);
  const onBack = vi.fn();
  render(<CallHistory onBack={onBack} />);
  expect(screen.getByRole("status").textContent).toContain("Loading");
  expect(getHistory).toHaveBeenCalledWith({ limit: 20, offset: 0 });
  await act(async () => pending.resolve(page()));
  expect(await loaded()).toBeTruthy();
  expect(
    screen.getByText("Retained for 30 days, up to 10,000 records"),
  ).toBeTruthy();
  expect(
    (screen.getByRole("button", { name: "Previous page" }) as HTMLButtonElement)
      .disabled,
  ).toBe(true);
  expect(
    (screen.getByRole("button", { name: "Next page" }) as HTMLButtonElement)
      .disabled,
  ).toBe(true);
  fireEvent.click(screen.getByRole("button", { name: "Back to apps" }));
  expect(onBack).toHaveBeenCalledOnce();
});

it("renders an empty page without inventing a successful call", async () => {
  getHistory.mockResolvedValue(page([]));
  mount();
  await screen.findByText("No usage records yet");
  expect(screen.queryByText(/Core produced a result|Adopted/)).toBeNull();
  expect(
    (
      screen.getByRole("button", {
        name: "Clear this page",
      }) as HTMLButtonElement
    ).disabled,
  ).toBe(true);
});

it.each(["reject", "unavailable"])(
  "does not render unavailable data as an empty or successful result: %s",
  async (mode) => {
    if (mode === "reject")
      getHistory.mockRejectedValue(new Error("sensitive response body"));
    else getHistory.mockResolvedValue(page([event()], { available: false }));
    mount();
    await screen.findByRole("alert");
    expect(screen.queryByText("No usage records yet")).toBeNull();
    expect(
      screen.queryByText(/memory recall|om_recall|sensitive response body/),
    ).toBeNull();
    getHistory.mockResolvedValue(page());
    refresh();
    await loaded();
  },
);

it("hides the previous page during refresh and after failure", async () => {
  mount();
  await loaded();
  const pending = deferred<CallHistoryPage>();
  getHistory.mockReturnValueOnce(pending.promise);
  refresh();
  expect(row("event-1")).toBeNull();
  expect(
    screen.queryByRole("navigation", { name: "Record pagination" }),
  ).toBeNull();
  await act(async () => pending.reject(new Error("offline")));
  await screen.findByRole("alert");
  expect(screen.queryByText(/Core produced a result/)).toBeNull();
});

it("ignores late refresh responses when a newer request has completed", async () => {
  const old = deferred<CallHistoryPage>();
  getHistory.mockReturnValueOnce(old.promise);
  mount();
  getHistory.mockResolvedValueOnce(page([event("new")]));
  refresh();
  await loaded("new");
  await act(async () => old.resolve(page([event("old")])));
  expect(row("old")).toBeNull();
  expect(await loaded("new")).toBeTruthy();
});

it("paginates in bounded pages and resets the offset on host filtering", async () => {
  const first = page(
    Array.from({ length: 20 }, (_, index) => event(`first-${index}`)),
    { total: 21 },
  );
  getHistory
    .mockResolvedValueOnce(first)
    .mockResolvedValueOnce(page([event("last")], { offset: 20, total: 21 }));
  mount();
  await loaded("first-0");
  expect(
    within(screen.getByRole("list", { name: "Call records" })).getAllByRole(
      "listitem",
    ),
  ).toHaveLength(20);
  fireEvent.click(screen.getByRole("button", { name: "Next page" }));
  await loaded("last");
  expect(getHistory).toHaveBeenLastCalledWith({ limit: 20, offset: 20 });
  expect(row("first-0")).toBeNull();
  expect(
    (screen.getByRole("button", { name: "Next page" }) as HTMLButtonElement)
      .disabled,
  ).toBe(true);
  getHistory.mockResolvedValueOnce(first);
  fireEvent.click(screen.getByRole("button", { name: "Previous page" }));
  await loaded("first-0");
  expect(getHistory).toHaveBeenLastCalledWith({ limit: 20, offset: 0 });
  getHistory.mockResolvedValueOnce(page([event("filtered")]));
  fireEvent.change(screen.getByRole("combobox", { name: "Filter by app" }), {
    target: { value: "cursor" },
  });
  await loaded("filtered");
  expect(getHistory).toHaveBeenLastCalledWith({
    limit: 20,
    offset: 0,
    hostId: "cursor",
  });
});

it("ignores an older host response and allows returning to the unfiltered list", async () => {
  getHistory.mockResolvedValueOnce(
    page([event(), event("other", { host_id: "codex" })]),
  );
  mount();
  await loaded();
  const old = deferred<CallHistoryPage>();
  getHistory
    .mockReturnValueOnce(old.promise)
    .mockResolvedValueOnce(page([event("codex", { host_id: "codex" })]));
  const filter = screen.getByRole("combobox");
  fireEvent.change(filter, { target: { value: "cursor" } });
  fireEvent.change(filter, { target: { value: "codex" } });
  await loaded("codex");
  await act(async () => old.resolve(page([event("old")])));
  expect(row("old")).toBeNull();
  fireEvent.change(filter, { target: { value: "" } });
  await loaded();
  expect(getHistory).toHaveBeenLastCalledWith({ limit: 20, offset: 0 });
});

it("shows exact identity, timestamp, result counts and metadata-only references", async () => {
  mount();
  fireEvent.click(await loaded());
  const detail = screen.getByRole("article", { name: "Call details" });
  expect(
    within(detail).getByText("2026-09-20T11:03:37.197+08:00"),
  ).toBeTruthy();
  expect(within(detail).getByText("Memory lookup")).toBeTruthy();
  expect(within(detail).queryByText("om_recall")).toBeNull();
  // The metadata table must not leak the raw Core state enum next to the
  // The humanized state must not leak the raw Core enum.
  expect(within(detail).queryByText("grounded")).toBeNull();
  expect(within(detail).getAllByText("Query succeeded").length).toBeGreaterThan(
    0,
  );
  expect(
    within(detail).getByText(
      "Found 2 source excerpts · memories 1 · elapsed 0.0 seconds",
    ),
  ).toBeTruthy();
  expect(
    within(detail).getByText((value) =>
      value.includes("Cursor found 3 related items"),
    ),
  ).toBeTruthy();
  expect(within(detail).getAllByText("source-1").length).toBeGreaterThan(0);
  expect(within(detail).queryByText(/revision 3|revision 2/)).toBeNull();
  expect(within(detail).queryByRole("link")).toBeNull();
  expect(
    within(detail).queryByText(/Call succeeded|Already adopted/),
  ).toBeNull();
  expect(
    within(detail).queryByText(
      /Delivery unconfirmed|Adoption unconfirmed|Core/,
    ),
  ).toBeNull();
  expect(document.activeElement).toBe(
    screen.getByRole("heading", { name: "This memory lookup" }),
  );
  fireEvent.click(screen.getByRole("button", { name: "Back to records" }));
  expect(document.activeElement).toBe(await loaded());
});
it("shows delivered delivery without claiming adoption", async () => {
  getHistory.mockResolvedValue(
    page([
      event("delivered", {
        delivery: "delivered",
        host_session_id: "session-1",
        injection_elapsed_ms: 31.4,
      }),
    ]),
  );
  mount();
  fireEvent.click(await loaded("delivered"));
  const detail = screen.getByRole("article", { name: "Call details" });
  expect(
    within(detail).getAllByText("Provided to Cursor").length,
  ).toBeGreaterThan(0);
  expect(within(detail).queryByText(/Adopted/)).toBeNull();
});
it("does not claim delivery for a call Core never completed", async () => {
  getHistory.mockResolvedValue(
    page([
      event("unavailable", {
        state: "unavailable",
        delivery: "unknown",
        raw_count: 0,
        memory_count: 0,
        references: [],
      }),
    ]),
  );
  mount();
  fireEvent.click(
    await screen.findByRole("button", {
      name: "Cursor cannot look up memory right now",
    }),
  );
  const detail = screen.getByRole("article", { name: "Call details" });
  expect(within(detail).queryByText(/Returned to/)).toBeNull();
  expect(
    within(detail).getByText(
      "One Memory is unavailable right now; this answer did not include memory.",
    ),
  ).toBeTruthy();
});
it("does not restate an undelivered result as returned", async () => {
  getHistory.mockResolvedValue(
    page([event("not-delivered", { delivery: "not_delivered" })]),
  );
  mount();
  fireEvent.click(await loaded("not-delivered"));
  const detail = screen.getByRole("article", { name: "Call details" });
  expect(within(detail).getAllByText("Not provided to the app")).toHaveLength(
    1,
  );
  expect(within(detail).queryByText(/Returned to/)).toBeNull();
});
it("shows delivery state in the list without turning it into adoption", async () => {
  getHistory.mockResolvedValue(
    page([
      event("delivered-list", { delivery: "delivered" }),
      event("not-delivered-list", { delivery: "not_delivered" }),
      event("unknown-list"),
    ]),
  );
  mount();
  await loaded("delivered-list");
  const list = screen.getByRole("list", { name: "Call records" });
  expect(within(list).getByText("Provided to Cursor")).toBeTruthy();
  expect(within(list).getByText("Not provided to the app")).toBeTruthy();
  expect(within(list).getByText("Query succeeded")).toBeTruthy();
  expect(within(list).queryByText(/Adopted/)).toBeNull();
});

it("keeps detail bound to event ID across refresh reordering, then removes missing details", async () => {
  getHistory.mockResolvedValueOnce(page([event("one"), event("two")]));
  mount();
  fireEvent.click(await loaded("two"));
  getHistory.mockResolvedValueOnce(
    page([event("two", { memory_count: 7 }), event("one")]),
  );
  refresh();
  expect(screen.queryByRole("article")).toBeNull();
  await screen.findByText(
    "Found 2 source excerpts · memories 7 · elapsed 0.0 seconds",
  );
  expect(screen.getByRole("article").textContent).not.toContain("memories 1");
  getHistory.mockResolvedValueOnce(page([event("one")]));
  refresh();
  await screen.findByText(
    "This record is no longer on the current page. Go back to the records to view it.",
  );
  expect(screen.queryByRole("article")).toBeNull();
});

it.each([
  [
    "grounded",
    "Query succeeded",
    "Direct call (diagnostics) found 0 related items",
  ],
  [
    "empty",
    "Query succeeded, no related memory found",
    "Direct call (diagnostics) found no related memory",
  ],
  [
    "partial",
    "Query succeeded, some content is still being organized",
    "Direct call (diagnostics) found 0 related items",
  ],
  [
    "timed_out",
    "Query timed out, no memory was provided",
    "Direct call (diagnostics) did not get memory in time this time",
  ],
  [
    "unavailable",
    "One Memory is unavailable right now",
    "Direct call (diagnostics) cannot look up memory right now",
  ],
  [
    "denied",
    "The current authorization does not allow this query",
    "Direct call (diagnostics) was not allowed to look up memory",
  ],
] as const)(
  "renders the %s state without claiming external adoption",
  async (state, label, headline) => {
    getHistory.mockResolvedValue(
      page([
        event("state", {
          state,
          host_id: null,
          installation_id: null,
          references: [],
          raw_count: 0,
          memory_count: 0,
        }),
      ]),
    );
    mount();
    fireEvent.click(
      await screen.findByRole("button", {
        name: headline,
      }),
    );
    expect(
      screen.getByRole("heading", {
        name: /^Not recorded · 09\/20 \d{2}:03$/,
      }),
    ).toBeTruthy();
    expect(screen.getByText("No related content found")).toBeTruthy();
    expect(screen.queryByText(/Already adopted|Call succeeded/)).toBeNull();
    // The humanized state label appears both as the stage line and in the
    // Record details table; the raw enum must appear nowhere.
    expect(screen.getAllByText(new RegExp(label)).length).toBeGreaterThan(0);
    expect(screen.queryByText(state)).toBeNull();
  },
);

it("does not call an unavailable call returned", async () => {
  getHistory.mockResolvedValue(
    page([
      event("unavailable-call", {
        state: "unavailable",
        raw_count: 0,
        memory_count: 0,
        references: [],
      }),
    ]),
  );
  mount();
  fireEvent.click(
    await screen.findByRole("button", {
      name: "Cursor cannot look up memory right now",
    }),
  );
  const detail = screen.getByRole("article", { name: "Call details" });
  expect(within(detail).queryByText(/· Returned/)).toBeNull();
  expect(
    within(detail).getAllByText(/One Memory is unavailable right now/).length,
  ).toBeGreaterThan(0);
  expect(within(detail).queryByText(/Adoption unconfirmed|Core/)).toBeNull();
});

it.each(["om_recall", "om_conversations", "om_ask", "om_matters"] as const)(
  "renders %s as Memory lookup instead of the API name",
  async (tool) => {
    getHistory.mockResolvedValue(page([event("tool", { tool })]));
    mount();
    expect(
      await screen.findByRole("button", {
        name: "Cursor found 3 related items",
      }),
    ).toBeTruthy();
    fireEvent.click(
      screen.getByRole("button", { name: "Cursor found 3 related items" }),
    );
    expect(
      within(screen.getByRole("article")).getByText("Memory lookup"),
    ).toBeTruthy();
    expect(screen.queryByText(tool)).toBeNull();
  },
);

it("warns when recording is incomplete without hiding readable historical metadata", async () => {
  getHistory.mockResolvedValue(page([event()], { recording_error: true }));
  mount();
  await loaded();
  expect(screen.getByRole("alert").textContent).toContain(
    "the list may be incomplete",
  );
});

it("requires a distinct confirmation for clearing all retained records", async () => {
  getHistory.mockResolvedValueOnce(page([event()], { total: 21, offset: 0 }));
  mount();
  await loaded();
  fireEvent.click(screen.getByRole("button", { name: "Clear all records" }));
  const dialog = await screen.findByRole("dialog");
  expect(
    within(dialog).getByRole("heading", { name: "Clear all usage records?" }),
  ).toBeTruthy();
  expect(dialog.textContent).toContain(
    "not just this page or the current filter",
  );
  fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
  expect(clearAllEvents).not.toHaveBeenCalled();
  expect(clearEvents).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Clear all records" }));
  getHistory.mockResolvedValueOnce(page([], { total: 0 }));
  fireEvent.click(
    within(await screen.findByRole("dialog")).getByRole("button", {
      name: "Confirm clearing all",
    }),
  );
  await screen.findByText("No usage records yet");
  // A clear is a real write: it now carries a trace id so the ui.action /
  // ui.result lines join Core's kernel-call/return in one log.
  expect(clearAllEvents).toHaveBeenCalledTimes(1);
  expect(clearAllEvents.mock.calls[0][0]).toEqual(expect.any(String));
  expect(clearEvents).not.toHaveBeenCalled();
  expect(getHistory).toHaveBeenLastCalledWith({ limit: 20, offset: 0 });
});

it("requires confirmation and cancelling or Escape sends no clear request", async () => {
  mount();
  await loaded();
  let dialog = await openClear();
  expect(
    within(dialog).getByRole("heading", {
      name: "Clear 1 record on this page?",
    }),
  ).toBeTruthy();
  fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
  expect(clearEvents).not.toHaveBeenCalled();
  dialog = await openClear();
  fireEvent(dialog, new Event("cancel", { bubbles: true, cancelable: true }));
  expect(screen.queryByRole("dialog")).toBeNull();
  expect(clearEvents).not.toHaveBeenCalled();
});

it("clears only the confirmed current-page IDs, suppresses double submit, and rereads", async () => {
  getHistory
    .mockResolvedValueOnce(page([event("first")], { total: 21 }))
    .mockResolvedValueOnce(page([event("last")], { offset: 20, total: 21 }));
  mount();
  await loaded("first");
  fireEvent.click(screen.getByRole("button", { name: "Next page" }));
  await loaded("last");
  const pending = deferred<void>();
  clearEvents.mockReturnValueOnce(pending.promise);
  const dialog = await openClear();
  const confirm = within(dialog).getByRole("button", {
    name: "Confirm clearing",
  });
  fireEvent.click(confirm);
  fireEvent.click(confirm);
  expect(clearEvents).toHaveBeenCalledTimes(1);
  expect(clearEvents.mock.calls[0][0]).toEqual(["last"]);
  expect(clearEvents.mock.calls[0][1]).toEqual(expect.any(String));
  expect(row("last")).toBeNull();
  expect(
    (
      within(dialog).getByRole("button", {
        name: "Cancel",
      }) as HTMLButtonElement
    ).disabled,
  ).toBe(true);
  getHistory.mockResolvedValueOnce(
    page([event("new-arrival")], { offset: 20, total: 21 }),
  );
  await act(async () => pending.resolve());
  await loaded("new-arrival");
  expect(getHistory).toHaveBeenLastCalledWith({ limit: 20, offset: 20 });
  expect(clearEvents).toHaveBeenCalledTimes(1);
  expect(screen.getByText("Selected records cleared.")).toBeTruthy();
});

it("rereads after an unknown clear outcome without replaying the deletion", async () => {
  mount();
  await loaded();
  clearEvents.mockRejectedValueOnce(new Error("private body"));
  getHistory.mockResolvedValueOnce(page([event("retained")]));
  const dialog = await openClear();
  fireEvent.click(
    within(dialog).getByRole("button", { name: "Confirm clearing" }),
  );
  await loaded("retained");
  expect(screen.getByText(/clear result is unconfirmed/)).toBeTruthy();
  expect(
    screen.queryByText(/private body|Selected records cleared/),
  ).toBeNull();
  refresh();
  await loaded();
  expect(clearEvents).toHaveBeenCalledTimes(1);
  expect(clearEvents.mock.calls[0][0]).toEqual(["event-1"]);
});

it("keeps read failure and unknown deletion visible together", async () => {
  mount();
  await loaded();
  clearEvents.mockRejectedValueOnce(new Error("lost response"));
  getHistory.mockRejectedValueOnce(new Error("offline"));
  const dialog = await openClear();
  fireEvent.click(
    within(dialog).getByRole("button", { name: "Confirm clearing" }),
  );
  await screen.findByText("Couldn't read the records. Refresh and try again.");
  expect(screen.getByText(/clear result is unconfirmed/)).toBeTruthy();
  expect(screen.queryByText("No usage records yet")).toBeNull();
  expect(clearEvents).toHaveBeenCalledTimes(1);
});

it("reports a refused clear as a definite failure, not as unconfirmed", async () => {
  // Core answers `409 call_history_disabled` when the store is disabled: that
  // is a refusal, so nothing was deleted. The old code logged `unknown` and
  // said "unconfirmed" for this case, contradicting a definite Core answer.
  mount();
  await loaded();
  clearEvents.mockRejectedValueOnce(
    Object.assign(new Error("refused"), {
      response: { status: 409, data: { detail: "call_history_disabled" } },
    }),
  );
  getHistory.mockResolvedValueOnce(page([event("retained")]));
  const dialog = await openClear();
  fireEvent.click(
    within(dialog).getByRole("button", { name: "Confirm clearing" }),
  );
  expect(
    await screen.findByText(
      /Couldn't clear the selected records. The existing records are unchanged/,
    ),
  ).toBeTruthy();
  expect(screen.queryByText(/clear result is unconfirmed/)).toBeNull();
  // The records are still there and are never auto-deleted twice.
  expect(clearEvents).toHaveBeenCalledTimes(1);
});

it("reports a refused clear-all as a definite failure", async () => {
  mount();
  await loaded();
  clearAllEvents.mockRejectedValueOnce(
    Object.assign(new Error("refused"), {
      response: { status: 409, data: { detail: "call_history_disabled" } },
    }),
  );
  getHistory.mockResolvedValueOnce(page([event("retained")]));
  fireEvent.click(screen.getByRole("button", { name: "Clear all records" }));
  fireEvent.click(
    within(await screen.findByRole("dialog")).getByRole("button", {
      name: "Confirm clearing all",
    }),
  );
  expect(
    await screen.findByText(
      /Couldn't clear the usage records. The existing records are unchanged/,
    ),
  ).toBeTruthy();
  expect(screen.queryByText(/clear result is unconfirmed/)).toBeNull();
  expect(clearAllEvents).toHaveBeenCalledTimes(1);
});

it("ignores late reads after unmount and never persists metadata locally", async () => {
  const storage = vi.spyOn(Storage.prototype, "setItem");
  const old = deferred<CallHistoryPage>();
  getHistory.mockReturnValueOnce(old.promise);
  const view = mount();
  view.unmount();
  mount();
  await loaded();
  await act(async () => old.resolve(page([event("old")])));
  expect(row("old")).toBeNull();
  expect(storage).not.toHaveBeenCalled();
});

it("does not issue recovery reads after unmount during deletion", async () => {
  const view = mount();
  await loaded();
  const pending = deferred<void>();
  clearEvents.mockReturnValueOnce(pending.promise);
  const dialog = await openClear();
  fireEvent.click(
    within(dialog).getByRole("button", { name: "Confirm clearing" }),
  );
  view.unmount();
  await act(async () => pending.reject(new Error("lost response")));
  await waitFor(() => expect(getHistory).toHaveBeenCalledTimes(1));
  expect(clearEvents).toHaveBeenCalledTimes(1);
});

it("can return from a page emptied by retention or clearing", async () => {
  getHistory
    .mockResolvedValueOnce(page([event()], { total: 21 }))
    .mockResolvedValueOnce(page([], { offset: 20, total: 20 }));
  mount();
  await loaded();
  fireEvent.click(screen.getByRole("button", { name: "Next page" }));
  await screen.findByText("No records on this page");
  expect(
    (screen.getByRole("button", { name: "Next page" }) as HTMLButtonElement)
      .disabled,
  ).toBe(true);
  expect(
    (screen.getByRole("button", { name: "Previous page" }) as HTMLButtonElement)
      .disabled,
  ).toBe(false);
  fireEvent.click(screen.getByRole("button", { name: "Previous page" }));
  await loaded();
  expect(getHistory).toHaveBeenLastCalledWith({ limit: 20, offset: 0 });
});

it("offers host filters even when that host is absent from the current page", async () => {
  mount();
  await loaded();
  getHistory.mockResolvedValueOnce(page([]));
  fireEvent.change(screen.getByRole("combobox"), {
    target: { value: "workbuddy" },
  });
  await screen.findByText("No usage records yet");
  expect(getHistory).toHaveBeenLastCalledWith({
    limit: 20,
    offset: 0,
    hostId: "workbuddy",
  });
});

it("preserves the filter and offset after an unknown clear result", async () => {
  mount();
  await loaded();
  getHistory
    .mockResolvedValueOnce(page([event()], { total: 21 }))
    .mockResolvedValueOnce(page([event("last")], { total: 21, offset: 20 }));
  fireEvent.change(screen.getByRole("combobox"), {
    target: { value: "cursor" },
  });
  await loaded();
  fireEvent.click(screen.getByRole("button", { name: "Next page" }));
  await loaded("last");
  clearEvents.mockRejectedValueOnce(new Error("unknown"));
  getHistory.mockResolvedValueOnce(
    page([event("last")], { total: 21, offset: 20 }),
  );
  const dialog = await openClear();
  fireEvent.click(
    within(dialog).getByRole("button", { name: "Confirm clearing" }),
  );
  await screen.findByText(/clear result is unconfirmed/);
  await loaded("last");
  expect(getHistory).toHaveBeenLastCalledWith({
    limit: 20,
    offset: 20,
    hostId: "cursor",
  });
  expect(clearEvents).toHaveBeenCalledTimes(1);
  expect(clearEvents.mock.calls[0][0]).toEqual(["last"]);
});

function resolved(call: CallEvent, title: string): ResolvedCallReference[] {
  return call.references.map((reference) => ({
    ...reference,
    available: true,
    title: `${title}-${reference.kind}`,
    reason: null,
  }));
}

it("resolves titles only on detail selection and shows explicit loading without invented content", async () => {
  const call = event();
  const pending = deferred<ResolvedCallReference[]>();
  getReferences.mockReturnValueOnce(pending.promise);
  const storage = vi.spyOn(Storage.prototype, "setItem");
  mount();
  const row = await loaded();
  expect(getReferences).not.toHaveBeenCalled();
  fireEvent.click(row);
  expect(getReferences).toHaveBeenCalledExactlyOnceWith(call);
  expect(screen.getByText("Loading related content")).toBeTruthy();
  expect(screen.queryByText("Source text · permission-checked-raw")).toBeNull();
  await act(async () => pending.resolve(resolved(call, "permission-checked")));
  expect(screen.getByText("Source text · permission-checked-raw")).toBeTruthy();
  expect(screen.getByText("Memory · permission-checked-memory")).toBeTruthy();
  expect(screen.getByText("source-1")).toBeTruthy();
  expect(screen.queryByRole("link")).toBeNull();
  expect(screen.getByRole("button", { name: "View source text" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "View memory" })).toBeTruthy();
  expect(storage).not.toHaveBeenCalled();
  expect(call.references[0]).not.toHaveProperty("title");
});

it("opens currently permitted memory or evidence and hides withdrawn titles", async () => {
  const onOpenMemory = vi.fn();
  const onOpenEvidence = vi.fn();
  getReferences.mockResolvedValueOnce(resolved(event(), "permission-checked"));
  render(
    <CallHistory
      onBack={vi.fn()}
      onOpenMemory={onOpenMemory}
      onOpenEvidence={onOpenEvidence}
    />,
  );
  fireEvent.click(await loaded());
  await screen.findByText("Source text · permission-checked-raw");
  fireEvent.click(screen.getByRole("button", { name: "View source text" }));
  fireEvent.click(screen.getByRole("button", { name: "View memory" }));
  expect(onOpenEvidence).toHaveBeenCalledExactlyOnceWith("raw-1");
  expect(onOpenMemory).toHaveBeenCalledExactlyOnceWith("memory-1");
  getReferences.mockResolvedValueOnce([
    {
      ...event().references[0],
      available: false,
      title: null,
      reason: "changed",
    },
    {
      ...event().references[1],
      available: false,
      title: null,
      reason: "unavailable",
    },
  ]);
  refresh();
  await screen.findByText("Source text · Version changed");
  expect(
    screen.queryByRole("button", { name: /View memory|View source text/ }),
  ).toBeNull();
});

it("drops a late reference response after selecting a different event", async () => {
  const first = event("one");
  const second = event("two");
  getHistory.mockResolvedValue(page([first, second]));
  const old = deferred<ResolvedCallReference[]>();
  getReferences
    .mockReturnValueOnce(old.promise)
    .mockResolvedValueOnce(resolved(second, "current"));
  mount();
  fireEvent.click(await loaded("one"));
  fireEvent.click(screen.getByRole("button", { name: "Back to records" }));
  fireEvent.click(await loaded("two"));
  await screen.findByText("Source text · current-raw");
  await act(async () => old.resolve(resolved(first, "old-private-title")));
  expect(screen.queryByText(/old-private-title/)).toBeNull();
  expect(screen.getByText("Source text · current-raw")).toBeTruthy();
  expect(getReferences).toHaveBeenNthCalledWith(2, second);
});

it("removes titles immediately on refresh and keeps them hidden when rechecking fails", async () => {
  getReferences.mockResolvedValueOnce(resolved(event(), "previous-title"));
  mount();
  fireEvent.click(await loaded());
  await screen.findByText("Source text · previous-title-raw");
  const pending = deferred<ResolvedCallReference[]>();
  getReferences.mockReturnValueOnce(pending.promise);
  refresh();
  expect(screen.queryByText(/previous-title/)).toBeNull();
  await screen.findByText("Loading related content");
  expect(screen.queryByText(/previous-title/)).toBeNull();
  await act(async () => pending.reject(new Error("private error content")));
  await screen.findByText(
    "Related content can't be read right now. Refresh and try again.",
  );
  expect(screen.queryByText(/previous-title|private error content/)).toBeNull();
  expect(screen.getByText("Source text · Unavailable")).toBeTruthy();
  getReferences.mockResolvedValueOnce(resolved(event(), "fresh-title"));
  refresh();
  await screen.findByText("Source text · fresh-title-raw");
});

it("does not restore a pre-refresh title after a newer reference recheck fails", async () => {
  const pending = deferred<ResolvedCallReference[]>();
  getReferences
    .mockReturnValueOnce(pending.promise)
    .mockRejectedValueOnce(new Error("offline"));
  mount();
  fireEvent.click(await loaded());
  refresh();
  await screen.findByText(
    "Related content can't be read right now. Refresh and try again.",
  );
  await act(async () => pending.resolve(resolved(event(), "stale-title")));
  expect(screen.queryByText(/stale-title/)).toBeNull();
  expect(
    screen.getByText(
      "Related content can't be read right now. Refresh and try again.",
    ),
  ).toBeTruthy();
});

it("shows changed or unavailable references without retaining previously permitted titles", async () => {
  getReferences.mockResolvedValueOnce(resolved(event(), "revoked-title"));
  mount();
  fireEvent.click(await loaded());
  await screen.findByText("Memory · revoked-title-memory");
  getReferences.mockResolvedValueOnce([
    {
      ...event().references[0],
      available: false,
      title: null,
      reason: "changed",
    },
    {
      ...event().references[1],
      available: false,
      title: null,
      reason: "unavailable",
    },
  ]);
  refresh();
  await screen.findByText("Source text · Version changed");
  expect(screen.getByText("Memory · Unavailable")).toBeTruthy();
  expect(screen.queryByText(/revoked-title/)).toBeNull();
  expect(screen.queryByRole("link")).toBeNull();
});

it("never renders a title from an unavailable reference even if a malformed mock includes one", async () => {
  getReferences.mockResolvedValue(
    event().references.map((reference) => ({
      ...reference,
      available: false,
      title: "forbidden-title",
      reason: "unavailable",
    })),
  );
  mount();
  fireEvent.click(await loaded());
  await screen.findByText("Source text · Unavailable");
  expect(screen.queryByText(/forbidden-title/)).toBeNull();
});

it("drops late reference content after leaving detail or unmounting", async () => {
  const pending = deferred<ResolvedCallReference[]>();
  getReferences.mockReturnValueOnce(pending.promise);
  const view = mount();
  fireEvent.click(await loaded());
  fireEvent.click(screen.getByRole("button", { name: "Back to records" }));
  await act(async () => pending.resolve(resolved(event(), "late-title")));
  expect(screen.queryByText(/late-title/)).toBeNull();
  const unmounted = deferred<ResolvedCallReference[]>();
  getReferences.mockReturnValueOnce(unmounted.promise);
  fireEvent.click(await loaded());
  view.unmount();
  mount();
  await loaded();
  await act(async () =>
    unmounted.resolve(resolved(event(), "unmounted-title")),
  );
  expect(screen.queryByText(/unmounted-title/)).toBeNull();
});

it("does not resolve references for calls with no references", async () => {
  getHistory.mockResolvedValue(page([event("empty", { references: [] })]));
  mount();
  fireEvent.click(await loaded("empty"));
  expect(screen.getByText("No related content found")).toBeTruthy();
  expect(getReferences).not.toHaveBeenCalled();
});
