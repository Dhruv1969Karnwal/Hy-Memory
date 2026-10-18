// @vitest-environment jsdom

import type {
  ConversationSearchResponse,
  ConversationTranscript,
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
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ConversationSearch } from "./ConversationSearch";

const {
  searchConversations,
  getConversation,
  getConversationBrowser,
  getConversationMemories,
} = vi.hoisted(() => ({
  searchConversations: vi.fn(),
  getConversation: vi.fn(),
  getConversationBrowser: vi.fn(),
  getConversationMemories: vi.fn(),
}));

vi.mock("./api", () => ({
  apiErrorCode: (reason: unknown) =>
    reason instanceof Error ? reason.message : null,
  searchConversations,
  getConversation,
  getConversationBrowser,
  getConversationMemories,
}));

const found: ConversationSearchResponse = {
  hits: [
    {
      evidence_id: "evidence-1",
      conversation_id: "conversation-1",
      source_id: "cursor",
      workspace: "/Users/k/one-memory",
      project: "one-memory",
      created_at: "2026-09-03T01:17:00Z",
      excerpt:
        "Last night I rebuilt the package and the extraction finally ran",
    },
  ],
  empty_reason: null,
  imported_messages: 6740,
};

const transcript: ConversationTranscript = {
  conversation_id: "conversation-1",
  source_id: "cursor",
  messages: [
    {
      evidence_id: "evidence-1",
      role: "user",
      written_by: "person",
      created_at: "2026-09-03T01:17:00Z",
      verbatim:
        "Last night I rebuilt the package and the extraction finally ran",
    },
    {
      evidence_id: "evidence-2",
      role: "assistant",
      written_by: "agent",
      created_at: "2026-09-03T01:18:00Z",
      verbatim: "Then ship the package first",
    },
  ],
};

const scrollIntoView = vi.fn();

beforeEach(() => {
  searchConversations.mockReset();
  getConversation.mockReset();
  getConversationBrowser.mockReset().mockResolvedValue({
    items: [],
    total: 0,
    offset: 0,
    limit: 50,
  });
  getConversationMemories.mockReset().mockResolvedValue({ items: [] });
  searchConversations.mockResolvedValue(found);
  getConversation.mockResolvedValue(transcript);
  scrollIntoView.mockReset();
  // jsdom has no layout, so the component's scroll call would throw here.
  Element.prototype.scrollIntoView = scrollIntoView;
});

afterEach(cleanup);

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

const secondHit = {
  ...found.hits[0],
  evidence_id: "evidence-b",
  conversation_id: "conversation-b",
  source_id: "workbuddy",
  excerpt: "Second hit excerpt",
};
const secondTranscript: ConversationTranscript = {
  conversation_id: "conversation-b",
  source_id: "workbuddy",
  messages: [
    {
      evidence_id: "evidence-b",
      role: "user",
      written_by: "person",
      created_at: null,
      verbatim: "B's full original text",
    },
  ],
};

function hitArticle(excerpt: string) {
  const article = screen.getByText(excerpt).closest("article");
  if (!article) throw new Error("missing hit article");
  return article;
}

async function openA(user: ReturnType<typeof userEvent.setup>) {
  await user.click(
    within(hitArticle(found.hits[0].excerpt)).getByRole("button", {
      name: "View source text",
    }),
  );
}

async function openB(user: ReturnType<typeof userEvent.setup>) {
  await user.click(
    within(hitArticle(secondHit.excerpt)).getByRole("button", {
      name: "View source text",
    }),
  );
}

async function search(
  text = "extraction finally",
  props: Parameters<typeof ConversationSearch>[0] = {},
) {
  const user = userEvent.setup();
  render(<ConversationSearch {...props} />);
  await user.type(screen.getByLabelText("Search conversation history"), text);
  await user.click(screen.getByRole("button", { name: "Search" }));
  return user;
}

it("lists recent conversations before the user searches", async () => {
  getConversationBrowser.mockResolvedValue({
    items: [
      {
        evidence_id: "recent-1",
        conversation_id: "conversation-recent",
        source_id: "workbuddy",
        workspace: null,
        project: "One Memory",
        title: "A recent real conversation",
        latest_at: "2026-10-04T08:00:00Z",
        turn_count: 3,
        memory_count: 2,
      },
    ],
    total: 1,
    offset: 0,
    limit: 50,
  });

  render(<ConversationSearch />);

  expect(await screen.findByText("A recent real conversation")).toBeTruthy();
  expect(searchConversations).not.toHaveBeenCalled();
});

it("shows memories associated with the selected conversation", async () => {
  const onOpenMemory = vi.fn();
  getConversationMemories.mockResolvedValue({
    items: [
      {
        memory_id: "memory-1",
        user_id: "local",
        kind: "chat",
        status: "confirmed",
        content: "Signature and notarization must be confirmed before release.",
        scopes: ["all-apps"],
        origin: "hy",
        tags: [],
        revision: 1,
        created_at: "2026-10-04T08:00:00Z",
        updated_at: "2026-10-04T08:00:00Z",
        deleted_at: null,
        undo_cursor_revision: null,
      },
    ],
  });
  const user = await search("extraction finally", { onOpenMemory });
  await openA(user);

  await user.click(
    await screen.findByRole("button", {
      name: /Signature and notarization must be confirmed before release/,
    }),
  );
  expect(onOpenMemory).toHaveBeenCalledWith("memory-1");
});

it("filters the conversation list by source", async () => {
  getConversationBrowser.mockResolvedValueOnce({
    items: [
      {
        evidence_id: "cursor-1",
        conversation_id: "c1",
        source_id: "cursor",
        workspace: null,
        project: null,
        title: "Cursor session",
        latest_at: "2026-10-04T08:00:00Z",
        turn_count: 2,
        memory_count: 0,
      },
      {
        evidence_id: "workbuddy-1",
        conversation_id: "w1",
        source_id: "workbuddy",
        workspace: null,
        project: null,
        title: "WorkBuddy session",
        latest_at: "2026-10-04T08:01:00Z",
        turn_count: 3,
        memory_count: 1,
      },
    ],
    total: 2,
    offset: 0,
    limit: 50,
  });
  getConversationBrowser.mockResolvedValueOnce({
    items: [],
    total: 0,
    offset: 0,
    limit: 50,
  });
  const user = userEvent.setup();
  render(<ConversationSearch />);
  await screen.findByText("Cursor session");

  await user.selectOptions(screen.getByLabelText("App"), "cursor");

  await waitFor(() =>
    expect(getConversationBrowser).toHaveBeenLastCalledWith({
      sourceId: "cursor",
    }),
  );
});

it("loads older complete turns into the middle column", async () => {
  getConversation.mockResolvedValueOnce({
    ...transcript,
    total_turns: 25,
    offset: 0,
    limit: 20,
    has_more: true,
  });
  getConversation.mockResolvedValueOnce({
    ...transcript,
    messages: [
      {
        ...transcript.messages[0],
        evidence_id: "older-user",
        verbatim: "An earlier turn",
        turn_index: 1,
      },
    ],
    total_turns: 25,
    offset: 20,
    limit: 20,
    has_more: false,
  });
  const user = await search();
  await openA(user);
  await user.click(
    await screen.findByRole("button", { name: "Load older conversations" }),
  );

  expect(await screen.findByText("An earlier turn")).toBeTruthy();
  expect(getConversation).toHaveBeenLastCalledWith("evidence-1", {
    offset: 20,
    limit: 20,
  });
});

it("opens a search hit on the page containing its matching turn", async () => {
  searchConversations.mockResolvedValue({
    ...found,
    hits: [{ ...found.hits[0], turn_offset: 40 }],
  });
  const user = await search();
  await openA(user);

  await waitFor(() =>
    expect(getConversation).toHaveBeenCalledWith("evidence-1", {
      offset: 40,
      limit: 20,
    }),
  );
});

it("Presents results as original conversations, never posing as memories", async () => {
  await search();

  await waitFor(() => {
    expect(
      screen.getByText(
        "Last night I rebuilt the package and the extraction finally ran",
      ),
    ).toBeTruthy();
  });
  // The wording carries the whole distinction: these are the user's own
  // sentences, not anything the product concluded about them.
  expect(
    screen.getByRole("region", {
      name: "Imported conversation search results",
    }),
  ).toBeTruthy();
  expect(screen.getByText(/original conversation/)).toBeTruthy();
  expect(screen.queryByText(/memories have been generated/)).toBeNull();
});

it("When search works but nothing matches, it says no matching conversations were found", async () => {
  searchConversations.mockResolvedValue({
    hits: [],
    empty_reason: "no_matches",
    imported_messages: 6740,
  });

  await search("quantum entanglement");

  await waitFor(() => {
    expect(
      screen.getByText("No matching conversations were found"),
    ).toBeTruthy();
  });
});

it("With no history imported yet, it says to import history first", async () => {
  searchConversations.mockResolvedValue({
    hits: [],
    empty_reason: "nothing_imported",
    imported_messages: 0,
  });

  await search();

  await waitFor(() => {
    expect(
      screen.getByText("Import history first, then you can search"),
    ).toBeTruthy();
  });
  expect(screen.queryByText("No matching conversations were found")).toBeNull();
});

it("When the search itself is broken it errors clearly, not faking no results", async () => {
  // The shape last night's packaging bug took: a core that could not search
  // at all was indistinguishable, on screen, from a user who had never said
  // the thing they were looking for.
  searchConversations.mockRejectedValue(
    new Error("conversation_search_unavailable"),
  );

  await search();

  await waitFor(() => {
    expect(screen.getByRole("alert")).toBeTruthy();
  });
  expect(screen.getByRole("alert").textContent).toContain(
    "The search index is unavailable",
  );
  expect(screen.queryByText("No matching conversations were found")).toBeNull();
});

it("Opening a result expands its source text in place", async () => {
  const user = await search();

  await waitFor(() => {
    expect(
      screen.getByRole("button", { name: /View source text/ }),
    ).toBeTruthy();
  });
  await user.click(screen.getByRole("button", { name: /View source text/ }));

  await waitFor(() => {
    expect(screen.getByText("Then ship the package first")).toBeTruthy();
  });
  expect(getConversation).toHaveBeenCalledWith("evidence-1");
});

it("explains model configuration before generating memory", () => {
  render(<ConversationSearch />);
  expect(
    screen.getByText(
      /Memories start being generated once a model is configured/,
    ),
  ).toBeTruthy();
  expect(screen.queryByText(/once the generation model is set up/)).toBeNull();
});

it("shows every original author and timestamp without inferring person from role=user", async () => {
  const verbatim = "  Keep spaces\nSecond line <source text>\n\nLast line  ";
  getConversation.mockResolvedValue({
    ...transcript,
    messages: [
      { ...transcript.messages[0], verbatim },
      transcript.messages[1],
      {
        evidence_id: "unknown",
        role: "user",
        written_by: "unknown",
        created_at: null,
        verbatim: "User role but author unknown",
      },
      {
        evidence_id: "machine",
        role: "user",
        written_by: "agent",
        created_at: "not-a-time",
        verbatim: "A machine reminder with the user role",
      },
      {
        evidence_id: "tool",
        role: "tool",
        written_by: "tool",
        created_at: null,
        verbatim: "Raw tool record",
      },
    ],
  });
  const user = await search();
  await openA(user);
  const region = await screen.findByRole("region", {
    name: "Complete original conversation",
  });
  const items = within(region).getAllByRole("listitem");
  expect(items).toHaveLength(1);
  expect(items[0].querySelector("p")?.textContent).toBe(verbatim);
  expect(within(items[0]).getByText("User's own words")).toBeTruthy();
  expect(items[0].querySelector("time")?.dateTime).toBe(
    transcript.messages[0].created_at,
  );
  expect(within(items[0]).getAllByText("Assistant's own words")).toHaveLength(
    2,
  );
  expect(within(items[0]).getByText("Author unknown")).toBeTruthy();
  expect(within(items[0]).getByText("Tool record")).toBeTruthy();
  expect(within(region).getAllByText("Time unknown")).toHaveLength(2);
  expect(region.textContent).not.toContain("not-a-time");
});

it.each([null, "invalid-time"])(
  "shows unknown original hit time for %s",
  async (created_at) => {
    searchConversations.mockResolvedValue({
      ...found,
      hits: [{ ...found.hits[0], created_at }],
    });
    await search();
    expect(
      within(hitArticle(found.hits[0].excerpt)).getByText("Time unknown"),
    ).toBeTruthy();
  },
);

it("A resolving after B cannot display A's original text under B", async () => {
  const a = deferred<ConversationTranscript>(),
    b = deferred<ConversationTranscript>();
  searchConversations.mockResolvedValue({
    ...found,
    hits: [found.hits[0], secondHit],
  });
  getConversation.mockImplementation((id) =>
    id === "evidence-1" ? a.promise : b.promise,
  );
  const user = await search();
  await openA(user);
  await openB(user);
  await act(async () => b.resolve(secondTranscript));
  expect(screen.getByText("B's full original text")).toBeTruthy();
  await act(async () => a.resolve(transcript));
  expect(screen.getByText("B's full original text")).toBeTruthy();
  expect(screen.queryByText("Then ship the package first")).toBeNull();
});

it("an obsolete A error never replaces B's success", async () => {
  const a = deferred<ConversationTranscript>();
  searchConversations.mockResolvedValue({
    ...found,
    hits: [found.hits[0], secondHit],
  });
  getConversation.mockImplementation((id) =>
    id === "evidence-1" ? a.promise : Promise.resolve(secondTranscript),
  );
  const user = await search();
  await openA(user);
  await openB(user);
  await screen.findByText("B's full original text");
  await act(async () => a.reject(new Error("source_not_authorized")));
  expect(screen.queryByRole("alert")).toBeNull();
  expect(screen.getByText("B's full original text")).toBeTruthy();
});

it("collapse and reopen invalidates a late response for the same hit", async () => {
  const first = deferred<ConversationTranscript>(),
    second = deferred<ConversationTranscript>();
  getConversation
    .mockReturnValueOnce(first.promise)
    .mockReturnValueOnce(second.promise);
  const user = await search();
  await openA(user);
  await user.click(
    screen.getByRole("button", { name: "Collapse source text" }),
  );
  await openA(user);
  await act(async () => first.resolve(transcript));
  expect(screen.queryByText("Then ship the package first")).toBeNull();
  expect(screen.getByText("Reading the source text...")).toBeTruthy();
  await act(async () =>
    second.resolve({
      ...transcript,
      messages: [
        { ...transcript.messages[0], verbatim: "Newly read source text" },
      ],
    }),
  );
  expect(screen.getByText("Newly read source text")).toBeTruthy();
});

it("collapsing discards a late rejected read without leaving an error or loading state", async () => {
  const pending = deferred<ConversationTranscript>();
  getConversation.mockReturnValue(pending.promise);
  const user = await search();
  await openA(user);
  await user.click(
    screen.getByRole("button", { name: "Collapse source text" }),
  );
  await act(async () => pending.reject(new Error("offline")));
  expect(screen.queryByRole("alert")).toBeNull();
  expect(screen.queryByText("Reading the source text...")).toBeNull();
  expect(
    screen
      .getByRole("button", { name: "View source text" })
      .getAttribute("aria-expanded"),
  ).toBe("false");
});

it("new search removes old hits immediately and invalidates pending original-text reads", async () => {
  const reading = deferred<ConversationTranscript>(),
    searching = deferred<ConversationSearchResponse>();
  getConversation.mockReturnValue(reading.promise);
  const user = await search();
  await openA(user);
  searchConversations.mockReturnValueOnce(searching.promise);
  await user.clear(screen.getByLabelText("Search conversation history"));
  await user.type(
    screen.getByLabelText("Search conversation history"),
    "second attempt",
  );
  await user.click(screen.getByRole("button", { name: "Search" }));
  expect(screen.queryByText(found.hits[0].excerpt)).toBeNull();
  await act(async () => reading.resolve(transcript));
  await act(async () => searching.resolve({ ...found, hits: [secondHit] }));
  expect(screen.queryByText("Then ship the package first")).toBeNull();
  expect(screen.getByText(secondHit.excerpt)).toBeTruthy();
});

it("a later search wins over an older success or failure while duplicate submits stay single-flight", async () => {
  const a = deferred<ConversationSearchResponse>(),
    b = deferred<ConversationSearchResponse>();
  searchConversations
    .mockReturnValueOnce(a.promise)
    .mockReturnValueOnce(b.promise);
  const view = render(<ConversationSearch />);
  const input = screen.getByLabelText("Search conversation history");
  const form = view.container.querySelector("form");
  if (!form) throw new Error("missing search form");
  fireEvent.change(input, { target: { value: "first" } });
  act(() => {
    fireEvent.submit(form);
    fireEvent.submit(form);
  });
  expect(searchConversations).toHaveBeenCalledTimes(1);
  fireEvent.change(input, { target: { value: "second" } });
  fireEvent.submit(form);
  await act(async () => b.resolve({ ...found, hits: [secondHit] }));
  await act(async () => a.reject(new Error("conversation_search_unavailable")));
  expect(screen.getByText(secondHit.excerpt)).toBeTruthy();
  expect(screen.queryByRole("alert")).toBeNull();
  expect(screen.queryByRole("button", { name: "Searching..." })).toBeNull();
});

it("read errors stop loading and expose an inline retry for the same hit", async () => {
  getConversation
    .mockRejectedValueOnce(new Error("offline"))
    .mockResolvedValueOnce(transcript);
  const user = await search();
  await openA(user);
  expect((await screen.findByRole("alert")).textContent).toContain(
    "The source text read did not complete",
  );
  expect(screen.queryByText("Reading the source text...")).toBeNull();
  expect(
    screen.getByRole("button", { name: "Retry reading the source text" }),
  ).toBeTruthy();
  await user.click(
    screen.getByRole("button", { name: "Retry reading the source text" }),
  );
  expect(await screen.findByText("Then ship the package first")).toBeTruthy();
  expect(getConversation.mock.calls).toEqual([["evidence-1"], ["evidence-1"]]);
});

it("a double retry submits one read and new failures remain retryable", async () => {
  const pending = deferred<ConversationTranscript>();
  getConversation
    .mockRejectedValueOnce(null)
    .mockReturnValueOnce(pending.promise);
  const user = await search();
  await openA(user);
  const retry = await screen.findByRole("button", {
    name: "Retry reading the source text",
  });
  act(() => {
    fireEvent.click(retry);
    fireEvent.click(retry);
  });
  expect(getConversation).toHaveBeenCalledTimes(2);
  await act(async () => pending.reject(new Error("again")));
  expect(
    screen.getByRole("button", { name: "Retry reading the source text" }),
  ).toBeTruthy();
  expect(screen.queryByText("Reading the source text...")).toBeNull();
});

it.each([
  "source_not_authorized",
  "conversation_not_found",
  "invalid_local_token",
])(
  "%s clears cached excerpts and original text, then allows a fresh search",
  async (code) => {
    searchConversations.mockResolvedValue({
      ...found,
      hits: [found.hits[0], { ...secondHit, source_id: "cursor" }],
    });
    const user = await search();
    await openA(user);
    await screen.findByText("Then ship the package first");
    getConversation.mockRejectedValueOnce(new Error(code));
    await openB(user);
    await screen.findByRole("alert");
    expect(screen.queryByText(found.hits[0].excerpt)).toBeNull();
    expect(screen.queryByText(secondHit.excerpt)).toBeNull();
    expect(screen.queryByText("Then ship the package first")).toBeNull();
    expect(screen.queryByText("Reading the source text...")).toBeNull();
    await user.click(screen.getByRole("button", { name: "Search again" }));
    expect(searchConversations).toHaveBeenCalledTimes(2);
  },
);

it("search permission rejection clears all prior original contents", async () => {
  const user = await search();
  await openA(user);
  await screen.findByText("Then ship the package first");
  searchConversations.mockRejectedValueOnce(new Error("source_not_authorized"));
  await user.click(screen.getByRole("button", { name: "Search" }));
  await screen.findByRole("alert");
  expect(screen.queryByText(found.hits[0].excerpt)).toBeNull();
  expect(screen.queryByText("Then ship the package first")).toBeNull();
});

it.each([
  { ...transcript, source_id: "other" },
  { ...transcript, conversation_id: "other" },
  {
    ...transcript,
    messages: [{ ...transcript.messages[1], evidence_id: "not-requested" }],
  },
])(
  "rejects a transcript with a mismatched response identity",
  async (response) => {
    getConversation.mockResolvedValue(response);
    const user = await search();
    await openA(user);
    expect((await screen.findByRole("alert")).textContent).toContain(
      "The source text read did not complete",
    );
    expect(screen.queryByText("Then ship the package first")).toBeNull();
    expect(screen.queryByText("Reading the source text...")).toBeNull();
  },
);

it("Opening a result scrolls to the matched message and highlights the matched words", async () => {
  const user = await search();
  await openA(user);

  const marked = await screen.findByText("extraction finally", {
    selector: "mark",
  });
  const hit = marked.closest("section[data-match]");
  expect(hit?.getAttribute("data-match")).toBe("true");
  expect(hit?.textContent).toContain("Last night I rebuilt the package");
  expect(scrollIntoView).toHaveBeenCalledTimes(1);
  expect(scrollIntoView.mock.instances[0]).toBe(hit);
  // The rest of the transcript is shown as plain text, not as a hit.
  expect(
    screen.queryByText("Then ship the package first", { selector: "mark" }),
  ).toBeNull();
});

it("unmount invalidates both search and original requests without another API call", async () => {
  const reading = deferred<ConversationTranscript>();
  getConversation.mockReturnValue(reading.promise);
  const user = await search();
  await openA(user);
  cleanup();
  await act(async () => reading.reject(new Error("source_not_authorized")));
  expect(document.body.textContent).toBe("");
  const searching = deferred<ConversationSearchResponse>();
  searchConversations.mockReturnValueOnce(searching.promise);
  const view = render(<ConversationSearch />);
  fireEvent.change(screen.getByLabelText("Search conversation history"), {
    target: { value: "late" },
  });
  const form = view.container.querySelector("form");
  if (!form) throw new Error("missing search form");
  fireEvent.submit(form);
  view.unmount();
  await act(async () => searching.resolve(found));
  expect(screen.queryByText(found.hits[0].excerpt)).toBeNull();
  expect(getConversation).toHaveBeenCalledTimes(1);
});

describe("BUG-06: a turn that is still waiting for its answer", () => {
  it("says so on the turn, and that it is left out of extraction", async () => {
    getConversation.mockResolvedValueOnce({
      ...transcript,
      trailing_turn_unanswered: true,
    });
    const user = await search();
    await openA(user);
    expect(
      await screen.findByText(
        "This turn has not received an answer yet, so it is left out of memory processing.",
      ),
    ).toBeTruthy();
  });

  it("says nothing when the turn got its answer", async () => {
    const user = await search();
    await openA(user);
    await screen.findByText("Then ship the package first");
    expect(
      screen.queryByText(
        "This turn has not received an answer yet, so it is left out of memory processing.",
      ),
    ).toBeNull();
  });
});
