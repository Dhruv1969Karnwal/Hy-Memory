// @vitest-environment jsdom

import type {
  AskMemoryResponse,
  GovernedMemory,
  MemoryDetail,
  MemoryGraphNode,
  MemoryGraphResponse,
  MemoryProcessingSummary,
  SourceResponse,
} from "@one-memory/contracts";
import {
  cleanup,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Workbench } from "./Workbench";

const getConversation = vi.fn();

vi.mock("./api", () => ({
  getConversation: (...args: unknown[]) => getConversation(...args),
}));

// The graph draws with WebGL, which jsdom has none of. The test doubles it
// with something that can be looked for, because what matters here is whether
// a graph is on screen at all — the complaint that started this was a page
// with no graph on it.
vi.mock("./MemoryGraph3D", () => ({
  MemoryGraph3D: ({
    graph,
    onSelectNode,
  }: {
    graph: MemoryGraphResponse;
    onSelectNode(node: MemoryGraphNode): void;
  }) => (
    <div data-testid="graph-3d" data-nodes={graph.nodes.length}>
      {graph.nodes.map((node) => (
        // Clickable, so a test can select a node the way a person does and
        // reach the panel that reads out the lines hanging off it.
        <button
          key={node.node_id}
          type="button"
          onClick={() => onSelectNode(node)}
        >
          {node.label}
        </button>
      ))}
    </div>
  ),
}));

function memory(
  id: string,
  overrides: Partial<GovernedMemory> = {},
): GovernedMemory {
  return {
    memory_id: id,
    user_id: "local",
    kind: "chat",
    status: "confirmed",
    content: `Memory ${id}`,
    scopes: ["all-apps"],
    evidence_ids: [`ev-${id}`],
    origin: "extracted",
    tags: [],
    revision: 1,
    created_at: "2026-09-01T00:00:00Z",
    updated_at: "2026-09-01T00:00:00Z",
    deleted_at: null,
    ...overrides,
  };
}

function source(
  sourceId: string,
  overrides: Partial<SourceResponse> = {},
): SourceResponse {
  return {
    source_id: sourceId,
    display_name: sourceId === "cursor" ? "Cursor" : sourceId,
    storage_kinds: ["jsonl"],
    available: true,
    authorized: true,
    detected: true,
    root_label: null,
    candidate_path: `~/.${sourceId}`,
    candidate_kind: "default",
    imported_messages: 40,
    imported_conversations: 12,
    capabilities: {
      historical_import: "implemented",
      incremental_import: "implemented",
      live_sync: "not_implemented",
      recall: "not_implemented",
      installer: "not_implemented",
      verification: "real_replayed",
    },
    sync_completeness: "additions_only",
    ...overrides,
  } as SourceResponse;
}

const processing = {
  pending: 0,
  running: 0,
  fresh_pending: 0,
  backlog_pending: 0,
  recent_completions: [],
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
  seconds_per_completion: null,
} as MemoryProcessingSummary;

const detail: MemoryDetail = {
  memory: memory("memory-1", { content: "Standardize on an axios wrapper" }),
  evidence: [
    {
      evidence_id: "ev-old",
      available: true,
      source_id: "cursor",
      role: "user",
      written_by: "user",
      created_at: "2026-07-01T10:00:00Z",
      verbatim: "What was said in July",
      extraction_state: "extracted",
    },
    {
      evidence_id: "ev-new",
      available: true,
      source_id: "claude-code",
      role: "user",
      written_by: "user",
      created_at: "2026-08-20T10:00:00Z",
      verbatim: "What was said in August",
      extraction_state: "extracted",
    },
  ],
  history: [
    {
      memory_id: "memory-1",
      revision: 1,
      action: "create",
      status: "confirmed",
      content: "Standardize on an axios wrapper",
      scopes: ["all-apps"],
      created_at: "2026-09-01T00:00:00Z",
    },
  ],
  index_sync: null,
} as MemoryDetail;

const EMPTY_GRAPH = {
  total_memories: 0,
  projected_memories: 0,
  hy_graph_available: false,
  nodes: [],
  edges: [],
} as MemoryGraphResponse;

function mount(props: Partial<Parameters<typeof Workbench>[0]> = {}) {
  const handlers = {
    onDraft: vi.fn(),
    onScopes: vi.fn(),
    onOpenMemory: vi.fn(),
    onCloseDetail: vi.fn(),
    onSave: vi.fn(),
    onAction: vi.fn(),
    onSyncIndex: vi.fn(),
    onAsk: vi.fn(),
    onBuildRelations: vi.fn(),
    onConnect: vi.fn(),
    onConfigureModel: vi.fn(),
    onSaveAnswer: vi.fn(),
  };
  const view = render(
    <Workbench
      memories={[
        memory("memory-1", { content: "Standardize on an axios wrapper" }),
      ]}
      graph={{
        total_memories: 1,
        projected_memories: 1,
        hy_graph_available: false,
        nodes: [
          {
            node_id: "memory:memory-1",
            node_type: "memory",
            label: "Standardize on an axios wrapper",
            memory_id: "memory-1",
            source_id: null,
            kind: "chat",
            status: "confirmed",
            confidence: null,
          },
        ],
        edges: [],
      }}
      sources={[source("cursor"), source("claude-code")]}
      processing={processing}
      loaded
      detail={null}
      draft=""
      scopes=""
      busy={false}
      answer={null}
      streaming={null}
      requestIds={{ current: new Map() }}
      {...handlers}
      {...props}
    />,
  );
  return { ...handlers, view };
}

it("a selected graph node links the library and the detail together (graph -> list -> source text)", async () => {
  // The goal requires the graph, the memory list and the original text to be
  // linked in both directions. The list→graph direction is covered elsewhere
  // (the graph receives `selectedNodeId`); this covers graph→list→detail, which
  // had no test even though the wiring exists in `Workbench.tsx`.
  const user = userEvent.setup();
  const { onOpenMemory } = mount();
  // The mock renders one clickable button per node, which is how a person
  // selects a node without WebGL. The same label also appears in the memory
  // list, so scope the query to the graph.
  const graph = screen.getByTestId("graph-3d");
  await user.click(
    within(graph).getByRole("button", {
      name: "Standardize on an axios wrapper",
    }),
  );
  expect(onOpenMemory).toHaveBeenCalledWith("memory-1");
});

it("expanding a quote and then selecting a graph node collapses the expansion instead of hanging old source text on the new node", async () => {
  // Graph selection must reset the evidence panel's *expanded* state. Leaving
  // it open would show one memory's raw conversation under a different node's
  // label, reading as if that conversation belonged to the newly selected
  // memory. The verbatim excerpt is always listed, so the assertion is on the
  // expanded body (`data-open`), which is what the reset actually controls.
  const user = userEvent.setup();
  mount({
    detail: {
      memory: memory("memory-1", {
        content: "Standardize on an axios wrapper",
      }),
      evidence: [
        {
          evidence_id: "ev-old",
          available: true,
          source_id: "cursor",
          role: "user",
          written_by: "user",
          created_at: "2026-07-01T10:00:00Z",
          verbatim: "Source text of the previous memory",
          extraction_state: "extracted",
        },
      ],
      history: [],
      index_sync: null,
    } as never,
  });
  // Open the evidence card.
  await user.click(
    screen.getByRole("button", { name: /Source text of the previous memory/ }),
  );
  expect(
    document.querySelector('[data-open="true"]'),
    "the evidence card should start expanded",
  ).not.toBeNull();
  // Selecting a graph node must collapse it.
  const graph = screen.getByTestId("graph-3d");
  await user.click(
    within(graph).getByRole("button", {
      name: "Standardize on an axios wrapper",
    }),
  );
  expect(
    document.querySelector('[data-open="true"]'),
    "graph selection must collapse the previously opened evidence",
  ).toBeNull();
});

beforeEach(() => {
  vi.clearAllMocks();
});

afterEach(() => {
  cleanup();
});

describe("all three columns are there with what belongs in each", () => {
  it("the left column groups by topic and clicking one leaves only its memories", async () => {
    // This used to click "Personal profile", the grouping by scope. In a real
    // library a scope is workspace:v1:<sha256>, all hash, so this axis is topics now.
    const user = userEvent.setup();
    mount({
      memories: [
        memory("a", { content: "Memory A", tags: ["Front end", "Used once"] }),
        memory("b", { content: "Memory B", tags: ["Front end"] }),
        memory("c", { content: "Memory C", tags: ["Scheduling"] }),
        memory("d", { content: "Memory D", tags: ["Scheduling"] }),
      ],
    });

    // Default is "All", so all four are present
    expect(screen.getByRole("button", { name: /Memory A/ })).toBeTruthy();
    expect(screen.getByRole("button", { name: /Memory C/ })).toBeTruthy();

    await user.click(screen.getByRole("button", { name: /^Front end/ }));

    expect(screen.getByRole("button", { name: /Memory A/ })).toBeTruthy();
    expect(screen.getByRole("button", { name: /Memory B/ })).toBeTruthy();
    expect(screen.queryByRole("button", { name: /Memory C/ })).toBeNull();
  });

  it("the left column never shows a whole authorization hash", async () => {
    // Positive control: the case above only proves topics are present, not that hashes
    // are absent. What the user saw was nine rows of v1:0027abad... under "All".
    mount({
      memories: [
        memory("a", {
          scopes: ["workspace:v1:0027abad1745330f1b57671635e0e9b3"],
          tags: ["Front end"],
        }),
        memory("b", {
          scopes: ["workspace:v1:daf248ff48ea3e6baf35cc7f1fb46d74"],
          tags: ["Front end"],
        }),
      ],
    });

    const column = screen.getByRole("navigation", { name: "Topics" });
    expect(column.textContent).toContain("Front end");
    expect(column.textContent).not.toContain("v1:");
    expect(column.textContent).not.toContain("0027abad");
  });

  it("the right column is newest first and names the app each quote came from", async () => {
    mount({ detail });

    const cards = screen.getAllByText(/What was said in (July|August)/);
    expect(cards[0].textContent).toContain("August");
    expect(cards[1].textContent).toContain("July");
    expect(screen.getByText("Cursor")).toBeTruthy();
    expect(screen.getByText("claude-code")).toBeTruthy();
  });

  it("clicking a quote expands the full conversation", async () => {
    getConversation.mockResolvedValue({
      conversation_id: "conv-1",
      source_id: "cursor",
      messages: [
        {
          evidence_id: "ev-new",
          role: "user",
          written_by: "user",
          created_at: "2026-08-20T10:00:00Z",
          verbatim: "What was said in August",
        },
        {
          evidence_id: "ev-reply",
          role: "assistant",
          written_by: "assistant",
          created_at: "2026-08-20T10:01:00Z",
          verbatim: "The assistant reply at the time",
        },
      ],
    });
    const user = userEvent.setup();
    mount({ detail });

    await user.click(
      screen.getByRole("button", { name: /What was said in August/ }),
    );
    await user.click(
      screen.getByRole("button", { name: "View full conversation" }),
    );

    expect(
      await screen.findByText("The assistant reply at the time"),
    ).toBeTruthy();
    expect(getConversation).toHaveBeenCalledWith("ev-new");
  });

  it("with no memory selected the right column says how to get content", () => {
    mount();

    expect(screen.getByText(/Click any circle in the graph/)).toBeTruthy();
    expect(
      screen.queryByRole("button", { name: "View full conversation" }),
    ).toBeNull();
  });
});

describe("the graph is never empty", () => {
  it("when the graph query succeeds with 0 nodes a skeleton is drawn from imported sources", () => {
    // This is the real state of a fresh install: extraction has not run and the graph
    // endpoint answers with an empty graph. This was fed graph: null (request failure only),
    // so the skeleton never appeared on the one screen that needed it; a real run found it.
    mount({ memories: [], graph: EMPTY_GRAPH });

    expect(screen.getByTestId("graph-3d").getAttribute("data-nodes")).toBe("2");
    expect(screen.getByText("Cursor · 12 conversations")).toBeTruthy();
  });

  it("a completely failed graph endpoint still gets a skeleton instead of a blank screen", () => {
    mount({ memories: [], graph: null });

    expect(screen.getByTestId("graph-3d").getAttribute("data-nodes")).toBe("2");
  });

  it("with nothing imported it draws no fake circles and says to authorize a source", async () => {
    // The case above is the positive control for this reverse guard: still no memories,
    // only the source was never imported, so the skeleton has to disappear.
    const user = userEvent.setup();
    const { onConnect } = mount({
      memories: [],
      graph: EMPTY_GRAPH,
      sources: [source("cursor", { imported_conversations: 0 })],
    });

    expect(screen.queryByTestId("graph-3d")).toBeNull();
    await user.click(
      screen.getByRole("button", { name: "Authorize a source" }),
    );
    expect(onConnect).toHaveBeenCalled();
  });

  it("an unauthorized source does not count", () => {
    mount({
      memories: [],
      graph: EMPTY_GRAPH,
      sources: [source("cursor", { authorized: false })],
    });

    expect(screen.queryByTestId("graph-3d")).toBeNull();
  });

  it("while still loading it says it is reading, not that you have no memories", () => {
    mount({ memories: [], graph: EMPTY_GRAPH, sources: [], loaded: false });

    expect(screen.getByText("Reading memories...")).toBeTruthy();
    expect(screen.queryByText(/Authorize a source/)).toBeNull();
  });
});

describe("the findings strip", () => {
  it(`clicking "Show in graph" opens the memory the finding rests on`, async () => {
    const user = userEvent.setup();
    const { onOpenMemory } = mount({
      memories: [
        memory("a", {
          content: "Standardize on an axios wrapper",
          evidence_ids: ["ev-1"],
        }),
        memory("b", {
          content: "Do not call fetch bare",
          evidence_ids: ["ev-1"],
        }),
      ],
    });

    expect(
      screen.getByText(/extracted from the same source quote/),
    ).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Show in graph" }));

    expect(onOpenMemory).toHaveBeenCalledWith("a");
  });

  it("with no findings at all the whole strip is absent, not an empty shell", () => {
    mount({ memories: [memory("a")] });

    expect(screen.queryByText("What I noticed for you")).toBeNull();
  });
});

describe("memories can be rewritten and deleted", () => {
  it("once a memory is selected, rewrite and delete are at hand", async () => {
    const user = userEvent.setup();
    const { onAction } = mount({
      detail,
      draft: "Standardize on an axios wrapper",
      scopes: "all-apps",
    });

    expect(screen.getByLabelText("Memory content")).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Save correction" }),
    ).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "Delete" }));
    expect(onAction).toHaveBeenCalledWith("delete");
  });

  it("editing the text voids the prepared request id so new content is not sent with the old id", async () => {
    const user = userEvent.setup();
    const requestIds = { current: new Map([["memory-1:correct:1", "old"]]) };
    mount({
      detail,
      draft: "Standardize on an axios wrapper",
      scopes: "all-apps",
      requestIds,
    });

    await user.type(screen.getByLabelText("Memory content"), " x");

    expect(requestIds.current.size).toBe(0);
  });

  it("a deleted memory no longer shows a delete button", () => {
    mount({
      detail: {
        ...detail,
        memory: { ...detail.memory, status: "deleted" },
      } as MemoryDetail,
      draft: "Standardize on an axios wrapper",
      scopes: "all-apps",
    });

    expect(screen.queryByRole("button", { name: "Delete" })).toBeNull();
    expect(screen.queryByLabelText("Memory content")).toBeNull();
  });
});

describe("asking", () => {
  it("the question sent is the sentence the user typed", async () => {
    const user = userEvent.setup();
    const { onAsk } = mount();

    await user.type(
      screen.getByLabelText("Ask about this matter"),
      "What are my requirements for real data",
    );
    await user.click(screen.getByRole("button", { name: "Ask" }));

    await waitFor(() =>
      expect(onAsk).toHaveBeenCalledWith(
        "What are my requirements for real data",
      ),
    );
  });

  it(`a memory is usable as soon as it is produced, with no "not confirmed yet" chip`, () => {
    mount({ memories: [memory("a")] });

    expect(
      screen.queryByRole("button", { name: "Which ones are unconfirmed" }),
    ).toBeNull();
  });
});

const ANSWER = {
  answer:
    "Taken together, you always go through an axios wrapper on the front end.",
  memories: [
    memory("m1", { evidence_ids: ["ev-a", "ev-b"] }),
    memory("m2", { evidence_ids: ["ev-b", "ev-c"] }),
  ],
  mode: "generated",
} as AskMemoryResponse;

describe("an answer can be kept", () => {
  it("saving carries the provenance of the memories it rests on", async () => {
    const user = userEvent.setup();
    const { onSaveAnswer } = mount({ answer: ANSWER });

    await user.click(screen.getByRole("button", { name: "Save as memory" }));

    await waitFor(() =>
      expect(onSaveAnswer).toHaveBeenCalledWith({
        content:
          "Taken together, you always go through an axios wrapper on the front end.",
        evidence_ids: ["ev-a", "ev-b", "ev-c"],
      }),
    );
  });

  it("an answer that matched no memory cannot be saved", () => {
    mount({
      answer: {
        answer: "No related memories found.",
        memories: [],
        mode: "empty",
      } as AskMemoryResponse,
    });

    expect(screen.queryByRole("button", { name: "Save as memory" })).toBeNull();
  });

  it("no saving while the stream is still running; what gets saved has to be complete", () => {
    mount({
      answer: null,
      streaming: { text: "Taken to", memories: [memory("m1")] },
    });

    expect(screen.queryByRole("button", { name: "Save as memory" })).toBeNull();
  });

  it("an answer too long to be a memory is pointed at the matter page instead of forced into the library", () => {
    mount({
      answer: { ...ANSWER, answer: "word".repeat(601) } as AskMemoryResponse,
    });

    expect(screen.queryByRole("button", { name: "Save as memory" })).toBeNull();
    expect(screen.getByText(/too long/)).toBeTruthy();
  });
});

describe("topic tags", () => {
  it("a memory row carries its tags underneath", () => {
    mount({ memories: [memory("a", { tags: ["Scheduling", "Work habits"] })] });

    expect(screen.getByText("Scheduling")).toBeTruthy();
    expect(screen.getByText("Work habits")).toBeTruthy();
  });

  it("a tag used once is marked and one used twice is not", () => {
    mount({
      memories: [
        memory("a", { tags: ["Scheduling", "Used once"] }),
        memory("b", { tags: ["Scheduling"] }),
      ],
    });

    expect(screen.getByText("Used once").getAttribute("data-once")).toBe(
      "true",
    );
    // Both memories carry "Scheduling", so as a tag it already has two nodes, which is
    // exactly why it must not be marked "used once". Count only tags, not the topic row in
    // the left column: "Scheduling" is both now and getAllByText counts it in as well.
    const shared = screen
      .getAllByText("Scheduling")
      .filter((node) => node.className === "wb-tag");
    expect(shared).toHaveLength(2);
    for (const node of shared) {
      expect(node.getAttribute("data-once")).toBeNull();
    }
  });

  it("the header reports two counts instead of a warning", () => {
    mount({
      memories: [
        memory("a", { tags: ["Scheduling", "Used once"] }),
        memory("b", { tags: ["Scheduling"] }),
      ],
    });

    // What it says has to match what is on screen: only a formed topic gets a row and a tag
    // used once does not, so the two counts are stated separately instead of one "2 topic
    // tags", which would read 2 while the eye counts a single row.
    expect(screen.getByText("Topics formed: 1, appeared once: 1")).toBeTruthy();
  });

  it("with no tags at all it takes up no space", () => {
    mount({ memories: [memory("a")] });

    expect(screen.queryByText(/Topics formed/)).toBeNull();
  });
});

describe("the matter summary", () => {
  it("the question carries the name of this matter", async () => {
    const user = userEvent.setup();
    // The fake scope fed in here used to be workspace:v1:one-memory, which happened to be
    // readable, so this test could never catch an unreadable real hash. It selects by topic now.
    const { onAsk } = mount({
      memories: [
        memory("a", { tags: ["one-memory"] }),
        memory("b", { tags: ["one-memory"] }),
      ],
    });
    await user.click(screen.getByRole("button", { name: /^one-memory/ }));

    await user.click(
      screen.getByRole("button", { name: "Read a summary of this matter" }),
    );

    await waitFor(() =>
      expect(onAsk).toHaveBeenCalledWith(expect.stringContaining("one-memory")),
    );
  });

  it("a summary cannot be saved as a memory because it is rebuilt every time by definition", async () => {
    const user = userEvent.setup();
    mount({ answer: ANSWER });

    await user.click(
      screen.getByRole("button", { name: "Read a summary of this matter" }),
    );

    expect(screen.queryByRole("button", { name: "Save as memory" })).toBeNull();
    expect(screen.getByText(/regenerated every time/)).toBeTruthy();
  });

  it("switching back to a normal question makes saving available again", async () => {
    const user = userEvent.setup();
    mount({ answer: ANSWER, memories: [memory("a")] });
    await user.click(
      screen.getByRole("button", { name: "Read a summary of this matter" }),
    );

    await user.click(
      screen.getByRole("button", { name: "What changed here recently" }),
    );

    expect(screen.getByRole("button", { name: "Save as memory" })).toBeTruthy();
  });

  it("when the matter holds no memories there is no summary to read", () => {
    mount({ memories: [] });

    expect(
      screen.queryByRole("button", { name: "Read a summary of this matter" }),
    ).toBeNull();
  });
});

describe("clustering by topic", () => {
  afterEach(cleanup);

  it("a shared-topic edge has its own wording and its own count in the graph", () => {
    // With few memories the only thing in the graph holding several is the source app, so it
    // looks like clustering by app -- exactly the wrong thing for it to express. Before the
    // topic axis existed, tags carried back from HY were only text in the left column.
    mount({
      graph: {
        total_memories: 2,
        projected_memories: 2,
        hy_graph_available: false,
        nodes: [
          { node_id: "topic:axios", node_type: "topic", label: "axios" },
          {
            node_id: "memory:a",
            node_type: "memory",
            label: "Always go through axios",
          },
          { node_id: "memory:b", node_type: "memory", label: "Tests first" },
        ],
        edges: [
          {
            edge_id: "e1",
            source_node_id: "topic:axios",
            target_node_id: "memory:a",
            relation: "shared_tag",
            origin: "shared_tag",
          },
          {
            edge_id: "e2",
            source_node_id: "topic:axios",
            target_node_id: "memory:b",
            relation: "shared_tag",
            origin: "shared_tag",
          },
        ],
      } as MemoryGraphResponse,
    });

    // The receipt counts per origin bucket. If the extra bucket is not recognized,
    // counts[edge.origin] += 1 yields NaN and the whole receipt line breaks silently.
    const receipt = screen.getByText(/Quote source/).textContent ?? "";
    expect(receipt).not.toContain("NaN");
    expect(receipt).toContain("Same topic 2 lines");
  });

  it("with no shared topic that bucket stays out of the receipt", () => {
    // Positive control: without it the case above could be fooled by always appending the
    // sentence. The graph fed here has edges but no topic edge; an empty graph takes the
    // skeleton branch with no receipt line at all, so absence would pass vacuously.
    mount({
      graph: {
        total_memories: 1,
        projected_memories: 1,
        hy_graph_available: false,
        nodes: [
          { node_id: "source:cursor", node_type: "source", label: "Cursor" },
          {
            node_id: "memory:a",
            node_type: "memory",
            label: "Always go through axios",
          },
        ],
        edges: [
          {
            edge_id: "e1",
            source_node_id: "source:cursor",
            target_node_id: "memory:a",
            relation: "evidence_source",
            origin: "provenance",
          },
        ],
      } as MemoryGraphResponse,
    });

    const receipt = screen.getByText(/Quote source/).textContent ?? "";
    expect(receipt).not.toContain("Same topic");
  });

  it("opening a topic names the memories hanging off it and where the line came from", async () => {
    // edgeOriginLabel is only reachable through the detail panel. It is an if chain that
    // ends in Quote source, so a missing case never throws; it just labels topic edges wrong.
    const user = userEvent.setup();
    mount({
      graph: {
        total_memories: 2,
        projected_memories: 2,
        hy_graph_available: false,
        nodes: [
          { node_id: "topic:axios", node_type: "topic", label: "axios" },
          {
            node_id: "memory:a",
            node_type: "memory",
            label: "Always go through axios",
          },
          { node_id: "memory:b", node_type: "memory", label: "Tests first" },
        ],
        edges: [
          {
            edge_id: "e1",
            source_node_id: "topic:axios",
            target_node_id: "memory:a",
            relation: "shared_tag",
            origin: "shared_tag",
          },
          {
            edge_id: "e2",
            source_node_id: "topic:axios",
            target_node_id: "memory:b",
            relation: "shared_tag",
            origin: "shared_tag",
          },
        ],
      } as MemoryGraphResponse,
    });

    await user.click(screen.getByRole("button", { name: "axios" }));

    const panel = screen.getByLabelText("This node");
    expect(panel.textContent).toContain("Topic");
    expect(panel.textContent).toContain("Always go through axios");
    expect(panel.textContent).toContain("Same topic");
    expect(panel.textContent).not.toContain("Quote source");
  });
});
