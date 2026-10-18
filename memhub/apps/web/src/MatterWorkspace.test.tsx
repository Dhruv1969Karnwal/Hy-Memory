// @vitest-environment jsdom
import type {
  Matter,
  MatterContext,
  MemoryDetail,
} from "@one-memory/contracts";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ComponentProps } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { MatterWorkspace } from "./MatterWorkspace";
import type { MemoryGraph3D } from "./MemoryGraph3D";

const api = vi.hoisted(() => ({
  getMatters: vi.fn(),
  getMatterRuns: vi.fn(),
  patchMatter: vi.fn(),
  getMatterEvidence: vi.fn(),
  getMatterContext: vi.fn(),
  getMatterHistory: vi.fn(),
  addMatterNote: vi.fn(),
  discoverMatters: vi.fn(),
  refreshMatter: vi.fn(),
  getMemories: vi.fn(),
  getMemoryGraph: vi.fn(),
  getMemory: vi.fn(),
}));
vi.mock("./api", () => api);
vi.mock("./MemoryGraph3D", () => ({
  MemoryGraph3D: ({
    graph,
    selectedNodeId,
    onSelectNode,
  }: ComponentProps<typeof MemoryGraph3D>) => (
    <div data-testid="matter-map" data-selected={selectedNodeId}>
      {graph.nodes.map((node) => (
        <button
          type="button"
          key={node.node_id}
          onClick={() => onSelectNode(node)}
        >
          {`Map node ${node.label}`}
        </button>
      ))}
    </div>
  ),
}));

const matter: Matter = {
  source_engine: "legacy_direct",
  content_review: "needs_review",
  matter_id: "m1",
  title: "Atlas product development",
  purpose: "Ship a usable product",
  boundary: "Product design and development",
  anchors: ["e1"],
  tags: ["Product"],
  uncertainties: ["Windows is not accepted yet"],
  version: 1,
  scope_confirmed: false,
  agent_access: false,
  availability: "ready",
  model: "test-model",
  updated_at: "2026-09-11T00:00:00Z",
  source_count: 1,
  evidence_count: 1,
  claims: [
    {
      statement: "The launch target is September 23",
      memory_refs: [],
      state: "current",
      change_note: "",
      kind: "decision",
      scope: "Open-source release",
      basis: "human_statement",
      verification: "model_reviewed",
      citations: [
        {
          evidence_id: "e1",
          quote: "plan to open source on September 23",
          revision: 1,
          source_id: "cursor",
          written_by: "person",
          created_at: null,
        },
      ],
    },
  ],
};

beforeEach(() => {
  vi.resetAllMocks();
  api.getMatters.mockResolvedValue([matter]);
  api.getMatterRuns.mockResolvedValue([]);
  api.getMemories.mockResolvedValue([]);
  api.getMemoryGraph.mockResolvedValue(null);
  api.getMatterHistory.mockResolvedValue([matter]);
  api.getMatterEvidence.mockResolvedValue({
    evidence_id: "e1",
    source_id: "cursor",
    written_by: "person",
    role: "user",
    created_at: null,
    observed_at: "2026-09-11T00:00:00Z",
    text: "We plan to open source on September 23 and finish acceptance.",
  });
  api.patchMatter.mockResolvedValue({
    ...matter,
    version: 2,
    scope_confirmed: true,
  });
});
afterEach(cleanup);

it("separates write verification from provider failures without automatic retries", async () => {
  api.getMatters.mockResolvedValue([]);
  api.getMatterRuns.mockResolvedValue([
    {
      run_id: "failed-run",
      status: "failed",
      phase: "Organization incomplete",
      error: "hy_memory_write_verification_failed",
      observations: [],
      selected_records: 3,
      hy_input_records: 0,
      omitted_records: 0,
      coverage_complete: true,
    },
  ]);
  render(<MatterWorkspace onOpenSources={() => {}} />);
  expect(
    await screen.findByText(
      /The memory content did not pass source and author verification; the current batch is incomplete, and the original text is kept./,
    ),
  ).toBeTruthy();
  expect(api.discoverMatters).not.toHaveBeenCalled();
});

it("does not turn a failed matters read into zero matters", async () => {
  api.getMatters.mockRejectedValue(new Error("offline"));
  api.getMatterRuns.mockRejectedValue(new Error("offline"));
  render(<MatterWorkspace onOpenSources={() => {}} />);
  const list = await screen.findByRole("complementary", {
    name: "Matter list",
  });
  expect(list.textContent).toContain("Pending verification");
  expect(list.textContent).not.toContain("0");
  expect(
    screen.queryByRole("heading", { name: "No matters organized yet" }),
  ).toBeNull();
  expect((await screen.findByRole("alert")).textContent).toContain(
    "Could not read the matter files",
  );
});

it("does not turn a failed matters read plus a query into no matches", async () => {
  const user = userEvent.setup();
  api.getMatters.mockRejectedValue(new Error("offline"));
  api.getMatterRuns.mockRejectedValue(new Error("offline"));
  render(<MatterWorkspace onOpenSources={() => {}} />);
  await screen.findByRole("alert");
  await user.type(
    screen.getByRole("textbox", { name: "Search matters" }),
    "nonexistent",
  );
  expect(screen.queryByText("No matching matters")).toBeNull();
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((complete) => {
    resolve = complete;
  });
  return { promise, resolve };
}

const context: MatterContext = {
  matter_id: matter.matter_id,
  version: matter.version,
  availability: "ready",
  instruction_policy: "reference_only",
  truncated: false,
  text: "Only the original authorized matter can show this context.",
};

it("opens source evidence and preserves unknown message time", async () => {
  render(<MatterWorkspace onOpenSources={() => {}} />);
  await screen.findByRole("heading", { name: "Atlas product development" });
  await userEvent.click(screen.getByRole("tab", { name: "Knowledge file" }));
  await userEvent.click(screen.getByRole("button", { name: /View quote/ }));
  expect(await screen.findByText("Message time unknown")).toBeTruthy();
  expect(document.querySelector(".evidence-verbatim")?.textContent).toBe(
    "We plan to open source on September 23 and finish acceptance.",
  );
  expect(document.querySelector(".evidence-verbatim mark")?.textContent).toBe(
    "plan to open source on September 23",
  );
  expect(api.getMatterEvidence).toHaveBeenCalledWith("m1", "e1");
});

it("links real governed memories, map nodes and raw evidence without losing the graph", async () => {
  const item = {
    memory_id: "base-1",
    revision: 1,
    status: "confirmed",
    kind: "working",
    content: "The project weekly meeting is on Wednesday",
    evidence_ids: ["e1"],
    tags: [],
    scopes: ["all-apps"],
  };
  api.getMatters.mockResolvedValue([
    {
      ...matter,
      source_engine: "hy-memory",
      claims: [
        {
          ...matter.claims[0],
          memory_refs: [{ memory_id: "base-1", revision: 1 }],
        },
      ],
    },
  ]);
  api.getMemories.mockResolvedValue([item]);
  api.getMemory.mockResolvedValue({
    memory: item,
    evidence: [
      {
        evidence_id: "e1",
        available: true,
        verbatim: "The weekly meeting moved to Wednesday.",
        written_by: "person",
        source_id: "codex",
        created_at: null,
      },
    ],
    history: [],
  });
  render(<MatterWorkspace onOpenSources={() => {}} />);
  const node = await screen.findByRole("button", {
    name: "Map node The project weekly meeting is on Wednesday",
  });
  await userEvent.click(node);
  expect(
    await screen.findByText("The weekly meeting moved to Wednesday."),
  ).toBeTruthy();
  const canvas = screen.getByTestId("matter-map");
  expect(canvas.dataset.selected).toBe("memory:base-1");
  await userEvent.click(screen.getByRole("tab", { name: "Knowledge file" }));
  await userEvent.click(screen.getByRole("tab", { name: "Memory map" }));
  expect(screen.getByTestId("matter-map")).toBe(canvas);
  expect(canvas.dataset.selected).toBe("memory:base-1");
  await userEvent.click(
    screen.getByRole("button", {
      name: "Select memory The project weekly meeting is on Wednesday",
    }),
  );
  expect(api.getMemory).toHaveBeenCalledWith("base-1");
});

it("confirms scope using optimistic version without confirming claims", async () => {
  render(<MatterWorkspace onOpenSources={() => {}} />);
  await userEvent.click(
    await screen.findByRole("button", { name: "Confirm matter scope" }),
  );
  await waitFor(() =>
    expect(api.patchMatter).toHaveBeenCalledWith(
      "m1",
      {
        expected_version: 1,
        scope_confirmed: true,
      },
      expect.any(String),
    ),
  );
});

it("shows a failed read rather than a misleading empty library", async () => {
  api.getMatters.mockRejectedValue(new Error("offline"));
  render(<MatterWorkspace onOpenSources={() => {}} />);
  expect(await screen.findByRole("alert")).toBeTruthy();
  expect(screen.queryByText("No matters organized yet")).toBeNull();
});

it("blocks stale context after new evidence", async () => {
  api.getMatters.mockResolvedValue([
    { ...matter, availability: "needs_update" },
  ]);
  render(<MatterWorkspace onOpenSources={() => {}} />);
  const button = await screen.findByRole("button", { name: "Task context" });
  expect((button as HTMLButtonElement).disabled).toBe(true);
  expect(screen.getByText(/New material awaits organization/)).toBeTruthy();
});

it.each(["source_unavailable", "needs_update", "memory_changed"] as const)(
  "closes an already open context when the matter becomes %s",
  async (availability) => {
    api.getMatterContext.mockResolvedValue(context);
    render(<MatterWorkspace onOpenSources={() => {}} />);
    await userEvent.click(
      await screen.findByRole("button", { name: "Task context" }),
    );
    expect(await screen.findByDisplayValue(context.text)).toBeTruthy();
    api.getMatters.mockResolvedValue([{ ...matter, availability, claims: [] }]);
    fireEvent(document, new Event("visibilitychange"));
    await waitFor(() =>
      expect(
        (
          screen.getByRole("button", {
            name: "Task context",
          }) as HTMLButtonElement
        ).disabled,
      ).toBe(true),
    );
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.queryByDisplayValue(context.text)).toBeNull();
  },
);

it("does not reopen context from a request started before switching matters", async () => {
  const response = deferred<MatterContext>();
  api.getMatterContext.mockReturnValue(response.promise);
  api.getMatters.mockResolvedValue([
    matter,
    { ...matter, matter_id: "m2", title: "Cedar plan" },
  ]);
  render(<MatterWorkspace onOpenSources={() => {}} />);
  await userEvent.click(
    await screen.findByRole("button", { name: "Task context" }),
  );
  await userEvent.click(screen.getByRole("button", { name: /Cedar plan/ }));
  await screen.findByRole("heading", { name: "Cedar plan" });
  await act(async () => {
    response.resolve(context);
  });
  expect(screen.queryByRole("dialog")).toBeNull();
  expect(screen.queryByDisplayValue(context.text)).toBeNull();
});

it("rejects a context returned for a different revision", async () => {
  api.getMatterContext.mockResolvedValue({ ...context, version: 2 });
  render(<MatterWorkspace onOpenSources={() => {}} />);
  await userEvent.click(
    await screen.findByRole("button", { name: "Task context" }),
  );
  await waitFor(() => expect(api.getMatterContext).toHaveBeenCalledOnce());
  expect(screen.queryByRole("dialog")).toBeNull();
});

it("clears history and ignores its late response after source revocation", async () => {
  const response = deferred<Matter[]>();
  api.getMatterHistory.mockReturnValue(response.promise);
  render(<MatterWorkspace onOpenSources={() => {}} />);
  await userEvent.click(
    await screen.findByRole("tab", { name: "Version history" }),
  );
  const privateHistory = { ...matter, title: "A private historical title" };
  api.getMatters.mockResolvedValue([
    { ...matter, availability: "source_unavailable", claims: [] },
  ]);
  fireEvent(document, new Event("visibilitychange"));
  await screen.findByText(/the old file is hidden/);
  await act(async () => {
    response.resolve([privateHistory]);
  });
  expect(screen.queryByText(privateHistory.title)).toBeNull();
});

it("clears already loaded history when its matter becomes unavailable", async () => {
  api.getMatterHistory.mockResolvedValue([
    { ...matter, title: "Private prior version" },
  ]);
  render(<MatterWorkspace onOpenSources={() => {}} />);
  await userEvent.click(
    await screen.findByRole("tab", { name: "Version history" }),
  );
  await screen.findByText("Private prior version");
  api.getMatters.mockResolvedValue([
    { ...matter, availability: "source_unavailable", claims: [] },
  ]);
  fireEvent(document, new Event("visibilitychange"));
  await screen.findByText(/the old file is hidden/);
  expect(screen.queryByText("Private prior version")).toBeNull();
});

it("does not restore a late context after a known revocation", async () => {
  const response = deferred<MatterContext>();
  api.getMatterContext.mockReturnValue(response.promise);
  render(<MatterWorkspace onOpenSources={() => {}} />);
  await userEvent.click(
    await screen.findByRole("button", { name: "Task context" }),
  );
  api.getMatters.mockResolvedValue([
    { ...matter, availability: "source_unavailable", claims: [] },
  ]);
  fireEvent(document, new Event("visibilitychange"));
  await screen.findByText(/the old file is hidden/);
  await act(async () => {
    response.resolve(context);
  });
  expect(screen.queryByDisplayValue(context.text)).toBeNull();
  expect(screen.queryByRole("dialog")).toBeNull();
});

it("applies a successful source invalidation even when run-status loading fails", async () => {
  api.getMatterContext.mockResolvedValue(context);
  render(<MatterWorkspace onOpenSources={() => {}} />);
  await userEvent.click(
    await screen.findByRole("button", { name: "Task context" }),
  );
  await screen.findByDisplayValue(context.text);
  api.getMatters.mockResolvedValue([
    { ...matter, availability: "source_unavailable", claims: [] },
  ]);
  api.getMatterRuns.mockRejectedValue(new Error("status endpoint unavailable"));
  fireEvent(document, new Event("visibilitychange"));
  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  expect(screen.queryByDisplayValue(context.text)).toBeNull();
  expect(
    (screen.getByRole("button", { name: "Task context" }) as HTMLButtonElement)
      .disabled,
  ).toBe(true);
});

it("does not wait for a pending progress request to apply source revocation", async () => {
  api.getMatterContext.mockResolvedValue(context);
  render(<MatterWorkspace onOpenSources={() => {}} />);
  await userEvent.click(
    await screen.findByRole("button", { name: "Task context" }),
  );
  await screen.findByDisplayValue(context.text);
  const progress = deferred<[]>();
  api.getMatterRuns.mockReturnValue(progress.promise);
  api.getMatters.mockResolvedValue([
    { ...matter, availability: "source_unavailable", claims: [] },
  ]);
  fireEvent(document, new Event("visibilitychange"));
  try {
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(
      (
        screen.getByRole("button", {
          name: "Task context",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
  } finally {
    await act(async () => {
      progress.resolve([]);
    });
  }
});

it.each([false, true])(
  "invalidates memory details on version change (pending: %s)",
  async (pending) => {
    const memory: MemoryDetail["memory"] = {
      memory_id: "base-1",
      user_id: "local",
      revision: 1,
      status: "confirmed",
      kind: "working",
      content: "Memory before update",
      evidence_ids: ["e1"],
      tags: [],
      scopes: ["all-apps"],
      origin: "extracted",
      created_at: "2026-09-12",
      updated_at: "2026-09-12",
      deleted_at: null,
    };
    const item = {
      ...matter,
      source_engine: "hy-memory",
      claims: [
        {
          ...matter.claims[0],
          memory_refs: [
            { memory_id: memory.memory_id, revision: memory.revision },
          ],
        },
      ],
    };
    api.getMatters.mockResolvedValue([item]);
    api.getMemories.mockResolvedValue([memory]);
    const response = deferred<MemoryDetail>();
    const detail: MemoryDetail = {
      memory,
      history: [],
      index_sync: null,
      evidence: [
        {
          evidence_id: "e1",
          available: true,
          verbatim: "Private source before update",
          written_by: "person",
          source_id: "codex",
          role: "user",
          created_at: null,
        },
      ],
    };
    api.getMemory.mockReturnValue(
      pending ? response.promise : Promise.resolve(detail),
    );
    render(<MatterWorkspace onOpenSources={() => {}} />);
    await userEvent.click(
      await screen.findByRole("button", {
        name: "Select memory Memory before update",
      }),
    );
    if (!pending) await screen.findByText("Private source before update");
    api.getMatters.mockResolvedValue([
      { ...item, availability: "needs_update" },
    ]);
    fireEvent(document, new Event("visibilitychange"));
    await screen.findByText(/New material awaits organization/);
    if (pending)
      await act(async () => {
        response.resolve(detail);
      });
    expect(screen.queryByText("Private source before update")).toBeNull();
  },
);

it.each([
  { revision: 2 },
  { status: "deleted" },
  { memory_id: "different-memory" },
])(
  "rejects a detail that no longer matches the selected reference: %j",
  async (change) => {
    const memory = {
      memory_id: "base-1",
      revision: 1,
      status: "confirmed",
      kind: "working",
      content: "Referenced memory",
      evidence_ids: ["e1"],
      tags: [],
      scopes: ["all-apps"],
    };
    api.getMatters.mockResolvedValue([
      {
        ...matter,
        source_engine: "hy-memory",
        claims: [
          {
            ...matter.claims[0],
            memory_refs: [{ memory_id: memory.memory_id, revision: 1 }],
          },
        ],
      },
    ]);
    api.getMemories.mockResolvedValue([memory]);
    api.getMemory.mockResolvedValue({
      memory: { ...memory, ...change },
      history: [],
      index_sync: null,
      evidence: [
        {
          evidence_id: "e1",
          available: true,
          verbatim: "Mismatched source text",
        },
      ],
    });
    render(<MatterWorkspace onOpenSources={() => {}} />);
    await userEvent.click(
      await screen.findByRole("button", {
        name: "Select memory Referenced memory",
      }),
    );
    await screen.findByText(/This base memory has changed or cannot be read/);
    expect(screen.queryByText("Mismatched source text")).toBeNull();
  },
);

it("separates full input coverage from failed dossier checks", async () => {
  api.getMatterRuns.mockResolvedValue([
    {
      run_id: "r1",
      status: "failed",
      phase: "Organization incomplete",
      failed_phase: "Organizing current conclusions and change history",
      selected_records: 77,
      hy_input_records: 77,
      omitted_records: 0,
      coverage_complete: true,
      required_memory_coverage_complete: false,
      error: "matter_synthesis_coverage_incomplete",
      observations: [],
    },
  ]);
  render(<MatterWorkspace onOpenSources={() => {}} />);
  expect(await screen.findByText(/Read 77 records/)).toBeTruthy();
  expect(screen.getByText(/HY input 77 records/)).toBeTruthy();
  expect(screen.getByText(/Key memories were not fully retained/)).toBeTruthy();
  expect(screen.queryByText("File pending verification")).toBeNull();
});

it("reports deferred conversations instead of implying complete coverage", async () => {
  api.getMatterRuns.mockResolvedValue([
    {
      run_id: "r2",
      status: "failed",
      phase: "Organization incomplete",
      selected_records: 3,
      hy_input_records: 0,
      omitted_records: 42,
      coverage_complete: false,
      error: "matter_collection_incomplete",
      observations: [],
    },
  ]);
  render(<MatterWorkspace onOpenSources={() => {}} />);
  expect(await screen.findByText(/42 records not processed/)).toBeTruthy();
  expect(screen.getByText(/Incomplete read this run/)).toBeTruthy();
});

it("refreshes idle matters so external revocation cannot leave old claims visible", async () => {
  render(<MatterWorkspace onOpenSources={() => {}} />);
  await screen.findByRole("heading", { name: matter.title });
  api.getMatters.mockResolvedValue([
    {
      ...matter,
      availability: "source_unavailable",
      claims: [],
      title: "Source unavailable",
      purpose: "",
    },
  ]);
  await waitFor(
    () =>
      expect(
        screen.getByText(
          "The sources or base memories have changed; the old file is hidden.",
        ),
      ).toBeTruthy(),
    { timeout: 7500 },
  );
  expect(
    (screen.getByRole("button", { name: "Task context" }) as HTMLButtonElement)
      .disabled,
  ).toBe(true);
}, 9000);
