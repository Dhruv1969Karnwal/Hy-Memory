// @vitest-environment jsdom

import type { MemoryDetail } from "@one-memory/contracts";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { StrictMode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { SelectedMemoryDetail } from "./SelectedMemoryDetail";

const { correct, read, remove, errorCode } = vi.hoisted(() => ({
  correct: vi.fn(),
  read: vi.fn(),
  remove: vi.fn(),
  errorCode: vi.fn(),
}));
const trace = vi.hoisted(() => ({
  done: vi.fn(),
  traceId: "trace-selected-memory",
}));
vi.mock("./api", () => ({
  correctMemory: correct,
  getMemory: read,
  mutateMemory: remove,
  apiErrorCode: errorCode,
}));
vi.mock("./ui-trace", () => ({
  traceUiAction: () => trace,
}));
vi.mock("./MemoryRecallScope", () => ({ MemoryRecallScope: () => null }));
const detail = {
  memory: {
    memory_id: "m1",
    revision: 2,
    content:
      "Confirm the scope before the release. A design preview is not product sign-off.",
    status: "confirmed",
    scopes: ["private"],
  },
  evidence: [
    {
      evidence_id: "e1",
      source_id: "workbuddy",
      written_by: "person",
      created_at: "2026-09-19T00:00:00Z",
      available: true,
      verbatim: "Confirm the scope first this time.",
    },
  ],
  history: [],
} as unknown as MemoryDetail;
beforeEach(() => {
  correct.mockReset().mockResolvedValue(detail.memory);
  read.mockReset();
  remove.mockReset().mockResolvedValue(detail.memory);
  errorCode.mockReset();
  trace.done.mockReset();
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});
it("OBS-21: names a manually added memory Added manually, not the internal source id", () => {
  render(
    <SelectedMemoryDetail
      detail={{
        ...detail,
        memory: { ...detail.memory, origin: "manual" },
        evidence: [{ ...detail.evidence[0], source_id: "one-memory-manual" }],
      }}
      sources={[]}
      onChanged={vi.fn()}
      hideActions
    />,
  );
  expect(screen.getByText(/^Source Manually added · /)).toBeTruthy();
  expect(screen.queryByText(/one-memory-manual/)).toBeNull();
});

it("keeps the simplified memory inspector informational", () => {
  render(
    <SelectedMemoryDetail
      detail={{
        ...detail,
        history: [
          { ...detail.memory, revision: 1, action: "create" },
          { ...detail.memory, revision: 2, action: "correct" },
        ],
      }}
      sources={[]}
      onChanged={vi.fn()}
      hideActions
    />,
  );
  expect(
    screen.getByRole("heading", {
      name: "Confirm the scope before the release. A design preview is not product sign-off.",
    }),
  ).toBeTruthy();
  for (const label of [
    "Edit",
    "View source text",
    "Version history",
    "More management",
  ])
    expect(screen.queryByText(label)).toBeNull();
});
it.each([
  ["confirmed", "Stored"],
  ["deleted", "Deleted"],
  ["superseded", "Superseded"],
] as const)("labels a %s memory with its actual state", (status, label) => {
  render(
    <SelectedMemoryDetail
      detail={{
        ...detail,
        memory: { ...detail.memory, status },
      }}
      sources={[]}
      onChanged={vi.fn()}
      hideActions
    />,
  );
  expect(screen.getByText(`${label} · revision 2`)).toBeTruthy();
  expect(screen.queryByText("Pending verification · revision 2")).toBeNull();
});
it("opens full editor and saves content without rewriting scope", async () => {
  const changed = vi.fn();
  render(
    <SelectedMemoryDetail detail={detail} sources={[]} onChanged={changed} />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Edit" }));
  fireEvent.change(screen.getByLabelText("Memory content"), {
    target: { value: "New body text" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
  await waitFor(() =>
    expect(correct).toHaveBeenCalledWith(
      "m1",
      2,
      "New body text",
      undefined,
      expect.any(String),
      expect.any(String),
    ),
  );
  expect(changed).toHaveBeenCalledWith("m1");
});
it("keeps draft after conflict until latest revision is explicitly acknowledged", async () => {
  correct.mockRejectedValueOnce(Error("conflict"));
  errorCode.mockReturnValue("memory_revision_conflict");
  read.mockResolvedValue({
    ...detail,
    memory: {
      ...detail.memory,
      revision: 3,
      content: "The new remote version",
    },
  });
  render(
    <SelectedMemoryDetail detail={detail} sources={[]} onChanged={() => {}} />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Edit" }));
  fireEvent.change(screen.getByLabelText("Memory content"), {
    target: { value: "My draft" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
  await screen.findByText("A newer version exists");
  expect(screen.getByLabelText("Memory content")).toHaveProperty(
    "value",
    "My draft",
  );
  expect(screen.getByRole("button", { name: "Save changes" })).toHaveProperty(
    "disabled",
    true,
  );
  fireEvent.click(screen.getByRole("button", { name: "View the new version" }));
  await screen.findByText("The new remote version");
  fireEvent.click(
    screen.getByRole("button", { name: "Keep editing from this version" }),
  );
  correct.mockResolvedValue(detail.memory);
  fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
  await waitFor(() =>
    expect(correct).toHaveBeenLastCalledWith(
      "m1",
      3,
      "My draft",
      undefined,
      expect.any(String),
      expect.any(String),
    ),
  );
});
it("original page never substitutes unavailable evidence with stale text", () => {
  const view = render(
    <SelectedMemoryDetail detail={detail} sources={[]} onChanged={() => {}} />,
  );
  fireEvent.click(screen.getByRole("button", { name: "View source text" }));
  expect(
    screen.getAllByText("Confirm the scope first this time.").length,
  ).toBeGreaterThan(0);
  view.rerender(
    <SelectedMemoryDetail
      detail={{
        ...detail,
        evidence: [{ ...detail.evidence[0], available: false }],
      }}
      sources={[]}
      onChanged={() => {}}
    />,
  );
  expect(screen.queryByText("Confirm the scope first this time.")).toBeNull();
  expect(
    screen.getByText(
      "Source access changed or the source text is unavailable.",
    ),
  ).toBeTruthy();
});

it.each(["Edit", "View source text", "Delete memory"])(
  "returns focus to %s without scrolling the background",
  (name) => {
    render(
      <StrictMode>
        <div className="r6-shell">
          <SelectedMemoryDetail
            detail={detail}
            sources={[]}
            onChanged={vi.fn()}
          />
        </div>
      </StrictMode>,
    );
    if (name === "Delete memory")
      fireEvent.click(screen.getByText("More management"));
    const trigger = screen.getByRole("button", { name });
    trigger.focus();
    const restore = vi.spyOn(trigger, "focus");
    fireEvent.click(trigger);
    const title = screen.getByRole("heading", { level: 1 });
    expect(document.activeElement).toBe(title);
    fireEvent.keyDown(title, { key: "Escape" });
    expect(document.activeElement).toBe(trigger);
    expect(restore).toHaveBeenCalledWith({ preventScroll: true });
    expect(document.querySelector<HTMLElement>(".r6-shell")?.inert).toBe(false);
  },
);

it.each(["escape", "cancel"])(
  "retains dirty draft when %s confirmation is declined and only discards after acceptance",
  (mode) => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    render(
      <SelectedMemoryDetail detail={detail} sources={[]} onChanged={vi.fn()} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Edit" }));
    fireEvent.change(screen.getByLabelText("Memory content"), {
      target: { value: "An unsaved draft" },
    });
    const close = () =>
      mode === "escape"
        ? fireEvent.keyDown(screen.getByLabelText("Memory content"), {
            key: "Escape",
          })
        : fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    close();
    expect(confirm).toHaveBeenCalledWith("Discard unsaved changes?");
    expect(screen.getByLabelText("Memory content")).toHaveProperty(
      "value",
      "An unsaved draft",
    );
    expect(screen.getByText("Confirm the scope first this time.")).toBeTruthy();
    expect(correct).not.toHaveBeenCalled();
    confirm.mockReturnValue(true);
    close();
    expect(screen.queryByLabelText("Memory content")).toBeNull();
  },
);

it("keeps draft and request identity after a failed save; pending escape cannot discard", async () => {
  let reject!: (reason: Error) => void;
  correct.mockReturnValueOnce(
    new Promise((_, r) => {
      reject = r;
    }),
  );
  const confirm = vi.spyOn(window, "confirm");
  render(
    <SelectedMemoryDetail detail={detail} sources={[]} onChanged={vi.fn()} />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Edit" }));
  fireEvent.change(screen.getByLabelText("Memory content"), {
    target: { value: "Kept draft" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
  fireEvent.keyDown(screen.getByLabelText("Memory content"), { key: "Escape" });
  expect(confirm).not.toHaveBeenCalled();
  await act(async () => reject(Error("offline")));
  expect(screen.getByRole("alert").textContent).toContain("the draft is kept");
  expect(screen.getByLabelText("Memory content")).toHaveProperty(
    "value",
    "Kept draft",
  );
  fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
  await waitFor(() => expect(correct).toHaveBeenCalledTimes(2));
  // The durable request id must be reused across a retry; the trace id is a
  // fresh id per user attempt so each try forms its own four-line chain.
  expect(correct.mock.calls[1][0]).toBe(correct.mock.calls[0][0]);
  expect(correct.mock.calls[1][1]).toBe(correct.mock.calls[0][1]);
  expect(correct.mock.calls[1][2]).toBe(correct.mock.calls[0][2]);
  expect(correct.mock.calls[1][4]).toBe(correct.mock.calls[0][4]);
});

it("does not apply a late conflict read after its editor has closed", async () => {
  let resolve!: (value: MemoryDetail) => void;
  read.mockReturnValue(
    new Promise((r) => {
      resolve = r;
    }),
  );
  correct.mockRejectedValue(Error("conflict"));
  errorCode.mockReturnValue("memory_revision_conflict");
  const view = render(
    <SelectedMemoryDetail detail={detail} sources={[]} onChanged={vi.fn()} />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Edit" }));
  fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
  await screen.findByText("A newer version exists");
  fireEvent.click(screen.getByRole("button", { name: "View the new version" }));
  view.unmount();
  await act(async () =>
    resolve({
      ...detail,
      memory: { ...detail.memory, revision: 3, content: "late" },
    }),
  );
  expect(screen.queryByText("late")).toBeNull();
});

it("keeps failed deletion on its confirmation page and retries only the same version and request", async () => {
  remove.mockRejectedValue(Error("offline"));
  const changed = vi.fn();
  render(
    <SelectedMemoryDetail detail={detail} sources={[]} onChanged={changed} />,
  );
  fireEvent.click(screen.getByText("More management"));
  fireEvent.click(screen.getByRole("button", { name: "Delete memory" }));
  const buttons = screen.getAllByRole("button", { name: "Delete memory" });
  fireEvent.click(buttons[buttons.length - 1]);
  await screen.findByRole("alert");
  expect(
    screen.getByText(
      /the original conversation in the source app is not deleted/,
    ),
  ).toBeTruthy();
  fireEvent.click(buttons[buttons.length - 1]);
  await waitFor(() => expect(remove).toHaveBeenCalledTimes(2));
  expect(remove.mock.calls[1][0]).toBe(remove.mock.calls[0][0]);
  expect(remove.mock.calls[1][1]).toBe(remove.mock.calls[0][1]);
  expect(remove.mock.calls[1][2]).toBe(remove.mock.calls[0][2]);
  expect(remove.mock.calls[1][3]).toBe(remove.mock.calls[0][3]);
  expect(remove.mock.calls[0]).toEqual([
    "m1",
    "delete",
    2,
    expect.any(String),
    expect.any(String),
  ]);
  expect(changed).not.toHaveBeenCalled();
});

it("reports a definite save refusal as not submitted, not as an unknown write", async () => {
  correct.mockRejectedValueOnce({
    response: { status: 422, data: { detail: "request_validation_failed" } },
  });
  errorCode.mockReturnValue("request_validation_failed");
  render(
    <SelectedMemoryDetail detail={detail} sources={[]} onChanged={vi.fn()} />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Edit" }));
  fireEvent.change(screen.getByLabelText("Memory content"), {
    target: { value: "A rejected draft" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Save changes" }));

  expect((await screen.findByRole("alert")).textContent).toContain(
    "The change was not submitted",
  );
  expect(screen.queryByText(/The change is not confirmed yet/)).toBeNull();
  expect(trace.done).toHaveBeenLastCalledWith("failed", {
    reason: "memory_correct_failed",
  });
});

it("reports a definite delete refusal as not submitted and keeps the same request", async () => {
  remove.mockRejectedValueOnce({
    response: { status: 404, data: { detail: "memory_not_found" } },
  });
  errorCode.mockReturnValue("memory_not_found");
  render(
    <SelectedMemoryDetail detail={detail} sources={[]} onChanged={vi.fn()} />,
  );
  fireEvent.click(screen.getByText("More management"));
  fireEvent.click(screen.getByRole("button", { name: "Delete memory" }));
  const buttons = screen.getAllByRole("button", { name: "Delete memory" });
  fireEvent.click(buttons[buttons.length - 1]);

  expect((await screen.findByRole("alert")).textContent).toContain(
    "The delete was not submitted",
  );
  expect(screen.queryByText(/The delete is not confirmed yet/)).toBeNull();
  expect(trace.done).toHaveBeenLastCalledWith("failed", {
    reason: "memory_delete_failed",
  });

  fireEvent.click(buttons[buttons.length - 1]);
  await waitFor(() => expect(remove).toHaveBeenCalledTimes(2));
  expect(remove.mock.calls[1][3]).toBe(remove.mock.calls[0][3]);
});

it("reports a definite undo refusal as not submitted, not as an unknown write", async () => {
  remove.mockRejectedValueOnce({
    response: { status: 409, data: { detail: "memory_revision_conflict" } },
  });
  errorCode.mockReturnValue("memory_revision_conflict");
  render(
    <SelectedMemoryDetail
      detail={{
        ...detail,
        history: [
          { ...detail.memory, revision: 1, action: "create" },
          { ...detail.memory, revision: 2, action: "correct" },
        ],
      }}
      sources={[]}
      onChanged={vi.fn()}
    />,
  );
  fireEvent.click(screen.getByText("More management"));
  fireEvent.click(screen.getByRole("button", { name: "Undo last change" }));

  expect((await screen.findByRole("alert")).textContent).toContain(
    "The undo was not submitted",
  );
  expect(screen.queryByText(/The operation is not confirmed yet/)).toBeNull();
  expect(trace.done).toHaveBeenLastCalledWith("failed", {
    reason: "memory_undo_failed",
  });
});

it("keeps a timeout save as unknown and does not replace its request identity", async () => {
  let reject!: (reason: unknown) => void;
  correct.mockReturnValueOnce(
    new Promise((_, fail) => {
      reject = fail;
    }),
  );
  errorCode.mockReturnValue(null);
  render(
    <SelectedMemoryDetail detail={detail} sources={[]} onChanged={vi.fn()} />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Edit" }));
  fireEvent.change(screen.getByLabelText("Memory content"), {
    target: { value: "A timed-out draft" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
  await act(async () => reject(new Error("offline")));

  expect(screen.getByRole("alert").textContent).toContain(
    "The change is not confirmed yet",
  );
  expect(trace.done).toHaveBeenLastCalledWith("unknown", {
    reason: "memory_correct_result_unknown",
  });
  correct.mockResolvedValue(detail.memory);
  fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
  await waitFor(() => expect(correct).toHaveBeenCalledTimes(2));
  expect(correct.mock.calls[1][4]).toBe(correct.mock.calls[0][4]);
});

it("keeps the original CAS and draft when reading a conflict fails", async () => {
  correct.mockRejectedValue(Error("conflict"));
  errorCode.mockReturnValue("memory_revision_conflict");
  read.mockRejectedValue(Error("offline"));
  render(
    <SelectedMemoryDetail detail={detail} sources={[]} onChanged={vi.fn()} />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Edit" }));
  fireEvent.change(screen.getByLabelText("Memory content"), {
    target: { value: "A conflicting draft" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Save changes" }));
  await screen.findByText("A newer version exists");
  fireEvent.click(screen.getByRole("button", { name: "View the new version" }));
  expect((await screen.findByRole("alert")).textContent).toContain(
    "the draft is kept",
  );
  expect(screen.getByLabelText("Memory content")).toHaveProperty(
    "value",
    "A conflicting draft",
  );
  expect(screen.getByRole("button", { name: "Save changes" })).toHaveProperty(
    "disabled",
    true,
  );
  expect(
    screen.queryByRole("button", { name: "Keep editing from this version" }),
  ).toBeNull();
  expect(correct.mock.calls[0][1]).toBe(2);
});

it("suspends the editor without losing its draft or leaving the background inert", () => {
  const p = { detail, sources: [], onChanged: vi.fn() };
  const view = render(
    <div className="r6-shell">
      <SelectedMemoryDetail {...p} />
    </div>,
  );
  const trigger = screen.getByRole("button", { name: "Edit" });
  trigger.focus();
  fireEvent.click(trigger);
  fireEvent.change(screen.getByLabelText("Memory content"), {
    target: { value: "A parked draft" },
  });
  view.rerender(
    <div className="r6-shell">
      <SelectedMemoryDetail {...p} suspended />
    </div>,
  );
  expect(document.querySelector<HTMLElement>(".r6-shell")?.inert).toBe(false);
  expect(document.activeElement).toBe(trigger);
  view.rerender(
    <div className="r6-shell">
      <SelectedMemoryDetail {...p} />
    </div>,
  );
  expect(screen.getByLabelText("Memory content")).toHaveProperty(
    "value",
    "A parked draft",
  );
  expect(document.querySelector<HTMLElement>(".r6-shell")?.inert).toBe(true);
  expect(document.activeElement).toBe(
    screen.getByRole("heading", { level: 1 }),
  );
});

it("stamps today's source time the same way as 06A and 06E", () => {
  render(
    <SelectedMemoryDetail
      detail={{
        ...detail,
        evidence: [
          {
            ...detail.evidence[0],
            created_at: new Date().toISOString(),
          },
        ],
      }}
      sources={[]}
      onChanged={vi.fn()}
    />,
  );
  expect(screen.getByText(/Source workbuddy · Today /)).toBeTruthy();
  expect(screen.getByText(/Original conversation Today /)).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "View source text" }));
  expect(
    screen.getByRole("heading", { name: "Original conversation" }),
  ).toBeTruthy();
  expect(screen.getByText(/^workbuddy · Today /)).toBeTruthy();
});
