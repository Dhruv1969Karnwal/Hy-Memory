// @vitest-environment jsdom
import type { MemoryDetail, SourceResponse } from "@one-memory/contracts";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FirstUseGuide, FirstUseInvitation } from "./FirstUseGuide";

const getMemories = vi.fn(),
  getMemory = vi.fn(),
  getSources = vi.fn(),
  copyText = vi.fn(),
  openHandoff = vi.fn();
vi.mock("./api", () => ({
  getMemories: (...args: unknown[]) => getMemories(...args),
  getMemory: (...args: unknown[]) => getMemory(...args),
  getSources: () => getSources(),
}));
vi.mock("./copy-text", () => ({ copyText: (text: string) => copyText(text) }));
const source = {
  source_id: "cursor",
  display_name: "Cursor",
  authorized: true,
} as SourceResponse;
const detail: MemoryDetail = {
  memory: {
    memory_id: "m1",
    user_id: "local",
    content: "Avoid Fridays for the weekly meeting",
    kind: "chat",
    status: "confirmed",
    origin: "extracted",
    scopes: ["all-apps"],
    tags: ["Project weekly meeting"],
    evidence_ids: ["e1"],
    revision: 1,
    created_at: "2026-09-01T00:00:00Z",
    updated_at: "2026-09-01T00:00:00Z",
    deleted_at: null,
  },
  evidence: [
    {
      evidence_id: "e1",
      source_id: "cursor",
      written_by: "agent",
      role: "assistant",
      created_at: null,
      verbatim: "Suggest moving the weekly meeting to another weekday.",
      available: true,
    },
  ],
  history: [],
  index_sync: null,
};
const second: MemoryDetail = {
  ...detail,
  memory: {
    ...detail.memory,
    memory_id: "m2",
    content: "Second background",
    evidence_ids: ["e2"],
  },
  evidence: [{ ...detail.evidence[0], evidence_id: "e2", source_id: "codex" }],
};
beforeEach(() => {
  vi.resetAllMocks();
  getMemories.mockResolvedValue([detail.memory, second.memory]);
  getMemory.mockImplementation(async (id: string) =>
    id === "m1" ? detail : second,
  );
  getSources.mockResolvedValue([source]);
  copyText.mockResolvedValue(undefined);
  openHandoff.mockResolvedValue({ opened: true });
  window.oneMemory = {
    openHandoff,
    getRecallHosts: vi.fn().mockResolvedValue([
      {
        hostId: "cursor",
        label: "Cursor",
        present: true,
        installed: true,
        handoff: "open",
        selfCheck: { state: "wired" },
      },
      {
        hostId: "codex",
        label: "Codex",
        present: true,
        installed: false,
        handoff: "copy",
        selfCheck: null,
      },
    ]),
  } as unknown as Window["oneMemory"];
});
afterEach(() => {
  cleanup();
  delete window.oneMemory;
});
function show() {
  return render(
    <FirstUseGuide sources={[source]} onClose={vi.fn()} onSources={vi.fn()} />,
  );
}
async function ready() {
  await waitFor(() =>
    expect(
      (screen.getByLabelText("Full opener preview") as HTMLTextAreaElement)
        .value,
    ).toContain("Friday"),
  );
}

describe("first-use guide", () => {
  it("clears an old preview if the current memory read is denied", async () => {
    const user = userEvent.setup();
    show();
    await ready();
    getMemory.mockRejectedValue(new Error("access revoked"));
    await user.click(screen.getByRole("button", { name: "Copy opener" }));
    expect(copyText).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Full opener preview")).toHaveProperty(
      "value",
      "",
    );
  });
  it("opens without a cloud request, auto copy or launch; keeps author and unknown time", async () => {
    show();
    await ready();
    expect(screen.getByText(/Agent · Time unknown/)).toBeTruthy();
    expect(
      screen.getByText(/Organized automatically; needs your review/),
    ).toBeTruthy();
    expect(copyText).not.toHaveBeenCalled();
    expect(openHandoff).not.toHaveBeenCalled();
  });
  it("keeps all three scenarios optional and filters destinations", async () => {
    const user = userEvent.setup();
    show();
    await ready();
    expect(screen.queryByRole("option", { name: "Codex" })).toBeNull();
    await user.click(
      screen.getByRole("radio", { name: /Continue in another app/ }),
    );
    expect(screen.getByRole("option", { name: "Codex" })).toBeTruthy();
    expect(screen.queryByRole("option", { name: "Cursor" })).toBeNull();
    await user.click(
      screen.getByRole("radio", { name: /Let a new tool get to know me/ }),
    );
    expect(screen.getByRole("option", { name: "Cursor" })).toBeTruthy();
    expect(
      (screen.getByLabelText("Full opener preview") as HTMLTextAreaElement)
        .value,
    ).toContain("it is not a complete personal profile");
  });
  it("only enables query for a connected destination, and excludes the answer", async () => {
    const user = userEvent.setup();
    show();
    await ready();
    expect(
      screen.getByRole("radio", { name: "Let the app query" }),
    ).toHaveProperty("disabled", true);
    await user.selectOptions(
      screen.getByLabelText("Which app to try it in"),
      "cursor",
    );
    await user.click(screen.getByRole("radio", { name: "Let the app query" }));
    expect(
      (screen.getByLabelText("Full opener preview") as HTMLTextAreaElement)
        .value,
    ).not.toContain("Friday");
    await user.click(screen.getByRole("button", { name: "Copy opener" }));
    expect(copyText.mock.calls[0][0]).not.toContain("Friday");
    expect(
      screen.getByText(/No receipt for this external call yet/),
    ).toBeTruthy();
  });
  it("does not turn copy or user feedback into an automatic adoption pass", async () => {
    const user = userEvent.setup();
    show();
    await ready();
    await user.click(screen.getByRole("button", { name: "Copy opener" }));
    expect(getMemory).toHaveBeenCalledTimes(2);
    expect(getSources).toHaveBeenCalledOnce();
    expect(screen.getByRole("status").textContent).toContain("Not sent yet");
    await user.click(
      screen.getByRole("button", { name: "I verified the result" }),
    );
    expect(
      screen.getByText(/This is not an automatic acceptance record/),
    ).toBeTruthy();
  });
  it.each(["revision", "revocation", "evidence"] as const)(
    "rechecks %s before copying",
    async (kind) => {
      const user = userEvent.setup();
      show();
      await ready();
      if (kind === "revision")
        getMemory.mockResolvedValue({
          ...detail,
          memory: { ...detail.memory, revision: 2 },
        });
      if (kind === "revocation")
        getSources.mockResolvedValue([{ ...source, authorized: false }]);
      if (kind === "evidence")
        getMemory.mockResolvedValue({
          ...detail,
          evidence: [{ ...detail.evidence[0], verbatim: "New source text" }],
        });
      await user.click(screen.getByRole("button", { name: "Copy opener" }));
      expect(copyText).not.toHaveBeenCalled();
      expect(openHandoff).not.toHaveBeenCalled();
      expect(screen.getByRole("alert").textContent).toContain(
        "authorization has changed",
      );
      expect(screen.getByLabelText("Full opener preview")).toHaveProperty(
        "value",
        "",
      );
    },
  );
  it("ignores a late detail response after selecting another memory", async () => {
    let first: (data: MemoryDetail) => void = () => undefined;
    getMemory.mockImplementation((id: string) =>
      id === "m1"
        ? new Promise((resolve) => {
            first = resolve;
          })
        : Promise.resolve(second),
    );
    const user = userEvent.setup();
    show();
    await screen.findByLabelText("Choose a memory");
    await user.selectOptions(screen.getByLabelText("Choose a memory"), "m2");
    await waitFor(() =>
      expect(
        (screen.getByLabelText("Full opener preview") as HTMLTextAreaElement)
          .value,
      ).toContain("Second background"),
    );
    await act(async () => first(detail));
    expect(
      (screen.getByLabelText("Full opener preview") as HTMLTextAreaElement)
        .value,
    ).not.toContain("Friday");
  });
  it("does not launch after unmount while the pre-copy check is pending", async () => {
    let resolve: (data: MemoryDetail) => void = () => undefined;
    const user = userEvent.setup();
    const view = show();
    await ready();
    getMemory.mockImplementation(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    await user.selectOptions(
      screen.getByLabelText("Which app to try it in"),
      "cursor",
    );
    await user.click(
      screen.getByRole("button", { name: "Copy and open Cursor" }),
    );
    view.unmount();
    await act(async () => resolve(detail));
    expect(copyText).not.toHaveBeenCalled();
    expect(openHandoff).not.toHaveBeenCalled();
  });
  it("does not launch or claim success when the clipboard fails", async () => {
    copyText.mockRejectedValue(new Error("denied"));
    const user = userEvent.setup();
    show();
    await ready();
    await user.selectOptions(
      screen.getByLabelText("Which app to try it in"),
      "cursor",
    );
    await user.click(
      screen.getByRole("button", { name: "Copy and open Cursor" }),
    );
    expect(openHandoff).not.toHaveBeenCalled();
    expect(screen.getByRole("status").textContent).toContain(
      "Could not verify",
    );
    expect(
      screen.queryByRole("button", { name: "I verified the result" }),
    ).toBeNull();
  });
  it("keeps the empty state truthful and retryable", async () => {
    getMemories.mockResolvedValue([]);
    show();
    expect(
      await screen.findByText(
        "Leave behind one piece of background that really helps",
      ),
    ).toBeTruthy();
    expect(screen.queryByLabelText("Full opener preview")).toBeNull();
    expect(
      screen.getByRole("button", { name: "View imports and connections" }),
    ).toBeTruthy();
  });
  it("refuses missing evidence without replacing it with invented content", async () => {
    getMemory.mockResolvedValue({ ...detail, evidence: [] });
    show();
    await screen.findByText(
      /The source text or current authorization is unavailable/,
    );
    expect(screen.getByRole("button", { name: "Copy opener" })).toHaveProperty(
      "disabled",
      true,
    );
  });
  it("dismisses the invitation without changing jobs, memory or local storage", async () => {
    const user = userEvent.setup(),
      onDismiss = vi.fn(),
      onOpen = vi.fn();
    render(<FirstUseInvitation onDismiss={onDismiss} onOpen={onOpen} />);
    await user.click(
      screen.getByRole("button", { name: "Close getting-started invite" }),
    );
    expect(onDismiss).toHaveBeenCalledOnce();
    expect(onOpen).not.toHaveBeenCalled();
    expect(getMemories).not.toHaveBeenCalled();
  });
});
