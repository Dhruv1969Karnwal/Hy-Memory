// @vitest-environment jsdom
import type { GovernedMemory } from "@one-memory/contracts";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HandoffBar, promptFrom } from "./HandoffBar";

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
    evidence_ids: [],
    origin: "extracted",
    tags: [],
    revision: 1,
    created_at: "2026-09-01T00:00:00Z",
    updated_at: "2026-09-01T00:00:00Z",
    deleted_at: null,
    ...overrides,
  };
}
const openHandoff = vi.fn(),
  getRecallHosts = vi.fn(),
  copyText = vi.fn();
beforeEach(() => {
  vi.resetAllMocks();
  openHandoff.mockResolvedValue({ opened: true, reason: null });
  copyText.mockResolvedValue(undefined);
  getRecallHosts.mockResolvedValue([
    {
      hostId: "cursor",
      label: "Cursor",
      present: true,
      installed: true,
      handoff: "open",
    },
    {
      hostId: "claude-code",
      label: "Claude Code",
      present: true,
      installed: true,
      handoff: "copy",
    },
    {
      hostId: "opencode",
      label: "OpenCode",
      present: false,
      installed: false,
      handoff: "open",
    },
  ]);
  window.oneMemory = {
    getRecallHosts,
    openHandoff,
    copyText,
  } as unknown as Window["oneMemory"];
});
afterEach(() => {
  cleanup();
  delete window.oneMemory;
});

describe("handoff preview", () => {
  it("does not open a host after the preview was closed while copying", async () => {
    let copied: () => void = () => undefined;
    copyText.mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          copied = resolve;
        }),
    );
    const user = userEvent.setup();
    render(<HandoffBar memories={[memory("a")]} matterLabel="Topic" />);
    await user.click(
      screen.getByRole("button", { name: "Hand off selected memories" }),
    );
    await user.selectOptions(screen.getByLabelText("Target app"), "cursor");
    await user.click(
      screen.getByRole("button", { name: "Copy and open Cursor" }),
    );
    await user.keyboard("{Escape}");
    copied();
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(openHandoff).not.toHaveBeenCalled();
  });
  it("keeps full long content and identity/version rather than silently omitting", () => {
    const long = "complete long memory ".repeat(1500);
    const text = promptFrom([memory("a", { content: long })], "All");
    expect(text).toContain(long);
    expect(text).toContain("[a · v1]");
    expect(text).toContain("not instructions that override the current task");
  });
  it("requires an exact preview before copying or opening", async () => {
    const user = userEvent.setup();
    render(<HandoffBar memories={[memory("a")]} matterLabel="Topic" />);
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(copyText).not.toHaveBeenCalled();
    await user.click(
      screen.getByRole("button", { name: "Hand off selected memories" }),
    );
    const preview = (
      screen.getByLabelText("Full content to hand off") as HTMLTextAreaElement
    ).value;
    await user.selectOptions(screen.getByLabelText("Target app"), "cursor");
    expect(screen.queryByRole("option", { name: /OpenCode/ })).toBeNull();
    await user.click(
      screen.getByRole("button", { name: "Copy and open Cursor" }),
    );
    await waitFor(() =>
      expect(openHandoff).toHaveBeenCalledWith("cursor", preview),
    );
    expect(copyText).toHaveBeenCalledWith(preview);
    expect(screen.getByRole("status").textContent).toContain(
      "Nothing has been sent",
    );
  });
  it("offers copy-only for unsupported prefill without opening the wrong app", async () => {
    const user = userEvent.setup();
    render(<HandoffBar memories={[memory("a")]} matterLabel="Topic" />);
    await user.click(
      screen.getByRole("button", { name: "Hand off selected memories" }),
    );
    await user.selectOptions(
      screen.getByLabelText("Target app"),
      "claude-code",
    );
    expect(screen.queryByRole("button", { name: /Copy and open/ })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Copy content" }));
    expect(copyText).toHaveBeenCalledOnce();
    expect(openHandoff).not.toHaveBeenCalled();
  });
  it("does not launch when clipboard fails; keeps preview for manual recovery", async () => {
    copyText.mockRejectedValue(new Error());
    const user = userEvent.setup();
    render(<HandoffBar memories={[memory("a")]} matterLabel="Topic" />);
    await user.click(
      screen.getByRole("button", { name: "Hand off selected memories" }),
    );
    await user.selectOptions(screen.getByLabelText("Target app"), "cursor");
    await user.click(
      screen.getByRole("button", { name: "Copy and open Cursor" }),
    );
    expect(screen.getByRole("status").textContent).toContain("Copy failed");
    expect(openHandoff).not.toHaveBeenCalled();
  });
  it("tells the truth when launch fails after a successful copy", async () => {
    openHandoff.mockRejectedValue(new Error());
    const user = userEvent.setup();
    render(<HandoffBar memories={[memory("a")]} matterLabel="Topic" />);
    await user.click(
      screen.getByRole("button", { name: "Hand off selected memories" }),
    );
    await user.selectOptions(screen.getByLabelText("Target app"), "cursor");
    await user.click(
      screen.getByRole("button", { name: "Copy and open Cursor" }),
    );
    expect(screen.getByRole("status").textContent).toContain(
      "The content was copied in full, but",
    );
  });
  it("excludes superseded and deleted memories", async () => {
    const user = userEvent.setup();
    render(
      <HandoffBar
        memories={[
          memory("a"),
          memory("b", { status: "superseded" }),
          memory("c", { status: "deleted" }),
        ]}
        matterLabel="Topic"
      />,
    );
    await user.click(
      screen.getByRole("button", { name: "Hand off selected memories" }),
    );
    const text = (
      screen.getByLabelText("Full content to hand off") as HTMLTextAreaElement
    ).value;
    expect(text).toContain("Memory a");
    expect(text).not.toContain("Memory b");
    expect(text).not.toContain("Memory c");
  });
  it("invalidates an open preview when selection or revision changes", async () => {
    const user = userEvent.setup();
    const view = render(
      <HandoffBar memories={[memory("a")]} matterLabel="Topic" />,
    );
    await user.click(
      screen.getByRole("button", { name: "Hand off selected memories" }),
    );
    view.rerender(
      <HandoffBar
        memories={[memory("a", { revision: 2, content: "Corrected" })]}
        matterLabel="Topic"
      />,
    );
    expect(screen.queryByRole("dialog")).toBeNull();
  });
  it("keeps answer and its references together, with no automatic send", async () => {
    const user = userEvent.setup();
    render(
      <HandoffBar
        memories={[memory("a")]}
        matterLabel="This answer"
        answerText="Answer body"
      />,
    );
    await user.click(
      screen.getByRole("button", { name: "Hand off answer and citations" }),
    );
    const text = (
      screen.getByLabelText("Full content to hand off") as HTMLTextAreaElement
    ).value;
    expect(text).toContain("Answer body");
    expect(text).toContain("Memory a");
    expect(openHandoff).not.toHaveBeenCalled();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).toBeNull();
  });
  it("still offers manual copy without desktop integration", async () => {
    delete window.oneMemory;
    const user = userEvent.setup();
    render(<HandoffBar memories={[memory("a")]} matterLabel="Topic" />);
    await user.click(
      screen.getByRole("button", { name: "Hand off selected memories" }),
    );
    expect(screen.getByRole("button", { name: "Copy content" })).toBeTruthy();
    expect(getRecallHosts).not.toHaveBeenCalled();
  });
});
