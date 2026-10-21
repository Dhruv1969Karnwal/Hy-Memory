// @vitest-environment jsdom
import type { ImportJob, SourceResponse } from "@one-memory/contracts";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { FirstImportReview } from "./FirstImportReview";

const api = vi.hoisted(() => ({ receipt: vi.fn(), conversation: vi.fn() }));
vi.mock("./api", () => ({
  getSourceImportReceipt: api.receipt,
  getConversation: api.conversation,
}));
const source = (id: string): SourceResponse => ({
  source_id: id,
  display_name: id,
  authorized: true,
  available: true,
  detected: true,
  storage_kinds: ["jsonl"],
  candidate_kind: "default",
  candidate_path: `/isolated/${id}`,
  root_label: id,
  root_id: id,
  grant_revision: 1,
  imported_messages: 7,
  imported_conversations: 1,
  sync_enabled: false,
  sync_completeness: "additions_only",
  capabilities: {
    historical_import: "implemented",
    incremental_import: "implemented",
    live_sync: "implemented",
    recall: "implemented",
    installer: "implemented",
    verification: "fixture_only",
  },
});
const job = (
  id: string,
  status: ImportJob["status"] = "running",
  operation: ImportJob["operation"] = "history",
): ImportJob => ({
  job_id: id,
  source_id: id,
  user_id: "local",
  operation,
  status,
  phase: "read",
  checkpoint: null,
  discovered: 0,
  discovered_targets: 0,
  emitted: 7,
  stored: 7,
  revised: 0,
  merged: 0,
  skipped: 0,
  errors: 0,
  created_at: "2026-09-19T00:00:00Z",
  updated_at: "2026-09-19T00:00:01Z",
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.resetAllMocks();
});
beforeEach(() =>
  api.receipt.mockImplementation(async (source_id: string) => ({
    source_id,
    query_examples: [],
    examples: [],
  })),
);

it("uses step2 frame and withholds unqualified pure-user feed from old receipts", async () => {
  const view = render(
    <FirstImportReview
      sourceIds={["a"]}
      sources={[source("a")]}
      jobs={[job("a")]}
      onContinue={vi.fn()}
    />,
  );
  expect(view.container.querySelector(".onboarding-r6")).toBeTruthy();
  expect(
    screen
      .getByRole("navigation", { name: "First-time setup progress" })
      .querySelector('[aria-current="step"]')?.textContent,
  ).toContain("Connect apps");
  expect(
    await screen.findByText(
      "The background keeps importing; you can safely move to the next step",
    ),
  ).toBeTruthy();
  expect(screen.queryByText("Import continuing...")).toBeNull();
  expect(api.receipt).not.toHaveBeenCalled();
  expect(screen.getByText("Saved 7 messages")).toBeTruthy();
  expect(
    screen.getByRole("heading", { name: "Importing your conversations" }),
  ).toBeTruthy();
  expect(
    screen.getByText(
      "The background keeps importing; you can safely move to the next step",
    ),
  ).toBeTruthy();
});

it("does not render saved excerpts or a transcript while import is still running", () => {
  api.receipt.mockClear();
  api.conversation.mockClear();
  render(
    <FirstImportReview
      sourceIds={["a"]}
      sources={[source("a")]}
      jobs={[job("a")]}
      onContinue={vi.fn()}
    />,
  );
  expect(
    screen.queryByText("Questions you can ask before the import finishes"),
  ).toBeNull();
  expect(screen.queryByRole("button", { name: "View source text" })).toBeNull();
  expect(screen.queryByRole("dialog")).toBeNull();
  expect(api.receipt).not.toHaveBeenCalled();
  expect(api.conversation).not.toHaveBeenCalled();
});

it("pauses only accepted history jobs, retains partial receipts and never touches sync or another source", async () => {
  const change = vi.fn().mockImplementation(async (id: string) => {
    if (id === "b") throw Error("secret");
    return { ...job(id), status: "paused", updated_at: "2026-09-19T00:00:02Z" };
  });
  render(
    <FirstImportReview
      sourceIds={["a", "b"]}
      sources={[source("a"), source("b")]}
      jobs={[
        job("a"),
        job("b"),
        { ...job("sync", "running", "sync"), source_id: "a" },
        job("outside"),
      ]}
      onChangeJob={change}
      onContinue={vi.fn()}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Pause import" }));
  await waitFor(() => expect(change).toHaveBeenCalledTimes(2));
  expect(change.mock.calls).toEqual([
    ["a", "pause"],
    ["b", "pause"],
  ]);
  expect(await screen.findByText(/Confirmed 1, 1 unconfirmed/)).toBeTruthy();
  expect(screen.queryByText(/secret/)).toBeNull();
});

it("stops remaining batch commands after cancellation without pretending the in-flight command was undone", async () => {
  let resolve!: (v: ImportJob) => void;
  const change = vi.fn().mockReturnValue(
    new Promise((r) => {
      resolve = r;
    }),
  );
  render(
    <FirstImportReview
      sourceIds={["a", "b"]}
      sources={[source("a"), source("b")]}
      jobs={[job("a"), job("b")]}
      onChangeJob={change}
      onContinue={vi.fn()}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Pause import" }));
  fireEvent.click(screen.getByRole("button", { name: "Stop later actions" }));
  await act(async () => resolve({ ...job("a"), status: "paused" }));
  expect(change).toHaveBeenCalledOnce();
  expect(screen.getByText(/Later actions stopped/)).toBeTruthy();
});

it("does not offer resume for a cancelled job and still permits safe pause when stale", () => {
  const change = vi.fn();
  const view = render(
    <FirstImportReview
      sourceIds={["a"]}
      sources={[source("a")]}
      jobs={[job("a", "cancelled")]}
      onChangeJob={change}
      onContinue={vi.fn()}
    />,
  );
  expect(screen.queryByRole("button", { name: "Resume import" })).toBeNull();
  view.rerender(
    <FirstImportReview
      sourceIds={["a"]}
      sources={[source("a")]}
      jobs={[job("a")]}
      sourceSnapshot={{ confirmedAt: null, stale: true }}
      onChangeJob={change}
      onContinue={vi.fn()}
    />,
  );
  expect(
    screen.getByRole<HTMLButtonElement>("button", { name: "Pause import" })
      .disabled,
  ).toBe(false);
  expect(change).not.toHaveBeenCalled();
});

it("does not present unwired connection choices as actionable", () => {
  render(
    <FirstImportReview
      sourceIds={["a"]}
      sources={[source("a")]}
      jobs={[job("a")]}
      onContinue={vi.fn()}
    />,
  );
  for (const input of screen.getAllByRole<HTMLInputElement>("checkbox"))
    expect(input.disabled).toBe(true);
  expect(screen.getByText(/Apps cannot be connected right now/)).toBeTruthy();
});

it("emits controlled connection drafts only and passes continuation without completing onboarding", () => {
  const draft = { hostSourceIds: [] },
    update = vi.fn(),
    next = vi.fn();
  render(
    <FirstImportReview
      sourceIds={["a"]}
      sources={[source("a")]}
      jobs={[job("a")]}
      connectionDraft={draft}
      onConnectionDraftChange={update}
      connectionExecutionAvailable
      onContinue={next}
    />,
  );
  fireEvent.click(
    screen.getByRole("checkbox", { name: "Connect apps in one click" }),
  );
  expect(update).toHaveBeenCalledWith({
    hostSourceIds: ["a"],
  });
  expect(
    screen.getByRole<HTMLInputElement>("checkbox", {
      name: "Connect apps in one click",
    }).checked,
  ).toBe(false);
  fireEvent.click(
    screen.getByRole("button", { name: "Configure a model to build memory" }),
  );
  expect(next).toHaveBeenCalledWith("settings");
  expect(api.conversation).not.toHaveBeenCalled();
});

it("keeps profile-based Harness and paused OpenClaw out of the atomic one-click host batch", () => {
  const update = vi.fn();
  render(
    <FirstImportReview
      sourceIds={["a", "deepseek-harness", "openclaw"]}
      sources={[source("a"), source("deepseek-harness"), source("openclaw")]}
      jobs={[job("a"), job("deepseek-harness"), job("openclaw")]}
      connectionDraft={{ hostSourceIds: [] }}
      onConnectionDraftChange={update}
      connectionExecutionAvailable
      onContinue={vi.fn()}
    />,
  );
  fireEvent.click(
    screen.getByRole("checkbox", { name: "Connect apps in one click" }),
  );
  expect(update).toHaveBeenCalledWith({
    hostSourceIds: ["a"],
  });
});

it("shows a disabled host option for a noncanonical test instance", () => {
  const update = vi.fn();
  render(
    <FirstImportReview
      sourceIds={["a"]}
      sources={[source("a")]}
      jobs={[job("a")]}
      connectionDraft={{ hostSourceIds: [] }}
      onConnectionDraftChange={update}
      connectionExecutionAvailable
      hostConnectionAvailable={false}
      connectionNote="Test instance does not connect apps"
      onContinue={vi.fn()}
    />,
  );

  const host = screen.getByRole<HTMLInputElement>("checkbox", {
    name: "Test instance does not connect apps",
  });
  expect(host.disabled).toBe(true);
  expect(host.checked).toBe(false);
  expect(
    screen.getAllByText("Test instance does not connect apps").length,
  ).toBeGreaterThan(0);
});

it("does not adopt later history jobs into the initially accepted batch", async () => {
  const change = vi.fn().mockImplementation(async (id: string) => ({
    ...job(id),
    status: "paused",
  }));
  const props = {
    sourceIds: ["a"],
    sources: [source("a")],
    onChangeJob: change,
    onContinue: vi.fn(),
  };
  const view = render(<FirstImportReview {...props} jobs={[job("a")]} />);
  view.rerender(
    <FirstImportReview
      {...props}
      jobs={[
        job("a"),
        { ...job("later"), source_id: "a", created_at: "2026-09-19T00:05:00Z" },
      ]}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Pause import" }));
  await waitFor(() => expect(change).toHaveBeenCalledOnce());
  expect(change).toHaveBeenCalledWith("a", "pause");
});

it("stops the remainder when source authority changes during a pending command", async () => {
  let resolve!: (value: ImportJob) => void;
  const change = vi.fn().mockReturnValue(
    new Promise((r) => {
      resolve = r;
    }),
  );
  const props = {
    sourceIds: ["a", "b"],
    jobs: [job("a"), job("b")],
    onChangeJob: change,
    onContinue: vi.fn(),
  };
  const view = render(
    <FirstImportReview {...props} sources={[source("a"), source("b")]} />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Pause import" }));
  view.rerender(
    <FirstImportReview
      {...props}
      sources={[{ ...source("a"), authorized: false }, source("b")]}
    />,
  );
  await act(async () => resolve({ ...job("a"), status: "paused" }));
  expect(change).toHaveBeenCalledOnce();
  expect(screen.getByText(/Confirmed 0, 1 unconfirmed/)).toBeTruthy();
});

it("does not accept a command receipt for another task", async () => {
  const change = vi
    .fn()
    .mockResolvedValue({ ...job("wrong"), status: "paused" });
  render(
    <FirstImportReview
      sourceIds={["a"]}
      sources={[source("a")]}
      jobs={[job("a")]}
      onChangeJob={change}
      onContinue={vi.fn()}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Pause import" }));
  expect(await screen.findByText(/Confirmed 0, 1 unconfirmed/)).toBeTruthy();
  expect(screen.getByRole("button", { name: "Pause import" })).toBeTruthy();
});

it("uses user-facing connection confirmation copy without implementation narration", () => {
  render(
    <FirstImportReview
      sourceIds={["a"]}
      sources={[source("a")]}
      jobs={[job("a")]}
      connectionDraft={{ hostSourceIds: [] }}
      onConnectionDraftChange={vi.fn()}
      connectionExecutionAvailable
      onContinue={vi.fn()}
    />,
  );
  expect(
    screen.getByRole("complementary", { name: "Connect apps" }),
  ).toBeTruthy();
  expect(
    screen.getByText(
      "After you press the bottom button, the background connects each app separately; one app failing does not affect the others.",
    ),
  ).toBeTruthy();
  expect(
    screen.queryByText(
      /Select draft|Per-app receipts|This page task controls are not connected/,
    ),
  ).toBeNull();
  expect(screen.getByText(/does not grant app read access/)).toBeTruthy();
  expect(
    screen.getByText(
      /Memory generation starts only after a model is configured/,
    ),
  ).toBeTruthy();
});

it.each(["paused", "failed", "cancelled", "interrupted"] as const)(
  "keeps the stable import heading while %s",
  (status) => {
    render(
      <FirstImportReview
        sourceIds={["a"]}
        sources={[source("a")]}
        jobs={[job("a", status)]}
        onContinue={vi.fn()}
      />,
    );
    expect(
      screen.queryByText(
        "The import continues in the background. No need to wait.",
      ),
    ).toBeNull();
    expect(
      screen.getByRole("heading", {
        name: "Your conversations are coming in; memory starts here",
      }),
    ).toBeTruthy();
    expect(
      screen.getByText(/Saved conversations are still here/, {
        selector: ".first-import-r6-footer-note p",
      }),
    ).toBeTruthy();
  },
);

it("does not turn a failed import into a claim that there was nothing to save", () => {
  const failed = {
    ...job("a"),
    status: "failed" as const,
    discovered: 0,
    stored: 0,
    errors: 1,
  };
  render(
    <FirstImportReview
      sourceIds={["a"]}
      sources={[source("a")]}
      jobs={[failed]}
      onContinue={vi.fn()}
    />,
  );
  expect(screen.queryByText("No history conversations to save")).toBeNull();
  expect(screen.getByText(/Import.*problem/)).toBeTruthy();
  expect(screen.getByText(/read error/)).toBeTruthy();
});

it.each(["stale", "revoked", "missing"])(
  "keeps the stable import heading for an unverified %s batch",
  (kind) => {
    render(
      <FirstImportReview
        sourceIds={["a"]}
        sources={[{ ...source("a"), authorized: kind !== "revoked" }]}
        jobs={kind === "missing" ? [] : [job("a")]}
        sourceSnapshot={{ confirmedAt: null, stale: kind === "stale" }}
        onContinue={vi.fn()}
      />,
    );
    if (kind === "missing")
      expect(
        screen.getByText(
          "The import continues in the background. No need to wait.",
        ),
      ).toBeTruthy();
    else
      expect(
        screen.queryByText(
          "The import continues in the background. No need to wait.",
        ),
      ).toBeNull();
    expect(
      screen.getByRole("heading", {
        name: "Your conversations are coming in; memory starts here",
      }),
    ).toBeTruthy();
    expect(api.conversation).not.toHaveBeenCalled();
  },
);

it("does not open a transcript when the saved count changes", () => {
  api.receipt.mockClear();
  api.conversation.mockClear();
  const view = render(
    <FirstImportReview
      sourceIds={["a"]}
      sources={[source("a")]}
      jobs={[job("a")]}
      onContinue={vi.fn()}
    />,
  );
  view.rerender(
    <FirstImportReview
      sourceIds={["a"]}
      sources={[source("a")]}
      jobs={[{ ...job("a"), stored: 9 }]}
      onContinue={vi.fn()}
    />,
  );
  expect(screen.getByText("Saved 9 messages")).toBeTruthy();
  expect(screen.queryByRole("dialog")).toBeNull();
  expect(screen.queryByRole("button", { name: "View source text" })).toBeNull();
  expect(api.receipt).not.toHaveBeenCalled();
  expect(api.conversation).not.toHaveBeenCalled();
});
