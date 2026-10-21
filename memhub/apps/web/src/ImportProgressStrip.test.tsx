// @vitest-environment jsdom

import type { ImportJob, SourceResponse } from "@one-memory/contracts";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ImportProgressStrip } from "./ImportProgressStrip";

function job(overrides: Partial<ImportJob> = {}): ImportJob {
  return {
    job_id: "job-1",
    user_id: "local",
    source_id: "cursor",
    status: "running",
    phase: "emit",
    checkpoint: null,
    discovered: 1240,
    discovered_targets: 18,
    emitted: 300,
    stored: 300,
    revised: 0,
    merged: 0,
    skipped: 0,
    errors: 0,
    created_at: "2026-09-04T00:00:01Z",
    updated_at: "2026-09-04T00:00:02Z",
    ...overrides,
  };
}

function source(
  sourceId: string,
  displayName: string,
  overrides: Partial<SourceResponse> = {},
): SourceResponse {
  return {
    source_id: sourceId,
    display_name: displayName,
    storage_kinds: ["jsonl"],
    available: true,
    authorized: true,
    detected: true,
    root_label: null,
    candidate_path: `~/.${sourceId}`,
    candidate_kind: "default",
    imported_messages: 0,
    imported_conversations: 0,
    capabilities: {
      historical_import: "implemented",
      incremental_import: "implemented",
      live_sync: "not_implemented",
      recall: "not_implemented",
      installer: "not_implemented",
      verification: "real_replayed",
    },
    sync_enabled: false,
    sync_completeness: "additions_only",
    ...overrides,
  };
}

const sources = [source("cursor", "Cursor"), source("claude", "Claude Code")];
const confirmedAt = "2026-09-14T02:20:30.000Z";
const staleSnapshot = { confirmedAt, stale: true };

function deferred() {
  let resolve!: (value?: unknown) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<unknown>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

describe("ImportProgressStrip", () => {
  afterEach(cleanup);

  it("keeps daily sync silent in the user-facing shell", () => {
    const { container } = render(
      <ImportProgressStrip
        userFacing
        jobs={[job({ operation: "sync" })]}
        sources={sources}
        backfilling={false}
        onOpenSources={vi.fn()}
      />,
    );
    expect(container.textContent).toBe("");
  });

  it("shows one plain sentence for first history import without details", () => {
    render(
      <ImportProgressStrip
        userFacing
        jobs={[job()]}
        sources={sources}
        backfilling={false}
        onOpenSources={vi.fn()}
      />,
    );
    expect(screen.getByRole("status").textContent).toContain(
      "Importing your conversation history",
    );
    expect(screen.queryByText("View progress and details")).toBeNull();
  });

  it("import jobs and backfill preparation alone do not assert active generation", () => {
    const { container, rerender } = render(
      <ImportProgressStrip
        jobs={[job()]}
        sources={sources}
        backfilling
        onOpenSources={vi.fn()}
      />,
    );
    expect(
      container.querySelector(".import-strip-headline")?.textContent,
    ).toContain("Importing conversations");
    expect(container.textContent).not.toContain(
      "more memories will appear soon",
    );
    rerender(
      <ImportProgressStrip
        jobs={[job({ status: "completed" })]}
        sources={sources}
        backfilling
        onOpenSources={vi.fn()}
      />,
    );
    expect(
      container.querySelector(".import-strip-headline")?.textContent,
    ).toContain("Preparing to organize");
    expect(container.textContent).not.toContain("Memories are taking shape");
  });

  it.each(["failed", "processing"] as const)(
    "shows %s generation separately from active imports and preserves every failure",
    (phase) => {
      const { container } = render(
        <ImportProgressStrip
          jobs={[job()]}
          sources={sources}
          backfilling
          onOpenSources={vi.fn()}
          arrival={{
            importing: true,
            importedMessages: 7865,
            importedConversations: 30,
            memoryCount: 0,
            phase,
            blockedStages: [
              {
                kind: "extract",
                reason: "hy_memory_model_request_failed",
                count: 1,
              },
              {
                kind: "extract",
                reason: "hy_memory_input_exceeds_window",
                count: 5,
              },
            ],
          }}
        />,
      );
      const headline = container.querySelector(
        ".import-strip-headline",
      )?.textContent;
      expect(headline).toContain(
        phase === "processing"
          ? "Memories are taking shape"
          : "Waiting to resume",
      );
      expect(headline).toContain("0 memories");
      expect(container.querySelectorAll(".import-strip-pulse")).toHaveLength(
        phase === "processing" ? 1 : 0,
      );
      expect(container.querySelector("details")?.textContent).toContain(
        "Importing Cursor",
      );
      expect(container.querySelector("details")?.textContent).toContain(
        "The model request did not succeed",
      );
      expect(container.querySelector("details")?.textContent).toContain(
        "This conversation contains oversized source text; not organized yet, source text kept",
      );
      // Named in the reader's language, never in the pipeline's.
      expect(container.querySelector("details")?.textContent).not.toContain(
        "hy_memory_",
      );
      expect(screen.queryByRole("alert")).toBeNull();
    },
  );

  it("shows one calm outer band and leaves partial errors in collapsed details", () => {
    const { container } = render(
      <ImportProgressStrip
        jobs={[job({ status: "completed", errors: 2 })]}
        sources={sources}
        backfilling={false}
        onOpenSources={vi.fn()}
      />,
    );
    expect(
      container.querySelector(".import-strip-headline")?.textContent,
    ).toContain("Waiting to resume");
    expect(
      container.querySelector(".import-strip-headline")?.textContent,
    ).not.toMatch(/failed|issues/i);
    expect(container.querySelector("details")?.open).toBe(false);
    expect(container.querySelector("details")?.textContent).toContain(
      "2 issues",
    );
    expect(container.querySelector(".import-strip-pulse")).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("shows active copy once while preserving separate receipts", () => {
    const { container } = render(
      <ImportProgressStrip
        jobs={[
          job(),
          job({
            job_id: "other",
            source_id: "claude",
            status: "completed",
            errors: 2,
          }),
        ]}
        sources={sources}
        arrival={{
          importing: true,
          importedMessages: 300,
          importedConversations: 2,
          phase: "processing",
        }}
        backfilling
        onOpenSources={vi.fn()}
      />,
    );
    expect(screen.getAllByText("Memories are taking shape")).toHaveLength(1);
    expect(
      screen.getByText(
        "Recent conversations are being organized first; more memories will appear soon",
      ),
    ).toBeTruthy();
    expect(container.querySelectorAll(".import-strip-pulse")).toHaveLength(1);
    expect(container.querySelector("details")?.textContent).toContain(
      "2 issues",
    );
    expect(
      container.querySelector(".import-strip-headline")?.textContent,
    ).not.toContain("issues");
  });

  it.each(["model-required", "failed"] as const)(
    "does not override critical %s with running import/backfill",
    (phase) => {
      const { container } = render(
        <ImportProgressStrip
          jobs={[job()]}
          sources={sources}
          backfilling
          onOpenSources={vi.fn()}
          arrival={{
            importing: true,
            importedMessages: 300,
            importedConversations: 2,
            phase,
          }}
          onConfigureModel={vi.fn()}
        />,
      );
      expect(
        container.querySelector(".import-strip-headline")?.textContent,
      ).not.toContain("taking shape");
      expect(container.querySelector(".import-strip-pulse")).toBeNull();
      expect(
        screen.getByRole("button", {
          name:
            phase === "model-required" ? "Set up model" : "Continue organizing",
        }),
      ).toBeTruthy();
    },
  );

  it("distinguishes zero-addition sync from completed history and preserves source totals", () => {
    render(
      <ImportProgressStrip
        jobs={[
          job({
            job_id: "history",
            operation: "history",
            status: "completed",
            stored: 81614,
          }),
          job({
            job_id: "sync",
            operation: "sync",
            stored: 0,
            merged: 81614,
            discovered: 0,
            created_at: "2026-09-14T00:00:00Z",
          }),
        ]}
        sources={[
          source("cursor", "Cursor", {
            imported_messages: 81614,
            imported_conversations: 1761,
          }),
        ]}
        backfilling={false}
        onOpenSources={vi.fn()}
      />,
    );
    const text = screen.getByRole("status").textContent;
    expect(text).toContain("Checking Cursor for new conversations");
    expect(text).toContain("0 new originals this run");
    expect(text).toContain("Total saved: Cursor 81,614 originals");
    expect(text).toContain("1,761 conversations");
    expect(text).toContain("81,614 already existed");
    expect(text).not.toMatch(
      /Importing|0 originals saved|81,614 new originals/,
    );
  });

  it("keeps mixed history and sync counters separate", () => {
    render(
      <ImportProgressStrip
        jobs={[
          job({ operation: "sync", stored: 0 }),
          job({
            job_id: "history",
            source_id: "claude",
            operation: "history",
            stored: 7,
          }),
        ]}
        sources={sources}
        backfilling={false}
        onOpenSources={vi.fn()}
      />,
    );
    const text = screen.getByRole("status").textContent;
    expect(text).toContain("Importing Claude Code history");
    expect(text).toContain("Checking Cursor for new conversations");
    expect(text).toContain("0 new originals this run");
    expect(text).toContain("7 originals saved");
    expect(text).not.toContain("7 new originals");
  });

  it.each([
    ["pending", "Cursor new-item sync waiting to check"],
    ["paused", "Cursor new-item sync paused"],
    ["interrupted", "Cursor new-item sync interrupted"],
    ["failed", "Cursor new-item sync failed"],
    [
      "completed",
      "Cursor this new-item check finished, but read issues remain",
    ],
  ] as const)(
    "names sync %s without implying unread history",
    (status, label) => {
      render(
        <ImportProgressStrip
          jobs={[
            job({
              operation: "sync",
              status,
              stored: 0,
              errors: status === "completed" ? 2 : 0,
            }),
          ]}
          sources={[source("cursor", "Cursor", { imported_messages: 81614 })]}
          backfilling={false}
          onOpenSources={vi.fn()}
        />,
      );
      const text = screen.getByRole("status").textContent;
      expect(text).toContain(label);
      expect(text).toContain("0 new originals this run");
      expect(text).toContain("81,614 originals");
      expect(text).not.toMatch(
        /history import not finished|failed to read through|Importing/,
      );
    },
  );

  it.each(["pending", "paused", "interrupted", "cancelled"] as const)(
    "does not animate %s-only jobs or promise background processing",
    (status) => {
      const { container } = render(
        <ImportProgressStrip
          jobs={[job({ status })]}
          sources={sources}
          backfilling={false}
          onOpenSources={vi.fn()}
        />,
      );
      expect(
        container.querySelector(".import-strip-headline")?.textContent,
      ).toContain("Waiting to resume");
      expect(container.querySelector(".import-strip-pulse")).toBeNull();
      expect(
        screen.queryByText(
          "Recent conversations are being organized first; more memories will appear soon",
        ),
      ).toBeNull();
      expect(
        screen.queryByText(
          "Large backlog; organizing in background batches to preserve quality",
        ),
      ).toBeNull();
    },
  );

  it("opens all supplied exact issues and task receipts without triggering processing", async () => {
    const open = vi.fn();
    const { container } = render(
      <ImportProgressStrip
        jobs={[job({ status: "completed", errors: 2 })]}
        sources={sources}
        backfilling={false}
        onOpenSources={open}
        issues={[
          {
            issue_id: 1,
            job_id: "job-1",
            kind: "error",
            reason: "source_permission_changed",
            conversation_id: null,
            created_at: confirmedAt,
          },
        ]}
      />,
    );
    const details = container.querySelector("details");
    expect(details?.open).toBe(false);
    await userEvent.click(screen.getByText("View progress and details"));
    expect(details?.open).toBe(true);
    // Said in the reader's language. A raw reason code, a job id or an ISO
    // timestamp in here is what this test exists to prevent.
    expect(details?.textContent).toContain(
      "The read authorization for this source changed",
    );
    expect(details?.textContent).not.toContain("source_permission_changed");
    expect(details?.textContent).not.toContain(confirmedAt);
    expect(details?.textContent).not.toContain("job-1");
    expect(
      screen.getByRole("list", { name: "Full job receipts" }),
    ).toBeTruthy();
    expect(details?.textContent).toContain("Completed");
    expect(details?.textContent).not.toContain("completed");
    expect(open).not.toHaveBeenCalled();
  });

  it("makes authentication failure actionable when no generation task is running", async () => {
    const configure = vi.fn();
    const { container } = render(
      <ImportProgressStrip
        jobs={[job()]}
        sources={sources}
        backfilling
        onOpenSources={vi.fn()}
        onConfigureModel={configure}
        arrival={{
          importing: true,
          importedMessages: 300,
          importedConversations: 2,
          phase: "failed",
          blockedStages: [
            {
              kind: "extract",
              reason: "hy_memory_model_authentication_failed",
              count: 1,
            },
          ],
        }}
      />,
    );
    expect(
      container.querySelector(".import-strip-headline")?.textContent,
    ).toContain("Waiting to resume");
    expect(container.querySelector(".import-strip-pulse")).toBeNull();
    expect(container.querySelector("details")?.textContent).toContain(
      "Model authentication failed. Update the key, then",
    );
    await userEvent.click(screen.getByRole("button", { name: "Set up model" }));
    expect(configure).toHaveBeenCalledOnce();
  });

  it("keeps stale sync counts as a last receipt without claiming a live check", () => {
    const { container } = render(
      <ImportProgressStrip
        jobs={[job({ operation: "sync", stored: 0 })]}
        sources={[source("cursor", "Cursor", { imported_messages: 81614 })]}
        backfilling={false}
        sourceSnapshot={staleSnapshot}
        onOpenSources={vi.fn()}
      />,
    );
    expect(screen.getByRole("status").textContent).toContain("New-item sync");
    expect(screen.getByRole("status").textContent).toContain(
      "0 new originals this run",
    );
    expect(screen.getByRole("status").textContent).toContain(
      "81,614 originals",
    );
    expect(screen.getByRole("status").textContent).not.toMatch(
      /Checking|Importing|organiz/,
    );
    expect(container.querySelector(".import-strip-pulse")).toBeNull();
  });

  it.each(["paused", "interrupted"] as const)(
    "keeps %s imports visible with recovery",
    (status) => {
      render(
        <ImportProgressStrip
          jobs={[job({ status })]}
          sources={sources}
          backfilling={false}
          onOpenSources={vi.fn()}
        />,
      );
      expect(screen.getByRole("status").textContent).toContain(
        "history import not finished",
      );
      expect(screen.getByRole("button", { name: "View sources" })).toBeTruthy();
    },
  );

  it("distinguishes discovered messages from stored originals and skipped records", () => {
    render(
      <ImportProgressStrip
        jobs={[job({ stored: 280, merged: 10, skipped: 8, errors: 2 })]}
        sources={sources}
        backfilling={false}
        onOpenSources={vi.fn()}
      />,
    );
    expect(screen.getByRole("status").textContent).toContain(
      "280 originals saved",
    );
    expect(screen.getByRole("status").textContent).toContain(
      "10 already existed",
    );
    expect(screen.getByRole("status").textContent).toContain("8 skipped");
    expect(screen.getByRole("status").textContent).toContain("2 issues");
  });

  it("reports the real source names while importing, counts are the sum across sources", () => {
    render(
      <ImportProgressStrip
        jobs={[
          job(),
          job({ job_id: "job-2", source_id: "claude", discovered: 60 }),
        ]}
        sources={sources}
        backfilling={false}
        onOpenSources={vi.fn()}
      />,
    );

    const strip = screen.getByRole("status");
    expect(strip.textContent).toContain("Cursor");
    expect(strip.textContent).toContain("Claude Code");
    expect(strip.textContent).toContain("1,300");
  });

  it("does not include completed sources in the active group but preserves their receipts", () => {
    render(
      <ImportProgressStrip
        jobs={[
          job(),
          job({ job_id: "job-2", source_id: "claude", status: "completed" }),
        ]}
        sources={sources}
        backfilling={false}
        onOpenSources={vi.fn()}
      />,
    );

    const strip = screen.getByRole("status");
    expect(strip.textContent).toContain("Cursor");
    expect(
      strip.querySelector(".import-strip-operation")?.textContent,
    ).not.toContain("Claude Code");
    expect(strip.querySelector("details")?.textContent).toContain(
      "Claude Code",
    );
  });

  it("the whole strip disappears when no jobs are running", () => {
    render(
      <ImportProgressStrip
        jobs={[job({ status: "completed" })]}
        sources={sources}
        backfilling={false}
        onOpenSources={vi.fn()}
      />,
    );

    expect(screen.queryByRole("status")).toBeNull();
  });

  it("distinguishes backfill preparation from running generation after import", () => {
    render(
      <ImportProgressStrip
        jobs={[job({ status: "completed" })]}
        sources={sources}
        backfilling
        onOpenSources={vi.fn()}
      />,
    );

    expect(screen.getByRole("status").textContent).toContain(
      "Preparing organizing jobs for authorized conversations",
    );
    expect(screen.getByRole("status").textContent).not.toContain(
      "Memories are taking shape",
    );
  });

  it("a failed source is not pretended to still be running, it is named with a way out", async () => {
    const user = userEvent.setup();
    const onOpenSources = vi.fn();
    render(
      <ImportProgressStrip
        jobs={[job({ status: "failed", errors: 2 })]}
        sources={sources}
        backfilling={false}
        onOpenSources={onOpenSources}
      />,
    );

    const strip = screen.getByRole("status");
    expect(strip.textContent).toContain("Cursor");
    expect(strip.textContent).toContain("failed to read through");

    await user.click(screen.getByRole("button", { name: "View sources" }));
    expect(onOpenSources).toHaveBeenCalledOnce();
  });

  it("for repeated jobs on one source only the latest is used", () => {
    render(
      <ImportProgressStrip
        jobs={[
          job({
            job_id: "old",
            status: "failed",
            updated_at: "2026-09-04T00:00:00Z",
          }),
          job({
            job_id: "new",
            status: "running",
            updated_at: "2026-09-04T00:05:00Z",
          }),
        ]}
        sources={sources}
        backfilling={false}
        onOpenSources={vi.fn()}
      />,
    );

    const strip = screen.getByRole("status");
    expect(strip.textContent).not.toContain("failed to read through");
  });

  it("keeps stale running jobs static with distinct source totals, job counts and confirmed time", () => {
    const { container } = render(
      <ImportProgressStrip
        jobs={[job({ stored: 280, merged: 10, skipped: 8, errors: 2 })]}
        sources={[
          source("cursor", "Cursor", {
            imported_messages: 900,
            imported_conversations: 21,
          }),
        ]}
        backfilling
        sourceSnapshot={staleSnapshot}
        onOpenSources={vi.fn()}
      />,
    );
    const strip = screen.getByRole("status");
    expect(strip.textContent).toContain("status pending verification");
    expect(strip.textContent).toContain("Cursor");
    expect(strip.textContent).toContain("Last confirmed sources");
    expect(strip.textContent).toContain("900 originals");
    expect(strip.textContent).toContain("21 conversations");
    expect(strip.textContent).toContain("Job receipts");
    expect(strip.textContent).toContain("280 originals saved");
    expect(strip.textContent).toContain("10 already existed");
    expect(strip.textContent).toContain("8 skipped");
    expect(strip.textContent).toContain("2 issues");
    expect(strip.textContent).not.toMatch(/Importing|organiz|%/);
    expect(container.querySelector(".import-strip-pulse")).toBeNull();
    expect(container.querySelector("time")?.dateTime).toBe(confirmedAt);
    expect(container.querySelector("time")?.textContent).toBe(
      new Date(confirmedAt).toLocaleString("zh-CN", { hour12: false }),
    );
  });

  it("keeps a stale strip visible even when the last confirmed jobs are complete", () => {
    const { container } = render(
      <ImportProgressStrip
        jobs={[job({ status: "completed" })]}
        sources={sources}
        backfilling={false}
        sourceSnapshot={staleSnapshot}
        onOpenSources={vi.fn()}
      />,
    );
    expect(screen.getByRole("status").textContent).toContain(
      "status pending verification",
    );
    expect(container.querySelector(".import-strip-pulse")).toBeNull();
    expect(screen.getByRole("button", { name: "View sources" })).toBeTruthy();
  });

  it.each([null, "not-a-date"])(
    "does not fabricate confirmed counts or time after an initial failure (%s)",
    (time) => {
      const { container } = render(
        <ImportProgressStrip
          jobs={[]}
          sources={sources}
          backfilling
          sourceSnapshot={{ confirmedAt: time, stale: true }}
          onOpenSources={vi.fn()}
        />,
      );
      const text = screen.getByRole("status").textContent;
      expect(text).toContain("status pending verification");
      expect(text).toContain("No confirmed source status yet");
      expect(text).not.toMatch(
        /0 originals|imported so far|organiz|Invalid Date/,
      );
      expect(container.querySelector("time")).toBeNull();
      expect(container.querySelector(".import-strip-pulse")).toBeNull();
    },
  );

  it.each(["failed", "paused", "interrupted"] as const)(
    "preserves stale %s source recovery without a live claim",
    async (status) => {
      const open = vi.fn();
      render(
        <ImportProgressStrip
          jobs={[job({ status })]}
          sources={sources}
          backfilling={false}
          sourceSnapshot={staleSnapshot}
          onOpenSources={open}
        />,
      );
      expect(screen.getByRole("status").textContent).toContain("Cursor");
      expect(screen.getByRole("status").textContent).toContain(
        status === "failed"
          ? "failed to read through"
          : "history import not finished",
      );
      await userEvent.click(
        screen.getByRole("button", { name: "View sources" }),
      );
      expect(open).toHaveBeenCalledOnce();
    },
  );

  it("refreshes through the callback only, single-flight, without certifying stale props on resolution", async () => {
    const pending = deferred();
    const refresh = vi.fn().mockReturnValue(pending.promise);
    const open = vi.fn();
    const { container } = render(
      <ImportProgressStrip
        jobs={[job()]}
        sources={sources}
        backfilling
        sourceSnapshot={staleSnapshot}
        onRefresh={refresh}
        onOpenSources={open}
      />,
    );
    const trigger = screen.getByRole<HTMLButtonElement>("button", {
      name: "Refresh and verify",
    });
    expect(trigger.querySelector("svg")).toBeTruthy();
    expect(refresh).not.toHaveBeenCalled();
    expect(
      container.querySelector(".import-strip-headline")?.textContent,
    ).toContain("Progress needs updating");
    act(() => {
      fireEvent.click(trigger);
      fireEvent.click(trigger);
    });
    expect(refresh.mock.calls).toEqual([[]]);
    expect(trigger.disabled).toBe(true);
    expect(screen.getByRole("status").getAttribute("aria-busy")).toBe("true");
    expect(screen.getByRole("status").textContent).toContain(
      "Checking source status",
    );
    expect(
      container.querySelector(".import-strip-headline")?.textContent,
    ).toContain("Saved content is kept; verifying progress");
    expect(container.querySelector(".import-strip-pulse")).toBeNull();
    await act(async () =>
      pending.resolve({ arbitrary: "not a certified source snapshot" }),
    );
    expect(screen.getByRole("status").textContent).toContain(
      "status pending verification",
    );
    expect(screen.getByRole("status").textContent).not.toMatch(
      /Importing|organiz/,
    );
    expect(container.querySelector("time")?.dateTime).toBe(confirmedAt);
    expect(trigger.disabled).toBe(false);
    expect(
      container.querySelector(".import-strip-headline")?.textContent,
    ).toContain("Progress needs updating");
    expect(
      container.querySelector(".import-strip-headline")?.textContent,
    ).not.toContain("Checking");
    expect(open).not.toHaveBeenCalled();
  });

  it("keeps the previous time and counts after refresh failure, and recovers only from fresh props", async () => {
    const refresh = vi
      .fn()
      .mockRejectedValueOnce(new Error("Local service unavailable"));
    const props = {
      jobs: [job()],
      sources,
      backfilling: true,
      sourceSnapshot: staleSnapshot,
      onRefresh: refresh,
      onOpenSources: vi.fn(),
    };
    const { container, rerender } = render(<ImportProgressStrip {...props} />);
    await userEvent.click(
      screen.getByRole("button", { name: "Refresh and verify" }),
    );
    expect(container.querySelector("details")?.textContent).toContain(
      "Verification failed",
    );
    expect(container.querySelector("details")?.textContent).toContain(
      "Local service unavailable",
    );
    expect(screen.queryByRole("alert")).toBeNull();
    expect(container.querySelector("time")?.dateTime).toBe(confirmedAt);
    expect(screen.getByRole("status").textContent).toContain(
      "300 originals saved",
    );
    refresh.mockResolvedValueOnce(undefined);
    await userEvent.click(
      screen.getByRole("button", { name: "Refresh and verify" }),
    );
    expect(refresh).toHaveBeenCalledTimes(2);
    expect(screen.queryByRole("alert")).toBeNull();
    expect(container.querySelector(".import-strip-refresh-error")).toBeNull();
    expect(screen.getByRole("status").textContent).toContain(
      "status pending verification",
    );
    rerender(
      <ImportProgressStrip
        {...props}
        sourceSnapshot={{ confirmedAt: "2026-09-14T02:21:00Z", stale: false }}
      />,
    );
    expect(screen.getByRole("status").textContent).toContain("Importing");
    expect(screen.getByRole("status").textContent).not.toContain(
      "status pending verification",
    );
    expect(container.querySelector(".import-strip-pulse")).toBeTruthy();
  });

  it("does not let an old refresh rejection overwrite a newer confirmed snapshot", async () => {
    const pending = deferred();
    const props = {
      jobs: [job()],
      sources,
      backfilling: false,
      onRefresh: vi.fn().mockReturnValue(pending.promise),
      onOpenSources: vi.fn(),
    };
    const { rerender } = render(
      <ImportProgressStrip {...props} sourceSnapshot={staleSnapshot} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Refresh and verify" }));
    rerender(
      <ImportProgressStrip
        {...props}
        sourceSnapshot={{ confirmedAt: "2026-09-14T02:22:00Z", stale: false }}
      />,
    );
    await act(async () => pending.reject(new Error("old request failed")));
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getByRole("status").textContent).toContain("Importing");
  });

  it("handles unknown refresh errors honestly and does not refresh on unmount", async () => {
    const pending = deferred();
    const refresh = vi
      .fn()
      .mockRejectedValueOnce(null)
      .mockReturnValueOnce(pending.promise);
    const { container, unmount } = render(
      <ImportProgressStrip
        jobs={[]}
        sources={[]}
        backfilling={false}
        sourceSnapshot={{ confirmedAt: null, stale: true }}
        onRefresh={refresh}
        onOpenSources={vi.fn()}
      />,
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Refresh and verify" }),
    );
    expect(container.querySelector("details")?.textContent).toContain(
      "Verification failed",
    );
    expect(screen.queryByRole("alert")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Refresh and verify" }));
    unmount();
    await act(async () => pending.reject(new Error("unmounted")));
    expect(refresh).toHaveBeenCalledTimes(2);
  });

  it("shows completed jobs with read errors rather than hiding the strip", () => {
    const { container } = render(
      <ImportProgressStrip
        jobs={[job({ status: "completed", errors: 2, stored: 280 })]}
        sources={sources}
        backfilling={false}
        onOpenSources={vi.fn()}
      />,
    );
    const text = screen.getByRole("status").textContent;
    expect(text).toContain("Cursor");
    expect(text).toContain("280 originals saved");
    expect(text).toContain("2 issues");
    expect(text).not.toContain("failed to read through");
    expect(container.querySelector(".import-strip-pulse")).toBeNull();
    expect(screen.getByRole("button", { name: "View sources" })).toBeTruthy();
  });

  it("selects the newest created job even when an older resumed job was updated later", () => {
    render(
      <ImportProgressStrip
        jobs={[
          job({
            job_id: "older-resumed",
            status: "failed",
            created_at: "2026-09-01T00:00:00Z",
            updated_at: "2026-09-14T00:00:00Z",
          }),
          job({
            job_id: "newer",
            status: "running",
            created_at: "2026-09-13T00:00:00Z",
            updated_at: "2026-09-13T00:00:01Z",
          }),
        ]}
        sources={sources}
        backfilling={false}
        onOpenSources={vi.fn()}
      />,
    );
    expect(screen.getByRole("status").textContent).toContain(
      "Importing Cursor",
    );
    expect(screen.getByRole("status").textContent).not.toContain(
      "failed to read through",
    );
  });

  it("compares creation instants rather than ISO string order and never mutates job order", () => {
    const jobs = Object.freeze([
      job({
        job_id: "older",
        status: "failed",
        created_at: "2026-09-14T08:30:00+08:00",
        updated_at: "2026-09-14T03:00:00Z",
      }),
      job({
        job_id: "newer",
        status: "completed",
        created_at: "2026-09-14T01:00:00Z",
        updated_at: "2026-09-14T01:00:01Z",
      }),
    ]);
    render(
      <ImportProgressStrip
        jobs={jobs as ImportJob[]}
        sources={sources}
        backfilling={false}
        onOpenSources={vi.fn()}
      />,
    );
    expect(screen.queryByRole("status")).toBeNull();
    expect(jobs.map((item) => item.job_id)).toEqual(["older", "newer"]);
  });

  it("a newer clean completion keeps old errors in history, not the outer summary", () => {
    const { container } = render(
      <ImportProgressStrip
        jobs={[
          job({
            status: "completed",
            errors: 5,
            created_at: "2026-09-01T00:00:00Z",
            updated_at: "2026-09-14T00:00:00Z",
          }),
          job({
            job_id: "new",
            status: "completed",
            created_at: "2026-09-13T00:00:00Z",
          }),
          job({ job_id: "claude", source_id: "claude", status: "failed" }),
        ]}
        sources={sources}
        backfilling={false}
        onOpenSources={vi.fn()}
      />,
    );
    expect(
      container.querySelector(".import-strip-headline")?.textContent,
    ).not.toContain("Cursor");
    expect(container.querySelector("details")?.textContent).toContain("Cursor");
    expect(container.querySelector("details")?.textContent).toContain(
      "5 issues",
    );
    expect(screen.getByRole("status").textContent).toContain(
      "Claude Code failed to read through",
    );
  });

  it("keeps the precipitation line and hides the source link and progress disclosure", () => {
    render(
      <ImportProgressStrip
        jobs={[job({ status: "completed", stored: 5171 })]}
        sources={sources}
        backfilling={false}
        onOpenSources={vi.fn()}
        showSourceLink={false}
        showProgressDetails={false}
        arrival={{
          importing: false,
          importedMessages: 5171,
          importedConversations: 511,
          memoryCount: 100,
          phase: "processing",
        }}
      />,
    );
    const status = screen.getByRole("status");
    expect(status.textContent).toContain("Memories are taking shape");
    expect(status.textContent).toContain(
      "Recent conversations are being organized first; more memories will appear soon",
    );
    expect(screen.queryByRole("button", { name: "View sources" })).toBeNull();
    expect(screen.queryByText("View progress and details")).toBeNull();
  });
});

describe("pausing and continuing imports from the strip", () => {
  afterEach(cleanup);

  type Action = "pause" | "resume" | "cancel";
  const cursorRunning = job();
  const claudeRunning = job({
    job_id: "job-2",
    source_id: "claude",
    created_at: "2026-09-04T00:00:03Z",
  });
  const paused = (item: ImportJob) => ({
    ...item,
    status: "paused" as const,
    updated_at: "2026-09-04T00:01:00Z",
  });
  const running = (item: ImportJob) => ({
    ...item,
    status: "running" as const,
    updated_at: "2026-09-04T00:02:00Z",
  });
  const settle = (id: string, action: Action, from: ImportJob[]) => {
    const item = from.find((entry) => entry.job_id === id) as ImportJob;
    return action === "pause" ? paused(item) : running(item);
  };

  /** Stands in for the controller: a receipt replaces the job it names. */
  function Harness({
    initial,
    action,
    arrival,
    onConfigureModel,
  }: {
    initial: ImportJob[];
    action(jobId: string, action: Action): Promise<ImportJob>;
    arrival?: Parameters<typeof ImportProgressStrip>[0]["arrival"];
    onConfigureModel?(): void;
  }) {
    const [jobs, setJobs] = useState(initial);
    return (
      <ImportProgressStrip
        jobs={jobs}
        sources={sources}
        backfilling={false}
        onOpenSources={() => {}}
        arrival={arrival}
        onConfigureModel={onConfigureModel}
        onJobAction={async (jobId, next) => {
          const receipt = await action(jobId, next);
          if (receipt?.job_id)
            setJobs((all) =>
              all.map((item) =>
                item.job_id === receipt.job_id ? receipt : item,
              ),
            );
          return receipt;
        }}
      />
    );
  }

  function actionNames() {
    const strip = screen.getByRole("status");
    const actions = strip.querySelector(".import-strip-actions");
    return Array.from(actions?.querySelectorAll("button") ?? []).map(
      (button) => button.textContent,
    );
  }

  it("puts the pause button first, before setting up the model", () => {
    render(
      <Harness
        initial={[cursorRunning]}
        action={vi.fn()}
        arrival={{ phase: "model-required" } as never}
        onConfigureModel={() => {}}
      />,
    );
    const names = actionNames();
    expect(names[0]).toBe("Pause all");
    expect(names).toContain("Set up model");
    const pause = screen.getByRole("button", { name: "Pause all" });
    expect(pause.className).toContain("quiet");
    expect(pause.querySelector("svg")).toBeTruthy();
  });

  it("pauses every running import at once, without asking, and waits for each receipt", async () => {
    const user = userEvent.setup();
    const gate = deferred();
    const initial = [cursorRunning, claudeRunning];
    const action = vi.fn(async (id: string, next: Action) => {
      await gate.promise;
      return settle(id, next, initial);
    });
    render(<Harness initial={initial} action={action} />);
    await user.click(screen.getByRole("button", { name: "Pause all" }));
    const busy = screen.getByRole("button", { name: "Pausing..." });
    expect((busy as HTMLButtonElement).disabled).toBe(true);
    expect(screen.queryByRole("dialog")).toBeNull();
    await act(async () => gate.resolve());
    expect(
      await screen.findByRole("button", { name: "Continue all" }),
    ).toBeTruthy();
    expect(action.mock.calls).toEqual([
      ["job-1", "pause"],
      ["job-2", "pause"],
    ]);
    expect(screen.getByRole("status").textContent).toContain(
      "Import paused; saved content is kept; continue anytime.",
    );
  });

  it("with some paused and some running, pauses only the running ones", async () => {
    const user = userEvent.setup();
    const initial = [paused(cursorRunning), claudeRunning];
    const action = vi.fn(async (id: string, next: Action) =>
      settle(id, next, initial),
    );
    render(<Harness initial={initial} action={action} />);
    await user.click(screen.getByRole("button", { name: "Pause all" }));
    await screen.findByRole("button", { name: "Continue all" });
    expect(action.mock.calls).toEqual([["job-2", "pause"]]);
  });

  it("continues every paused import", async () => {
    const user = userEvent.setup();
    const gate = deferred();
    const initial = [paused(cursorRunning), paused(claudeRunning)];
    const action = vi.fn(async (id: string, next: Action) => {
      await gate.promise;
      return settle(id, next, initial);
    });
    render(<Harness initial={initial} action={action} />);
    await user.click(screen.getByRole("button", { name: "Continue all" }));
    expect(
      (screen.getByRole("button", { name: "Resuming..." }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    await act(async () => gate.resolve());
    expect(
      await screen.findByRole("button", { name: "Pause all" }),
    ).toBeTruthy();
    expect(action.mock.calls).toEqual([
      ["job-1", "resume"],
      ["job-2", "resume"],
    ]);
  });

  it("names a job the backend refused, and retries only that one", async () => {
    const user = userEvent.setup();
    const initial = [cursorRunning, claudeRunning];
    const action = vi
      .fn(async (id: string, next: Action) => settle(id, next, initial))
      .mockImplementationOnce(async (id, next) => settle(id, next, initial))
      .mockImplementationOnce(async () => {
        throw new Error(
          "The source operation did not complete; verify the current status and retry.",
        );
      });
    render(<Harness initial={initial} action={action} />);
    await user.click(screen.getByRole("button", { name: "Pause all" }));
    expect((await screen.findByRole("alert")).textContent).toBe(
      "Claude Code could not pause: The source operation did not complete; verify the current status and retry.",
    );
    await user.click(screen.getByRole("button", { name: "Retry pause" }));
    await screen.findByRole("button", { name: "Continue all" });
    expect(action.mock.calls.slice(2)).toEqual([["job-2", "pause"]]);
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("does not take a receipt that is still running as paused", async () => {
    const user = userEvent.setup();
    const action = vi.fn(async () => running(cursorRunning));
    render(<Harness initial={[cursorRunning]} action={action} />);
    await user.click(screen.getByRole("button", { name: "Pause all" }));
    expect((await screen.findByRole("alert")).textContent).toBe(
      "Cursor could not pause: The backend has not confirmed this operation yet.",
    );
    expect(screen.getByRole("button", { name: "Retry pause" })).toBeTruthy();
  });

  it("does not take a receipt for another job", async () => {
    const user = userEvent.setup();
    const action = vi.fn(async () => paused(claudeRunning));
    render(<Harness initial={[cursorRunning]} action={action} />);
    await user.click(screen.getByRole("button", { name: "Pause all" }));
    expect((await screen.findByRole("alert")).textContent).toBe(
      "Cursor could not pause: No valid receipt received for this job.",
    );
  });

  it("offers nothing for new-conversation checks, or while the snapshot is stale", () => {
    const sync = job({ operation: "sync" } as Partial<ImportJob>);
    const { unmount } = render(<Harness initial={[sync]} action={vi.fn()} />);
    expect(screen.queryByRole("button", { name: "Pause all" })).toBeNull();
    unmount();
    render(
      <ImportProgressStrip
        jobs={[cursorRunning]}
        sources={sources}
        backfilling={false}
        onOpenSources={() => {}}
        sourceSnapshot={staleSnapshot}
        onJobAction={vi.fn()}
      />,
    );
    expect(screen.queryByRole("button", { name: "Pause all" })).toBeNull();
  });
});
