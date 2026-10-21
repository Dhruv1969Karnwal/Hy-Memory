// @vitest-environment jsdom

import type { ImportJob, SourceResponse } from "@one-memory/contracts";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SourceCard } from "./SourceCard";

const source: SourceResponse = {
  source_id: "claude-code",
  display_name: "Claude Code",
  storage_kinds: ["jsonl"],
  available: true,
  authorized: false,
  detected: null,
  root_label: null,
  candidate_path: "~/.claude/projects",
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
};

const job: ImportJob = {
  job_id: "job-1",
  user_id: "local",
  source_id: "claude-code",
  status: "running",
  phase: "read",
  checkpoint: "byte:10",
  discovered: 9,
  discovered_targets: 2,
  emitted: 7,
  stored: 5,
  revised: 0,
  merged: 2,
  skipped: 1,
  errors: 1,
  created_at: "2026-09-01T00:00:00Z",
  updated_at: "2026-09-01T00:00:01Z",
};

function callbacks() {
  return {
    onAuthorize: vi.fn(),
    onDetect: vi.fn(),
    onImport: vi.fn(),
    onChangeJob: vi.fn(),
    onRevoke: vi.fn(),
  };
}

describe("SourceCard", () => {
  it("describes an export as a snapshot without promising source changes", () => {
    render(
      <SourceCard
        source={{ ...source, sync_completeness: "one_time_snapshot" }}
        busyKey={null}
        {...callbacks()}
      />,
    );
    expect(
      screen.getByText(
        /One-time snapshot; does not follow changes in the source app/,
      ),
    ).toBeTruthy();
    expect(
      screen.queryByText(
        /Add-only sync; deletions at the source are not followed/,
      ),
    ).toBeNull();
  });
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("requires an explicit keyboard-accessible authorization action", async () => {
    const user = userEvent.setup();
    const actions = callbacks();
    render(<SourceCard source={source} busyKey={null} {...actions} />);

    await user.tab();
    await user.keyboard("{Enter}");

    expect(actions.onAuthorize).toHaveBeenCalledWith(source, false);
  });

  it("shows real counters without inventing a completion percentage", () => {
    const actions = callbacks();
    const { container } = render(
      <SourceCard
        source={{ ...source, authorized: true, detected: true }}
        job={job}
        busyKey={null}
        {...actions}
      />,
    );

    expect(screen.getByText("Importing in the background")).toBeTruthy();
    expect(screen.getByText("5")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Pause" })).toBeTruthy();
    expect(container.textContent).not.toContain("%");
  });

  it("states recall support without claiming this machine is connected", () => {
    const actions = callbacks();
    const { container } = render(
      <SourceCard
        source={{
          ...source,
          capabilities: { ...source.capabilities, recall: "implemented" },
        }}
        busyKey={null}
        {...actions}
      />,
    );

    // Whether recall is installed here is the recall section's fact, not ours.
    expect(container.textContent).toContain("In-app recall supported");
    expect(container.textContent).not.toContain("In-app recall wired up");
  });

  it("does not call an additions-only connection real-time sync", () => {
    const actions = callbacks();
    const { container } = render(
      <SourceCard
        source={{
          ...source,
          capabilities: { ...source.capabilities, live_sync: "implemented" },
          sync_completeness: "additions_only",
        }}
        busyKey={null}
        {...actions}
      />,
    );

    // We poll on a server-side interval and cannot see upstream deletions, so
    // a memory can outlive the sentence it came from. Saying "live sync" would
    // tell the user their copy tracks the source.
    expect(container.textContent).toContain("Add-only sync");
    expect(container.textContent).not.toContain("live sync");
  });

  it("requires a second explicit confirmation for environment paths", async () => {
    const user = userEvent.setup();
    const actions = callbacks();
    const confirm = vi.spyOn(window, "confirm").mockReturnValueOnce(false);
    const custom = {
      ...source,
      candidate_kind: "environment_override" as const,
      candidate_path: "/custom/agent/history",
    };
    render(<SourceCard source={custom} busyKey={null} {...actions} />);

    await user.click(
      screen.getByRole("button", {
        name: "Confirm and authorize custom directory",
      }),
    );
    expect(confirm).toHaveBeenCalledWith(
      expect.stringContaining(custom.candidate_path),
    );
    expect(actions.onAuthorize).not.toHaveBeenCalled();

    confirm.mockReturnValueOnce(true);
    await user.click(
      screen.getByRole("button", {
        name: "Confirm and authorize custom directory",
      }),
    );
    expect(actions.onAuthorize).toHaveBeenCalledWith(custom, true);
  });

  it("tells the person how many sessions were found before they grant", () => {
    const actions = callbacks();
    render(
      <SourceCard
        source={{ ...source, detected: true, session_count: 128 }}
        busyKey={null}
        {...actions}
      />,
    );

    expect(screen.getByText("Found 128 conversations")).toBeTruthy();
  });

  it("says the host is installed but empty rather than showing a bare zero", () => {
    const actions = callbacks();
    render(
      <SourceCard
        source={{ ...source, detected: true, session_count: 0 }}
        busyKey={null}
        {...actions}
      />,
    );

    expect(
      screen.getByText("Installed, no conversations found yet"),
    ).toBeTruthy();
    expect(screen.queryByText("Found 0 conversations")).toBeNull();
  });

  it("shows no session count when the core has not reported one", () => {
    const actions = callbacks();
    const { container } = render(
      <SourceCard
        source={{ ...source, detected: true }}
        busyKey={null}
        {...actions}
      />,
    );

    expect(container.textContent).not.toContain("conversations");
    expect(container.textContent).not.toContain(
      "Installed, no conversations found yet",
    );
  });
});
