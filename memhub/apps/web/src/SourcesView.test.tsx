// @vitest-environment jsdom

import type { ConnectSourcesResponse, ImportJob } from "@one-memory/contracts";
import {
  act,
  cleanup,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SourcesView } from "./SourcesView";
import type { useImportSources } from "./use-import-sources";

afterEach(cleanup);

describe("SourcesView", () => {
  afterEach(() => {
    cleanup();
    delete window.oneMemory;
  });

  it("does not mark a completed scan with parsing errors as a healthy connection", () => {
    const source = {
      source_id: "openclaw",
      display_name: "OpenClaw",
      available: true,
      authorized: true,
      candidate_kind: "default",
      candidate_path: "~/.openclaw",
      imported_messages: 0,
      imported_conversations: 0,
      capabilities: {
        verification: "fixture_only",
        live_sync: "not_implemented",
      },
    };
    const job = {
      source_id: "openclaw",
      job_id: "partial",
      status: "completed",
      stored: 0,
      merged: 0,
      skipped: 0,
      errors: 2,
    };
    const imports = {
      sources: [source],
      jobs: [job],
      latestJobBySource: new Map([["openclaw", job]]),
      loading: false,
      busyKey: null,
      error: null,
      lastConnectResult: null,
      connectSelected: vi.fn(),
      syncAll: vi.fn(),
      startImport: vi.fn(),
      changeJob: vi.fn(),
    } as unknown as ReturnType<typeof useImportSources>;
    render(<SourcesView imports={imports} />);
    expect(screen.getByText("Import has issues")).toBeTruthy();
    expect(screen.getByText(/2 issues/)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Rescan" })).toBeTruthy();
  });

  it("turns one source switch into one explicit authorization and import", async () => {
    const user = userEvent.setup();
    const connectSelected = vi.fn().mockResolvedValue({
      accepted_source_ids: ["workbuddy"],
      failed_sources: {},
      jobs: [],
    });
    const imports = {
      health: null,
      healthStale: false,
      sources: [
        {
          source_id: "workbuddy",
          display_name: "WorkBuddy",
          storage_kinds: ["jsonl"],
          available: true,
          authorized: false,
          detected: null,
          root_label: null,
          candidate_path: "~/.workbuddy/projects",
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
          sync_completeness: "additions_only",
        },
      ],
      jobs: [],
      loading: false,
      busyKey: null,
      error: null,
      lastConnectResult: null,
      latestJobBySource: new Map(),
      refresh: vi.fn(),
      connectSelected,
      syncAll: vi.fn(),
      detect: vi.fn(),
      startImport: vi.fn(),
      changeJob: vi.fn(),
    } as unknown as ReturnType<typeof useImportSources>;

    render(<SourcesView imports={imports} />);
    await user.click(screen.getByRole("checkbox", { name: /WorkBuddy/ }));
    await waitFor(() =>
      expect(connectSelected).toHaveBeenCalledWith(
        ["workbuddy"],
        [
          expect.objectContaining({
            source_id: "workbuddy",
            candidate_path: "~/.workbuddy/projects",
          }),
        ],
        // The interaction's trace id, so the connect request carries the kernel
        // legs that join the renderer's ui.action / ui.result pair.
        { traceId: expect.any(String) },
      ),
    );
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("one timeout is not written as a failure: the result is unconfirmed and no retry is offered", async () => {
    // A timeout means Core may or may not have applied the switch. Reporting it
    // as "could not turn reading on" would assert something the client cannot know, and
    // "check and retry" invites a retry of a write that may already have landed.
    // The trace must say `unknown` for the same reason.
    const emitted: Array<Record<string, unknown>> = [];
    (
      window as unknown as {
        oneMemory: { logUiTrace: (input: Record<string, unknown>) => void };
      }
    ).oneMemory = { logUiTrace: (input) => emitted.push(input) };
    const user = userEvent.setup();
    const timeout = Object.assign(new Error("timeout of 30000ms exceeded"), {
      code: "ECONNABORTED",
    });
    const imports = {
      health: null,
      healthStale: false,
      sources: [
        {
          source_id: "workbuddy",
          display_name: "WorkBuddy",
          storage_kinds: ["jsonl"],
          available: true,
          authorized: true,
          detected: true,
          root_label: null,
          candidate_path: "~/.workbuddy/projects",
          candidate_kind: "default",
          imported_messages: 0,
          imported_conversations: 0,
          grant_revision: 4,
          sync_enabled: true,
          capabilities: {
            historical_import: "implemented",
            incremental_import: "implemented",
            live_sync: "not_implemented",
            recall: "not_implemented",
            installer: "not_implemented",
            verification: "replayed",
          },
          sync_completeness: "additions_only",
        },
      ],
      jobs: [],
      loading: false,
      busyKey: null,
      error: null,
      lastConnectResult: null,
      latestJobBySource: new Map(),
      refresh: vi.fn(),
      connectSelected: vi.fn(),
      syncAll: vi.fn(),
      setSync: vi.fn().mockRejectedValue(timeout),
      detect: vi.fn(),
      startImport: vi.fn(),
      changeJob: vi.fn(),
    } as unknown as ReturnType<typeof useImportSources>;
    try {
      render(<SourcesView imports={imports} />);
      await user.click(screen.getByRole("checkbox", { name: /WorkBuddy/ }));
      const result = await waitFor(() => {
        const found = emitted.find((item) => item.phase === "result");
        expect(found).toBeTruthy();
        return found;
      });
      expect(result?.status).toBe("unknown");
      // The copy must not claim failure, and must not invite a retry.
      expect(
        screen.queryByText("Could not turn reading off, check and retry."),
      ).toBeNull();
      expect(
        screen.getByText(
          /processing switch result is unconfirmed; refresh and verify; there is no automatic retry/,
        ),
      ).toBeTruthy();
    } finally {
      delete (window as unknown as { oneMemory?: unknown }).oneMemory;
    }
  });

  it("keeps the environment override confirmation after the source switch", async () => {
    const user = userEvent.setup();
    const connectSelected = vi.fn().mockResolvedValue({
      accepted_source_ids: ["cursor"],
      failed_sources: {},
      jobs: [],
    });
    const imports = {
      health: null,
      healthStale: false,
      sources: [
        {
          source_id: "cursor",
          display_name: "Cursor",
          storage_kinds: ["sqlite"],
          available: true,
          authorized: false,
          detected: true,
          root_label: null,
          candidate_path: "/Volumes/custom/cursor",
          candidate_kind: "environment_override",
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
          sync_completeness: "additions_only",
        },
      ],
      jobs: [],
      loading: false,
      busyKey: null,
      error: null,
      lastConnectResult: null,
      latestJobBySource: new Map(),
      connectSelected,
      syncAll: vi.fn(),
      changeJob: vi.fn(),
    } as unknown as ReturnType<typeof useImportSources>;

    render(<SourcesView imports={imports} />);
    expect(screen.getByText(/Environment variable override path/)).toBeTruthy();
    await user.click(screen.getByRole("checkbox", { name: /Cursor/ }));

    expect(connectSelected).not.toHaveBeenCalled();
    const dialog = screen.getByRole("dialog", {
      name: "Confirm reading these custom paths?",
    });
    expect(within(dialog).getByText("/Volumes/custom/cursor")).toBeTruthy();

    await user.click(
      screen.getByRole("button", {
        name: "Confirm paths and authorize import",
      }),
    );
    expect(connectSelected).toHaveBeenCalledWith(
      ["cursor"],
      [expect.objectContaining({ source_id: "cursor" })],
      { traceId: expect.any(String) },
    );
  });

  it("hides unavailable ungranted sources and shows per-source batch results", () => {
    const imports = {
      health: null,
      healthStale: false,
      sources: [
        {
          source_id: "ok",
          display_name: "Connected AI",
          storage_kinds: ["jsonl"],
          available: true,
          authorized: true,
          detected: true,
          root_label: "default",
          candidate_path: "~/.connected",
          candidate_kind: "default",
          imported_messages: 8,
          imported_conversations: 2,
          capabilities: {
            historical_import: "implemented",
            incremental_import: "implemented",
            live_sync: "not_implemented",
            recall: "not_implemented",
            installer: "not_implemented",
            verification: "real_replayed",
          },
          sync_completeness: "additions_only",
        },
        {
          source_id: "missing",
          display_name: "Missing AI",
          storage_kinds: ["jsonl"],
          available: false,
          authorized: false,
          detected: false,
          root_label: null,
          candidate_path: "~/.missing",
          candidate_kind: "default",
          imported_messages: 0,
          imported_conversations: 0,
          capabilities: {
            historical_import: "implemented",
            incremental_import: "implemented",
            live_sync: "not_implemented",
            recall: "not_implemented",
            installer: "not_implemented",
            verification: "fixture_only",
          },
          sync_completeness: "additions_only",
        },
      ],
      jobs: [],
      loading: false,
      busyKey: null,
      error: null,
      lastConnectResult: {
        accepted_source_ids: ["ok"],
        failed_sources: { cursor: "grant_not_available" },
        jobs: [],
      },
      latestJobBySource: new Map(),
      syncAll: vi.fn(),
      changeJob: vi.fn(),
    } as unknown as ReturnType<typeof useImportSources>;

    render(<SourcesView imports={imports} />);

    expect(document.querySelector(".source-table")?.textContent).not.toContain(
      "Missing AI",
    );
    expect(
      document.querySelector(".application-coverage")?.textContent,
    ).toContain("Missing AI");
    expect(screen.getAllByText("Connected AI")).toHaveLength(3);
    expect(screen.getByText("Authorized, import job started")).toBeTruthy();
    expect(
      screen.getByText("Could not authorize: Grant record not created"),
    ).toBeTruthy();
  });

  it("shows the edit and installs only once it is approved", async () => {
    const user = userEvent.setup();
    const installedPermission = {
      hostId: "claude-code",
      state: "install-only" as const,
      reason: null,
      receipt: {
        installationId: "24361b44-5a15-4f87-ad09-05571dc281c3",
        hostId: "claude-code",
        revision: 1,
        status: "install-only" as const,
        allowedTools: [],
        allowedScopes: [],
      },
    };
    const approvedPermission = {
      ...installedPermission,
      state: "approved" as const,
      receipt: {
        ...installedPermission.receipt,
        revision: 2,
        status: "approved" as const,
        allowedTools: ["om_recall", "om_ask"] as const,
        allowedScopes: ["all-apps"],
      },
    };
    const installRecall = vi.fn().mockResolvedValue({
      hostId: "claude-code",
      installed: true,
      permission: {
        hostId: "claude-code",
        state: "install-only",
        reason: null,
        receipt: {
          installationId: "24361b44-5a15-4f87-ad09-05571dc281c3",
          hostId: "claude-code",
          revision: 1,
          status: "install-only",
          allowedTools: [],
          allowedScopes: [],
        },
      },
    });
    const previewRecallInstall = vi.fn().mockResolvedValue({
      hostId: "claude-code",
      hostLabel: "Claude Code",
      fingerprint: "fp-1",
      files: [
        {
          path: "/Users/example/.claude.json",
          owner: "host",
          existed: true,
          hunk: {
            startLine: 4,
            removed: ['  "mcpServers": {}'],
            added: ['  "mcpServers": {', '    "one-memory": {}', "  }"],
            contextBefore: ["{"],
            contextAfter: ["}"],
          },
        },
      ],
    });
    const getRecallHosts = vi
      .fn()
      .mockResolvedValueOnce([
        { hostId: "cursor", label: "Cursor", present: true, installed: true },
        {
          hostId: "claude-code",
          label: "Claude Code",
          present: true,
          installed: false,
        },
      ])
      .mockResolvedValue([
        { hostId: "cursor", label: "Cursor", present: true, installed: true },
        {
          hostId: "claude-code",
          label: "Claude Code",
          present: true,
          installed: true,
          permission: approvedPermission,
        },
      ]);
    const getRecallPermission = vi.fn().mockResolvedValue(installedPermission);
    const setRecallPermission = vi.fn().mockResolvedValue(approvedPermission);
    window.oneMemory = {
      getApiToken: vi.fn(),
      getApiBaseUrl: vi.fn(),
      openHandoff: vi.fn(),
      probeRecallHost: vi.fn(),
      previewSkillSync: vi.fn(),
      applySkillSync: vi.fn(),
      getRecallHosts,
      previewRecallInstall,
      installRecall,
      previewRecallUninstall: vi.fn(),
      uninstallRecall: vi.fn(),
      getRecallPermission,
      setRecallPermission,
      revokeRecallPermission: vi.fn(),
    };
    const imports = emptyImports();

    render(<SourcesView imports={imports} />);

    // Cursor is already installed, so only Claude Code offers an install.
    const button = await screen.findByRole("button", {
      name: "Install to Claude Code in one click",
    });
    await user.click(button);

    // The click asks what would change; it does not change anything. This is
    // someone else's settings file, so nothing is written unseen.
    expect(previewRecallInstall).toHaveBeenCalledWith("claude-code");
    expect(installRecall).not.toHaveBeenCalled();
    const dialog = await screen.findByRole("dialog");
    expect(dialog.textContent).toContain("/Users/example/.claude.json");
    expect(dialog.textContent).toContain('+     "one-memory": {}');
    expect(dialog.textContent).toContain('-   "mcpServers": {}');

    expect(dialog.textContent).toContain("Confirming also allows Claude Code");
    expect(setRecallPermission).not.toHaveBeenCalled();
    await user.click(
      screen.getByRole("button", { name: "Connect and allow memory use" }),
    );

    // The approval carries the fingerprint of the state that was shown, so it
    // cannot be spent on a file that changed in between.
    expect(installRecall).toHaveBeenCalledWith("claude-code", "fp-1");
    expect(installRecall).toHaveBeenCalledTimes(1);
    await waitFor(() =>
      expect(
        within(
          screen.getByRole("region", { name: "Claude Code call permissions" }),
        ).getByRole("button", { name: "Adjust call permissions" }),
      ).toHaveProperty("disabled", false),
    );
    expect(
      await screen.findAllByRole("button", {
        name: "Installed · click to disconnect",
      }),
    ).toHaveLength(2);
    expect(getRecallPermission).toHaveBeenCalledExactlyOnceWith("claude-code");
    expect(getRecallHosts).toHaveBeenCalledTimes(2);
    // No source is offered: grant governed-memory tools, never raw-conversation access.
    expect(setRecallPermission).toHaveBeenCalledExactlyOnceWith("claude-code", {
      installationId: "24361b44-5a15-4f87-ad09-05571dc281c3",
      expectedRevision: 1,
      allowedTools: ["om_recall", "om_ask", "om_diagnostic_trace"],
      allowedScopes: ["all-apps"],
      userConfirmed: true,
    });
    expect(installRecall.mock.invocationCallOrder[0]).toBeLessThan(
      getRecallPermission.mock.invocationCallOrder[0],
    );
    expect(getRecallPermission.mock.invocationCallOrder[0]).toBeLessThan(
      setRecallPermission.mock.invocationCallOrder[0],
    );
  });

  it("writes nothing when the edit is declined", async () => {
    const user = userEvent.setup();
    const installRecall = vi.fn();
    window.oneMemory = {
      getApiToken: vi.fn(),
      getApiBaseUrl: vi.fn(),
      openHandoff: vi.fn(),
      probeRecallHost: vi.fn(),
      previewSkillSync: vi.fn(),
      applySkillSync: vi.fn(),
      getRecallHosts: vi.fn().mockResolvedValue([
        {
          hostId: "claude-code",
          label: "Claude Code",
          present: true,
          installed: false,
        },
      ]),
      previewRecallInstall: vi.fn().mockResolvedValue({
        hostId: "claude-code",
        hostLabel: "Claude Code",
        fingerprint: "fp-1",
        files: [
          {
            path: "/Users/example/.claude.json",
            owner: "host",
            existed: true,
            hunk: {
              startLine: 2,
              removed: [],
              added: ['  "one-memory": {}'],
              contextBefore: ["{"],
              contextAfter: ["}"],
            },
          },
        ],
      }),
      installRecall,
      previewRecallUninstall: vi.fn(),
      uninstallRecall: vi.fn(),
    };

    render(<SourcesView imports={emptyImports()} />);

    await user.click(
      await screen.findByRole("button", {
        name: "Install to Claude Code in one click",
      }),
    );
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(installRecall).not.toHaveBeenCalled();
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(
      screen.getByRole("button", {
        name: "Install to Claude Code in one click",
      }),
    ).toBeTruthy();
  });

  it("shows the edit before disconnecting, too", async () => {
    const user = userEvent.setup();
    const uninstallRecall = vi
      .fn()
      .mockResolvedValue({ hostId: "cursor", installed: false });
    const previewRecallUninstall = vi.fn().mockResolvedValue({
      hostId: "cursor",
      hostLabel: "Cursor",
      fingerprint: "fp-out",
      files: [
        {
          path: "/Users/example/.cursor/mcp.json",
          owner: "host",
          deleted: false,
          hunk: {
            startLine: 2,
            removed: ['    "one-memory": {}'],
            added: [],
            contextBefore: ['  "mcpServers": {'],
            contextAfter: ["  }"],
          },
        },
        {
          path: "/Users/example/.cursor/skills/one-memory-recall/SKILL.md",
          owner: "one-memory",
          deleted: true,
          hunk: null,
        },
      ],
    });
    window.oneMemory = {
      getApiToken: vi.fn(),
      getApiBaseUrl: vi.fn(),
      openHandoff: vi.fn(),
      getRecallHosts: vi
        .fn()
        .mockResolvedValue([
          { hostId: "cursor", label: "Cursor", present: true, installed: true },
        ]),
      probeRecallHost: vi.fn(),
      previewSkillSync: vi.fn(),
      applySkillSync: vi.fn(),
      previewRecallInstall: vi.fn(),
      installRecall: vi.fn(),
      previewRecallUninstall,
      uninstallRecall,
    };

    render(<SourcesView imports={emptyImports()} />);

    await user.click(
      await screen.findByRole("button", {
        name: "Installed · click to disconnect",
      }),
    );

    // Disconnecting rewrites the host's own file just as installing did, so it
    // asks first rather than acting on the click.
    expect(previewRecallUninstall).toHaveBeenCalledWith("cursor");
    expect(uninstallRecall).not.toHaveBeenCalled();
    const dialog = await screen.findByRole("dialog");
    expect(dialog.textContent).toContain('-     "one-memory": {}');
    // A file that goes away is said to go away, not rendered as a diff.
    expect(dialog.textContent).toContain("The whole file is removed");

    await user.click(
      screen.getByRole("button", { name: "Confirm disconnect" }),
    );

    expect(uninstallRecall).toHaveBeenCalledWith("cursor", "fp-out");
    expect(
      await screen.findByRole("button", {
        name: "Install to Cursor in one click",
      }),
    ).toBeTruthy();
  });

  it("restores nothing when disconnecting is declined", async () => {
    const user = userEvent.setup();
    const uninstallRecall = vi.fn();
    window.oneMemory = {
      getApiToken: vi.fn(),
      getApiBaseUrl: vi.fn(),
      openHandoff: vi.fn(),
      getRecallHosts: vi
        .fn()
        .mockResolvedValue([
          { hostId: "cursor", label: "Cursor", present: true, installed: true },
        ]),
      probeRecallHost: vi.fn(),
      previewSkillSync: vi.fn(),
      applySkillSync: vi.fn(),
      previewRecallInstall: vi.fn(),
      installRecall: vi.fn(),
      previewRecallUninstall: vi.fn().mockResolvedValue({
        hostId: "cursor",
        hostLabel: "Cursor",
        fingerprint: "fp-out",
        files: [
          {
            path: "/Users/example/.cursor/mcp.json",
            owner: "host",
            deleted: true,
            hunk: null,
          },
        ],
      }),
      uninstallRecall,
    };

    render(<SourcesView imports={emptyImports()} />);

    await user.click(
      await screen.findByRole("button", {
        name: "Installed · click to disconnect",
      }),
    );
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(uninstallRecall).not.toHaveBeenCalled();
    expect(screen.queryByRole("dialog")).toBeNull();
    // Still installed, because declining changed nothing.
    expect(
      screen.getByRole("button", { name: "Installed · click to disconnect" }),
    ).toBeTruthy();
  });

  it("does not offer hosts that are not installed on this machine", async () => {
    window.oneMemory = {
      getApiToken: vi.fn(),
      getApiBaseUrl: vi.fn(),
      openHandoff: vi.fn(),
      probeRecallHost: vi.fn(),
      previewSkillSync: vi.fn(),
      applySkillSync: vi.fn(),
      getRecallHosts: vi.fn().mockResolvedValue([
        { hostId: "cursor", label: "Cursor", present: true, installed: false },
        {
          hostId: "claude-code",
          label: "Claude Code",
          present: false,
          installed: false,
        },
      ]),
      previewRecallInstall: vi.fn(),
      installRecall: vi.fn(),
      previewRecallUninstall: vi.fn(),
      uninstallRecall: vi.fn(),
    };

    render(<SourcesView imports={emptyImports()} />);

    expect(
      await screen.findByRole("button", {
        name: "Install to Cursor in one click",
      }),
    ).toBeTruthy();
    expect(
      screen.queryByRole("button", {
        name: "Install to Claude Code in one click",
      }),
    ).toBeNull();
  });

  it("says the link is not through, not merely that it is installed", async () => {
    // "Installed" is the word that hid the recorded failure: three hosts were
    // installed, and installed was all the list said, while recall returned
    // nothing. So a host whose last self-check did not go through must not
    // read as working.
    window.oneMemory = {
      getApiToken: vi.fn(),
      getApiBaseUrl: vi.fn(),
      openHandoff: vi.fn(),
      probeRecallHost: vi.fn().mockResolvedValue({
        hostId: "cursor",
        probe: {
          state: "unreachable",
          detail: "MemHub is not running (broker discovery is unavailable)",
          discoveryPath: "/tmp/wrong.json",
          discoveryPresent: false,
          probedAt: new Date().toISOString(),
        },
        skill: null,
        repoint: { from: "/tmp/wrong.json", to: "/right/broker.json" },
      }),
      previewSkillSync: vi.fn(),
      applySkillSync: vi.fn(),
      getRecallHosts: vi.fn().mockResolvedValue([
        {
          hostId: "cursor",
          label: "Cursor",
          present: true,
          installed: true,
          selfCheck: {
            state: "unreachable",
            detail: "broker discovery is unavailable",
            probed_at: new Date().toISOString(),
          },
        },
      ]),
      previewRecallInstall: vi.fn().mockResolvedValue({
        hostId: "cursor",
        hostLabel: "Cursor",
        files: [],
        supersedes: null,
        fingerprint: "f",
      }),
      installRecall: vi.fn(),
      previewRecallUninstall: vi.fn(),
      uninstallRecall: vi.fn(),
    };

    render(<SourcesView imports={emptyImports()} />);

    expect(await screen.findByText(/Not connected/)).toBeTruthy();

    // Diagnosing offers the repair, and the repair goes through the ordinary
    // preview rather than writing to their file unannounced.
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Diagnose" }));
    await user.click(await screen.findByRole("button", { name: "Repoint" }));

    expect(window.oneMemory?.previewRecallInstall).toHaveBeenCalledWith(
      "cursor",
    );
  });

  it("offers no repair when the app is merely not running", async () => {
    // The entry is already right, so repointing would change nothing and
    // saying "repoint" would send the user to fix a correct config while the
    // actual fix is to open the app. The probe reports this by offering no
    // repoint, and the view must respect that rather than keying off the
    // verdict.
    window.oneMemory = {
      getApiToken: vi.fn(),
      getApiBaseUrl: vi.fn(),
      openHandoff: vi.fn(),
      probeRecallHost: vi.fn().mockResolvedValue({
        hostId: "cursor",
        probe: {
          state: "reachable_but_blind",
          detail: "MemHub is not running or its recall broker is unavailable",
          discoveryPath: "/right/broker.json",
          discoveryPresent: true,
          probedAt: new Date().toISOString(),
        },
        skill: null,
        repoint: null,
      }),
      previewSkillSync: vi.fn(),
      applySkillSync: vi.fn(),
      getRecallHosts: vi.fn().mockResolvedValue([
        {
          hostId: "cursor",
          label: "Cursor",
          present: true,
          installed: true,
          selfCheck: {
            state: "reachable_but_blind",
            detail: "broker unavailable",
            probed_at: new Date().toISOString(),
          },
        },
      ]),
      previewRecallInstall: vi.fn(),
      installRecall: vi.fn(),
      previewRecallUninstall: vi.fn(),
      uninstallRecall: vi.fn(),
    };

    render(<SourcesView imports={emptyImports()} />);

    expect(
      await screen.findByText(/Installed, but no memories come back/),
    ).toBeTruthy();
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Diagnose" }));

    expect(
      await screen.findByText(/not running or its recall broker/),
    ).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Repoint" })).toBeNull();
  });

  it("still lists a host whose journal predates self-checks", async () => {
    // Installs made before self-checks existed have no verdict at all, and
    // reading one off a host that has none took the whole row's markup down
    // with it — the install button included. Found by the existing tests,
    // pinned here so it cannot come back.
    window.oneMemory = {
      getApiToken: vi.fn(),
      getApiBaseUrl: vi.fn(),
      openHandoff: vi.fn(),
      probeRecallHost: vi.fn(),
      previewSkillSync: vi.fn(),
      applySkillSync: vi.fn(),
      getRecallHosts: vi
        .fn()
        .mockResolvedValue([
          { hostId: "cursor", label: "Cursor", present: true, installed: true },
        ]),
      previewRecallInstall: vi.fn(),
      installRecall: vi.fn(),
      previewRecallUninstall: vi.fn(),
      uninstallRecall: vi.fn(),
    };

    render(<SourcesView imports={emptyImports()} />);

    expect(
      await screen.findByRole("button", {
        name: "Installed · click to disconnect",
      }),
    ).toBeTruthy();
  });

  it("shows call-rule state beside the link state", async () => {
    // R-33 + R-24 step 5: two different failure modes, side by side. A host
    // can be wired and still be running on yesterday's skill.
    window.oneMemory = {
      getApiToken: vi.fn(),
      getApiBaseUrl: vi.fn(),
      openHandoff: vi.fn(),
      probeRecallHost: vi.fn(),
      previewSkillSync: vi.fn(),
      applySkillSync: vi.fn(),
      getRecallHosts: vi.fn().mockResolvedValue([
        {
          hostId: "cursor",
          label: "Cursor",
          present: true,
          installed: true,
          selfCheck: {
            state: "wired",
            detail: "recalled 0 memories for the self-check",
            probed_at: new Date().toISOString(),
          },
          skillDrift: {
            state: "stale",
            installedVersion: 1,
            currentVersion: 2,
          },
        },
      ]),
      previewRecallInstall: vi.fn(),
      installRecall: vi.fn(),
      previewRecallUninstall: vi.fn(),
      uninstallRecall: vi.fn(),
    };

    render(<SourcesView imports={emptyImports()} />);

    expect(await screen.findByText(/Connected/)).toBeTruthy();
    expect(await screen.findByText(/Needs update/)).toBeTruthy();
  });

  it("warns before overwriting a hand-edited call rule", async () => {
    window.oneMemory = {
      getApiToken: vi.fn(),
      getApiBaseUrl: vi.fn(),
      openHandoff: vi.fn(),
      probeRecallHost: vi.fn(),
      applySkillSync: vi.fn(),
      getRecallHosts: vi.fn().mockResolvedValue([
        {
          hostId: "cursor",
          label: "Cursor",
          present: true,
          installed: true,
          selfCheck: {
            state: "wired",
            detail: "ok",
            probed_at: new Date().toISOString(),
          },
          skillDrift: {
            state: "modified",
            installedVersion: 2,
            currentVersion: 2,
          },
        },
      ]),
      previewRecallInstall: vi.fn(),
      installRecall: vi.fn(),
      previewRecallUninstall: vi.fn(),
      uninstallRecall: vi.fn(),
      previewSkillSync: vi.fn().mockResolvedValue({
        hostId: "cursor",
        hostLabel: "Cursor",
        files: [],
        fingerprint: "f",
        upToDate: false,
        conflict: true,
      }),
    };

    render(<SourcesView imports={emptyImports()} />);

    expect(await screen.findByText(/Manually edited by you/)).toBeTruthy();
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Update call rule" }));
    expect(await screen.findByText(/will overwrite your changes/)).toBeTruthy();
  });
});
function emptyImports(): ReturnType<typeof useImportSources> {
  return {
    sources: [],
    jobs: [],
    loading: false,
    busyKey: null,
    error: null,
    lastConnectResult: null,
    latestJobBySource: new Map(),
  } as unknown as ReturnType<typeof useImportSources>;
}

describe("SourcesView complementary sync Hook", () => {
  afterEach(() => {
    cleanup();
    delete window.oneMemory;
  });
  function hookHostApi() {
    const host = {
      hostId: "claude-code",
      label: "Claude Code",
      present: true,
      installed: true,
      selfCheck: null,
      skillDrift: null,
    };
    const api = {
      getApiToken: vi.fn(),
      getApiBaseUrl: vi.fn(),
      openHandoff: vi.fn(),
      getRecallHosts: vi
        .fn()
        .mockResolvedValue([
          host,
          { ...host, hostId: "cursor", label: "Cursor" },
        ]),
      probeRecallHost: vi.fn(),
      previewSkillSync: vi.fn(),
      applySkillSync: vi.fn(),
      previewRecallInstall: vi.fn(),
      installRecall: vi.fn(),
      previewRecallUninstall: vi.fn().mockResolvedValue({
        hostId: "claude-code",
        hostLabel: "Claude Code",
        fingerprint: "fp",
        files: [],
      }),
      uninstallRecall: vi.fn(),
      getSyncHookStatus: vi.fn().mockResolvedValue({
        hostId: "claude-code",
        state: "installed",
        reason: null,
      }),
      previewSyncHook: vi.fn(),
      applySyncHook: vi.fn(),
    };
    window.oneMemory = api;
    return api;
  }
  it("mounts only Claude Hook control alongside existing read permissions", async () => {
    const api = hookHostApi();
    render(<SourcesView imports={emptyImports()} />);
    expect(
      await screen.findByRole("region", { name: "Claude Code Sync Hook" }),
    ).toBeTruthy();
    expect(
      screen.getByRole("region", { name: "Claude Code call permissions" }),
    ).toBeTruthy();
    expect(
      screen.queryByRole("region", { name: "Cursor Sync Hook" }),
    ).toBeNull();
    await waitFor(() =>
      expect(api.getSyncHookStatus).toHaveBeenCalledExactlyOnceWith(
        "claude-code",
      ),
    );
    expect(api.applySyncHook).not.toHaveBeenCalled();
  });
  it.each(["preview", "apply"])(
    "retains the host list and actionable remove-first message on %s failure",
    async (stage) => {
      const api = hookHostApi();
      const error = new Error(
        "Error invoking remote method: sync_hook_remove_first",
      );
      if (stage === "preview")
        api.previewRecallUninstall.mockRejectedValueOnce(error);
      else api.uninstallRecall.mockRejectedValueOnce(error);
      render(<SourcesView imports={emptyImports()} />);
      await screen.findByText("Sync Hook installed");
      const claude = screen
        .getByRole("region", { name: "Claude Code Sync Hook" })
        .closest("li");
      if (!claude) throw new Error("missing Claude host row");
      await userEvent.click(
        within(claude).getByRole("button", {
          name: "Installed · click to disconnect",
        }),
      );
      if (stage === "apply")
        await userEvent.click(
          await screen.findByRole("button", { name: "Confirm disconnect" }),
        );
      expect(
        await screen.findByText(
          "Remove the Sync Hook before disconnecting the app connection.",
        ),
      ).toBeTruthy();
      expect(
        screen.queryByText("Cannot read the installation state."),
      ).toBeNull();
      expect(
        screen.getByRole("region", { name: "Claude Code Sync Hook" }),
      ).toBeTruthy();
      expect(
        screen.getByRole("button", { name: "Uninstall Sync Hook" }),
      ).toBeTruthy();
      expect(api.applySyncHook).not.toHaveBeenCalled();
    },
  );
});

function selectableImports(): ReturnType<typeof useImportSources> {
  return {
    ...emptyImports(),
    sources: [
      {
        source_id: "workbuddy",
        display_name: "WorkBuddy",
        available: true,
        authorized: false,
        candidate_path: "/isolated/workbuddy/projects",
        candidate_kind: "default",
        imported_messages: 0,
        imported_conversations: 0,
        storage_kinds: ["jsonl"],
        capabilities: {
          verification: "fixture_only",
          live_sync: "implemented",
        },
      },
    ],
    connectSelected: vi.fn().mockResolvedValue({
      accepted_source_ids: ["workbuddy"],
      failed_sources: {},
      jobs: [],
    }),
  } as unknown as ReturnType<typeof useImportSources>;
}

it("S1.3 COLD-19: application entry leaves every ungranted source unselected", () => {
  const imports = selectableImports();
  imports.sources = [
    imports.sources[0],
    {
      ...imports.sources[0],
      source_id: "cursor",
      display_name: "Cursor",
      candidate_path: "/isolated/cursor",
    },
  ];
  render(<SourcesView imports={imports} />);
  for (const name of [/WorkBuddy/, /Cursor/]) {
    expect(
      screen.getByRole<HTMLInputElement>("checkbox", { name }).checked,
    ).toBe(false);
  }
  expect(
    screen.getByRole<HTMLButtonElement>("button", {
      name: "Turn on source authorization one by one",
    }).disabled,
  ).toBe(true);
  expect(imports.connectSelected).not.toHaveBeenCalled();
});

it("S1.3 COLD-10: stale source state is qualified and refresh cannot start a connection", async () => {
  const imports = selectableImports();
  imports.sourceSnapshot = {
    confirmedAt: "2026-09-14T04:00:00.000Z",
    stale: true,
  };
  imports.refresh = vi.fn().mockRejectedValue(new Error("offline"));
  render(<SourcesView imports={imports} />);
  expect(screen.getByText("Status pending verification")).toBeTruthy();
  expect(
    screen.getByText(/Last confirmed/).querySelector("time")?.dateTime,
  ).toBe(imports.sourceSnapshot.confirmedAt);
  expect(document.querySelector(".coverage-line")?.textContent).toContain(
    "active jobs pending verification",
  );
  await userEvent.click(
    screen.getByRole("button", { name: "Refresh source status" }),
  );
  expect(imports.refresh).toHaveBeenCalledOnce();
  expect(imports.connectSelected).not.toHaveBeenCalled();
  expect(screen.getByRole("alert").textContent).toContain("No new status yet");
});

it("does not turn an unconfirmed source read into zero coverage", () => {
  const imports = selectableImports();
  imports.sourceSnapshot = { confirmedAt: null, stale: true };
  imports.jobs = [];
  render(<SourcesView imports={imports} />);
  const coverage = document.querySelector(".coverage-line")?.textContent ?? "";
  expect(coverage).toContain("The source state has not been verified");
  expect(coverage).not.toContain("0 apps connected");
  expect(coverage).not.toContain("0 can connect");
  expect(coverage).not.toContain("0 active jobs pending verification");
});

it("S1.3 COLD-19: late discovery and returning to applications do not infer selection", async () => {
  const imports = selectableImports();
  const view = render(<SourcesView imports={{ ...imports, sources: [] }} />);
  view.rerender(<SourcesView imports={imports} />);
  const source = screen.getByRole<HTMLInputElement>("checkbox", {
    name: /WorkBuddy/,
  });
  expect(source.checked).toBe(false);
  expect(source.disabled).toBe(false);
  view.unmount();
  render(<SourcesView imports={imports} />);
  expect(
    screen.getByRole<HTMLInputElement>("checkbox", { name: /WorkBuddy/ })
      .checked,
  ).toBe(false);
  expect(imports.connectSelected).not.toHaveBeenCalled();
});

it("IMP-013: a failed switch keeps the same source available for retry", async () => {
  const imports = selectableImports();
  vi.mocked(imports.connectSelected).mockRejectedValueOnce(
    new Error("connection failed"),
  );
  render(<SourcesView imports={imports} />);
  await userEvent.click(screen.getByRole("checkbox", { name: /WorkBuddy/ }));
  expect(await screen.findByRole("alert")).toBeTruthy();
  expect(screen.getByText("/isolated/workbuddy/projects")).toBeTruthy();
  expect(
    screen.getByRole<HTMLInputElement>("checkbox", { name: /WorkBuddy/ })
      .checked,
  ).toBe(false);
  await userEvent.click(screen.getByRole("checkbox", { name: /WorkBuddy/ }));
  expect(imports.connectSelected).toHaveBeenCalledTimes(2);
  cleanup();
});

it("IMP-009: polling cannot turn a pending switch into an authorized source", async () => {
  const imports = selectableImports();
  const view = render(<SourcesView imports={imports} />);
  view.rerender(
    <SourcesView
      imports={{
        ...imports,
        sources: imports.sources.map((item) => ({ ...item })),
      }}
    />,
  );
  expect(
    (screen.getByRole("checkbox", { name: /WorkBuddy/ }) as HTMLInputElement)
      .checked,
  ).toBe(false);
  expect(imports.connectSelected).not.toHaveBeenCalled();
  cleanup();
});

it("IMP-015: a source path change after starting is surfaced before retry", async () => {
  const imports = selectableImports();
  let first = true;
  vi.mocked(imports.connectSelected).mockImplementation(
    async (): Promise<ConnectSourcesResponse> => {
      if (!first)
        return {
          accepted_source_ids: ["workbuddy"],
          failed_sources: {},
          jobs: [],
        };
      first = false;
      return {
        accepted_source_ids: [],
        failed_sources: { workbuddy: "source_identity_changed" },
        jobs: [],
      };
    },
  );
  const view = render(<SourcesView imports={imports} />);
  await userEvent.click(screen.getByRole("checkbox", { name: /WorkBuddy/ }));
  expect(imports.connectSelected).toHaveBeenCalledTimes(1);
  expect(await screen.findByRole("alert")).toBeTruthy();
  view.rerender(
    <SourcesView
      imports={{
        ...imports,
        sources: imports.sources.map((item) => ({
          ...item,
          candidate_path: "/changed/private",
        })),
      }}
    />,
  );
  expect(screen.getByText("/changed/private")).toBeTruthy();
  await userEvent.click(screen.getByRole("checkbox", { name: /WorkBuddy/ }));
  expect(imports.connectSelected).toHaveBeenCalledTimes(2);
  expect(vi.mocked(imports.connectSelected).mock.calls[1][1]).toEqual([
    expect.objectContaining({ candidate_path: "/changed/private" }),
  ]);
  cleanup();
});

it("IMP-016: the source switch stays single-flight while authorization lands", async () => {
  const imports = selectableImports();
  let finish = () => {};
  vi.mocked(imports.connectSelected).mockImplementation(
    () =>
      new Promise((resolve) => {
        finish = () =>
          resolve({
            accepted_source_ids: ["workbuddy"],
            failed_sources: {},
            jobs: [],
          });
      }),
  );
  render(<SourcesView imports={imports} />);
  await userEvent.click(screen.getByRole("checkbox", { name: /WorkBuddy/ }));
  expect(
    screen.getByRole<HTMLInputElement>("checkbox", { name: /WorkBuddy/ })
      .disabled,
  ).toBe(true);
  await userEvent.dblClick(screen.getByRole("checkbox", { name: /WorkBuddy/ }));
  expect(imports.connectSelected).toHaveBeenCalledTimes(1);
  await act(async () => finish());
  cleanup();
});

function syncImports(): ReturnType<typeof useImportSources> {
  const imports = selectableImports();
  return {
    ...imports,
    sources: imports.sources.map((source) => ({
      ...source,
      authorized: true,
      grant_revision: 1,
      root_id: "root-workbuddy",
      sync_enabled: false,
      sync_completeness: "additions_only",
    })),
    setSync: vi.fn().mockResolvedValue(undefined),
    syncAll: vi.fn().mockResolvedValue(undefined),
  };
}

it.each([
  ["pending", "Waiting to check additions"],
  ["running", "Checking for additions"],
  ["paused", "Additions sync paused"],
  ["interrupted", "Additions sync interrupted"],
  ["failed", "Additions sync failed"],
  ["completed", "Additions check completed"],
  ["cancelled", "Additions sync cancelled"],
] as const)(
  "sync %s preserves cumulative history and removes engineering verification wording",
  (status, label) => {
    const imports = syncImports();
    imports.sources = imports.sources.map((source) => ({
      ...source,
      imported_messages: 81614,
      imported_conversations: 1761,
    }));
    const current: ImportJob = {
      job_id: "sync-zero",
      source_id: "workbuddy",
      user_id: "local",
      operation: "sync",
      status,
      phase: "read",
      checkpoint: null,
      discovered: 0,
      discovered_targets: 0,
      emitted: 0,
      stored: 0,
      merged: 81614,
      revised: 0,
      skipped: 0,
      errors: 0,
      created_at: "2026-09-14T00:00:00Z",
      updated_at: "2026-09-14T00:00:01Z",
    };
    imports.jobs = [current];
    imports.latestJobBySource = new Map([["workbuddy", current]]);
    const { container } = render(<SourcesView imports={imports} />);
    expect(container.querySelector(".source-state")?.textContent).toBe(label);
    expect(container.querySelector(".source-counts")?.textContent).toContain(
      "81614 messages",
    );
    expect(container.querySelector(".source-counts")?.textContent).toContain(
      "0 new from this sync",
    );
    expect(container.textContent).not.toContain("real sample replay");
    expect(
      container.querySelector(".coverage-line")?.textContent,
    ).not.toContain("importing history");
    if (status === "running" || status === "pending")
      expect(container.querySelector(".coverage-line")?.textContent).toContain(
        "1 checking for additions",
      );
  },
);

it("S1.3 COLD-12: source rows expose real job controls without coupling cancellation to revocation", async () => {
  const imports = syncImports();
  const job = {
    job_id: "ui-import-job",
    user_id: "local",
    source_id: "workbuddy",
    status: "running" as const,
    phase: "read" as const,
    checkpoint: null,
    discovered: 10,
    discovered_targets: 1,
    emitted: 8,
    stored: 8,
    revised: 0,
    merged: 0,
    skipped: 0,
    errors: 0,
    created_at: "2026-09-14T00:00:00Z",
    updated_at: "2026-09-14T00:00:01Z",
    operation: "history" as const,
  };
  imports.jobs = [job];
  imports.latestJobBySource = new Map([["workbuddy", job]]);
  imports.changeJob = vi.fn().mockResolvedValue({
    ...job,
    status: "paused",
    updated_at: "2026-09-14T00:00:02Z",
  });
  render(<SourcesView imports={imports} />);
  const actions = screen.getByRole("group", {
    name: "WorkBuddy import task actions",
  });
  await userEvent.click(
    within(actions).getByRole("button", { name: "Pause import" }),
  );
  expect(imports.changeJob).toHaveBeenCalledWith(job.job_id, "pause");
  expect(
    await within(actions).findByRole("button", { name: "Resume import" }),
  ).toBeTruthy();
  await userEvent.click(
    within(actions).getByRole("button", { name: "Cancel this import" }),
  );
  const dialog = screen.getByRole("dialog", {
    name: "Cancel this import for WorkBuddy?",
  });
  expect(within(dialog).getByText(/local copies are kept/)).toBeTruthy();
  expect(imports.changeJob).toHaveBeenCalledTimes(1);
  await userEvent.keyboard("{Escape}");
  await waitFor(() =>
    expect(
      screen.queryByRole("dialog", {
        name: "Cancel this import for WorkBuddy?",
      }),
    ).toBeNull(),
  );
  expect(imports.changeJob).toHaveBeenCalledTimes(1);
  expect(imports.setSync).not.toHaveBeenCalled();
});

it("Thaw and Freeze use the one Harness processing switch", async () => {
  const imports = syncImports();
  const view = render(<SourcesView imports={imports} />);
  const processing = screen.getByRole<HTMLInputElement>("checkbox", {
    name: "Process WorkBuddy",
  });
  expect(processing.checked).toBe(false);
  expect(
    screen.getByRole<HTMLButtonElement>("button", { name: "Sync sources" })
      .disabled,
  ).toBe(true);
  await userEvent.click(processing);
  expect(imports.setSync).toHaveBeenCalledWith(
    imports.sources[0],
    true,
    expect.any(String),
  );
  expect(imports.connectSelected).not.toHaveBeenCalled();
  view.rerender(
    <SourcesView
      imports={{
        ...imports,
        sources: imports.sources.map((source) => ({
          ...source,
          sync_enabled: true,
        })),
      }}
    />,
  );
  const thawed = screen.getByRole<HTMLInputElement>("checkbox", {
    name: "Process WorkBuddy",
  });
  expect(thawed.checked).toBe(true);
  await userEvent.click(screen.getByRole("button", { name: "Sync sources" }));
  expect(imports.syncAll).toHaveBeenCalledTimes(1);
  await userEvent.click(thawed);
  expect(imports.setSync).toHaveBeenLastCalledWith(
    expect.objectContaining({ source_id: "workbuddy", sync_enabled: true }),
    false,
    expect.any(String),
  );
});

it("S1.3: an incomplete permission receipt cannot activate global sync", () => {
  const imports = syncImports();
  render(
    <SourcesView
      imports={{
        ...imports,
        sources: imports.sources.map((source) => ({
          ...source,
          sync_enabled: true,
          root_id: null,
        })),
      }}
    />,
  );
  expect(
    screen.getByRole<HTMLInputElement>("checkbox", {
      name: "Process WorkBuddy",
    }).checked,
  ).toBe(true);
  expect(
    screen.getByRole<HTMLButtonElement>("button", { name: "Sync sources" })
      .disabled,
  ).toBe(true);
});
