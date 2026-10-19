// @vitest-environment jsdom

import type { SourceResponse } from "@one-memory/contracts";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { StrictMode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { HealthPanel } from "./HealthPanel";
import type { ImportSourcesController } from "./use-import-sources";

const getMemoryProcessing = vi.fn();
const getDetailedLogging = vi.fn(async () => false);
const setDetailedLogging = vi.fn(async (enabled: boolean) => enabled);

vi.mock("./api", () => ({
  getMemoryProcessing: (...args: unknown[]) => getMemoryProcessing(...args),
  getDetailedLogging: () => getDetailedLogging(),
  setDetailedLogging: (enabled: boolean) => setDetailedLogging(enabled),
}));

function source(
  sourceId: string,
  overrides: Partial<SourceResponse> = {},
): SourceResponse {
  return {
    source_id: sourceId,
    display_name: sourceId === "cursor" ? "Cursor" : "Claude Code",
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

function imports(
  overrides: Partial<ImportSourcesController> = {},
): ImportSourcesController {
  return {
    health: {
      status: "ok",
      service: "one-memory-core",
      version: "1.2.3",
      instance_id: "instance-1",
    },
    sources: [],
    jobs: [],
    loading: false,
    busyKey: null,
    error: null,
    lastConnectResult: null,
    latestJobBySource: new Map(),
    refresh: vi.fn(),
    connectForOnboarding: vi.fn(),
    connectSelected: vi.fn(),
    syncAll: vi.fn(),
    authorize: vi.fn(),
    detect: vi.fn(),
    startImport: vi.fn(),
    changeJob: vi.fn(),
    revoke: vi.fn(),
    ...overrides,
  } as ImportSourcesController;
}

function bridge(hosts: unknown[]) {
  window.oneMemory = {
    getApiToken: vi.fn(),
    getApiBaseUrl: vi.fn(),
    getRecallHosts: vi.fn().mockResolvedValue(hosts),
    previewRecallInstall: vi.fn(),
    probeRecallHost: vi.fn(),
    previewSkillSync: vi.fn(),
    applySkillSync: vi.fn(),
    installRecall: vi.fn(),
    previewRecallUninstall: vi.fn(),
    uninstallRecall: vi.fn(),
  } as unknown as NonNullable<Window["oneMemory"]>;
  return window.oneMemory;
}

describe("HealthPanel", () => {
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    delete window.oneMemory;
  });

  beforeEach(() => {
    vi.resetAllMocks();
    getDetailedLogging.mockResolvedValue(false);
    setDetailedLogging.mockImplementation(async (enabled: boolean) => enabled);
    getMemoryProcessing.mockResolvedValue({
      pending: 2,
      running: 1,
      fresh_pending: 2,
      backlog_pending: 0,
      recent_completions: [],
      waiting_for_model: 7,
      completed: 4,
      failed: 0,
      blocked_stages: [
        { kind: "extract", reason: "model_profile_not_active", count: 7 },
      ],
      outside_window_pending: 0,
      extraction_window_preset: "3m",
      manual_extract_batches_planned: 0,
      manual_extract_batches_current: 0,
      active_queue: 0,
      in_flight_count: 0,
      seconds_per_completion: 2,
      metrics: {
        raw_evidence: 100000,
        complete_turns: 50000,
        hy_calls: 120,
        completed: 100,
        unknown: 1,
        failed: 2,
        cancelled: 3,
        cancelled_reasons: { source_permission_changed: 3 },
        governed_memories: 80,
        workbuddy: { injected: 20, timed_out: 1, unavailable: 0 },
      },
    });
  });

  it("shows the complete import, HY and WorkBuddy counters", async () => {
    bridge([]);
    render(<HealthPanel imports={imports()} />);

    const details = await screen.findByLabelText(
      "Import and organizing details",
    );
    expect(details.textContent).toContain(
      "100000 originals · 50000 complete turns",
    );
    expect(details.textContent).toContain(
      "120 calls · 100 done · 1 uncertain result · 2 failed · 3 cancelled",
    );
    expect(details.textContent).toContain("80 memories");
    expect(details.textContent).toContain(
      "20 injections · 1 timeout · 0 unavailable",
    );
  });

  it("keeps detailed pipeline logging off by default and changes the Core setting", async () => {
    bridge([]);
    render(<HealthPanel imports={imports()} />);
    const toggle = await screen.findByRole("switch", {
      name: "Detailed logs (for debugging)",
    });
    expect(toggle.getAttribute("aria-checked")).toBe("false");
    fireEvent.click(toggle);
    await waitFor(() => expect(setDetailedLogging).toHaveBeenCalledWith(true));
    await waitFor(() =>
      expect(toggle.getAttribute("aria-checked")).toBe("true"),
    );
  });

  it("records a problem marker, copies its id and exports diagnostics", async () => {
    const desktop = bridge([]);
    desktop.markDiagnosticIncident = vi.fn().mockResolvedValue({
      incident_id: "11111111-1111-4111-8111-111111111111",
      session_id: "22222222-2222-4222-8222-222222222222",
      timestamp: "2026-09-24T01:00:00Z",
    });
    desktop.copyText = vi.fn().mockResolvedValue(undefined);
    desktop.exportDiagnostics = vi.fn().mockResolvedValue({
      state: "exported",
      filename: "MemHub-logs-3800-20261005-0300.zip",
      file_count: 12,
    });
    vi.spyOn(window, "prompt").mockReturnValue("Blueberries did not show up");
    render(<HealthPanel imports={imports()} />);

    fireEvent.click(screen.getByRole("button", { name: "Record issue time" }));
    expect(await screen.findByText(/Incident ID copied/)).toBeTruthy();
    expect(desktop.markDiagnosticIncident).toHaveBeenCalledWith(
      "Blueberries did not show up",
    );
    expect(desktop.copyText).toHaveBeenCalledWith(
      "11111111-1111-4111-8111-111111111111",
    );

    fireEvent.click(
      screen.getByRole("button", {
        name: "Export logs (send to developers for debugging)",
      }),
    );
    expect(await screen.findByText(/MemHub-logs-3800/)).toBeTruthy();
    expect(desktop.exportDiagnostics).toHaveBeenCalledOnce();
  });

  it("reports export failure instead of silently looking successful", async () => {
    const desktop = bridge([]);
    desktop.exportDiagnostics = vi
      .fn()
      .mockRejectedValue(new Error("disk_full"));
    render(<HealthPanel imports={imports()} />);

    fireEvent.click(
      screen.getByRole("button", {
        name: "Export logs (send to developers for debugging)",
      }),
    );

    expect(
      await screen.findByText("Failed to export diagnostics. Try again."),
    ).toBeTruthy();
    expect(screen.queryByText(/exported/i)).toBeNull();
  });

  it("the four status blocks each report their real numbers, not four placeholders", async () => {
    bridge([
      {
        hostId: "cursor",
        label: "Cursor",
        present: true,
        installed: true,
        selfCheck: { state: "wired", detail: "", probed_at: "" },
        skillDrift: null,
      },
    ]);

    render(
      <HealthPanel
        imports={imports({
          sources: [
            source("cursor", {
              imported_conversations: 6,
              imported_messages: 12,
            }),
          ],
        })}
      />,
    );

    expect(screen.getByLabelText("Local service").textContent).toContain(
      "1.2.3",
    );

    const sources = screen.getByLabelText("Per-source totals");
    expect(sources.textContent).toContain("Cursor");
    expect(sources.textContent).toContain("6");
    expect(sources.textContent).toContain("12");

    const pipeline = await screen.findByLabelText("Memory processing");
    expect(pipeline.textContent).toContain("Memory extraction");
    expect(pipeline.textContent).toContain(
      "No generation model is enabled yet",
    );

    expect(
      (await screen.findByLabelText("AI connections")).textContent,
    ).toContain("Connected");
  });

  it("the three status wordings reuse the sources-page vocabulary, nothing invented", async () => {
    bridge([
      {
        hostId: "cursor",
        label: "Cursor",
        present: true,
        installed: true,
        selfCheck: {
          state: "reachable_but_blind",
          detail: "",
          probed_at: "",
        },
        skillDrift: null,
      },
    ]);

    render(<HealthPanel imports={imports()} />);

    const hosts = await screen.findByLabelText("AI connections");
    expect(hosts.textContent).toContain("Installed, but no memories come back");
    expect(hosts.textContent).not.toContain("Connected");
  });

  it("with the desktop bridge absent it says not readable instead of falsely reporting not connected", async () => {
    render(<HealthPanel imports={imports()} />);

    const hosts = await screen.findByLabelText("AI connections");
    expect(hosts.textContent).toContain("Not readable in the browser");
    expect(hosts.textContent).not.toContain("Not connected");
  });

  it("the diagnostics panel does not print the pipeline's own error codes either", async () => {
    getMemoryProcessing.mockResolvedValue({
      pending: 0,
      running: 0,
      fresh_pending: 0,
      backlog_pending: 0,
      recent_completions: [],
      waiting_for_model: 0,
      completed: 0,
      failed: 3,
      blocked_stages: [
        { kind: "extract", reason: "model_api_key_missing", count: 1 },
        {
          kind: "extract",
          reason: "hy_memory_write_verification_clause_incomplete",
          count: 2,
        },
      ],
      outside_window_pending: 0,
      extraction_window_preset: "3m",
      manual_extract_batches_planned: 0,
      manual_extract_batches_current: 0,
      active_queue: 0,
      in_flight_count: 0,
      seconds_per_completion: null,
    });
    bridge([]);

    const { container } = render(<HealthPanel imports={imports()} />);

    const block = await screen.findByLabelText("Memory processing");
    expect(block.textContent).toContain(
      "Memory extraction stuck: Generation model API key missing (1 item)",
    );
    expect(block.textContent).toContain(
      "2 items: Failed to organize this round; retrying",
    );
    expect(container.textContent).not.toContain("hy_memory_");
  });

  it("does not report OK when Core is not connected", async () => {
    bridge([]);

    render(<HealthPanel imports={imports({ health: null })} />);

    const service = screen.getByLabelText("Local service");
    expect(service.textContent).toContain(
      "Not connected to the local Core yet",
    );
    expect(service.textContent).not.toContain("Running");
  });

  describe("Shared background service health", () => {
    const healthy = {
      state: "healthy",
      code: "background_verified",
      core: "running",
      broker: "running",
      instanceId: "instance-1",
    } as const;
    function background(value: unknown) {
      const desktop = bridge([]);
      desktop.getBackgroundHealth = vi.fn().mockResolvedValue(value);
      return vi.mocked(desktop.getBackgroundHealth);
    }

    it("double refresh does not re-check, late results after unmount do not affect the next open", async () => {
      const read = background(healthy);
      const { unmount } = render(<HealthPanel imports={imports()} />);
      await waitFor(() =>
        expect(
          screen.getByLabelText("Background services").textContent,
        ).toContain("Background services available"),
      );
      let resolve!: (value: typeof healthy) => void;
      read.mockReturnValueOnce(
        new Promise((done) => {
          resolve = done;
        }),
      );
      const button = screen.getByRole("button", { name: "Reload status" });
      fireEvent.click(button);
      fireEvent.click(button);
      expect(read).toHaveBeenCalledTimes(2);
      unmount();
      background({
        state: "stopped",
        code: "background_runtime_not_healthy",
        core: "stopped",
        broker: "not_checked",
      });
      render(<HealthPanel imports={imports()} />);
      await waitFor(() =>
        expect(
          screen.getByLabelText("Background services").textContent,
        ).toContain("Background services stopped"),
      );
      await act(async () => resolve(healthy));
      expect(
        screen.getByLabelText("Background services").textContent,
      ).not.toContain("Background services available");
    });

    it("reads shared background identity and bridge status, does not treat status checks as auto-start", async () => {
      const read = background(healthy);
      const state = imports();
      render(<HealthPanel imports={state} />);
      await waitFor(() =>
        expect(
          screen.getByLabelText("Background services").textContent,
        ).toContain("Background services available"),
      );
      const text = screen.getByLabelText("Background services").textContent;
      expect(text).toContain("Host bridge: Running");
      expect(text).toContain("Core:Running");
      expect(text).toContain("Last successful read");
      expect(read).toHaveBeenCalledExactlyOnceWith();
      expect(state.syncAll).not.toHaveBeenCalled();
      expect(window.oneMemory?.installRecall).not.toHaveBeenCalled();
      expect(window.oneMemory?.probeRecallHost).not.toHaveBeenCalled();
    });

    it.each([
      [
        {
          state: "degraded",
          code: "background_broker_not_healthy",
          core: "running",
          broker: "stopped",
        },
        "Host bridge: Stopped",
      ],
      [
        {
          state: "degraded",
          code: "background_broker_not_healthy",
          core: "running",
          broker: "unhealthy",
        },
        "Host bridge: Cannot connect",
      ],
      [
        {
          state: "untrusted",
          code: "background_broker_identity_mismatch",
          core: "running",
          broker: "untrusted",
        },
        "Host bridge: Identity failed verification",
      ],
      [
        {
          state: "stopped",
          code: "background_runtime_not_healthy",
          core: "stopped",
          broker: "not_checked",
        },
        "Background services stopped",
      ],
      [
        {
          state: "unknown",
          code: "background_identity_changed",
          core: "running",
          broker: "not_checked",
        },
        "Background status not confirmed yet",
      ],
    ])(
      "keeps the real background status instead of claiming all is well because Core is fine %#",
      async (value, expected) => {
        background(value);
        render(<HealthPanel imports={imports()} />);
        await waitFor(() =>
          expect(
            screen.getByLabelText("Background services").textContent,
          ).toContain(expected),
        );
        expect(
          screen.getByLabelText("Background services").textContent,
        ).not.toContain("Background services available");
      },
    );

    it.each([undefined, {}])(
      "explicitly reports not checked when the desktop background API is missing %#",
      async (desktop) => {
        if (desktop) bridge([]);
        render(<HealthPanel imports={imports()} />);
        await waitFor(() =>
          expect(
            screen.getByLabelText("Background services").textContent,
          ).toContain(
            "Background services cannot be checked in this environment",
          ),
        );
        expect(
          screen.getByLabelText("Background services").textContent,
        ).not.toContain("Background services available");
      },
    );

    it.each([
      null,
      { ...healthy, broker: "stopped" },
      { ...healthy, instanceId: undefined },
      { ...healthy, state: ["healthy"] },
      { ...healthy, core: "invented" },
    ])("rejects incomplete or contradictory receipts %#", async (value) => {
      background(value);
      render(<HealthPanel imports={imports()} />);
      await waitFor(() =>
        expect(
          screen.getByLabelText("Background services").textContent,
        ).toContain("Temporarily unavailable"),
      );
      expect(
        screen.getByLabelText("Background services").textContent,
      ).not.toContain("Background services available");
    });

    it("a failed reload marks data stale, shows fresh state after recovery, never prints exception text", async () => {
      const read = background(healthy);
      render(<HealthPanel imports={imports()} />);
      const block = screen.getByLabelText("Background services");
      await waitFor(() =>
        expect(block.textContent).toContain("Background services available"),
      );
      read.mockRejectedValueOnce(Error("PRIVATE_BACKGROUND_ERROR"));
      fireEvent.click(screen.getByRole("button", { name: "Reload status" }));
      await waitFor(() => expect(block.textContent).toContain("Data is stale"));
      expect(block.textContent).toContain(
        "Last read: Background services available",
      );
      expect(block.textContent).not.toContain("PRIVATE_BACKGROUND_ERROR");
      read.mockResolvedValue({
        state: "degraded",
        code: "background_broker_not_healthy",
        core: "running",
        broker: "stopped",
      });
      fireEvent.click(screen.getByRole("button", { name: "Reload status" }));
      await waitFor(() =>
        expect(block.textContent).toContain("Host bridge: Stopped"),
      );
      expect(block.textContent).not.toContain("Data is stale");
      expect(block.textContent).not.toContain("Background services available");
    });
  });

  describe("Read receipts", () => {
    beforeEach(() => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date("2026-09-19T02:10:00Z"));
    });

    async function settle() {
      await act(async () => {});
    }

    function deferred<T>() {
      let resolve!: (value: T) => void;
      let reject!: (reason: Error) => void;
      const promise = new Promise<T>((yes, no) => {
        resolve = yes;
        reject = no;
      });
      return { promise, resolve, reject };
    }

    it("first host read in flight does not report browser-unavailable or empty results", async () => {
      const desktop = bridge([]);
      vi.mocked(desktop.getRecallHosts).mockReturnValue(new Promise(() => {}));
      render(<HealthPanel imports={imports()} />);
      await settle();
      const text = screen.getByLabelText("AI connections").textContent;
      expect(text).toContain("Reading status");
      expect(text).not.toContain("browser");
      expect(text).not.toContain("AI tools were detected");
    });

    it("a rejected host read does not falsely report no tools and hides exception text", async () => {
      const desktop = bridge([]);
      vi.mocked(desktop.getRecallHosts).mockRejectedValue(
        new Error("private-transport-detail"),
      );
      render(<HealthPanel imports={imports()} />);
      await settle();
      const text = screen.getByLabelText("AI connections").textContent;
      expect(text).toContain("Temporarily unavailable");
      expect(text).toContain("Failed read at");
      expect(text).not.toContain("AI tools were detected");
      expect(text).not.toContain("private-transport-detail");
    });

    it("a synchronous desktop bridge throw also lands in recoverable read failure", async () => {
      const desktop = bridge([]);
      vi.mocked(desktop.getRecallHosts).mockImplementation(() => {
        throw new Error("bridge failed");
      });
      render(<HealthPanel imports={imports()} />);
      await settle();
      expect(screen.getByLabelText("AI connections").textContent).toContain(
        "Temporarily unavailable",
      );
    });

    it("only a successful empty host list shows the true empty state", async () => {
      bridge([{ hostId: "cursor", present: false }]);
      render(<HealthPanel imports={imports()} />);
      await settle();
      const text = screen.getByLabelText("AI connections").textContent;
      expect(text).toContain(
        "No connectable AI tools were detected on this machine",
      );
      expect(text).toContain("Last successful read");
    });

    it("a first extraction failure ends loading without a clean bill of health", async () => {
      getMemoryProcessing.mockRejectedValue(new Error("offline"));
      render(<HealthPanel imports={imports()} />);
      await settle();
      const text = screen.getByLabelText("Memory processing").textContent;
      expect(text).toContain("Temporarily unavailable");
      expect(text).not.toContain("Reading status");
      expect(text).not.toContain("Nothing needs attention right now");
    });

    it("a poll failure keeps numbers and marks stale data plus success/failure times, recovery clears errors", async () => {
      const healthy = {
        pending: 2,
        running: 1,
        completed: 4,
        failed: 0,
        blocked_stages: [],
      };
      getMemoryProcessing.mockResolvedValue(healthy);
      render(<HealthPanel imports={imports()} />);
      await settle();
      const block = screen.getByLabelText("Memory processing");
      expect(block.textContent).toContain("Nothing needs attention right now");
      getMemoryProcessing.mockRejectedValue(new Error("offline"));
      await act(async () => {
        await vi.advanceTimersByTimeAsync(5_000);
      });
      expect(block.textContent).toContain("Data is stale");
      expect(block.textContent).toContain("4 completed");
      expect(block.textContent).not.toContain(
        "Nothing needs attention right now",
      );
      expect(
        block.querySelector('time[datetime="2026-09-19T02:10:00.000Z"]'),
      ).not.toBeNull();
      expect(
        block.querySelector('time[datetime="2026-09-19T02:10:05.000Z"]'),
      ).not.toBeNull();
      getMemoryProcessing.mockResolvedValue(healthy);
      await act(async () => {
        await vi.advanceTimersByTimeAsync(5_000);
      });
      expect(block.textContent).not.toContain("Data is stale");
      expect(block.textContent).not.toContain("Failed read at");
      expect(block.textContent).toContain("Nothing needs attention right now");
    });

    it("a stale sourceSnapshot stops claiming the service is running or sources are empty", async () => {
      render(
        <HealthPanel
          imports={imports({
            sourceSnapshot: {
              stale: true,
              confirmedAt: "2026-09-19T02:06:00Z",
            },
            error: "offline",
          })}
        />,
      );
      await settle();
      const service = screen.getByLabelText("Local service");
      expect(service.textContent).toContain("Data is stale");
      expect(service.textContent).toContain("1.2.3");
      expect(service.textContent).not.toContain("Running");
      expect(service.querySelector("time")?.dateTime).toBe(
        "2026-09-19T02:06:00Z",
      );
      const sources = screen.getByLabelText("Per-source totals");
      expect(sources.textContent).toContain("Temporarily unavailable");
      expect(sources.textContent).not.toContain("No sources connected yet");
    });

    it("first source load/failure is separate from the confirmed empty state", async () => {
      const { rerender } = render(
        <HealthPanel
          imports={imports({
            health: null,
            healthStale: false,
            loading: true,
            sourceSnapshot: { stale: true, confirmedAt: null },
          })}
        />,
      );
      await settle();
      expect(screen.getByLabelText("Per-source totals").textContent).toContain(
        "Reading local service and source status...",
      );
      rerender(
        <HealthPanel
          imports={imports({
            health: null,
            healthStale: false,
            loading: false,
            error: "offline",
            sourceSnapshot: { stale: true, confirmedAt: null },
          })}
        />,
      );
      expect(screen.getByLabelText("Per-source totals").textContent).toContain(
        "Temporarily unavailable",
      );
      rerender(
        <HealthPanel
          imports={imports({
            sourceSnapshot: {
              stale: false,
              confirmedAt: "2026-09-19T02:10:00Z",
            },
          })}
        />,
      );
      expect(screen.getByLabelText("Per-source totals").textContent).toContain(
        "No sources connected yet",
      );
    });

    it("reload only calls existing read-only APIs and recovers host errors, no repeat submits on click", async () => {
      const desktop = bridge([]);
      const readHosts = vi.mocked(desktop.getRecallHosts);
      readHosts.mockRejectedValueOnce(new Error("offline"));
      const pending = deferred<never[]>();
      const state = imports({
        refresh: vi.fn().mockReturnValue(pending.promise),
      });
      render(<HealthPanel imports={state} />);
      await settle();
      const button = screen.getByRole("button", { name: "Reload status" });
      fireEvent.click(button);
      fireEvent.click(button);
      await settle();
      expect(state.refresh).toHaveBeenCalledTimes(1);
      expect(readHosts).toHaveBeenCalledTimes(2);
      expect(getMemoryProcessing).toHaveBeenCalledTimes(2);
      expect(desktop.probeRecallHost).not.toHaveBeenCalled();
      expect(desktop.installRecall).not.toHaveBeenCalled();
      expect(state.detect).not.toHaveBeenCalled();
      expect(state.syncAll).not.toHaveBeenCalled();
      expect(screen.getByLabelText("AI connections").textContent).not.toContain(
        "Temporarily unavailable",
      );
      pending.resolve([]);
      await settle();
      expect((button as HTMLButtonElement).disabled).toBe(false);
    });

    it("after host success a failed refresh keeps the old list without claiming currently connected", async () => {
      const desktop = bridge([
        {
          hostId: "cursor",
          label: "Cursor",
          present: true,
          installed: true,
          selfCheck: { state: "wired" },
        },
      ]);
      render(<HealthPanel imports={imports()} />);
      await settle();
      vi.mocked(desktop.getRecallHosts).mockRejectedValue(new Error("offline"));
      fireEvent.click(screen.getByRole("button", { name: "Reload status" }));
      await settle();
      const text = screen.getByLabelText("AI connections").textContent;
      expect(text).toContain("Data is stale");
      expect(text).toContain("Cursor");
      expect(text).toContain("Last read: Connected");
      expect(text).not.toContain("AI tools were detected");
    });

    it.each(["resolve", "reject"] as const)(
      "a late StrictMode legacy lifecycle %s does not overwrite newer receipts",
      async (outcome) => {
        const old = deferred<unknown>();
        getMemoryProcessing.mockReturnValueOnce(old.promise);
        const desktop = bridge([
          {
            hostId: "cursor",
            label: "Cursor",
            present: true,
            installed: true,
            selfCheck: { state: "wired" },
          },
        ]);
        const oldHosts = deferred<never[]>();
        vi.mocked(desktop.getRecallHosts).mockReturnValueOnce(oldHosts.promise);
        render(
          <StrictMode>
            <HealthPanel imports={imports()} />
          </StrictMode>,
        );
        await settle();
        if (outcome === "resolve") {
          old.resolve({
            pending: 99,
            running: 0,
            completed: 0,
            failed: 0,
            blocked_stages: [],
          });
          oldHosts.resolve([]);
        } else {
          old.reject(new Error("late"));
          oldHosts.reject(new Error("late"));
        }
        await settle();
        expect(
          screen.getByLabelText("Memory processing").textContent,
        ).toContain("4 completed");
        expect(
          screen.getByLabelText("Memory processing").textContent,
        ).not.toContain("Data is stale");
        expect(
          screen.getByLabelText("AI connections").textContent,
        ).not.toContain("Temporarily unavailable");
        expect(screen.getByLabelText("AI connections").textContent).toContain(
          "Cursor",
        );
      },
    );

    it.each(["resolve", "reject"] as const)(
      "receipts after an explicit refresh are not overwritten by stale request %s",
      async (outcome) => {
        const old = deferred<unknown>();
        const oldHosts = deferred<never[]>();
        getMemoryProcessing.mockReturnValueOnce(old.promise);
        const desktop = bridge([
          {
            hostId: "cursor",
            label: "Cursor",
            present: true,
            installed: true,
            selfCheck: { state: "wired" },
          },
        ]);
        vi.mocked(desktop.getRecallHosts).mockReturnValueOnce(oldHosts.promise);
        render(<HealthPanel imports={imports()} />);
        await settle();
        fireEvent.click(screen.getByRole("button", { name: "Reload status" }));
        await settle();
        if (outcome === "resolve") {
          old.resolve({
            pending: 99,
            running: 0,
            completed: 0,
            failed: 0,
            blocked_stages: [],
          });
          oldHosts.resolve([]);
        } else {
          old.reject(new Error("late"));
          oldHosts.reject(new Error("late"));
        }
        await settle();
        expect(
          screen.getByLabelText("Memory processing").textContent,
        ).toContain("4 completed");
        expect(
          screen.getByLabelText("Memory processing").textContent,
        ).not.toContain("Data is stale");
        expect(screen.getByLabelText("AI connections").textContent).toContain(
          "Cursor",
        );
        expect(
          screen.getByLabelText("AI connections").textContent,
        ).not.toContain("Temporarily unavailable");
      },
    );

    it("a source refresh failure does not hide other successful receipts, old errors clear when the parent snapshot recovers", async () => {
      bridge([]);
      const state = imports({
        refresh: vi.fn().mockRejectedValue(new Error("offline")),
        sourceSnapshot: { stale: true, confirmedAt: "2026-09-19T02:06:00Z" },
      });
      const { rerender } = render(<HealthPanel imports={state} />);
      await settle();
      fireEvent.click(screen.getByRole("button", { name: "Reload status" }));
      await settle();
      expect(screen.getByLabelText("Local service").textContent).toContain(
        "Failed read at",
      );
      expect(screen.getByLabelText("Memory processing").textContent).toContain(
        "4 completed",
      );
      expect(
        (
          screen.getByRole("button", {
            name: "Reload status",
          }) as HTMLButtonElement
        ).disabled,
      ).toBe(false);
      rerender(
        <HealthPanel
          imports={imports({
            sourceSnapshot: {
              stale: false,
              confirmedAt: "2026-09-19T02:11:00Z",
            },
          })}
        />,
      );
      expect(screen.getByLabelText("Local service").textContent).toContain(
        "Running",
      );
      expect(screen.getByLabelText("Local service").textContent).not.toContain(
        "Failed read at",
      );
    });

    it("a failed source operation does not overwrite a confirmed fresh read snapshot", async () => {
      render(
        <HealthPanel
          imports={imports({
            error: "operation failed",
            sourceSnapshot: {
              stale: false,
              confirmedAt: "2026-09-19T02:10:00Z",
            },
          })}
        />,
      );
      await settle();
      expect(screen.getByLabelText("Local service").textContent).toContain(
        "Running",
      );
      expect(screen.getByLabelText("Local service").textContent).not.toContain(
        "Data is stale",
      );
    });

    it("pending polls do not pile up, no polling after unmount and late failures are handled", async () => {
      const pending = deferred<unknown>();
      getMemoryProcessing.mockReturnValue(pending.promise);
      const { unmount } = render(<HealthPanel imports={imports()} />);
      await act(async () => {
        await vi.advanceTimersByTimeAsync(15_000);
      });
      expect(getMemoryProcessing).toHaveBeenCalledTimes(1);
      unmount();
      pending.reject(new Error("late"));
      await act(async () => {
        await vi.advanceTimersByTimeAsync(10_000);
      });
      expect(getMemoryProcessing).toHaveBeenCalledTimes(1);
      expect(vi.getTimerCount()).toBe(0);
    });
  });
});
