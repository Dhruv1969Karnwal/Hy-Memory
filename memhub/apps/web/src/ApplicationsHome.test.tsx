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
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ApplicationsHome,
  CALL_HISTORY_ENTRY_VISIBLE,
} from "./ApplicationsHome";
import * as api from "./api";
import { saveConnectionPreferences } from "./connection-preferences";
import { resetConnectionReconcile } from "./connection-reconcile";
import { queryClient } from "./query-client";

const { summary, processing, history, liveSync, recentConversations } =
  vi.hoisted(() => ({
    summary: vi.fn(),
    processing: vi.fn(),
    history: vi.fn(),
    liveSync: vi.fn(),
    recentConversations: vi.fn(),
  }));
vi.mock("./call-history-client", async (original) => ({
  ...(await original<typeof import("./call-history-client")>()),
  getCallHistory: history,
}));
vi.mock("./api", () => ({
  getHomeSummary: summary,
  getLiveSyncHealth: liveSync,
  getMemoryProcessing: processing,
  getRecentConversations: recentConversations,
  getSources: vi.fn().mockResolvedValue([]),
}));
const sources = Array.from(
  { length: 5 },
  (_, i) =>
    ({
      source_id: `app-${i}`,
      display_name: `App ${i}`,
      authorized: true,
      sync_enabled: false,
      available: true,
    }) as SourceResponse,
);
const imports = {
  sources,
  jobs: [],
  loading: false,
  health: { status: "ok" },
  sourceSnapshot: { stale: false },
} as unknown as Parameters<typeof ApplicationsHome>[0]["imports"];
beforeEach(() => {
  history
    .mockReset()
    .mockResolvedValue({ available: true, recording_error: false, items: [] });
  imports.refresh = vi.fn();
  summary.mockReset().mockResolvedValue({
    memory_count: 601,
    latest: {
      memory_id: "m1",
      content: "A real governed memory",
      truncated: false,
      revision: 2,
      source_ids: ["app-0"],
      updated_at: "2026-09-19T00:00:00Z",
    },
    observed_at: "2026-09-19T00:00:00Z",
  });
  processing.mockReset().mockResolvedValue({ running: 0, pending: 0 });
  liveSync.mockReset().mockResolvedValue([]);
  recentConversations.mockReset().mockResolvedValue({
    items: [],
    total: 0,
    offset: 0,
    limit: 1,
  });
  window.oneMemory = {
    restartBackground: vi.fn().mockResolvedValue({ state: "restarted" }),
    getRecallHosts: vi.fn().mockResolvedValue([
      {
        hostId: "app-0",
        label: "App 0",
        present: true,
        installed: true,
        permission: { state: "approved" },
      },
      {
        hostId: "app-1",
        label: "App 1",
        present: true,
        installed: true,
        permission: { state: "approved" },
      },
    ]),
  } as unknown as NonNullable<typeof window.oneMemory>;
});
it("hides a debug-frame dialogue excerpt from Recent updates and falls back to memory", async () => {
  recentConversations.mockResolvedValue({
    items: [
      {
        evidence_id: "ev-1",
        conversation_id: "c1",
        source_id: "app-1",
        role: "user",
        written_by: "user",
        created_at: "2026-10-12T06:06:00Z",
        imported_at: "2026-10-12T06:06:00Z",
        time_basis: "conversation",
        excerpt: '[write_stdout] {"chars":"{\\"session_id\\":93099}',
        truncated: false,
      },
    ],
    total: 1,
    offset: 0,
    limit: 1,
  });
  render(
    <ApplicationsHome
      imports={imports}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
    />,
  );
  await screen.findByText("A real governed memory");
  expect(screen.queryByText(/write_stdout/)).toBeNull();
});

it("shows latest recorded call separately from newly generated memory and opens its history", async () => {
  history.mockResolvedValue({
    available: true,
    recording_error: false,
    items: [
      {
        event_id: "event-1",
        host_id: "codex",
        state: "grounded",
        occurred_at: new Date().toISOString(),
        memory_count: 2,
        raw_count: 1,
        references: [
          { kind: "memory", id: "m1", revision: 2, source_id: "app-0" },
        ],
      },
    ],
  });
  const onCalls = vi.fn();
  const onMemories = vi.fn();
  render(
    <ApplicationsHome
      imports={imports}
      onManage={() => {}}
      onMemories={onMemories}
      onSettings={() => {}}
      onCalls={onCalls}
    />,
  );
  await screen.findByText("A real governed memory");
  fireEvent.click(screen.getByRole("tab", { name: "Recent calls" }));
  await screen.findByText("Codex found 3 related items");
  expect(document.querySelector(".r6-home-call-direction")).toBeTruthy();
  expect(screen.queryByText("A real governed memory")).toBeNull();
  expect(screen.getByText(/Today .+ · Query succeeded/)).toBeTruthy();
  fireEvent.click(screen.getByRole("tab", { name: "Recent updates" }));
  expect(screen.getByText("A real governed memory")).toBeTruthy();
  const memoryActions = screen.getAllByRole("button", {
    name: "View memories",
  });
  fireEvent.click(memoryActions[memoryActions.length - 1]);
  expect(onMemories).toHaveBeenCalledWith("m1");
  fireEvent.click(screen.getByRole("tab", { name: "Recent calls" }));
  expect(history).toHaveBeenCalledWith({ limit: 20, offset: 0 });
  if (CALL_HISTORY_ENTRY_VISIBLE) {
    fireEvent.click(screen.getByRole("button", { name: "View records" }));
    expect(onCalls).toHaveBeenCalledOnce();
  } else {
    // D96: the recent call still shows on the home page; only the way into
    // the call-history page is hidden.
    expect(screen.getByText(/Today .+ · Query succeeded/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: "View records" })).toBeNull();
    expect(onCalls).not.toHaveBeenCalled();
  }
});
it("shows only the host for memory hits whose references have no source", async () => {
  history.mockResolvedValue({
    available: true,
    recording_error: false,
    items: [
      {
        event_id: "event-1",
        host_id: "workbuddy",
        tool: "om_recall",
        state: "grounded",
        occurred_at: new Date().toISOString(),
        memory_count: 2,
        raw_count: 0,
        references: [
          { kind: "memory", id: "m1", revision: 2, source_id: null },
        ],
      },
    ],
  });
  render(
    <ApplicationsHome
      imports={imports}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
      onCalls={() => {}}
    />,
  );
  fireEvent.click(await screen.findByRole("tab", { name: "Recent calls" }));
  await screen.findByText("WorkBuddy found 2 related items");
  const direction = document.querySelector(".r6-home-call-direction");
  expect(direction?.querySelectorAll("img")).toHaveLength(1);
  expect(direction?.querySelector("img")?.getAttribute("src")).toContain(
    "/r6/workbuddy.png",
  );
  expect(direction?.querySelector("svg")).toBeNull();
  expect(direction?.querySelector(".r6-app-initial")).toBeNull();
});
it("keeps source to host direction for raw hits with a source", async () => {
  history.mockResolvedValue({
    available: true,
    recording_error: false,
    items: [
      {
        event_id: "event-1",
        host_id: "workbuddy",
        tool: "om_recall",
        state: "grounded",
        occurred_at: new Date().toISOString(),
        memory_count: 0,
        raw_count: 1,
        references: [
          { kind: "raw", id: "e1", revision: 1, source_id: "app-0" },
        ],
      },
    ],
  });
  render(
    <ApplicationsHome
      imports={imports}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
      onCalls={() => {}}
    />,
  );
  fireEvent.click(await screen.findByRole("tab", { name: "Recent calls" }));
  await screen.findByText("WorkBuddy found 1 related item");
  const direction = document.querySelector(".r6-home-call-direction");
  expect(direction?.querySelectorAll("img")).toHaveLength(1);
  expect(direction?.querySelector("svg")).toBeTruthy();
  expect(direction?.querySelector(".r6-app-initial")?.textContent).toBe("A");
});
it("shows unlinked apps and the latest returned call on matching rows", async () => {
  history.mockResolvedValue({
    available: true,
    recording_error: false,
    items: [
      {
        event_id: "event-1",
        host_id: "cursor",
        state: "grounded",
        occurred_at: new Date().toISOString(),
        memory_count: 2,
        raw_count: 1,
      },
    ],
  });
  render(
    <ApplicationsHome
      imports={
        {
          ...imports,
          sources: [
            {
              ...sources[0],
              source_id: "cursor",
              display_name: "Cursor",
            },
            {
              ...sources[1],
              source_id: "codebuddy",
              display_name: "CodeBuddy",
              authorized: false,
              detected: true,
            },
          ],
        } as typeof imports
      }
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
      onCalls={() => {}}
    />,
  );
  expect(
    await screen.findByText(/Cursor found 3 related items ·/),
  ).toBeTruthy();
  expect(screen.getByText("Not linked")).toBeTruthy();
});
it("keeps the latest call visible when recording is incomplete", async () => {
  history.mockResolvedValue({
    available: true,
    recording_error: true,
    items: [
      {
        event_id: "event-1",
        host_id: "cursor",
        state: "grounded",
        occurred_at: new Date().toISOString(),
        memory_count: 1,
        raw_count: 0,
      },
    ],
  });
  render(
    <ApplicationsHome
      imports={imports}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
      onCalls={() => {}}
    />,
  );
  fireEvent.click(await screen.findByRole("tab", { name: "Recent calls" }));
  await screen.findByText("Cursor found 1 related item");
  expect(screen.getByText(/a few calls were not recorded/)).toBeTruthy();
  expect(
    screen.queryByText("Call records cannot be read right now"),
  ).toBeNull();
  expect(screen.queryByText("No recorded calls yet")).toBeNull();
});
it("does not present failed or disabled call history as zero usage", async () => {
  history.mockRejectedValue(Error("private error"));
  render(
    <ApplicationsHome
      imports={imports}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
      onCalls={() => {}}
    />,
  );
  await screen.findByText("A real governed memory");
  expect(screen.queryByText("No recorded calls yet")).toBeNull();
  expect(document.body.textContent).not.toContain("private error");
  expect(
    screen.queryByText("Call records cannot be read right now"),
  ).toBeNull();
});
it("marks a retained application activity as last-read when call history cannot refresh", async () => {
  history
    .mockResolvedValueOnce({
      available: true,
      recording_error: false,
      items: [
        {
          event_id: "event-1",
          host_id: "app-0",
          state: "grounded",
          occurred_at: new Date().toISOString(),
          memory_count: 2,
          raw_count: 1,
          references: [
            { kind: "memory", id: "m1", revision: 2, source_id: "app-0" },
          ],
        },
      ],
    })
    .mockRejectedValueOnce(Error("refresh failed"));
  const view = render(
    <ApplicationsHome
      active
      imports={imports}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
      onCalls={() => {}}
    />,
  );

  await screen.findByText(/App 0 found 3 related items ·/);
  view.rerender(
    <ApplicationsHome
      active={false}
      imports={imports}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
      onCalls={() => {}}
    />,
  );
  view.rerender(
    <ApplicationsHome
      active
      imports={imports}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
      onCalls={() => {}}
    />,
  );
  await waitFor(() => expect(history).toHaveBeenCalledTimes(2));

  expect(await screen.findByText("A real governed memory")).toBeTruthy();
  expect(
    screen.getByText(/Last read: App 0 found 3 related items/),
  ).toBeTruthy();
  expect(screen.queryByText(/^App 0 found/)).toBeNull();
});
it("does not say an unavailable call returned memories", async () => {
  history.mockResolvedValue({
    available: true,
    recording_error: false,
    items: [
      {
        event_id: "event-1",
        host_id: "app-0",
        state: "unavailable",
        occurred_at: new Date().toISOString(),
        memory_count: 0,
        raw_count: 0,
        references: [],
      },
    ],
  });
  render(
    <ApplicationsHome
      imports={imports}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
      onCalls={() => {}}
    />,
  );

  expect(
    await screen.findByText(
      "The background service is not running, so apps cannot query memories right now",
    ),
  ).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Restart" }));
  await waitFor(() =>
    expect(window.oneMemory?.restartBackground).toHaveBeenCalledOnce(),
  );
  expect(
    screen.queryByText(/App 0 cannot query memories right now/),
  ).toBeNull();
  expect(screen.queryByText(/found [0-9]+ related items/)).toBeNull();
});

it("warns only after three consecutive timeouts and keeps them out of app rows", async () => {
  history.mockResolvedValue({
    available: true,
    recording_error: false,
    items: Array.from({ length: 3 }, (_, index) => ({
      event_id: `timeout-${index}`,
      host_id: "workbuddy",
      tool: "om_recall",
      state: "timed_out",
      occurred_at: new Date(Date.now() - index * 1000).toISOString(),
      memory_count: 0,
      raw_count: 0,
      references: [],
    })),
  });
  const onCalls = vi.fn();
  render(
    <ApplicationsHome
      imports={imports}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
      onCalls={onCalls}
    />,
  );

  expect(
    await screen.findByText(
      "WorkBuddy found no memories in its last few calls; the background service may be busy and will recover on its own",
    ),
  ).toBeTruthy();
  expect(
    screen.queryByText(/WorkBuddy did not get memories back in time/),
  ).toBeNull();
  if (CALL_HISTORY_ENTRY_VISIBLE) {
    fireEvent.click(screen.getByRole("button", { name: "View records" }));
    expect(onCalls).toHaveBeenCalledOnce();
  } else {
    // D96: the warning stays, without a button into the call history.
    expect(screen.queryByRole("button", { name: "View records" })).toBeNull();
  }
});
it("uses live-sync success time for an application row", async () => {
  liveSync.mockResolvedValue([
    {
      source_id: "app-0",
      mode: "watch_and_polling",
      state: "idle",
      last_job_id: "job-1",
      last_attempt_at: "2026-09-20T09:00:00Z",
      last_success_at: new Date().toISOString(),
      last_error_code: null,
    },
  ]);
  render(
    <ApplicationsHome
      imports={imports}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
    />,
  );
  expect(await screen.findByText(/Last sync Today/)).toBeTruthy();
});
it.each([
  [{ running: 1, pending: 0 }, "Distilling the first memories"],
  [{ running: 0, pending: 1 }, "Preparing to distill the first memories"],
  [
    { running: 0, pending: 0, waiting_for_model: 1 },
    "Preparing to distill the first memories",
  ],
])(
  "renders the first-memory state from processing: %j",
  async (value, label) => {
    summary.mockResolvedValue({
      memory_count: 0,
      latest: null,
      observed_at: "2026-09-20T09:00:00Z",
    });
    processing.mockResolvedValue(value);
    render(
      <ApplicationsHome
        imports={imports}
        onManage={() => {}}
        onMemories={() => {}}
        onSettings={() => {}}
        onCalls={() => {}}
      />,
    );
    expect(await screen.findByText(label)).toBeTruthy();
    expect(
      screen.getByText(
        "History is saved; generated memories will appear here.",
      ),
    ).toBeTruthy();
  },
);

afterEach(() => {
  cleanup();
  resetConnectionReconcile();
  window.localStorage.clear();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  delete document.documentElement.dataset.nativeReduceMotion;
  Reflect.deleteProperty(Element.prototype, "animate");
});
it("reads sources again on a later visit once the earlier read has gone stale", async () => {
  const getSources = vi.mocked(api.getSources);
  getSources.mockClear();
  const home = (active: boolean) => (
    <ApplicationsHome
      imports={imports}
      active={active}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
    />
  );
  const view = render(home(true));
  await waitFor(() => expect(getSources).toHaveBeenCalledTimes(1));
  await act(async () => {
    await getSources.mock.results[0]?.value;
  });
  const later = Date.now() + 10 * 60_000;
  vi.spyOn(Date, "now").mockReturnValue(later);
  view.rerender(home(false));
  view.rerender(home(true));
  await waitFor(() => expect(getSources).toHaveBeenCalledTimes(2));
});
it("uses full summary count, renders latest real memory and expands all applications", async () => {
  render(
    <ApplicationsHome
      imports={imports}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
    />,
  );
  await screen.findByText("601");
  expect(screen.getByText("memories")).toBeTruthy();
  expect(screen.getByText("Backend running")).toBeTruthy();
  expect(screen.getByText("2 apps connected")).toBeTruthy();
  expect(screen.getByText("A real governed memory")).toBeTruthy();
  expect(screen.queryByRole("button", { name: "Add app" })).toBeNull();
  expect(screen.queryByText("App 4")).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "View all 5 apps" }));
  expect(screen.getByText("App 4")).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Back to home" }));
  expect(screen.queryByText("App 4")).toBeNull();
});
it("keeps authoritative home and processing data when only the source snapshot becomes stale", async () => {
  const view = render(
    <ApplicationsHome
      imports={imports}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
    />,
  );
  await screen.findByText("A real governed memory");
  view.rerender(
    <ApplicationsHome
      imports={
        {
          ...imports,
          sourceSnapshot: { ...imports.sourceSnapshot, stale: true },
        } as typeof imports
      }
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
    />,
  );
  expect(screen.getByText("A real governed memory")).toBeTruthy();
  expect(screen.getByText("601")).toBeTruthy();
  expect(screen.getByText("Backend running")).toBeTruthy();
  expect(screen.getByText("2 apps connected")).toBeTruthy();
  expect(screen.getAllByText("All set")).toHaveLength(2);
  expect(screen.queryByText("Status pending verification")).toBeNull();
});
it("does not turn a failed home summary into zero or expose a private error", async () => {
  summary.mockRejectedValue(new Error("private offline detail"));
  render(
    <ApplicationsHome
      imports={imports}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
    />,
  );
  await waitFor(() =>
    expect(screen.getByText("Temporarily unreadable")).toBeTruthy(),
  );
  expect(screen.queryByText("0")).toBeNull();
  expect(document.body.textContent).not.toContain("private offline detail");
});
it("never calls a recall event a success when only stored memories are available", async () => {
  render(
    <ApplicationsHome
      imports={imports}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
    />,
  );
  await screen.findByText("A real governed memory");
  expect(screen.queryByText(/returned|adopted/)).toBeNull();
});

it.each([
  [{ running: 0, pending: 0 }, "All set"],
  [{ running: 0, pending: 0, waiting_for_model: 1 }, "Waiting for the model"],
  [
    { running: 0, pending: 0, blocked_stages: [{ count: 1 }] },
    "Queued for processing",
  ],
  [{ running: 0, pending: 0, failed: 1 }, "Needs attention"],
])("does not confuse task state with readiness: %j", async (value, label) => {
  processing.mockResolvedValue(value);
  render(
    <ApplicationsHome
      imports={imports}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
    />,
  );
  expect(await screen.findAllByText(label)).toHaveLength(2);
  expect(screen.queryByText("Ready")).toBeNull();
});

it("keeps the import-in-progress empty state stable without a progress link", async () => {
  summary.mockResolvedValue({
    memory_count: 0,
    latest: null,
    observed_at: "2026-09-19T00:00:00Z",
  });
  const importing = {
    ...imports,
    jobs: [{ source_id: "app-0", status: "running" }],
  } as unknown as Parameters<typeof ApplicationsHome>[0]["imports"];
  render(
    <ApplicationsHome
      imports={importing}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
    />,
  );
  expect(
    await screen.findByText(
      "Conversations are still importing; your memories will be distilled shortly",
    ),
  ).toBeTruthy();
  expect(screen.queryByRole("button", { name: "View progress" })).toBeNull();
});

it.each([null, "", "not-a-date"])(
  "keeps the home usable when the latest timestamp is unavailable: %s",
  async (updated_at) => {
    summary.mockResolvedValue({
      memory_count: 1,
      latest: {
        memory_id: "m1",
        content: "A real governed memory",
        revision: 2,
        source_ids: ["app-0"],
        updated_at,
      },
    });
    render(
      <ApplicationsHome
        imports={imports}
        onManage={() => {}}
        onMemories={() => {}}
        onSettings={() => {}}
      />,
    );
    await screen.findByText("A real governed memory");
    expect(screen.getByText(/Time unknown.*version 2/)).toBeTruthy();
  },
);

it("does not treat an available adapter as an installed application", async () => {
  render(
    <ApplicationsHome
      imports={{
        ...imports,
        sources: [
          {
            ...sources[0],
            authorized: false,
            detected: false,
            available: true,
            display_name: "Not installed",
          },
        ],
      }}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
    />,
  );
  await screen.findByText("601");
  expect(screen.queryByRole("heading", { name: "Not installed" })).toBeNull();
});

function mountHome(active = true) {
  return (
    <ApplicationsHome
      imports={imports}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
      active={active}
    />
  );
}

function mockMotion() {
  const media = new EventTarget() as EventTarget & { matches: boolean };
  media.matches = false;
  vi.stubGlobal("matchMedia", () => media);
  const jobs: {
    element: Element;
    frames: Keyframe[];
    options: KeyframeAnimationOptions;
    cancel: ReturnType<typeof vi.fn>;
    finish(): void;
  }[] = [];
  Object.defineProperty(Element.prototype, "animate", {
    configurable: true,
    value: vi.fn(function (
      this: Element,
      frames: Keyframe[],
      options: KeyframeAnimationOptions,
    ) {
      let finish = () => {};
      const finished = new Promise<void>((resolve) => {
        finish = resolve;
      });
      const cancel = vi.fn(finish);
      jobs.push({ element: this, frames, options, cancel, finish });
      return { finished, cancel };
    }),
  });
  return {
    jobs,
    media,
    finish: async () => {
      await act(async () => {
        jobs
          .filter((j) => !j.cancel.mock.calls.length)
          .forEach((j) => {
            j.finish();
          });
      });
    },
  };
}

it("sequences a quick overview exit before translation-only expansion and reverses it", async () => {
  const motion = mockMotion();
  const view = render(mountHome());
  await screen.findByText("601");
  const expand = screen.getByRole("button", { name: "View all 5 apps" });
  fireEvent.click(expand);
  expect(motion.jobs).toHaveLength(1);
  expect(motion.jobs[0].element.className).toBe("r6-home-top");
  expect(motion.jobs[0].options.duration).toBe(90);
  expect(screen.queryByText("App 4")).toBeNull();
  await motion.finish();
  expect(screen.getByText("App 4")).toBeTruthy();
  expect(
    motion.jobs.find((j) => j.element.className === "r6-home-apps")?.options
      .duration,
  ).toBe(190);
  expect(JSON.stringify(motion.jobs.map((j) => j.frames))).not.toMatch(/scale/);
  await motion.finish();
  expect(document.activeElement).toBe(
    screen.getByRole("button", { name: "Back to home" }),
  );
  fireEvent.click(screen.getByRole("button", { name: "Back to home" }));
  expect(
    view.container.querySelector(".r6-home")?.getAttribute("data-phase"),
  ).toBe("collapsing");
  await motion.finish();
  expect(motion.jobs.at(-1)?.options.duration).toBe(100);
  await motion.finish();
  expect(document.activeElement).toBe(expand);
  expect(expand.getAttribute("aria-expanded")).toBe("false");
});

it("expands on downward wheel and collapses on upward wheel at the list top", async () => {
  document.documentElement.dataset.nativeReduceMotion = "true";
  render(mountHome());
  await screen.findByText("601");
  const panel = screen.getByRole("region", { name: "My apps" });
  fireEvent.wheel(panel, { deltaY: 80 });
  expect(screen.getByText("App 4")).toBeTruthy();
  const list = screen.getByRole("region", { name: "App list" });
  list.scrollTop = 0;
  fireEvent.wheel(panel, { deltaY: -80 });
  expect(screen.queryByText("App 4")).toBeNull();
});

it("Escape interrupts overview exit without a stale expansion completing", async () => {
  const motion = mockMotion();
  const view = render(mountHome());
  await screen.findByText("601");
  fireEvent.click(screen.getByRole("button", { name: "View all 5 apps" }));
  fireEvent.keyDown(screen.getByRole("region", { name: "App home" }), {
    key: "Escape",
  });
  await motion.finish();
  await motion.finish();
  expect(motion.jobs[0].cancel).toHaveBeenCalled();
  expect(screen.queryByText("App 4")).toBeNull();
  expect(
    view.container.querySelector(".r6-home")?.getAttribute("data-phase"),
  ).toBe("idle");
});

it("Escape still interrupts after the disappearing trigger loses focus to body", async () => {
  const motion = mockMotion();
  render(mountHome());
  await screen.findByText("601");
  fireEvent.click(screen.getByRole("button", { name: "View all 5 apps" }));
  await motion.finish();
  fireEvent.keyDown(document.body, { key: "Escape" });
  await motion.finish();
  await motion.finish();
  expect(screen.queryByText("App 4")).toBeNull();
  expect(
    screen.getByRole("region", { name: "App home" }).getAttribute("data-phase"),
  ).toBe("idle");
});

it.each(["native", "prefers"])(
  "%s reduce motion settles an in-flight transition immediately",
  async (kind) => {
    const motion = mockMotion();
    const view = render(mountHome());
    await screen.findByText("601");
    fireEvent.click(screen.getByRole("button", { name: "View all 5 apps" }));
    await act(async () => {
      if (kind === "native")
        document.documentElement.dataset.nativeReduceMotion = "true";
      else {
        motion.media.matches = true;
        motion.media.dispatchEvent(new Event("change"));
      }
    });
    expect(screen.getByText("App 4")).toBeTruthy();
    expect(motion.jobs.every((j) => j.cancel.mock.calls.length > 0)).toBe(true);
    expect(
      view.container.querySelector(".r6-home")?.getAttribute("data-phase"),
    ).toBe("idle");
    const count = motion.jobs.length;
    fireEvent.click(screen.getByRole("button", { name: "Back to home" }));
    expect(motion.jobs).toHaveLength(count);
    expect(screen.queryByText("App 4")).toBeNull();
  },
);

it("preserves independent nonzero list scroll through collapse and inactive navigation", async () => {
  const view = render(mountHome());
  await screen.findByText("601");
  fireEvent.click(screen.getByRole("button", { name: "View all 5 apps" }));
  const list = screen.getByRole("region", { name: "App list" });
  list.scrollTop = 240;
  fireEvent.scroll(list);
  const detail = screen.getAllByRole("button", { name: "View details" })[3];
  detail.focus();
  view.rerender(mountHome(false));
  view.rerender(mountHome(true));
  expect(screen.getByText("App 4")).toBeTruthy();
  expect(list.scrollTop).toBe(240);
  expect(document.activeElement).toBe(detail);
  fireEvent.click(screen.getByRole("button", { name: "Back to home" }));
  fireEvent.click(screen.getByRole("button", { name: "View all 5 apps" }));
  expect(list.scrollTop).toBe(240);
});

it.each(["unmount", "inactive"])(
  "cancels animation work on %s without deferred layout or focus",
  async (mode) => {
    const motion = mockMotion();
    const raf = vi.spyOn(window, "requestAnimationFrame");
    const view = render(mountHome());
    await screen.findByText("601");
    fireEvent.click(screen.getByRole("button", { name: "View all 5 apps" }));
    if (mode === "unmount") view.unmount();
    else view.rerender(mountHome(false));
    await motion.finish();
    expect(motion.jobs.every((j) => j.cancel.mock.calls.length > 0)).toBe(true);
    expect(raf).not.toHaveBeenCalled();
    const calls = summary.mock.calls.length;
    expect(calls).toBeGreaterThan(0);
  },
);

it("restores expanded list and source focus after detail unmount, scoped to the import owner", async () => {
  const view = render(mountHome());
  await screen.findByText("601");
  fireEvent.click(screen.getByRole("button", { name: "View all 5 apps" }));
  const list = screen.getByRole("region", { name: "App list" });
  list.scrollTop = 220;
  fireEvent.scroll(list);
  screen.getAllByRole("button", { name: "View details" })[3].focus();
  view.unmount();
  const returned = render(mountHome());
  expect(screen.getByText("App 4")).toBeTruthy();
  expect(screen.getByRole("region", { name: "App list" }).scrollTop).toBe(220);
  expect(document.activeElement).toBe(
    screen.getAllByRole("button", { name: "View details" })[3],
  );
  returned.unmount();
  imports.refresh = vi.fn();
  render(mountHome());
  expect(screen.queryByText("App 4")).toBeNull();
});

it("does not show zero connected apps before host status is ready", async () => {
  window.oneMemory = {
    getRecallHosts: vi.fn(() => new Promise(() => {})),
  } as unknown as NonNullable<typeof window.oneMemory>;
  render(
    <ApplicationsHome
      imports={imports}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
    />,
  );
  await screen.findByText("601");
  expect(screen.getByText("Verifying")).toBeTruthy();
  expect(screen.queryByText("0 apps connected")).toBeNull();
});

it("does not keep saying sync is being checked after the remembered apply failed", async () => {
  saveConnectionPreferences({ syncSourceIds: ["app-0"], hostSourceIds: [] });
  const setSync = vi.fn().mockRejectedValue(new Error("offline"));
  render(
    <ApplicationsHome
      imports={{ ...imports, setSync } as typeof imports}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
    />,
  );

  expect(await screen.findByText(/did not turn on automatically/)).toBeTruthy();
  const sync = screen.getByText("Conversation sync").closest("div");
  expect(sync?.textContent).not.toContain("Checking status");
  expect(sync?.textContent).toContain("Not enabled");
  expect(screen.getByRole("button", { name: "Reconnect" })).toBeTruthy();
});

it("keeps the conversation-sync checking copy before the first home read", () => {
  summary.mockImplementation(() => new Promise(() => {}));
  render(
    <ApplicationsHome
      imports={imports}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
    />,
  );
  expect(screen.getByText("Checking status")).toBeTruthy();
  expect(screen.queryByText(/apps enabled/)).toBeNull();
});

it("does not claim there are no memories while the first summary read is pending", async () => {
  summary.mockImplementation(() => new Promise(() => {}));
  history.mockResolvedValue({
    available: true,
    recording_error: false,
    items: [],
  });
  processing.mockResolvedValue({ running: 0, pending: 0 });
  render(
    <ApplicationsHome
      imports={imports}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
    />,
  );
  await screen.findByText("Loading activity");
  expect(screen.queryByText("No memories generated yet")).toBeNull();
});

it("counts enabled apps for conversation sync and does not report a degraded run", async () => {
  liveSync.mockResolvedValue([
    {
      source_id: "app-0",
      mode: "polling",
      state: "degraded",
      last_job_id: "job-1",
      last_attempt_at: "2026-09-20T09:00:00Z",
      last_success_at: null,
      last_error_code: "live_sync_import_errors",
    },
  ]);
  render(
    <ApplicationsHome
      imports={
        {
          ...imports,
          sources: imports.sources.map((source, index) =>
            index < 2 ? { ...source, sync_enabled: true } : source,
          ),
          jobs: [{ source_id: "app-0", status: "running" }],
        } as unknown as Parameters<typeof ApplicationsHome>[0]["imports"]
      }
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
    />,
  );
  await screen.findByText("601");
  const sync = screen.getByText("Conversation sync").closest("div");
  expect(sync?.textContent).toContain("2 apps enabled");
  expect(sync?.textContent).not.toContain("partial errors");
  expect(sync?.textContent).not.toContain("Importing");
  expect(sync?.textContent).not.toContain("Continuous sync is on");
  expect(await screen.findByText("2 apps connected")).toBeTruthy();
});

it("does not show zero connected apps while cold-start insertion is still applying", async () => {
  saveConnectionPreferences({
    syncSourceIds: ["app-0"],
    hostSourceIds: ["cursor"],
  });
  window.oneMemory = {
    getRecallHosts: vi.fn().mockResolvedValue([]),
    previewRecallInstall: vi.fn(() => new Promise(() => {})),
    installRecall: vi.fn(),
    getRecallPermission: vi.fn(),
    setRecallPermission: vi.fn(),
  } as unknown as NonNullable<typeof window.oneMemory>;
  render(
    <ApplicationsHome
      imports={imports}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
    />,
  );
  await screen.findByText("601");
  expect(screen.getByText("Verifying")).toBeTruthy();
  expect(screen.queryByText("0 apps connected")).toBeNull();
});

it("shows a newly imported conversation ahead of an older memory", async () => {
  recentConversations.mockResolvedValue({
    items: [
      {
        evidence_id: "e-live",
        conversation_id: "c1",
        source_id: "app-0",
        role: "user",
        written_by: "person",
        created_at: "2026-01-01T00:00:00Z",
        imported_at: "2026-09-23T06:47:00Z",
        excerpt: "The exact words just synced from Cursor",
        truncated: false,
      },
    ],
    total: 1,
    offset: 0,
    limit: 1,
  });
  const onConversations = vi.fn();
  render(
    <ApplicationsHome
      imports={imports}
      onManage={() => {}}
      onMemories={() => {}}
      onConversations={onConversations}
      onSettings={() => {}}
    />,
  );
  expect(
    await screen.findByText("The exact words just synced from Cursor"),
  ).toBeTruthy();
  expect(screen.getAllByText("Recent updates").length).toBeGreaterThan(0);
  expect(screen.queryByText("Conversation")).toBeNull();
  expect(screen.queryByText("Memory")).toBeNull();
  expect(screen.queryByText(/raw conversation/)).toBeNull();
  // The View conversation entry is hidden from Recent updates as required by product (2026-10-12).
  expect(
    screen.queryByRole("button", { name: "View conversation" }),
  ).toBeNull();
  expect(onConversations).not.toHaveBeenCalled();
  expect(recentConversations).toHaveBeenCalledWith({
    limit: 1,
    recentActivity: true,
  });
});

function homeWithHosts(getRecallHosts: ReturnType<typeof vi.fn>) {
  queryClient.clear();
  window.oneMemory = { getRecallHosts } as unknown as NonNullable<
    typeof window.oneMemory
  >;
  render(
    <ApplicationsHome
      imports={imports}
      onManage={() => {}}
      onMemories={() => {}}
      onSettings={() => {}}
    />,
  );
}
function accessRow() {
  return screen.getByText("App access").closest("div")?.textContent ?? "";
}
const installed = (hostId: string, permission?: object) => ({
  hostId,
  label: hostId === "app-0" ? "App 0" : "App 1",
  present: true,
  installed: true,
  ...(permission
    ? { permission: { hostId, reason: null, ...permission } }
    : {}),
});
const unreadable = {
  state: "unavailable",
  reason: "permission_store_unavailable",
};

it("never renders an app whose permission could not be read as zero connected", async () => {
  const read = vi
    .fn()
    .mockResolvedValueOnce([installed("app-0", unreadable)])
    .mockResolvedValue([installed("app-0", { state: "approved" })]);
  homeWithHosts(read);
  await waitFor(() =>
    expect(accessRow()).toContain("1 app pending verification"),
  );
  expect(screen.queryByText("0 apps connected")).toBeNull();
  expect(
    screen.getByText(
      "The memory permission for App 0 could not be read right now, so the connection cannot be confirmed.",
    ),
  ).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Verify again" }));
  await waitFor(() => expect(accessRow()).toContain("1 app connected"));
  expect(read).toHaveBeenCalledTimes(2);
  expect(screen.queryByRole("button", { name: "Verify again" })).toBeNull();
});

it("treats a host without any permission answer as unchecked, not unconnected", async () => {
  homeWithHosts(vi.fn().mockResolvedValue([installed("app-0")]));
  await waitFor(() =>
    expect(accessRow()).toContain("1 app pending verification"),
  );
  expect(screen.queryByText("0 apps connected")).toBeNull();
});

it("keeps the known count and names only the unchecked rest", async () => {
  homeWithHosts(
    vi
      .fn()
      .mockResolvedValue([
        installed("app-0", { state: "approved" }),
        installed("app-1", unreadable),
      ]),
  );
  await waitFor(() =>
    expect(accessRow()).toContain("1 app connected, 1 pending verification"),
  );
  expect(
    screen.getByText(
      "The memory permission for App 1 could not be read right now, so the connection cannot be confirmed.",
    ),
  ).toBeTruthy();
});

it("says a failed host read is unchecked and offers to read again", async () => {
  const read = vi
    .fn()
    .mockRejectedValueOnce(new Error("host_read_timeout"))
    .mockResolvedValue([installed("app-0", { state: "approved" })]);
  homeWithHosts(read);
  await waitFor(() => expect(accessRow()).toContain("Pending verification"));
  expect(accessRow()).not.toContain("Verifying");
  expect(screen.queryByText("0 apps connected")).toBeNull();
  expect(
    screen.getByText(
      "App connection status could not be read right now, so the number of connected apps cannot be confirmed.",
    ),
  ).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Verify again" }));
  await waitFor(() => expect(accessRow()).toContain("1 app connected"));
  expect(read).toHaveBeenCalledTimes(2);
});

it("still shows a checked zero as zero", async () => {
  homeWithHosts(
    vi
      .fn()
      .mockResolvedValue([
        installed("app-0", { state: "install-only" }),
        { ...installed("app-1", { state: "not-installed" }), installed: false },
      ]),
  );
  await waitFor(() => expect(accessRow()).toContain("0 apps connected"));
  expect(screen.queryByRole("button", { name: "Verify again" })).toBeNull();
});

describe("Runtime status card: a failed refresh must not keep it saying it is running", () => {
  function renderHome(next: Partial<typeof imports>) {
    render(
      <ApplicationsHome
        imports={{ ...imports, ...next } as typeof imports}
        onManage={() => {}}
        onMemories={() => {}}
        onSettings={() => {}}
      />,
    );
  }

  it("says it is running while the last health read succeeded", async () => {
    renderHome({ healthStale: false });
    await screen.findByText("A real governed memory");
    expect(screen.getByText("Backend running")).toBeTruthy();
  });

  it("stops saying it is running once a health read has failed", async () => {
    renderHome({ healthStale: true });
    await screen.findByText("A real governed memory");
    expect(screen.queryByText("Backend running")).toBeNull();
    expect(
      screen.getByText("Connection status pending verification"),
    ).toBeTruthy();
    // The health object itself is kept: it names the instance that owns the
    // running backfill and the write receipts.
    expect(screen.getByText("601")).toBeTruthy();
  });
});
