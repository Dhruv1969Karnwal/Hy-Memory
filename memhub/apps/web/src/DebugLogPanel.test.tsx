// @vitest-environment jsdom
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { DebugLogPanel } from "./DebugLogPanel";
import {
  recordRendererDebugLog,
  resetRendererDebugLogsForTest,
} from "./renderer-debug-log";

const diagnosticLevel = vi.hoisted(() => ({
  get: vi.fn(async () => "info" as const),
  set: vi.fn(async (level: "error" | "warning" | "info" | "debug") => level),
}));

vi.mock("./api", () => ({
  getDiagnosticLogLevel: diagnosticLevel.get,
  setDiagnosticLogLevel: diagnosticLevel.set,
}));

const snapshot = {
  started_at: "2026-10-14T10:00:00.000Z",
  read_at: "2026-10-14T10:01:00.000Z",
  truncated: false,
  entries: [
    {
      id: "core-info",
      timestamp: "2026-10-14T10:00:01.000Z",
      level: "info" as const,
      service: "core" as const,
      component: "import.scan",
      event: "source.scanned",
      outcome: "succeeded",
      details: { source_id: "cursor", count: 3 },
      source: "diagnostic" as const,
    },
    {
      id: "desktop-error",
      timestamp: "2026-10-14T10:00:02.000Z",
      level: "error" as const,
      service: "desktop" as const,
      component: "desktop.action",
      event: "sync.failed",
      outcome: "failed",
      error_code: "sync_failed",
      source: "diagnostic" as const,
    },
  ],
};

beforeEach(() => {
  resetRendererDebugLogsForTest();
  diagnosticLevel.get.mockClear();
  diagnosticLevel.set.mockClear();
  window.oneMemory = {
    getDebugLogs: vi.fn().mockResolvedValue(snapshot),
  } as unknown as NonNullable<typeof window.oneMemory>;
});

afterEach(() => {
  cleanup();
  window.oneMemory = undefined;
  vi.useRealTimers();
});

it("opens from the floating button and filters by text, level and source", async () => {
  render(<DebugLogPanel />);
  await userEvent.click(
    screen.getByRole("button", { name: "Open Debug logs" }),
  );
  const panel = await screen.findByRole("complementary", {
    name: "Debug logs",
  });
  await waitFor(() =>
    expect(within(panel).getByText(/source\.scanned/)).toBeTruthy(),
  );

  await userEvent.type(
    within(panel).getByRole("searchbox", { name: "Search logs" }),
    "sync_failed",
  );
  expect(within(panel).getByText(/sync\.failed/)).toBeTruthy();
  expect(within(panel).queryByText(/source\.scanned/)).toBeNull();

  await userEvent.clear(
    within(panel).getByRole("searchbox", { name: "Search logs" }),
  );
  await userEvent.click(within(panel).getByRole("button", { name: "Error" }));
  expect(within(panel).getByText(/sync\.failed/)).toBeTruthy();
  expect(within(panel).queryByText(/source\.scanned/)).toBeNull();

  await userEvent.click(within(panel).getByRole("button", { name: "All" }));
  await userEvent.selectOptions(
    within(panel).getByRole("combobox", { name: "Log source" }),
    "core",
  );
  expect(within(panel).getByText(/source\.scanned/)).toBeTruthy();
  expect(within(panel).queryByText(/sync\.failed/)).toBeNull();
});

it("keeps the underlying page interactive while the log panel is open", async () => {
  const action = vi.fn();
  render(
    <>
      <button type="button" onClick={action}>
        Page action
      </button>
      <DebugLogPanel />
    </>,
  );
  await userEvent.click(
    screen.getByRole("button", { name: "Open Debug logs" }),
  );
  await screen.findByRole("complementary", { name: "Debug logs" });

  await userEvent.click(screen.getByRole("button", { name: "Page action" }));

  expect(action).toHaveBeenCalledOnce();
});

it("shows renderer logs immediately and polls desktop logs while live", async () => {
  vi.useFakeTimers();
  const read = window.oneMemory?.getDebugLogs;
  if (!read) throw new Error("missing debug log reader");
  const readMock = vi.mocked(read);
  render(<DebugLogPanel />);
  fireEvent.click(screen.getByRole("button", { name: "Open Debug logs" }));
  await act(async () => {
    await Promise.resolve();
  });

  await act(async () => {
    recordRendererDebugLog({
      level: "warn",
      event: "console.warn",
      outcome: "observed",
      message: "renderer live message",
    });
  });

  expect(screen.getByText("renderer live message")).toBeTruthy();
  await act(async () => vi.advanceTimersByTimeAsync(1_000));
  expect(readMock.mock.calls.length).toBeGreaterThanOrEqual(2);
});

it("changes the Core and HY capture level independently from display filtering", async () => {
  render(<DebugLogPanel />);
  await userEvent.click(
    screen.getByRole("button", { name: "Open Debug logs" }),
  );
  const panel = await screen.findByRole("complementary", {
    name: "Debug logs",
  });
  await waitFor(() => expect(diagnosticLevel.get).toHaveBeenCalledOnce());

  await userEvent.selectOptions(
    within(panel).getByRole("combobox", { name: "Capture level" }),
    "debug",
  );

  await waitFor(() =>
    expect(diagnosticLevel.set).toHaveBeenCalledWith("debug"),
  );
  expect(
    (
      within(panel).getByRole("combobox", {
        name: "Capture level",
      }) as HTMLSelectElement
    ).value,
  ).toBe("debug");
  expect(
    within(panel)
      .getByRole("button", { name: "All" })
      .getAttribute("aria-pressed"),
  ).toBe("true");
});
