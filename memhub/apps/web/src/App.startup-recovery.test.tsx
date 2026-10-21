// @vitest-environment jsdom
import type { OnboardingState } from "@one-memory/contracts";
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
import { App } from "./App";

const mocks = vi.hoisted(() => ({
  getOnboarding: vi.fn(),
  write: vi.fn(),
  flow: vi.fn(),
  importsLoading: false,
  health: null as { instance_id: string } | null,
}));
vi.mock("./api", () => ({ getOnboarding: mocks.getOnboarding }));
vi.mock("./use-import-sources", () => ({
  useImportSources: () => ({
    sources: [],
    jobs: [],
    loading: mocks.importsLoading,
    health: mocks.health,
    healthStale: false,
    startImport: mocks.write,
    connectSources: mocks.write,
  }),
}));
vi.mock("./ApplicationsHome", () => ({
  ApplicationsHome: () => <p>Applications home ready</p>,
}));
vi.mock("./MemorySpace", () => ({
  MemorySpace: () => <p>Memory workbench</p>,
}));
vi.mock("./R6Settings", () => ({ R6Settings: () => <p>Settings ready</p> }));
vi.mock("./OnboardingFlow", () => ({
  OnboardingFlow: ({ state }: { state: OnboardingState }) => {
    mocks.flow(state);
    return <p>Continue first-time setup: {state.status}</p>;
  },
}));
vi.mock("./learning-preferences", () => ({ saveLearningState: mocks.write }));
vi.mock("./SourcesView", () => ({ SourcesView: () => null }));
vi.mock("./ImportProgressStrip", () => ({ ImportProgressStrip: () => null }));
vi.mock("./MatterWorkspace", () => ({ MatterWorkspace: () => null }));
vi.mock("./HealthPanel", () => ({ HealthPanel: () => null }));
vi.mock("./FirstUseGuide", () => ({ FirstUseGuide: () => null }));

function deferred() {
  let resolve!: (state: { status: string }) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<{ status: string }>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}
beforeEach(() => {
  vi.clearAllMocks();
  mocks.getOnboarding.mockReset().mockResolvedValue({ status: "completed" });
  mocks.importsLoading = false;
  mocks.health = null;
  window.history.replaceState({}, "", "/");
  localStorage.clear();
  window.oneMemory = undefined;
});
afterEach(cleanup);

it("recovers an initial failed read into the completed user's homepage without writes", async () => {
  mocks.getOnboarding.mockRejectedValueOnce(new Error("SECRET_RAW_ERROR"));
  render(<App />);
  expect(await screen.findByRole("alert")).toBeTruthy();
  expect(screen.queryByText(/SECRET_RAW_ERROR/)).toBeNull();
  expect(screen.queryByText("Opening MemHub...")).toBeNull();
  expect(
    screen.queryByRole("navigation", { name: "Primary navigation" }),
  ).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Read again" }));
  await screen.findByText("Applications home ready");
  expect(mocks.getOnboarding).toHaveBeenCalledTimes(2);
  expect(mocks.flow).not.toHaveBeenCalled();
  expect(mocks.write).not.toHaveBeenCalled();
  expect(localStorage.length).toBe(0);
});

it("retains retry after another failure and single-flights rapid retry clicks", async () => {
  const retry = deferred();
  mocks.getOnboarding
    .mockRejectedValueOnce(new Error("offline"))
    .mockReturnValueOnce(retry.promise);
  render(<App />);
  await screen.findByRole("alert");
  const button = screen.getByRole("button", { name: "Read again" });
  fireEvent.click(button);
  fireEvent.click(button);
  fireEvent.click(button);
  await waitFor(() => expect(mocks.getOnboarding).toHaveBeenCalledTimes(2));
  expect(screen.getByText("Opening MemHub...")).toBeTruthy();
  expect(screen.queryByRole("alert")).toBeNull();
  await act(async () => retry.reject(new Error("still offline")));
  await screen.findByRole("alert");
  expect(screen.getByRole("button", { name: "Read again" })).toHaveProperty(
    "disabled",
    false,
  );
  expect(mocks.getOnboarding).toHaveBeenCalledTimes(2);
  fireEvent.click(screen.getByRole("button", { name: "Read again" }));
  await screen.findByText("Applications home ready");
  expect(mocks.getOnboarding).toHaveBeenCalledTimes(3);
  expect(mocks.write).not.toHaveBeenCalled();
});

it("preserves the exact unfinished onboarding state after retry instead of completing it", async () => {
  const state = {
    status: "new",
    selected_source_ids: ["cursor"],
    completed_at: null,
  };
  mocks.getOnboarding
    .mockRejectedValueOnce(new Error("offline"))
    .mockResolvedValueOnce(state);
  render(<App />);
  await screen.findByRole("alert");
  fireEvent.click(screen.getByRole("button", { name: "Read again" }));
  await screen.findByText("Continue first-time setup: new");
  expect(mocks.flow).toHaveBeenLastCalledWith(state);
  expect(screen.queryByText("Applications home ready")).toBeNull();
  expect(mocks.write).not.toHaveBeenCalled();
});

it.each(["initial", "retry"])(
  "ignores a late %s result after unmount and does not overwrite the new owner",
  async (phase) => {
    const late = deferred();
    if (phase === "retry")
      mocks.getOnboarding.mockRejectedValueOnce(new Error("offline"));
    mocks.getOnboarding.mockReturnValueOnce(late.promise);
    const old = render(<App />);
    if (phase === "retry") {
      await screen.findByRole("alert");
      fireEvent.click(screen.getByRole("button", { name: "Read again" }));
    }
    await act(async () => {});
    old.unmount();
    render(<App />);
    await screen.findByText("Applications home ready");
    const reads = mocks.getOnboarding.mock.calls.length;
    await act(async () => late.resolve({ status: "new" }));
    expect(screen.getByText("Applications home ready")).toBeTruthy();
    expect(mocks.flow).not.toHaveBeenCalled();
    expect(mocks.getOnboarding).toHaveBeenCalledTimes(reads);
    expect(mocks.write).not.toHaveBeenCalled();
  },
);

it("ignores a late rejection after unmount", async () => {
  const late = deferred();
  mocks.getOnboarding.mockReturnValueOnce(late.promise);
  const view = render(<App />);
  await act(async () => {});
  view.unmount();
  await act(async () => late.reject(new Error("late offline")));
  expect(screen.queryByRole("alert")).toBeNull();
  expect(mocks.getOnboarding).toHaveBeenCalledOnce();
  expect(mocks.write).not.toHaveBeenCalled();
});

it("shares the pending initial read across StrictMode effect replay", async () => {
  const read = deferred();
  mocks.getOnboarding.mockReturnValue(read.promise);
  render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
  await act(async () => {});
  expect(mocks.getOnboarding).toHaveBeenCalledOnce();
  await act(async () => read.resolve({ status: "completed" }));
  await screen.findByText("Applications home ready");
  expect(mocks.write).not.toHaveBeenCalled();
});

it("keeps the read error actionable even if import startup is also pending", async () => {
  mocks.importsLoading = true;
  mocks.getOnboarding.mockRejectedValueOnce(new Error("offline"));
  const view = render(<App />);
  await screen.findByRole("alert");
  fireEvent.click(screen.getByRole("button", { name: "Read again" }));
  await act(async () => {});
  expect(screen.queryByText("Applications home ready")).toBeNull();
  expect(screen.getByText("Opening MemHub...")).toBeTruthy();
  mocks.importsLoading = false;
  view.rerender(<App />);
  await screen.findByText("Applications home ready");
  expect(mocks.getOnboarding).toHaveBeenCalledTimes(2);
});

it("retains an explicit settings destination after successful retry", async () => {
  window.history.replaceState({}, "", "/?view=settings");
  mocks.getOnboarding.mockRejectedValueOnce(new Error("offline"));
  render(<App />);
  await screen.findByRole("alert");
  fireEvent.click(screen.getByRole("button", { name: "Read again" }));
  await screen.findByText("Settings ready");
  expect(window.location.search).toBe("?view=settings");
  expect(mocks.write).not.toHaveBeenCalled();
});

it("re-reads on its own once a session exists after the failed first read", async () => {
  mocks.getOnboarding
    .mockRejectedValueOnce(new Error("offline"))
    .mockResolvedValue({ status: "completed" });
  const view = render(<App />);
  expect(await screen.findByRole("alert")).toBeTruthy();
  mocks.health = { instance_id: "instance-1" };
  await act(async () => {
    view.rerender(<App />);
  });
  await screen.findByText("Applications home ready");
  expect(screen.queryByRole("alert")).toBeNull();
  expect(mocks.getOnboarding).toHaveBeenCalledTimes(2);
  expect(mocks.write).not.toHaveBeenCalled();
});

it("re-reads when the page comes back after the failed first read", async () => {
  mocks.getOnboarding
    .mockRejectedValueOnce(new Error("offline"))
    .mockResolvedValue({ status: "completed" });
  render(<App />);
  expect(await screen.findByRole("alert")).toBeTruthy();
  await act(async () => {
    fireEvent(document, new Event("visibilitychange"));
  });
  await screen.findByText("Applications home ready");
  expect(screen.queryByRole("alert")).toBeNull();
  expect(mocks.getOnboarding).toHaveBeenCalledTimes(2);
});
