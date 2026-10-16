// @vitest-environment jsdom

import {
  cleanup,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { App } from "./App";

const { getOnboarding, sources, modelProps, mutations } = vi.hoisted(() => ({
  getOnboarding: vi.fn(),
  sources: [
    {
      source_id: "cursor",
      display_name: "Cursor",
      authorized: true,
      capabilities: {
        historical_import: "implemented",
        incremental_import: "implemented",
        live_sync: "implemented",
        recall: "implemented",
        installer: "implemented",
        verification: "real_replayed",
      },
    },
  ],
  modelProps: vi.fn(),
  mutations: {
    activateModelProfile: vi.fn(),
    setModelContentPermission: vi.fn(),
  },
}));

vi.mock("./api", () => ({
  getOnboarding,
  getModelProfiles: vi.fn().mockResolvedValue([]),
  getModelContentPermission: vi.fn().mockResolvedValue({ allowed: false }),
  getSources: vi.fn().mockResolvedValue(sources),
  ...mutations,
}));
vi.mock("./use-import-sources", () => ({
  useImportSources: () => ({ sources, jobs: [], loading: false, health: null }),
}));
vi.mock("./MemorySpace", () => ({
  MemorySpace: ({
    onReviewProcessing,
    onConfigureModel,
  }: {
    onReviewProcessing?(): void;
    onConfigureModel(): void;
  }) => (
    <section aria-label="测试工作台">
      <button type="button" onClick={onReviewProcessing}>
        打开模型内容权限
      </button>
      <button type="button" onClick={onConfigureModel}>
        设置记忆整理
      </button>
    </section>
  ),
}));
vi.mock("./ModelSettings", () => ({
  ModelSettings: (props: {
    compactSingle?: boolean;
    controlsRef?: { current: unknown };
  }) => {
    modelProps(props);
    if (props.controlsRef)
      props.controlsRef.current = {
        edit: vi.fn(() => true),
        editProfile: vi.fn(() => true),
        add: vi.fn(() => true),
        test: vi.fn(),
      };
    return <input aria-label="保留的模型草稿" defaultValue="" />;
  },
}));
vi.mock("./ModelContentConsent", () => ({
  ModelContentConsent: ({
    onClose,
    onConfigureModel,
  }: {
    onClose(): void;
    onConfigureModel?(): void;
  }) => (
    <section role="dialog" aria-label="模型内容权限">
      <button type="button" onClick={onClose}>
        关闭
      </button>
      <button type="button" onClick={onConfigureModel}>
        配置模型
      </button>
    </section>
  ),
}));
vi.mock("./OnboardingFlow", () => ({
  OnboardingFlow: ({ onFinish }: { onFinish(target: string): void }) => (
    <div>
      <button type="button" onClick={() => onFinish("settings")}>
        完成导入去设置
      </button>
      <button type="button" onClick={() => onFinish("memory")}>
        后台继续去工作台
      </button>
    </div>
  ),
}));
vi.mock("./SourcesView", () => ({ SourcesView: () => <p>应用列表</p> }));
vi.mock("./ApplicationsHome", () => ({
  ApplicationsHome: () => <p>应用首页</p>,
}));
vi.mock("./ImportProgressStrip", () => ({ ImportProgressStrip: () => null }));
vi.mock("./HealthPanel", () => ({ HealthPanel: () => null }));
vi.mock("./FirstUseGuide", () => ({ FirstUseGuide: () => null }));
vi.mock("./MatterWorkspace", () => ({ MatterWorkspace: () => null }));

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  window.oneMemory = undefined;
  window.history.replaceState({}, "", "/");
  getOnboarding.mockResolvedValue({ status: "completed" });
});
afterEach(cleanup);

function noPermissionWrites() {
  for (const mutation of Object.values(mutations))
    expect(mutation).not.toHaveBeenCalled();
}

it("opens model-content permission over the memory workbench", async () => {
  window.history.replaceState({}, "", "/?view=memory");
  render(<App />);
  await userEvent.click(
    await screen.findByRole("button", { name: "打开模型内容权限" }),
  );
  expect(screen.getByRole("dialog", { name: "模型内容权限" })).toBeTruthy();
  expect(window.location.search).toBe("?view=memory");
  expect(screen.queryByText("应用列表")).toBeNull();
  await userEvent.click(screen.getByRole("button", { name: "关闭" }));
  expect(screen.queryByRole("dialog", { name: "模型内容权限" })).toBeNull();
  expect(getOnboarding).toHaveBeenCalledTimes(1);
  expect(screen.getByRole("region", { name: "测试工作台" })).toBeTruthy();
  noPermissionWrites();
});

it("enters D47 settings without a tour or mandatory second scope-confirmation flow", async () => {
  getOnboarding.mockResolvedValue({ status: "new" });
  render(<App />);
  await userEvent.click(
    await screen.findByRole("button", { name: "完成导入去设置" }),
  );
  expect(window.location.search).toBe("?view=settings");
  expect(screen.queryByRole("dialog", { name: "功能教学" })).toBeNull();
  // The teaching flow is gone: finishing setup must not write a learning
  // preference that nothing reads any more.
  expect(localStorage.getItem("one-memory.learning.s1-3")).toBeNull();
  await waitFor(() => expect(modelProps).toHaveBeenCalled());
  expect(modelProps.mock.calls.at(-1)?.[0].compactSingle).toBe(true);
  expect(
    screen.queryByRole("button", { name: "继续：确认整理范围" }),
  ).toBeNull();
  expect(screen.queryByRole("dialog", { name: "模型内容权限" })).toBeNull();
  noPermissionWrites();
  expect(screen.queryByRole("button", { name: "管理已有授权" })).toBeNull();
  expect(screen.getByRole("button", { name: "模型" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "连接其他应用" })).toBeTruthy();
  expect(screen.queryByRole("button", { name: "帮助与诊断" })).toBeNull();
  noPermissionWrites();
});

it("enters the workbench without an automatic tour or removed help section", async () => {
  getOnboarding.mockResolvedValue({ status: "new" });
  render(<App />);
  await userEvent.click(
    await screen.findByRole("button", { name: "后台继续去工作台" }),
  );
  expect(window.location.search).toBe("?view=memory");
  expect(screen.getByRole("region", { name: "测试工作台" })).toBeTruthy();
  expect(screen.queryByRole("dialog", { name: "功能教学" })).toBeNull();
  expect(localStorage.getItem("one-memory.learning.s1-3")).toBeNull();
  const nav = screen.getByRole("navigation", { name: "主要导航" });
  await userEvent.click(within(nav).getByRole("button", { name: "设置" }));
  expect(screen.queryByRole("button", { name: "帮助与诊断" })).toBeNull();
  expect(screen.queryByRole("dialog", { name: "功能教学" })).toBeNull();
  noPermissionWrites();
});

it("does not resume a pending tour over a settings deep link", async () => {
  // A leftover pending preference from an older build must not resurrect the
  // removed teaching flow.
  localStorage.setItem(
    "one-memory.learning.s1-3",
    JSON.stringify({ version: 1, state: "pending" }),
  );
  window.history.replaceState({}, "", "/?view=settings");
  render(<App />);
  await waitFor(() => expect(modelProps).toHaveBeenCalled());
  expect(modelProps.mock.calls.at(-1)?.[0].compactSingle).toBe(true);
  expect(screen.queryByRole("dialog", { name: "功能教学" })).toBeNull();
  expect(screen.queryByRole("dialog", { name: "模型内容权限" })).toBeNull();
  expect(window.location.search).toBe("?view=settings");
  // Still pending: nothing read or rewrote it.
  expect(
    JSON.parse(localStorage.getItem("one-memory.learning.s1-3") ?? "null")
      ?.state,
  ).toBe("pending");
  noPermissionWrites();
});

it("keeps the applications default while explicit model configuration remains reachable", async () => {
  render(<App />);
  await screen.findByText("应用首页");
  expect(screen.queryByRole("region", { name: "测试工作台" })).toBeNull();
  expect(screen.queryByRole("dialog")).toBeNull();
  const nav = screen.getByRole("navigation", { name: "主要导航" });
  await userEvent.click(within(nav).getByRole("button", { name: "记忆" }));
  await userEvent.click(screen.getByRole("button", { name: "设置记忆整理" }));
  await waitFor(() => expect(modelProps).toHaveBeenCalled());
  expect(window.location.search).toBe("?view=settings");
  expect(modelProps.mock.calls.at(-1)?.[0].compactSingle).toBe(true);
  expect(screen.queryByRole("dialog")).toBeNull();
  noPermissionWrites();
});

it("keeps an open model draft while switching settings groups", async () => {
  window.history.replaceState({}, "", "/?view=settings");
  render(<App />);
  await waitFor(() => expect(modelProps).toHaveBeenCalled());
  await userEvent.click(screen.getByRole("button", { name: "配置模型" }));
  await userEvent.type(
    screen.getByLabelText("保留的模型草稿"),
    "unsaved draft",
  );
  await userEvent.click(screen.getByRole("button", { name: "连接其他应用" }));
  await userEvent.click(screen.getByRole("button", { name: "模型" }));
  expect(window.location.search).toBe("?view=settings");
  expect(screen.getByLabelText("保留的模型草稿")).toHaveProperty(
    "value",
    "unsaved draft",
  );
  noPermissionWrites();
});
