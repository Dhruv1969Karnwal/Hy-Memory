// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { R6Settings } from "./R6Settings";

const permission = vi.hoisted(() => ({
  update: vi.fn(),
}));

vi.mock("./api", () => ({
  getModelProfiles: vi.fn().mockResolvedValue([
    {
      profile_id: "profile-1",
      name: "当前模型",
      model: "model-1",
      base_url: "https://example.test/v1",
      is_active: true,
    },
  ]),
}));
vi.mock("./ModelSettings", () => ({
  ModelSettings: () => null,
}));
vi.mock("./ConnectionFallback", () => ({
  ConnectionFallback: () => null,
}));
vi.mock("./runtime-state", () => ({
  refreshLoginPreference: vi.fn(),
  setLoginPreference: vi.fn(),
  useLoginPreference: () => null,
}));
vi.mock("./diagnostics-export", () => ({
  exportDiagnosticsFile: vi.fn(),
}));
vi.mock("./use-model-content-permission", () => ({
  useModelContentPermission: () => ({
    allowed: false,
    reading: false,
    writing: false,
    error: null,
    refresh: vi.fn(),
    update: permission.update,
  }),
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it("renders one global model-content permission switch", async () => {
  const user = userEvent.setup();
  render(<R6Settings onDiagnostics={vi.fn()} />);

  const toggle = await screen.findByRole("switch", {
    name: "允许当前配置模型读取本地会话内容",
  });
  expect(toggle.getAttribute("aria-checked")).toBe("false");

  await user.click(toggle);
  expect(permission.update).toHaveBeenCalledWith(true);
  expect(screen.queryByText(/模型处理范围|后续新增/)).toBeNull();
});

it("hides diagnostics by default and exposes it only when configured", async () => {
  const user = userEvent.setup();
  const view = render(<R6Settings onDiagnostics={vi.fn()} />);
  expect(screen.queryByRole("button", { name: "帮助与诊断" })).toBeNull();

  view.rerender(<R6Settings onDiagnostics={vi.fn()} showDiagnostics={true} />);
  await user.click(screen.getByRole("button", { name: "帮助与诊断" }));
  expect(screen.getByRole("heading", { name: "帮助与诊断" })).toBeTruthy();
  expect(screen.getByRole("heading", { name: "导出日志" })).toBeTruthy();
});
