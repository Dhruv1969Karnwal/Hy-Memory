// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { R6Settings } from "./R6Settings";

vi.mock("./api", () => ({
  getModelProfiles: vi.fn().mockResolvedValue([
    {
      profile_id: "profile-1",
      name: "Current model",
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
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it("shows the help and diagnostics group by default and hides it when opted out", async () => {
  const user = userEvent.setup();
  const view = render(<R6Settings onDiagnostics={vi.fn()} />);
  await user.click(
    screen.getByRole("button", { name: "Help and diagnostics" }),
  );
  expect(
    screen.getByRole("heading", { name: "Help and diagnostics" }),
  ).toBeTruthy();
  expect(screen.getByRole("heading", { name: "Export logs" })).toBeTruthy();

  view.rerender(<R6Settings onDiagnostics={vi.fn()} showDiagnostics={false} />);
  expect(
    screen.queryByRole("button", { name: "Help and diagnostics" }),
  ).toBeNull();
});
