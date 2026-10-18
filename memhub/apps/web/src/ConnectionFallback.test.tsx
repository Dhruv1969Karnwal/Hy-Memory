// @vitest-environment jsdom
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ConnectionFallback } from "./ConnectionFallback";

afterEach(() => {
  cleanup();
  delete window.oneMemory;
});
describe("ConnectionFallback", () => {
  it("loads machine-local material on demand and never calls copy a connection", async () => {
    const user = userEvent.setup();
    const getConnectionMaterial = vi.fn().mockResolvedValue({
      mcp: '{"mcpServers":{}}',
      shell: "echo query",
      prompt: "connect local only",
    });
    const copyText = vi.fn().mockResolvedValue(undefined);
    window.oneMemory = {
      getConnectionMaterial,
      copyText,
    } as unknown as Window["oneMemory"];
    render(<ConnectionFallback />);
    expect(getConnectionMaterial).not.toHaveBeenCalled();
    await user.click(screen.getByText("Connect other tools"));
    expect(
      await screen.findByLabelText("Local connection material"),
    ).toHaveProperty("value", "connect local only");
    await user.click(screen.getByRole("button", { name: "Copy prompt" }));
    expect(copyText).toHaveBeenCalledWith("connect local only");
    expect(screen.getByRole("status").textContent).toContain(
      "no connection has been established",
    );
    await user.click(screen.getByRole("tab", { name: "MCP config" }));
    expect(screen.getByLabelText("Local connection material")).toHaveProperty(
      "value",
      '{"mcpServers":{}}',
    );
  });
  it("offers retry and clipboard failure recovery without fake success", async () => {
    const user = userEvent.setup();
    const getConnectionMaterial = vi
      .fn()
      .mockRejectedValueOnce(new Error())
      .mockResolvedValue({ mcp: "{}", shell: null, prompt: "local" });
    window.oneMemory = {
      getConnectionMaterial,
      copyText: vi.fn().mockRejectedValue(new Error()),
    } as unknown as Window["oneMemory"];
    render(<ConnectionFallback />);
    await user.click(screen.getByText("Connect other tools"));
    await user.click(await screen.findByRole("button", { name: "Retry" }));
    await screen.findByLabelText("Local connection material");
    expect(screen.getByRole("tab", { name: "Command" })).toHaveProperty(
      "disabled",
      true,
    );
    await user.click(screen.getByRole("button", { name: "Copy prompt" }));
    await waitFor(() =>
      expect(screen.getByRole("status").textContent).toContain("Copy failed"),
    );
  });
  it("does not invent runtime paths in the browser", () => {
    render(<ConnectionFallback />);
    expect(screen.queryByLabelText("Local connection material")).toBeNull();
    expect(
      screen.getByText(
        "Local connection material is only available in the desktop app.",
      ),
    ).toBeTruthy();
  });
});
