// @vitest-environment jsdom
import { afterEach, expect, it, vi } from "vitest";
import {
  installRendererDebugLogCapture,
  rendererDebugLogs,
  resetRendererDebugLogsForTest,
} from "./renderer-debug-log";

afterEach(() => resetRendererDebugLogsForTest());

it("captures console output and uncaught renderer failures", () => {
  const debug = vi.spyOn(console, "debug").mockImplementation(() => undefined);
  const error = vi.spyOn(console, "error").mockImplementation(() => undefined);
  const stop = installRendererDebugLogCapture();
  console.debug("debug-value", { count: 2 });
  window.dispatchEvent(new ErrorEvent("error", { message: "render exploded" }));

  expect(rendererDebugLogs()).toEqual([
    expect.objectContaining({
      level: "debug",
      event: "console.debug",
      message: 'debug-value {"count":2}',
    }),
    expect.objectContaining({
      level: "error",
      event: "window.error",
      message: "render exploded",
    }),
  ]);

  stop();
  debug.mockRestore();
  error.mockRestore();
});
