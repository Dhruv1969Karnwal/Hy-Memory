// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { MemoryArrivalPanel } from "./memory-arrival";
import type { MemoryArrivalPhase } from "./memory-arrival-state";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});
const base = {
  importing: false,
  importedMessages: 12,
  importedConversations: 2,
};
const noop = () => {};

it("import-only feedback does not imply HY generation", () => {
  const { container } = render(
    <MemoryArrivalPanel
      arrival={{ ...base, importing: true, phase: "importing" }}
      onConfigureModel={noop}
      onConnect={noop}
    />,
  );
  expect(screen.getByRole("status").textContent).toBe(
    "Importing conversations",
  );
  expect(container.textContent).not.toContain(
    "more memories will appear as they arrive",
  );
  expect(container.textContent).not.toContain("Settling memories");
});

it("confirmed running generation retains partial errors without looking stopped", () => {
  const { container } = render(
    <MemoryArrivalPanel
      arrival={{
        ...base,
        phase: "processing",
        blockedStages: [
          {
            kind: "extract",
            reason: "hy_memory_model_authentication_failed",
            count: 1,
          },
          {
            kind: "extract",
            reason: "hy_memory_input_exceeds_window",
            count: 5,
          },
        ],
      }}
      onConfigureModel={noop}
      onConnect={noop}
    />,
  );
  expect(screen.getByRole("status").textContent).toBe("Settling memories");
  expect(container.querySelector(".memory-arrival-spinner")).toBeTruthy();
  expect(container.querySelector("details")?.textContent).toContain(
    "Model authentication failed. Update the key",
  );
  expect(container.querySelector("details")?.textContent).toContain("5 items");
});

it.each([
  ["checking", "Progress needs refreshing", false],
  ["importing", "Importing conversations", true],
  ["model-required", "No organizing model set up", false],
  ["processing", "Settling memories", true],
  ["failed", "Waiting to resume", false],
  ["idle", "Waiting to resume", false],
  ["stopped", "Waiting to resume", false],
  ["unavailable", "Progress needs refreshing", false],
] satisfies [MemoryArrivalPhase, string, boolean][])(
  "renders honest %s without invented percentage",
  (phase, title, spinner) => {
    const { container } = render(
      <MemoryArrivalPanel
        arrival={{ ...base, phase }}
        onConfigureModel={noop}
        onConnect={noop}
      />,
    );
    expect(screen.getByRole("status").textContent).toBe(title);
    expect(!!container.querySelector(".memory-arrival-spinner")).toBe(spinner);
    expect(container.textContent).not.toContain("%");
    expect(screen.queryByRole("progressbar")).toBeNull();
  },
);

it("offers explicit connection only for the truly empty idle state", async () => {
  const connect = vi.fn();
  const view = render(
    <MemoryArrivalPanel
      arrival={{
        importing: false,
        importedMessages: 0,
        importedConversations: 0,
        phase: "idle",
      }}
      onConfigureModel={noop}
      onConnect={connect}
    />,
  );
  expect(screen.getByText("No conversations saved yet")).toBeDefined();
  expect(screen.queryByText(/Saved/)).toBeNull();
  expect(connect).not.toHaveBeenCalled();
  await userEvent.click(screen.getByRole("button", { name: "Connect an app" }));
  expect(connect).toHaveBeenCalledOnce();
  view.rerender(
    <MemoryArrivalPanel
      arrival={{ ...base, phase: "idle" }}
      onConfigureModel={noop}
      onConnect={connect}
    />,
  );
  expect(screen.queryByRole("button", { name: "Connect an app" })).toBeNull();
});

it.each(["suspended", "reduced-motion"])(
  "stops the processing indicator for %s",
  (mode) => {
    vi.stubGlobal("matchMedia", () => ({
      matches: mode === "reduced-motion",
      addEventListener() {},
      removeEventListener() {},
    }));
    const { container } = render(
      <MemoryArrivalPanel
        arrival={{ ...base, phase: "processing" }}
        suspended={mode === "suspended"}
        onConfigureModel={noop}
        onConnect={noop}
      />,
    );
    expect(
      container
        .querySelector(".memory-arrival-spinner")
        ?.getAttribute("data-moving"),
    ).toBe("false");
  },
);

it("keeps idle existing memories free of an empty-state message", () => {
  const view = render(
    <MemoryArrivalPanel
      arrival={{ ...base, phase: "idle" }}
      hasMemories
      onConfigureModel={noop}
      onConnect={noop}
    />,
  );
  expect(screen.queryByRole("region")).toBeNull();
  view.rerender(
    <MemoryArrivalPanel
      arrival={{ ...base, importing: true, phase: "idle" }}
      hasMemories
      onConfigureModel={noop}
      onConnect={noop}
    />,
  );
  expect(screen.getByText("Source text still importing")).toBeDefined();
  expect(
    screen.queryByText(
      "Conversations are saved, but no memories have been generated yet",
    ),
  ).toBeNull();
});

it("keeps background progress compact and exact failures collapsed", () => {
  const { container } = render(
    <MemoryArrivalPanel
      arrival={{
        ...base,
        phase: "processing",
        backlogPending: 20,
        blockedStages: [
          {
            kind: "extract",
            reason: "hy_memory_input_exceeds_window",
            count: 3,
          },
        ],
      }}
      onConfigureModel={noop}
      onConnect={noop}
    />,
  );
  expect(screen.getByRole("status").textContent).toBe("Settling memories");
  expect(
    screen.getByText(
      "Recent conversations are organized first; more memories will appear as they arrive",
    ),
  ).toBeTruthy();
  const details = container.querySelector("details");
  expect(details?.open).toBe(false);
  expect(details?.textContent).toContain(
    "This conversation contains oversized source text; not organized yet, source text kept",
  );
  expect(details?.textContent).toContain("3 items");
  // The code stays on the task row in Core. This panel is read by whoever
  // hit the stall, and `hy_memory_input_exceeds_window` is not an answer to
  // anything they can do.
  expect(details?.textContent).not.toContain("hy_memory_input_exceeds_window");
  expect(details?.textContent).toContain(
    "There is a large backlog; batches are being organized in the background to protect quality",
  );
  expect(screen.queryByRole("alert")).toBeNull();
});

it("does not spin or say checking without a live refresh", () => {
  const view = render(
    <MemoryArrivalPanel
      arrival={{ ...base, phase: "unavailable" }}
      onConfigureModel={noop}
      onConnect={noop}
    />,
  );
  expect(screen.getByRole("status").textContent).toBe(
    "Progress needs refreshing",
  );
  expect(view.container.querySelector(".memory-arrival-spinner")).toBeNull();
  view.rerender(
    <MemoryArrivalPanel
      arrival={{ ...base, phase: "unavailable", refreshing: true }}
      onConfigureModel={noop}
      onConnect={noop}
    />,
  );
  expect(screen.getByRole("status").textContent).toBe(
    "Saved content kept; checking progress",
  );
  view.rerender(
    <MemoryArrivalPanel
      arrival={{ ...base, phase: "unavailable", refreshing: false }}
      onConfigureModel={noop}
      onConnect={noop}
    />,
  );
  expect(screen.getByRole("status").textContent).toBe(
    "Progress needs refreshing",
  );
});

it("keeps a stopped authentication failure neutral without claiming processing never began", () => {
  const { container } = render(
    <MemoryArrivalPanel
      arrival={{
        ...base,
        phase: "failed",
        blockedStages: [
          {
            kind: "extract",
            reason: "hy_memory_model_authentication_failed",
            count: 1,
          },
        ],
      }}
      hasMemories
      onConfigureModel={noop}
      onConnect={noop}
    />,
  );
  expect(screen.getByRole("status").textContent).toBe("Waiting to resume");
  expect(container.querySelector(".memory-arrival-spinner")).toBeNull();
  expect(container.textContent).not.toContain("Nothing has started yet");
  expect(container.querySelector("details")?.textContent).toContain(
    "Model authentication failed. Update the key",
  );
  expect(container.textContent).not.toContain("hy_memory_");
  expect(
    screen.getByRole("button", { name: "Set up memory organizing" }),
  ).toBeTruthy();
});
