// @vitest-environment jsdom

import type { ImportJob, SourceResponse } from "@one-memory/contracts";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { R6ImportProgress } from "./R6ImportProgress";

afterEach(cleanup);
const source = {
  source_id: "cursor",
  display_name: "Cursor",
  authorized: true,
  imported_messages: 9999,
} as SourceResponse;
const job = {
  job_id: "j1",
  source_id: "cursor",
  status: "running",
  stored: 17,
  errors: 0,
  created_at: "2026-09-19T00:00:00Z",
  updated_at: "2026-09-19T00:01:00Z",
} as ImportJob;
it("uses the selected job's message count, never the source total or fake percentage", () => {
  const { container } = render(
    <R6ImportProgress
      sourceIds={["cursor", "cursor"]}
      sources={[source]}
      jobs={[job]}
    />,
  );
  expect(screen.getByText("Saved 17 messages")).toBeTruthy();
  expect(container.querySelectorAll("article")).toHaveLength(1);
  expect(container.querySelector("article")?.getAttribute("data-running")).toBe(
    "true",
  );
  expect(container.textContent).not.toContain("9999");
  expect(container.textContent).not.toContain("%");
  expect(container.textContent).not.toContain("Continuously importing");
});
it.each(["paused", "interrupted", "failed", "completed", "cancelled"] as const)(
  "does not animate %s as active work",
  (status) => {
    const { container } = render(
      <R6ImportProgress
        sourceIds={["cursor"]}
        sources={[source]}
        jobs={[{ ...job, status }]}
      />,
    );
    expect(
      container.querySelector("article")?.getAttribute("data-running"),
    ).toBe("false");
  },
);
it("stops animation when source state is stale or revoked, without converting old counts to zero", () => {
  const view = render(
    <R6ImportProgress
      sourceIds={["cursor"]}
      sources={[source]}
      jobs={[job]}
      stale
    />,
  );
  expect(screen.getByText("Status pending verification")).toBeTruthy();
  expect(
    view.container.querySelector("article")?.getAttribute("data-running"),
  ).toBe("false");
  view.rerender(
    <R6ImportProgress
      sourceIds={["cursor"]}
      sources={[{ ...source, authorized: false }]}
      jobs={[job]}
    />,
  );
  expect(screen.getByText("Read authorization unavailable")).toBeTruthy();
  expect(screen.getByText("Saved 17 messages")).toBeTruthy();
});
it.each(["missing", "pending"] as const)(
  "shows %s work optimistically as a single importing status",
  (kind) => {
    const { container } = render(
      <R6ImportProgress
        sourceIds={["cursor"]}
        sources={[source]}
        jobs={kind === "pending" ? [{ ...job, status: "pending" }] : []}
      />,
    );
    expect(screen.getByText("Importing")).toBeTruthy();
    expect(container.textContent).not.toContain("Waiting for a save receipt");
    expect(container.textContent).not.toContain("Continuously importing");
    expect(
      container.querySelector("article")?.getAttribute("data-running"),
    ).toBe("true");
  },
);

it("centers one or two apps and keeps three or more in a scrolling row", () => {
  function shown(ids: string[]) {
    const view = render(
      <R6ImportProgress
        sourceIds={ids}
        sources={ids.map((id) => ({
          ...source,
          source_id: id,
          display_name: id,
        }))}
        jobs={[]}
      />,
    );
    const count = view.container
      .querySelector(".r6-import-progress")
      ?.getAttribute("data-count");
    view.unmount();
    return count;
  }
  expect(shown(["cursor"])).toBe("1");
  expect(shown(["cursor", "codex"])).toBe("2");
  expect(shown(["cursor", "codex", "claude-code"])).toBe("many");
  expect(shown(["cursor", "codex", "claude-code", "workbuddy"])).toBe("many");
  const four = render(
    <R6ImportProgress
      sourceIds={["cursor", "codex", "claude-code", "workbuddy"]}
      sources={["cursor", "codex", "claude-code", "workbuddy"].map((id) => ({
        ...source,
        source_id: id,
        display_name: id,
      }))}
      jobs={[]}
    />,
  );
  expect(four.container.querySelectorAll("article")).toHaveLength(4);
  expect(
    four.container
      .querySelector(".r6-import-progress")
      ?.getAttribute("data-count"),
  ).toBe("many");
});

it("keeps the first app anchored across receipts unless the user scrolls", () => {
  const sources = ["cursor", "codex", "claude-code", "workbuddy"].map((id) => ({
    ...source,
    source_id: id,
    display_name: id,
  }));
  const view = render(
    <R6ImportProgress
      sourceIds={sources.map((item) => item.source_id)}
      sources={sources}
      jobs={[]}
    />,
  );
  const progress = screen.getByRole("region", {
    name: "Import progress for this app",
  });
  progress.scrollLeft = 420;
  fireEvent.scroll(progress);
  view.rerender(
    <R6ImportProgress
      sourceIds={sources.map((item) => item.source_id)}
      sources={sources}
      jobs={[job]}
    />,
  );
  expect(progress.scrollLeft).toBe(0);

  fireEvent.wheel(progress, { deltaX: 420 });
  progress.scrollLeft = 420;
  fireEvent.scroll(progress);
  view.rerender(
    <R6ImportProgress
      sourceIds={sources.map((item) => item.source_id)}
      sources={sources}
      jobs={[{ ...job, stored: 18 }]}
    />,
  );
  expect(progress.scrollLeft).toBe(420);
});

it("does not claim importing when the source is revoked", () => {
  const { container } = render(
    <R6ImportProgress
      sourceIds={["cursor"]}
      sources={[{ ...source, authorized: false }]}
      jobs={[]}
    />,
  );
  expect(screen.getByText("Read authorization unavailable")).toBeTruthy();
  expect(container.querySelector("article")?.getAttribute("data-running")).toBe(
    "false",
  );
});
