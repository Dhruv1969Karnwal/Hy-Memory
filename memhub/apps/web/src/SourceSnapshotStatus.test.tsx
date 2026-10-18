// @vitest-environment jsdom

import { act, cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { SourceSnapshotStatus } from "./SourceSnapshotStatus";

afterEach(cleanup);
const confirmedAt = "2026-09-14T04:00:00.000Z";

it("labels the last full-source observation without inventing a source event time", () => {
  render(
    <SourceSnapshotStatus
      snapshot={{ confirmedAt, stale: false }}
      onRefresh={vi.fn()}
    />,
  );
  const stamp = screen.getByText(/Status confirmed at/).querySelector("time");
  expect(stamp?.dateTime).toBe(confirmedAt);
  expect(screen.queryByText("Status pending verification")).toBeNull();
});

it("retains the exact prior time in stale state and offers an explicit read-only refresh", async () => {
  const onRefresh = vi.fn().mockResolvedValue(undefined);
  render(
    <SourceSnapshotStatus
      snapshot={{ confirmedAt, stale: true }}
      onRefresh={onRefresh}
    />,
  );
  expect(screen.getByText("Status pending verification")).toBeTruthy();
  expect(
    screen.getByText(/Last confirmed/).querySelector("time")?.dateTime,
  ).toBe(confirmedAt);
  await userEvent.click(
    screen.getByRole("button", { name: "Refresh source status" }),
  );
  expect(onRefresh).toHaveBeenCalledOnce();
  expect(screen.getByText("Status pending verification")).toBeTruthy();
});

it.each([null, "not-a-time"])(
  "does not fabricate an observation for %s",
  (confirmedAt) => {
    render(
      <SourceSnapshotStatus
        snapshot={{ confirmedAt, stale: true }}
        onRefresh={vi.fn()}
      />,
    );
    expect(screen.getByText("No confirmed status yet")).toBeTruthy();
    expect(document.querySelector("time")).toBeNull();
  },
);

it("coalesces repeated refresh clicks and retains the error until a real new snapshot", async () => {
  let fail!: (reason: Error) => void;
  const onRefresh = vi.fn(
    () =>
      new Promise((_, reject) => {
        fail = reject;
      }),
  );
  const view = render(
    <SourceSnapshotStatus
      snapshot={{ confirmedAt, stale: true }}
      onRefresh={onRefresh}
    />,
  );
  await userEvent.dblClick(
    screen.getByRole("button", { name: "Refresh source status" }),
  );
  expect(onRefresh).toHaveBeenCalledOnce();
  expect(
    screen.getByRole<HTMLButtonElement>("button", {
      name: "Refresh source status",
    }).disabled,
  ).toBe(true);
  await act(async () => fail(new Error("offline")));
  expect(screen.getByRole("alert").textContent).toContain("No new status yet");
  expect(
    screen.getByText(/Last confirmed/).querySelector("time")?.dateTime,
  ).toBe(confirmedAt);
  view.rerender(
    <SourceSnapshotStatus
      snapshot={{ confirmedAt: "2026-09-14T04:01:00.000Z", stale: false }}
      onRefresh={onRefresh}
    />,
  );
  expect(screen.queryByRole("alert")).toBeNull();
  expect(screen.queryByText("Status pending verification")).toBeNull();
});

it("reports a failed refresh even when the last snapshot was not stale", async () => {
  const onRefresh = vi.fn().mockRejectedValue(new Error("offline"));
  render(
    <SourceSnapshotStatus
      snapshot={{ confirmedAt, stale: false }}
      onRefresh={onRefresh}
    />,
  );
  await userEvent.click(
    screen.getByRole("button", { name: "Refresh source status" }),
  );
  expect((await screen.findByRole("alert")).textContent).toContain(
    "The refresh failed; the last confirmed status is still shown.",
  );
  expect(
    screen.getByText(/Status confirmed at/).querySelector("time")?.dateTime,
  ).toBe(confirmedAt);
  expect(screen.queryByText("No confirmed status yet")).toBeNull();
});

it("clears a failed refresh only after a newer authoritative snapshot arrives", async () => {
  const onRefresh = vi.fn().mockRejectedValue(new Error("offline"));
  const view = render(
    <SourceSnapshotStatus
      snapshot={{ confirmedAt, stale: false }}
      onRefresh={onRefresh}
    />,
  );
  await userEvent.click(
    screen.getByRole("button", { name: "Refresh source status" }),
  );
  await screen.findByRole("alert");
  view.rerender(
    <SourceSnapshotStatus
      snapshot={{ confirmedAt: "2026-09-14T04:01:00.000Z", stale: false }}
      onRefresh={onRefresh}
    />,
  );
  expect(screen.queryByRole("alert")).toBeNull();
});

it("has no UI for legacy callers with no snapshot", () => {
  render(<SourceSnapshotStatus onRefresh={vi.fn()} />);
  expect(screen.queryByRole("button")).toBeNull();
});
