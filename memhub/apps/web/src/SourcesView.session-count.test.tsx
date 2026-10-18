// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { SourcesView } from "./SourcesView";
import type { useImportSources } from "./use-import-sources";

afterEach(cleanup);

function renderUnauthorizedSource(sessionCount: number | null | undefined) {
  const source = {
    source_id: "workbuddy",
    display_name: "WorkBuddy",
    available: true,
    authorized: false,
    candidate_kind: "default",
    candidate_path: "~/.workbuddy/projects",
    imported_messages: 0,
    imported_conversations: 0,
    capabilities: {
      verification: "fixture_only",
      live_sync: "not_implemented",
    },
    ...(sessionCount === undefined ? {} : { session_count: sessionCount }),
  };
  const imports = {
    sources: [source],
    jobs: [],
    latestJobBySource: new Map(),
    loading: false,
    busyKey: null,
    error: null,
    lastConnectResult: null,
    connectSelected: vi.fn(),
    syncAll: vi.fn(),
    startImport: vi.fn(),
    changeJob: vi.fn(),
  } as unknown as ReturnType<typeof useImportSources>;
  render(<SourcesView imports={imports} />);
}

it("says how many sessions a source holds before anyone grants it", () => {
  renderUnauthorizedSource(12);
  expect(screen.getByText(/Found 12 conversations/)).toBeTruthy();
});

it("says an installed host holds none in words rather than a bare zero", () => {
  renderUnauthorizedSource(0);
  expect(
    screen.getByText(/Installed, no conversations found yet/),
  ).toBeTruthy();
  expect(screen.queryByText(/Found 0 conversations/)).toBeNull();
});

it("prints no number it was not given", () => {
  renderUnauthorizedSource(null);
  expect(screen.getByText(/Saved directory found/)).toBeTruthy();
});
