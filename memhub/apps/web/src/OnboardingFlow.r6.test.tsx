// @vitest-environment jsdom
import type { SourceResponse } from "@one-memory/contracts";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { OnboardingFlow } from "./OnboardingFlow";
import type { ImportSourcesController } from "./use-import-sources";

vi.mock("./api", () => ({ completeOnboarding: vi.fn() }));
vi.mock("./FirstImportReview", () => ({
  FirstImportReview: ({ sourceIds }: { sourceIds: string[] }) => (
    <p>Accepted {sourceIds.join(",")}</p>
  ),
  defaultHostSourceIds: () => [],
}));
beforeEach(() => {
  delete window.oneMemory;
});
afterEach(() => {
  cleanup();
  delete window.oneMemory;
  vi.useRealTimers();
});
const state = {
  status: "not_started",
  selected_source_ids: [],
  started_at: null,
  completed_at: null,
} as const;
function source(
  id: string,
  overrides: Partial<SourceResponse> = {},
): SourceResponse {
  return {
    source_id: id,
    display_name: id,
    available: true,
    authorized: false,
    detected: false,
    storage_kinds: ["jsonl"],
    root_label: null,
    candidate_path: `/fixture/${id}`,
    candidate_kind: "default",
    candidate_root_id: `root-${id}`,
    location_state: "found",
    imported_messages: 0,
    imported_conversations: 0,
    sync_enabled: false,
    sync_completeness: "additions_only",
    capabilities: {
      historical_import: "implemented",
      incremental_import: "implemented",
      live_sync: "not_implemented",
      recall: "not_implemented",
      installer: "not_implemented",
      verification: "fixture_only",
    },
    ...overrides,
  };
}
function controller(sources: SourceResponse[]) {
  return {
    sources,
    jobs: [],
    loading: false,
    refresh: vi.fn().mockResolvedValue({ sources, jobs: [], health: null }),
    connectForOnboarding: vi.fn().mockResolvedValue({
      accepted_source_ids: [sources[0]?.source_id],
      failed_sources: {},
      jobs: [],
    }),
    detect: vi.fn(),
  } as unknown as ImportSourcesController;
}
function mount(imports: ImportSourcesController) {
  return render(
    <OnboardingFlow
      state={{ ...state, selected_source_ids: [] }}
      imports={imports}
      onFinish={() => {}}
    />,
  );
}
it("renders the final shell, double title, step one, branded rows and exact source metadata", async () => {
  const imports = controller([
    source("workbuddy"),
    source("cursor"),
    source("codebuddy"),
  ]);
  const view = mount(imports);
  await waitFor(() =>
    expect(view.container.querySelectorAll(".onboarding-r6-row").length).toBe(
      3,
    ),
  );
  expect(view.container.querySelector(".onboarding-r6-shell")).toBeTruthy();
  expect(view.container.querySelector(".cold-start-layout")).toBeNull();
  expect(
    screen.getByRole("heading", { name: "Bring the context you know in here" }),
  ).toBeTruthy();
  expect(screen.getByText("Your memory starts connecting here")).toBeTruthy();
  expect(
    screen
      .getByRole("navigation", { name: "First-time setup progress" })
      .querySelector('[aria-current="step"]')?.textContent,
  ).toContain("Import history");
  expect(
    screen.getByRole("heading", { name: "AI apps on this machine" }),
  ).toBeTruthy();
  expect(
    screen.getByText(
      "Before you confirm, only app metadata is checked: no conversations are read and no connections are installed.",
    ),
  ).toBeTruthy();
  expect(
    screen.queryByRole("button", { name: "Import history package" }),
  ).toBeNull();
  expect(
    screen.getByText("/fixture/workbuddy · Default location"),
  ).toBeTruthy();
  expect(
    view.container.querySelector('img[src$="r6/workbuddy.png"]'),
  ).toBeTruthy();
  expect(
    await within(
      screen.getByRole("complementary", { name: "This import" }),
    ).findByText("3"),
  ).toBeTruthy();
  expect(
    screen.getByText("Less repeating yourself, more being understood."),
  ).toBeTruthy();
  expect(imports.detect).not.toHaveBeenCalled();
});
it("does not fabricate found rows while metadata is pending, then uses actual source count", async () => {
  let done!: (
    v: Awaited<ReturnType<ImportSourcesController["refresh"]>>,
  ) => void;
  const imports = controller([]);
  vi.mocked(imports.refresh).mockReturnValueOnce(
    new Promise((resolve) => {
      done = resolve;
    }),
  );
  const view = mount(imports);
  expect(
    view.container.querySelectorAll(".onboarding-r6-skeleton").length,
  ).toBeGreaterThan(0);
  expect(screen.queryByText("Found")).toBeNull();
  await act(async () =>
    done({
      sources: [source("codex")],
      jobs: [],
      health: {} as Awaited<
        ReturnType<ImportSourcesController["refresh"]>
      >["health"],
    }),
  );
  expect(screen.getByText("/fixture/codex · Default location")).toBeTruthy();
  expect(screen.getAllByText("Found")).toHaveLength(1);
});
it("starts location discovery once and keeps a single initial focus owner", async () => {
  const imports = controller([source("codex")]);
  mount(imports);
  await waitFor(() => expect(imports.refresh).toHaveBeenCalledTimes(1));
  expect(document.activeElement).toBe(
    screen.getByRole("heading", { name: "AI apps on this machine" }),
  );
});
it("restores the first-step focus when the native window becomes active", async () => {
  const imports = controller([source("codex")]);
  mount(imports);
  const firstStep = screen.getByRole("heading", {
    name: "AI apps on this machine",
  });
  (document.activeElement as HTMLElement | null)?.blur();
  expect(document.activeElement).not.toBe(firstStep);

  act(() => window.dispatchEvent(new Event("focus")));

  expect(document.activeElement).toBe(firstStep);
});
it("keeps the summary count real and large enough to see while metadata resolves", async () => {
  const imports = controller([
    source("workbuddy"),
    source("cursor"),
    source("codex"),
  ]);
  const view = mount(imports);
  const summary = screen.getByRole("complementary", { name: "This import" });
  const count = summary.querySelector(".onboarding-r6-count");
  expect(count?.classList.contains("onboarding-r6-count")).toBe(true);
  expect(count?.textContent).toBe("1");
  expect(count?.getAttribute("aria-live")).toBe("polite");
  expect(count?.getAttribute("data-unknown")).toBe("true");
  fireEvent.click(screen.getByRole("button", { name: "Re-check locations" }));
  await waitFor(() =>
    expect(summary.querySelector(".onboarding-r6-count")?.textContent).toBe(
      "3",
    ),
  );
  expect(
    summary.querySelector(".onboarding-r6-count")?.getAttribute("data-unknown"),
  ).toBe("false");
  expect(view.container.querySelector(".onboarding-r6-list-area")).toBeTruthy();
});
it("shows Incomplete, never a number or zero, when the location check fails", async () => {
  // D51 item 3: only a *confirmed* source snapshot may show a numeric count.
  // A failed detection must say Incomplete — a number here (especially `0`) would be
  // a factual claim about the user's machine that the UI cannot support.
  const imports = controller([source("workbuddy"), source("cursor")]);
  vi.mocked(imports.refresh).mockRejectedValue(new Error("offline"));
  const view = mount(imports);
  const count = () =>
    view.container.querySelector(".onboarding-r6-count")?.textContent;
  await waitFor(() => expect(count()).toBe("Incomplete"));
  expect(count()).not.toBe("0");
  expect(count()).not.toBe("2");
  // The unknown-state styling must still be applied, so a failure is not
  // presented in the large confirmed-number treatment.
  expect(
    view.container
      .querySelector(".onboarding-r6-count")
      ?.getAttribute("data-unknown"),
  ).toBe("true");
  // And it must be announced as a failure in words, not only as a glyph.
  expect(screen.getByRole("alert").textContent).toContain(
    "Location check did not finish",
  );
});
it("keeps the first step inside the selected R6 cold-start shell", () => {
  const view = mount(controller([source("workbuddy")]));
  const shell = view.container.querySelector<HTMLElement>(
    ".onboarding-r6-shell",
  );
  const discovery = view.container.querySelector<HTMLElement>(
    ".onboarding-r6-discovery",
  );
  expect(shell).toBeTruthy();
  expect(discovery).toBeTruthy();
  expect(
    discovery?.parentElement?.classList.contains("onboarding-r6-body"),
  ).toBe(true);
  expect(shell?.getAttribute("data-step")).toBe("1");
  expect(view.container.querySelector(".onboarding-r6-titlebar")).toBeTruthy();
});
it("shows the scroll hint only when the first-step list actually clips rows", async () => {
  const imports = controller([
    source("workbuddy"),
    source("cursor"),
    source("codex"),
    source("claude"),
  ]);
  const view = mount(imports);
  await waitFor(() =>
    expect(view.container.querySelectorAll(".onboarding-r6-row").length).toBe(
      4,
    ),
  );
  const list = view.container.querySelector<HTMLElement>(".onboarding-r6-list");
  if (!list) throw new Error("missing source list");
  vi.spyOn(list, "getBoundingClientRect").mockReturnValue({
    bottom: 100,
  } as DOMRect);
  for (const child of Array.from(list.children))
    vi.spyOn(child, "getBoundingClientRect").mockReturnValue({
      bottom: 80,
    } as DOMRect);
  fireEvent.scroll(list);
  expect(screen.queryByText(/Scroll down for more/)).toBeNull();
  vi.mocked(list.children[3].getBoundingClientRect).mockReturnValue({
    bottom: 140,
  } as DOMRect);
  fireEvent.scroll(list);
  expect(
    await screen.findByText("Scroll down for more · 1 not shown"),
  ).toBeTruthy();
});
it("holds the native discovery placeholder, then shows every detected app ready to continue", async () => {
  vi.useFakeTimers();
  window.oneMemory = {} as NonNullable<Window["oneMemory"]>;
  const sources = ["workbuddy", "cursor", "codebuddy", "codex"].map((id) =>
    source(id),
  );
  const imports = controller(sources);
  const view = mount(imports);

  expect(view.container.querySelectorAll(".onboarding-r6-row")).toHaveLength(1);
  expect(
    view.container.querySelectorAll(".onboarding-r6-skeleton"),
  ).toHaveLength(2);
  expect(screen.getByText("Checking other apps...")).toBeTruthy();
  expect(screen.getByText("Found")).toBeTruthy();
  expect(screen.getByRole("button", { name: "Start import" })).toHaveProperty(
    "disabled",
    true,
  );
  expect(
    within(
      screen.getByRole("complementary", { name: "This import" }),
    ).getByText("1"),
  ).toBeTruthy();

  await act(async () => vi.advanceTimersByTimeAsync(850));
  expect(view.container.querySelectorAll(".onboarding-r6-row")).toHaveLength(4);
  expect(
    view.container.querySelectorAll(".onboarding-r6-skeleton"),
  ).toHaveLength(0);
  expect(
    screen.getByLabelText<HTMLInputElement>("Import cursor history").checked,
  ).toBe(true);
  expect(
    screen.getByLabelText<HTMLInputElement>("Import codebuddy history").checked,
  ).toBe(true);
  expect(
    screen.getByLabelText<HTMLInputElement>("Import codex history").checked,
  ).toBe(true);
  expect(screen.getByRole("button", { name: "Start import" })).toHaveProperty(
    "disabled",
    false,
  );
});
it("custom source cancel does not grant or select and accepted import retains review branch", async () => {
  const imports = controller([
    source("workbuddy"),
    source("codex", { candidate_kind: "environment_override" }),
  ]);
  mount(imports);
  await waitFor(() =>
    expect(
      (screen.getByLabelText("Import workbuddy history") as HTMLInputElement)
        .checked,
    ).toBe(true),
  );
  const codex = await screen.findByLabelText("Import codex history");
  await waitFor(() => expect((codex as HTMLInputElement).disabled).toBe(false));
  fireEvent.click(codex);
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(
    (screen.getByLabelText("Import codex history") as HTMLInputElement).checked,
  ).toBe(false);
  fireEvent.click(screen.getByRole("button", { name: "Start import" }));
  await screen.findByText("Accepted workbuddy");
  expect(vi.mocked(imports.connectForOnboarding).mock.calls[0]?.[0]).toEqual([
    "workbuddy",
  ]);
});

it("does not submit authorization when the first-step owner leaves during final metadata check", async () => {
  const imports = controller([source("workbuddy")]);
  const view = mount(imports);
  await waitFor(() =>
    expect(
      (screen.getByLabelText("Import workbuddy history") as HTMLInputElement)
        .checked,
    ).toBe(true),
  );
  let done!: (
    value: Awaited<ReturnType<ImportSourcesController["refresh"]>>,
  ) => void;
  vi.mocked(imports.refresh).mockReturnValueOnce(
    new Promise((resolve) => {
      done = resolve;
    }),
  );
  fireEvent.click(screen.getByRole("button", { name: "Start import" }));
  view.unmount();
  await act(async () =>
    done({
      sources: imports.sources,
      jobs: [],
      health: {} as Awaited<
        ReturnType<ImportSourcesController["refresh"]>
      >["health"],
    }),
  );
  expect(imports.connectForOnboarding).not.toHaveBeenCalled();
});
