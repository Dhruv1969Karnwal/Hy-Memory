// @vitest-environment jsdom
import { webcrypto } from "node:crypto";
import type {
  ImportJob,
  OnboardingState,
  SourceResponse,
} from "@one-memory/contracts";
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
import * as api from "./api";
import type { FirstImportReviewProps } from "./FirstImportReview";
import { OnboardingFlow } from "./OnboardingFlow";
import { useImportSources } from "./use-import-sources";

vi.mock("./api", () => ({
  getHealth: vi.fn(),
  getSources: vi.fn(),
  getImportJobs: vi.fn(),
  connectSources: vi.fn(),
  completeOnboarding: vi.fn(),
  getSourceConnection: vi.fn(),
  createImportJob: vi.fn(),
  detectSource: vi.fn(),
  setSourceSync: vi.fn(),
  changeImportJob: vi.fn(),
  getImportJob: vi.fn(),
  apiErrorCode: vi.fn(() => null),
}));
vi.mock("./FirstImportReview", () => ({
  defaultHostSourceIds: () => [],
  FirstImportReview: (props: FirstImportReviewProps) => (
    <section aria-label="Import result">
      <output aria-label="Accepted sources">{props.sourceIds.join(",")}</output>
      <output aria-label="Kept jobs">
        {props.jobs.map((job) => `${job.job_id}:${job.status}`).join(",")}
      </output>
      <output aria-label="Failed sources">
        {Object.keys(props.failedSources ?? {}).join(",")}
      </output>
      <button type="button" onClick={() => props.onContinue("settings")}>
        Continue to model setup
      </button>
      <button type="button" onClick={() => props.onContinue("sources")}>
        Go to apps
      </button>
    </section>
  ),
}));
vi.mock("./OnboardingModelController", () => ({
  OnboardingModelController: ({ onBack }: { onBack(): void }) => (
    <section aria-label="Model step">
      <button type="button" onClick={onBack}>
        Back to import
      </button>
    </section>
  ),
}));
const state: OnboardingState = {
  status: "not_started",
  selected_source_ids: [],
  started_at: null,
  completed_at: null,
};
let sources: SourceResponse[];
let jobs: ImportJob[];
const finish = vi.fn();
function source(
  id: string,
  patch: Partial<SourceResponse> = {},
): SourceResponse {
  return {
    source_id: id,
    display_name: id,
    authorized: false,
    available: true,
    detected: false,
    storage_kinds: ["jsonl"],
    candidate_path: `/fixture/${id}`,
    candidate_kind: "default",
    candidate_root_id: `root-${id}`,
    location_state: "found",
    root_label: null,
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
    ...patch,
  };
}
function Harness() {
  const imports = useImportSources({ automaticSync: false });
  return imports.loading ? (
    <p>Reading initial sources</p>
  ) : (
    <OnboardingFlow state={state} imports={imports} onFinish={finish} />
  );
}
function checkbox(id: string) {
  return screen.getByRole<HTMLInputElement>("checkbox", {
    name: `Import ${id} history`,
  });
}
function count() {
  return within(
    screen.getByRole("complementary", { name: "This import" }),
  ).getByText(/^\d+$/).textContent;
}
async function ready() {
  render(<Harness />);
  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: "Re-check locations" }),
    ).toHaveProperty("disabled", false),
  );
}
function noUnrequestedWrites() {
  for (const operation of [
    api.createImportJob,
    api.detectSource,
    api.setSourceSync,
  ])
    expect(operation).not.toHaveBeenCalled();
}
beforeEach(() => {
  vi.resetAllMocks();
  localStorage.clear();
  vi.stubGlobal("crypto", webcrypto);
  delete window.oneMemory;
  sources = [];
  jobs = [];
  vi.mocked(api.getHealth).mockResolvedValue({
    status: "ok",
    service: "one-memory-core",
    version: "fixture",
    instance_id: "fixture-instance",
  });
  vi.mocked(api.getSources).mockImplementation(async () =>
    sources.map((item) => ({ ...item })),
  );
  vi.mocked(api.getImportJobs).mockImplementation(async () =>
    jobs.map((job) => ({ ...job })),
  );
  vi.mocked(api.completeOnboarding).mockResolvedValue({
    ...state,
    status: "completed",
  });
  vi.mocked(api.connectSources).mockImplementation(
    async (ids, _confirmed, options) => ({
      request_id: options?.requestId ?? null,
      accepted_source_ids: ids,
      failed_sources: {},
      jobs,
    }),
  );
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  localStorage.clear();
});

it("defaults to all three readable apps without authorization and preserves explicit deselection after refresh", async () => {
  sources = [source("workbuddy"), source("cursor"), source("codex")];
  await ready();
  await waitFor(() => expect(count()).toBe("3"));
  expect(
    screen.getByRole("list", { name: "Candidate apps" }).children,
  ).toHaveLength(3);
  for (const item of sources)
    expect(checkbox(item.source_id).checked).toBe(true);
  expect(api.connectSources).not.toHaveBeenCalled();
  noUnrequestedWrites();
  fireEvent.click(checkbox("cursor"));
  expect(count()).toBe("2");
  fireEvent.click(screen.getByRole("button", { name: "Re-check locations" }));
  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: "Re-check locations" }),
    ).toHaveProperty("disabled", false),
  );
  expect(checkbox("cursor").checked).toBe(false);
  expect(count()).toBe("2");
  fireEvent.click(screen.getByRole("button", { name: "Start import" }));
  await screen.findByRole("region", { name: "Import result" });
  expect(api.connectSources).toHaveBeenCalledExactlyOnceWith(
    ["codex", "workbuddy"],
    true,
    expect.objectContaining({
      onboarding: true,
      fullRescan: true,
      sourceRoots: { codex: "root-codex", workbuddy: "root-workbuddy" },
    }),
  );
  noUnrequestedWrites();
});

it("canceling custom location does not select it on later metadata refresh", async () => {
  sources = [
    source("workbuddy"),
    source("codex", { candidate_kind: "environment_override" }),
  ];
  await ready();
  await waitFor(() => expect(count()).toBe("1"));
  fireEvent.click(checkbox("codex"));
  const dialog = screen.getByRole("dialog", {
    name: "Confirm custom location",
  });
  fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
  fireEvent.click(screen.getByRole("button", { name: "Re-check locations" }));
  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: "Re-check locations" }),
    ).toHaveProperty("disabled", false),
  );
  expect(checkbox("codex").checked).toBe(false);
  expect(count()).toBe("1");
  expect(api.connectSources).not.toHaveBeenCalled();
  noUnrequestedWrites();
});

it("keeps the single-source submit frozen when another source arrives during final refresh", async () => {
  sources = [source("workbuddy")];
  await ready();
  await waitFor(() => expect(count()).toBe("1"));
  let resolve!: (items: SourceResponse[]) => void;
  vi.mocked(api.getSources).mockReturnValueOnce(
    new Promise((done) => {
      resolve = done;
    }),
  );
  fireEvent.click(screen.getByRole("button", { name: "Start import" }));
  expect(api.connectSources).not.toHaveBeenCalled();
  sources = [...sources, source("codex")];
  await act(async () => resolve(sources));
  await screen.findByRole("region", { name: "Import result" });
  expect(api.connectSources).toHaveBeenCalledExactlyOnceWith(
    ["workbuddy"],
    true,
    expect.objectContaining({ sourceRoots: { workbuddy: "root-workbuddy" } }),
  );
  expect(screen.getByLabelText("Accepted sources").textContent).toBe(
    "workbuddy",
  );
  noUnrequestedWrites();
});

it("empty discovery shows zero and allows explicit entry without a connection request", async () => {
  await ready();
  expect(count()).toBe("0");
  expect(screen.getByRole("button", { name: "Start import" })).toHaveProperty(
    "disabled",
    true,
  );
  fireEvent.click(
    screen.getByRole("button", { name: "Add later, continue now" }),
  );
  await waitFor(() =>
    expect(finish).toHaveBeenCalledExactlyOnceWith("sources"),
  );
  expect(api.completeOnboarding).toHaveBeenCalledOnce();
  expect(api.connectSources).not.toHaveBeenCalled();
  noUnrequestedWrites();
});

it("partial acceptance preserves the successful job and failure through model-step back/continue without resubmission", async () => {
  sources = [source("workbuddy"), source("cursor")];
  const job = {
    job_id: "accepted-job",
    user_id: "local",
    source_id: "workbuddy",
    status: "completed",
    created_at: "2026-09-19T00:00:00Z",
    updated_at: "2026-09-19T00:00:01Z",
  } as ImportJob;
  vi.mocked(api.connectSources).mockImplementationOnce(
    async (_ids, _confirmed, options) => {
      sources = [
        source("workbuddy", { authorized: true, grant_revision: 1 }),
        source("cursor"),
      ];
      jobs = [job];
      return {
        request_id: options?.requestId ?? null,
        accepted_source_ids: ["workbuddy"],
        failed_sources: { cursor: "source_unavailable" },
        jobs: [job],
      };
    },
  );
  await ready();
  await waitFor(() => expect(count()).toBe("2"));
  fireEvent.click(screen.getByRole("button", { name: "Start import" }));
  await screen.findByRole("region", { name: "Import result" });
  expect(screen.getByLabelText("Accepted sources").textContent).toBe(
    "workbuddy",
  );
  expect(screen.getByLabelText("Failed sources").textContent).toBe("cursor");
  expect(screen.getByLabelText("Kept jobs").textContent).toBe(
    "accepted-job:completed",
  );
  fireEvent.click(
    screen.getByRole("button", { name: "Continue to model setup" }),
  );
  fireEvent.click(
    await screen.findByRole("button", { name: "Back to import" }),
  );
  expect(screen.getByLabelText("Kept jobs").textContent).toBe(
    "accepted-job:completed",
  );
  expect(screen.getByLabelText("Failed sources").textContent).toBe("cursor");
  expect(api.completeOnboarding).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Go to apps" }));
  await waitFor(() =>
    expect(finish).toHaveBeenCalledExactlyOnceWith("sources"),
  );
  expect(api.connectSources).toHaveBeenCalledOnce();
  noUnrequestedWrites();
});
