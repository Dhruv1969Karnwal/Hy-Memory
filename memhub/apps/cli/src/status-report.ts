import type {
  HomeSummary,
  LiveSyncHealth,
  MemoryProcessingSummary,
  ModelProfile,
  SourceResponse,
} from "@one-memory/contracts";
import type {
  BackgroundHealth,
  RuntimeStatus,
} from "@one-memory/runtime-control";
import type { CoreReader, ReadOutcome } from "./core-client.js";
import {
  validateHomeSummary,
  validateLiveSync,
  validateModelProfiles,
  validateProcessing,
  validateSources,
} from "./core-responses.js";
import { nounLabel } from "./count-label.js";
import { PRODUCT } from "./version.js";

export type SectionState =
  | "ok"
  | "denied"
  | "unavailable"
  | "invalid"
  | "not_checked";

export type Section = { state: SectionState; code?: string } & Record<
  string,
  unknown
>;

export interface StatusDocument {
  schema_version: 1;
  product: string;
  data_dir: string;
  runtime: {
    state: RuntimeStatus["state"];
    code: string;
    instance_id?: string;
    host?: string;
    port?: number;
    version?: string;
  };
  bridge: {
    state: BackgroundHealth["state"] | "not_checked";
    code: string;
    core?: BackgroundHealth["core"];
    broker?: BackgroundHealth["broker"];
  };
  sections: Record<string, Section>;
  summary: {
    state: "ok" | "partial";
    degraded: string[];
    not_checked: string[];
  };
}

const SECTION_NAMES = [
  "sources",
  "sync",
  "processing",
  "model",
  "memories",
] as const;

const BRIDGE_LABELS: Record<StatusDocument["bridge"]["state"], string> = {
  healthy: "Healthy",
  stopped: "Not running",
  degraded: "Degraded",
  untrusted: "Identity not trusted",
  unknown: "Unknown",
  not_checked: "Not checked",
};

const SECTION_LABELS: Record<string, string> = {
  sources: "Sources",
  sync: "Sync",
  processing: "Generation jobs",
  model: "Model",
  memories: "Memories",
};

export async function composeStatusReport(options: {
  dataDir: string;
  runtime: RuntimeStatus;
  background: BackgroundHealth | null;
  reader: CoreReader | null;
  attachFailure?: string;
}): Promise<StatusDocument> {
  const sections: Record<string, Section> = {};
  if (options.reader === null) {
    const state: SectionState =
      options.attachFailure === undefined ? "not_checked" : "unavailable";
    const code = options.attachFailure ?? options.runtime.code;
    for (const name of SECTION_NAMES) sections[name] = { state, code };
  } else {
    const reader = options.reader;
    const [sources, sync, processing, model, memories] = await Promise.all([
      reader<unknown>("/api/v1/sources"),
      reader<unknown>("/api/v1/live-sync"),
      reader<unknown>("/api/v1/memory-processing"),
      reader<unknown>("/api/v1/model-profiles"),
      reader<unknown>("/api/v1/home-summary"),
    ]);
    sections.sources = section(sources, validateSources, describeSources);
    sections.sync = section(sync, validateLiveSync, describeSync);
    sections.processing = section(
      processing,
      validateProcessing,
      describeProcessing,
    );
    sections.model = section(model, validateModelProfiles, describeModel);
    sections.memories = section(
      memories,
      validateHomeSummary,
      describeMemories,
    );
  }
  // A bridge we could not inspect is unknown, not broken and not fine.
  const bridge: StatusDocument["bridge"] =
    options.background === null
      ? { state: "not_checked", code: "bridge_inspection_failed" }
      : {
          state: options.background.state,
          code: options.background.code,
          core: options.background.core,
          broker: options.background.broker,
        };
  const degraded: string[] = SECTION_NAMES.filter((name) =>
    ["denied", "unavailable", "invalid"].includes(
      String(sections[name]?.state),
    ),
  );
  if (bridge.state === "degraded" || bridge.state === "untrusted") {
    degraded.push("bridge");
  }
  const notChecked: string[] = SECTION_NAMES.filter(
    (name) => sections[name]?.state === "not_checked",
  );
  if (bridge.state === "not_checked" || bridge.state === "unknown") {
    notChecked.push("bridge");
  }
  return {
    schema_version: 1,
    product: PRODUCT,
    data_dir: options.dataDir,
    runtime: {
      state: options.runtime.state,
      code: options.runtime.code,
      instance_id: options.runtime.instanceId,
      host: options.runtime.host,
      port: options.runtime.port,
      version: options.runtime.version,
    },
    bridge,
    sections,
    summary: {
      state:
        degraded.length === 0 && notChecked.length === 0 ? "ok" : "partial",
      degraded,
      not_checked: notChecked,
    },
  };
}

export function statusExitCode(report: StatusDocument): number {
  if (report.runtime.state === "stopped") return 3;
  if (report.runtime.state !== "running") return 1;
  return report.summary.state === "ok" ? 0 : 1;
}

export function renderStatusText(
  report: StatusDocument,
  heading: string,
): string {
  const lines = [
    `${heading}: ${runtimeLine(report)}`,
    `Data space: ${report.data_dir}`,
    `App connections: ${BRIDGE_LABELS[report.bridge.state]}`,
  ];
  for (const name of SECTION_NAMES) {
    lines.push(
      `${SECTION_LABELS[name]}: ${sectionLine(report.sections[name])}`,
    );
  }
  if (report.summary.state === "partial") {
    const notes = [
      report.summary.degraded.length > 0
        ? `${report.summary.degraded.join(", ")} ${nounLabel(
            report.summary.degraded.length,
            "issue",
          )}`
        : "",
      report.summary.not_checked.length > 0
        ? `${report.summary.not_checked.join(", ")} could not be read`
        : "",
    ].filter((note) => note.length > 0);
    lines.push(
      `Note: ${notes.join("; ")}; the above is not a complete status.`,
    );
  }
  lines.push("");
  return lines.join("\n");
}

function runtimeLine(report: StatusDocument): string {
  if (report.runtime.state !== "running") {
    return report.runtime.state === "stopped"
      ? "Not running"
      : "Cannot confirm yet";
  }
  return `Running (${report.runtime.host}:${report.runtime.port}${
    report.runtime.version ? `, version ${report.runtime.version}` : ""
  })`;
}

function sectionLine(section: Section | undefined): string {
  if (!section) return "Not checked";
  if (section.state === "not_checked") return "Not checked";
  if (section.state !== "ok")
    return "Temporarily unavailable; try again shortly";
  return Object.entries(section)
    .filter(([key]) => key !== "state" && key !== "code")
    .map(([key, value]) => `${fieldLabel(key)} ${formatValue(value)}`)
    .join(" ");
}

function formatValue(value: unknown): string {
  if (Array.isArray(value))
    return value.length === 0
      ? "None"
      : value.map((item) => formatValue(item)).join("; ");
  if (value === null || value === undefined) return "None";
  if (typeof value === "object") {
    return Object.entries(value as Record<string, unknown>)
      .map(([key, item]) => `${fieldLabel(key)} ${formatValue(item)}`)
      .join("/");
  }
  return valueLabel(String(value));
}

function fieldLabel(key: string): string {
  const labels: Record<string, string> = {
    total: "Total",
    authorized: "Authorized",
    available: "Discovered",
    imported_conversations: "Conversations",
    imported_messages: "Originals",
    sources: "Sources",
    running: "Running",
    idle: "Idle",
    degraded: "Needs attention",
    unavailable: "Unavailable",
    pending: "Pending",
    fresh_pending: "New conversations pending",
    backlog_pending: "History pending",
    outside_window_pending: "Process later",
    waiting_for_model: "Waiting for model",
    completed: "Completed",
    failed: "Failed",
    blocked: "Stuck",
    kind: "Kind",
    reason: "Reason",
    count: "Count",
    profiles: "Profiles",
    active: "Current",
    memory_count: "Memories",
  };
  return labels[key] ?? key;
}

function valueLabel(value: string): string {
  const labels: Record<string, string> = {
    extract: "Organize conversations",
    waiting_for_model: "Waiting for model",
    input_too_long: "Content too long",
    background_verified: "Background verified",
    additions_only: "Only new content is synced",
  };
  return labels[value] ?? value;
}

function section<T>(
  outcome: ReadOutcome<unknown>,
  validate: (value: unknown) => T | null,
  describe: (data: T) => Record<string, unknown>,
): Section {
  if (outcome.state !== "ok") {
    // A status route that is missing is a Runtime we cannot read, not an empty one.
    return {
      state: outcome.state === "denied" ? "denied" : "unavailable",
      code: outcome.code,
    };
  }
  const data = validate(outcome.data);
  // Answered, but not with something we can read: not a smaller truth.
  return data === null
    ? { state: "invalid", code: "response_invalid" }
    : { state: "ok", ...describe(data) };
}

function describeSources(sources: SourceResponse[]): Record<string, unknown> {
  return {
    total: sources.length,
    authorized: sources.filter((source) => source.authorized).length,
    available: sources.filter((source) => source.available).length,
    imported_conversations: sum(sources, "imported_conversations"),
    imported_messages: sum(sources, "imported_messages"),
  };
}

function describeSync(health: LiveSyncHealth[]): Record<string, unknown> {
  return {
    sources: health.length,
    running: health.filter((item) => item.state === "running").length,
    idle: health.filter((item) => item.state === "idle").length,
    degraded: health.filter((item) => item.state === "degraded").length,
    unavailable: health.filter((item) => item.state === "unavailable").length,
  };
}

function describeProcessing(
  summary: MemoryProcessingSummary,
): Record<string, unknown> {
  return {
    pending: summary.pending,
    fresh_pending: summary.fresh_pending,
    backlog_pending: summary.backlog_pending,
    outside_window_pending: summary.outside_window_pending,
    running: summary.running,
    waiting_for_model: summary.waiting_for_model,
    completed: summary.completed,
    failed: summary.failed,
    blocked: (summary.blocked_stages ?? []).map((block) => ({
      kind: block.kind,
      reason: block.reason,
      count: block.count,
    })),
  };
}

/** Projection, not passthrough: a Core that leaks a key must not make us leak. */
function describeModel(profiles: ModelProfile[]): Record<string, unknown> {
  const active = profiles.find((profile) => profile.is_active);
  return {
    profiles: profiles.length,
    active: active
      ? {
          name: active.name,
          model: active.model,
        }
      : null,
  };
}

function describeMemories(summary: HomeSummary): Record<string, unknown> {
  return {
    memory_count: summary.memory_count,
  };
}

function sum(
  sources: SourceResponse[],
  key: "imported_conversations" | "imported_messages",
): number {
  return sources.reduce((total, source) => total + (source[key] ?? 0), 0);
}
