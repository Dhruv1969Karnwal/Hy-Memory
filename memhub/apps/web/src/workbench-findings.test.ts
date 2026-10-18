import type {
  GovernedMemory,
  MemoryProcessingSummary,
  SourceResponse,
} from "@one-memory/contracts";
import { describe, expect, it } from "vitest";

import { findingsOf, mattersOf, memoriesInMatter } from "./workbench-findings";

const NOW = new Date("2026-09-10T00:00:00Z");

function memory(
  id: string,
  overrides: Partial<GovernedMemory> = {},
): GovernedMemory {
  return {
    memory_id: id,
    user_id: "local",
    kind: "chat",
    status: "confirmed",
    content: `Memory ${id}`,
    scopes: ["all-apps"],
    evidence_ids: [`ev-${id}`],
    origin: "extracted",
    tags: [],
    revision: 1,
    created_at: "2026-09-01T00:00:00Z",
    updated_at: "2026-09-01T00:00:00Z",
    deleted_at: null,
    ...overrides,
  };
}

function source(
  sourceId: string,
  overrides: Partial<SourceResponse> = {},
): SourceResponse {
  return {
    source_id: sourceId,
    display_name: sourceId,
    storage_kinds: ["jsonl"],
    available: true,
    authorized: true,
    detected: true,
    root_label: null,
    candidate_path: `~/.${sourceId}`,
    candidate_kind: "default",
    imported_messages: 10,
    imported_conversations: 2,
    capabilities: {
      historical_import: "implemented",
      incremental_import: "implemented",
      live_sync: "not_implemented",
      recall: "not_implemented",
      installer: "not_implemented",
      verification: "real_replayed",
    },
    sync_enabled: false,
    sync_completeness: "additions_only",
    ...overrides,
  };
}

const processing: MemoryProcessingSummary = {
  pending: 0,
  running: 0,
  fresh_pending: 0,
  backlog_pending: 0,
  recent_completions: [],
  waiting_for_model: 0,
  completed: 0,
  failed: 0,
  blocked_stages: [],
  outside_window_pending: 0,
  extraction_window_preset: "3m",
  manual_extract_batches_planned: 0,
  manual_extract_batches_current: 0,
  active_queue: 0,
  in_flight_count: 0,
  seconds_per_completion: null,
} as MemoryProcessingSummary;

describe("mattersOf", () => {
  // Three tests used to guard "group by scope" here. That axis was judged wrong:
  // scope is an authorization range; in a real library it reads workspace:v1:<sha256>,
  // so listing it as matters was a column of hashes. They were deleted not to go green,
  // but because the behavior they guarded should not exist at all.
  // An authorization scope still has to be sayable, only now inside the duplicate-finding
  it("authorization scopes are still stated, just no longer as the matter axis", () => {
    const matters = mattersOf(
      [memory("a", { scopes: ["all-apps"], tags: [] })],
      [source("cursor")],
    );

    // The left column no longer adds a row because of a scope.
    expect(matters.map((item) => item.label)).toEqual(["All"]);
    // Fetching memories by scope still works: a saved selection must not become unfindable after the axis change.
    expect(
      memoriesInMatter(
        [memory("a", { scopes: ["all-apps"], tags: [] })],
        "all-apps",
      ),
    ).toHaveLength(1);
  });

  it("a duplicate finding never prints a whole hash", () => {
    // scopesText reads out each memory's authorization scopes. In a real library that is 64
    // hex characters, which fills the line and says nothing. Only the short id needs to be recognizable here; the long string must not appear.
    const said = findingsOf({
      memories: [
        memory("a", {
          scopes: ["workspace:v1:0027abad1745330f1b57671635e0e9b3bd0cc3d2"],
          evidence_ids: ["ev-1"],
        }),
        memory("b", {
          scopes: ["workspace:v1:daf248ff48ea3e6baf35cc7f1fb46d74d5dc5360"],
          evidence_ids: ["ev-1"],
        }),
      ],
      processing: null,
      now: new Date("2026-09-10T00:00:00Z"),
    })
      .map((item) => item.text)
      .join(" ");

    expect(said).not.toContain("0027abad1745330f");
    // but the two projects must still be tellable apart, otherwise the sentence says nothing.
    expect(said).toContain("0027abad");
    expect(said).toContain("daf248ff");
  });

  it("with no memories the source count is 0, without padding with imported sources", () => {
    const matters = mattersOf([], [source("cursor")]);

    expect(matters[0]).toMatchObject({ memoryCount: 0, sourceCount: 0 });
  });
});

describe("findingsOf", () => {
  it("points out two memories extracted from the same sentence and gives each owner", () => {
    const findings = findingsOf({
      memories: [
        memory("a", {
          content: "Wrap everything with axios, do not call fetch directly",
          scopes: ["project:om"],
          evidence_ids: ["ev-1"],
        }),
        memory("b", {
          content: "Wrap everything with axios, no direct fetch calls",
          scopes: ["all-apps"],
          evidence_ids: ["ev-1", "ev-2"],
        }),
      ],
      processing,
      now: NOW,
    });

    const duplicate = findings.find((item) => item.kind === "duplicate");
    expect(duplicate?.memoryIds).toEqual(["a", "b"]);
    expect(duplicate?.text).toContain("om");
    expect(duplicate?.text).toContain("Personal profile");
  });

  it("two memories with different provenance are not duplicates even when worded alike", () => {
    // Measured in practice: surface similarity cannot separate them. "PersonaMem is stuck
    // at the extraction layer" and "leave the retrieval layer alone" scored 0.346, higher than several genuine duplicate pairs, so the criterion is provenance only.
    const findings = findingsOf({
      memories: [
        memory("a", {
          content: "Wrap everything with axios, do not call fetch directly",
          evidence_ids: ["ev-1"],
        }),
        memory("b", {
          content: "Wrap everything with axios, no direct fetch calls",
          evidence_ids: ["ev-2"],
        }),
      ],
      processing,
      now: NOW,
    });

    expect(findings.find((item) => item.kind === "duplicate")).toBeUndefined();
  });

  it("deleted and superseded memories do not take part in duplicate detection", () => {
    const findings = findingsOf({
      memories: [
        memory("a", {
          content: "Wrap everything with axios, do not call fetch directly",
        }),
        memory("b", {
          content: "Wrap everything with axios, no direct fetch calls",
          status: "deleted",
        }),
      ],
      processing,
      now: NOW,
    });

    expect(findings.find((item) => item.kind === "duplicate")).toBeUndefined();
  });

  it("memories already deleted or superseded are not paired", () => {
    // Positive control: the same two entries, same evidence, reported while both are live.
    const same = { evidence_ids: ["ev-1"] };
    const live = findingsOf({
      memories: [memory("a", same), memory("b", same)],
      processing,
      now: NOW,
    });
    expect(live.find((item) => item.kind === "duplicate")).toBeDefined();

    // Flip to negative: once one of them has been processed away, stop bothering the user with it.
    const gone = findingsOf({
      memories: [
        memory("a", same),
        memory("b", { ...same, status: "deleted" }),
      ],
      processing,
      now: NOW,
    });
    expect(gone.find((item) => item.kind === "duplicate")).toBeUndefined();

    const refused = findingsOf({
      memories: [
        memory("a", same),
        memory("b", { ...same, status: "superseded" }),
      ],
      processing,
      now: NOW,
    });
    expect(refused.find((item) => item.kind === "duplicate")).toBeUndefined();
  });

  it("reports at most two duplicate pairs so other findings are not buried", () => {
    const same = { evidence_ids: ["ev-1"] };
    const findings = findingsOf({
      memories: [
        memory("a", same),
        memory("b", same),
        memory("c", same),
        memory("d", same),
      ],
      processing,
      now: NOW,
    });

    expect(findings.filter((item) => item.kind === "duplicate")).toHaveLength(
      2,
    );
  });

  it("reports dormancy only for three or more untouched memories, not two", () => {
    const stale = { updated_at: "2026-06-01T00:00:00Z" };
    const two = findingsOf({
      memories: [
        memory("a", { ...stale, content: "A" }),
        memory("b", { ...stale, content: "B" }),
      ],
      processing,
      now: NOW,
    });
    expect(two.find((item) => item.kind === "dormant")).toBeUndefined();

    const three = findingsOf({
      memories: [
        memory("a", { ...stale, content: "A" }),
        memory("b", { ...stale, content: "B" }),
        memory("c", { ...stale, content: "C" }),
      ],
      processing,
      now: NOW,
    });
    expect(three.find((item) => item.kind === "dormant")?.text).toContain(
      "3 memories",
    );
  });

  it("reports the real count when the pipeline is stuck instead of a vague in-progress line", () => {
    const findings = findingsOf({
      memories: [],
      processing: {
        ...processing,
        blocked_stages: [
          { kind: "extract", reason: "model_profile_not_active", count: 7 },
        ],
      } as MemoryProcessingSummary,
      now: NOW,
    });

    expect(findings.find((item) => item.kind === "blocked")?.text).toContain(
      "7 conversation segments",
    );
  });

  it("does not invent findings when there is nothing", () => {
    expect(findingsOf({ memories: [], processing, now: NOW })).toEqual([]);
  });
});

describe("the matter axis in the left column", () => {
  it("lists topics, not the authorization scope hash", () => {
    // In a real library scopes look like this: workspace:v1:<sha256>. mattersOf used to group
    // by scope and take the label after the colon, so "matters" listed 9 rows of v1:0027abad... --
    // that is permissions, not matters, and nobody can read it.
    const matters = mattersOf(
      [
        memory("a", {
          scopes: ["workspace:v1:0027abad1745330f1b57671635e0e9b3"],
          tags: ["axios", "frontend"],
        }),
        memory("b", {
          scopes: ["workspace:v1:daf248ff48ea3e6baf35cc7f1fb46d74"],
          tags: ["axios", "testing"],
        }),
      ],
      [source("cursor")],
    );

    const labels = matters.map((item) => item.label);
    expect(labels).toContain("All");
    expect(labels).toContain("axios");
    // Not a single hash may appear in front of the user.
    expect(labels.filter((label) => label.includes("v1:"))).toEqual([]);
    expect(labels.filter((label) => label.startsWith("workspace"))).toEqual([]);
  });

  it("a topic used by a single memory does not get its own row", () => {
    // Positive control: without this, the previous test could be fooled by "one row per tag" --
    // axios would still be there, only beside frontend and testing rows carrying one memory each,
    // turning the column back into one row per memory, which is no categorization at all.
    const matters = mattersOf(
      [
        memory("a", { scopes: [], tags: ["axios", "frontend"] }),
        memory("b", { scopes: [], tags: ["axios", "testing"] }),
      ],
      [],
    );

    const labels = matters.map((item) => item.label);
    expect(labels).toContain("axios");
    expect(labels).not.toContain("frontend");
    expect(labels).not.toContain("testing");
  });

  it("picking a topic filters the memories carrying that tag", () => {
    const memories = [
      memory("a", { scopes: [], tags: ["axios"] }),
      memory("b", { scopes: [], tags: ["testing"] }),
    ];

    const picked = memoriesInMatter(memories, "topic:axios");

    expect(picked.map((item) => item.memory_id)).toEqual(["a"]);
  });

  it("topics with more entries sort first and All is always first", () => {
    const matters = mattersOf(
      [
        memory("a", { scopes: [], tags: ["Few", "Many"] }),
        memory("b", { scopes: [], tags: ["Many"] }),
        memory("c", { scopes: [], tags: ["Many"] }),
        memory("d", { scopes: [], tags: ["Few"] }),
      ],
      [],
    );

    expect(matters.map((item) => item.label)).toEqual(["All", "Many", "Few"]);
  });

  it("with no tagged memories the column keeps All instead of going blank", () => {
    const matters = mattersOf([memory("a", { scopes: [], tags: [] })], []);

    expect(matters.map((item) => item.label)).toEqual(["All"]);
    expect(matters[0].memoryCount).toBe(1);
  });
});
