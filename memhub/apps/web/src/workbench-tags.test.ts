import type { GovernedMemory } from "@one-memory/contracts";
import { describe, expect, it } from "vitest";

import { singletonTags, tagCounts } from "./workbench-tags";

function memory(
  id: string,
  tags: string[],
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
    tags,
    revision: 1,
    created_at: "2026-09-01T00:00:00Z",
    updated_at: "2026-09-01T00:00:00Z",
    deleted_at: null,
    ...overrides,
  };
}

describe("topic tag usage", () => {
  it("sorts by most used, and breaks ties by name", () => {
    const counted = tagCounts([
      memory("a", ["scheduling", "work habits"]),
      memory("b", ["scheduling"]),
      memory("c", ["frontend"]),
    ]);

    expect(counted).toEqual([
      ["scheduling", 2],
      ["frontend", 1],
      ["work habits", 1],
    ]);
  });

  it("deleted and superseded memories do not inflate tag counts", () => {
    const counted = tagCounts([
      memory("a", ["scheduling"]),
      memory("b", ["scheduling"], { status: "deleted" }),
      memory("c", ["scheduling"], { status: "superseded" }),
      memory("d", ["frontend"]),
    ]);

    expect(counted).toEqual([
      ["frontend", 1],
      ["scheduling", 1],
    ]);
  });

  it("picks out tags used only once; this is a measure, not a verdict", () => {
    expect(
      singletonTags([
        memory("a", ["scheduling", "work habits"]),
        memory("b", ["scheduling"]),
      ]),
    ).toEqual(["work habits"]);
  });

  it("a library with no tags at all reports nothing", () => {
    expect(tagCounts([memory("a", [])])).toEqual([]);
    expect(singletonTags([])).toEqual([]);
  });
});
