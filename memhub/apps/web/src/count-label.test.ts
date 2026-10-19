import { describe, expect, it } from "vitest";

import { countLabel, groupedCountLabel } from "./count-label";

describe("count label", () => {
  it("drops the plural ending when the count is one", () => {
    expect(countLabel(1, "app")).toBe("1 app");
    expect(countLabel(1, "memory")).toBe("1 memory");
    expect(countLabel(0, "app")).toBe("0 apps");
    expect(countLabel(3, "app")).toBe("3 apps");
  });

  it("inflects the nouns this UI counts", () => {
    expect(countLabel(2, "memory")).toBe("2 memories");
    expect(countLabel(2, "entry")).toBe("2 entries");
    expect(countLabel(2, "batch")).toBe("2 batches");
    expect(countLabel(2, "issue")).toBe("2 issues");
    expect(countLabel(2, "character")).toBe("2 characters");
  });

  it("takes an explicit plural where the rule would guess wrong", () => {
    expect(countLabel(2, "person", "people")).toBe("2 people");
    expect(countLabel(1, "person", "people")).toBe("1 person");
  });

  it("keeps a plain count plain", () => {
    expect(countLabel(1_234, "original")).toBe("1234 originals");
    expect(countLabel(1, "complete turn")).toBe("1 complete turn");
  });

  it("groups the counts the counters grouped", () => {
    expect(groupedCountLabel(1_234, "original")).toBe("1,234 originals");
    expect(groupedCountLabel(1, "message")).toBe("1 message");
    expect(groupedCountLabel(81_614, "conversation")).toBe(
      "81,614 conversations",
    );
  });
});
