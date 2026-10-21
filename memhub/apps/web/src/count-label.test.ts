import { describe, expect, it, vi } from "vitest";

import { countLabel, groupedCountLabel, groupedNumber } from "./count-label";

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

  it("groups a bare count the way it groups a counted noun", () => {
    expect(groupedNumber(1)).toBe("1");
    expect(groupedNumber(1_234)).toBe("1,234");
    expect(groupedNumber(81_614)).toBe("81,614");
    expect(groupedCountLabel(81_614, "conversation")).toBe(
      `${groupedNumber(81_614)} conversations`,
    );
  });

  it("groups the same way wherever the app is run", () => {
    // The locale is an argument, not an accident: without it the same build prints
    // "1.234.567,5" under de_DE, "1 234 567,5" under fr_FR and "12,34,567.5" under
    // hi_IN - a different grouping, not just a different separator. The digits are
    // pinned for the reader, and the argument is asserted so this still bites on a
    // runner whose default locale happens to agree with the expected output.
    const locale = vi.spyOn(Number.prototype, "toLocaleString");
    try {
      expect(groupedNumber(1_234_567)).toBe("1,234,567");
      expect(groupedCountLabel(1_234_567, "conversation")).toBe(
        "1,234,567 conversations",
      );
      expect(locale).toHaveBeenCalledWith("en-US");
      expect(countLabel(1, "conversation")).toBe("1 conversation");
      expect(countLabel(17, "conversation")).toBe("17 conversations");
    } finally {
      locale.mockRestore();
    }
  });
});
