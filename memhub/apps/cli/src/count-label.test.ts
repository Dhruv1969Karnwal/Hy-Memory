import { describe, expect, it } from "vitest";

import { countLabel, nounLabel } from "./count-label.js";

describe("count label", () => {
  it("drops the plural ending when the count is one", () => {
    expect(countLabel(1, "app")).toBe("1 app");
    expect(countLabel(0, "app")).toBe("0 apps");
    expect(countLabel(3, "app")).toBe("3 apps");
  });

  it("inflects the nouns this CLI counts", () => {
    expect(countLabel(2, "character")).toBe("2 characters");
    expect(countLabel(2, "source")).toBe("2 sources");
    expect(countLabel(1, "second")).toBe("1 second");
  });

  it("prints the count the way the counters it replaced did", () => {
    expect(countLabel(12_480, "message")).toBe("12480 messages");
  });

  it("agrees with a list that stands in for the count", () => {
    expect(nounLabel(1, "issue")).toBe("issue");
    expect(nounLabel(2, "issue")).toBe("issues");
  });
});
