import path from "node:path";
import { describe, expect, it } from "vitest";

import { resolveWebResource } from "./app-protocol";

describe("packaged web protocol", () => {
  it("serves only files inside the packaged web root", () => {
    const root = path.resolve("/bundle/web");

    expect(resolveWebResource(root, "one-memory://app/")).toBe(
      path.join(root, "index.html"),
    );
    expect(resolveWebResource(root, "one-memory://app/assets/app.js")).toBe(
      path.join(root, "assets/app.js"),
    );
    expect(
      resolveWebResource(root, "one-memory://other/index.html"),
    ).toBeNull();
    expect(
      resolveWebResource(root, "one-memory://app/%2F..%2Fsecret"),
    ).toBeNull();
  });
});
