import { describe, expect, it, vi } from "vitest";
import { processMcpLine } from "./mcp-protocol.js";
import { McpSession } from "./mcp-session.js";

const FOUND = {
  hits: [
    {
      evidence_id: "ev-1",
      conversation_id: "conv-1",
      source_id: "cursor",
      workspace: null,
      project: null,
      created_at: "2026-08-20T10:00:00Z",
      excerpt:
        "We agreed to do the import first for this task and not block on it",
    },
  ],
  empty_reason: null,
  imported_messages: 4210,
};

function session(search = vi.fn().mockResolvedValue(FOUND)) {
  return {
    search,
    instance: new McpSession(
      "/fixture/discovery.json",
      undefined,
      undefined,
      undefined,
      search,
    ),
  };
}

async function call(
  instance: McpSession,
  args: unknown,
  name = "om_conversations",
) {
  return JSON.parse(
    (await processMcpLine(
      instance,
      JSON.stringify({
        jsonrpc: "2.0",
        id: 1,
        method: "tools/call",
        params: { name, arguments: args },
      }),
    )) ?? "",
  );
}

describe("om_conversations", () => {
  it("returns the original conversation quote with the app it came from and when", async () => {
    const { search, instance } = session();

    const response = await call(instance, {
      query: "do the import first",
      limit: 5,
    });

    expect(search).toHaveBeenCalledWith("/fixture/discovery.json", {
      query: "do the import first",
      limit: 5,
    });
    const hit = response.result.structuredContent.hits[0];
    expect(hit.source_id).toBe("cursor");
    expect(hit.created_at).toBe("2026-08-20T10:00:00Z");
    expect(hit.excerpt).toContain("do the import first");
  });

  it("the tool description spells out how it divides work with om_recall", async () => {
    const { instance } = session();

    const listed = JSON.parse(
      (await processMcpLine(
        instance,
        JSON.stringify({ jsonrpc: "2.0", id: 2, method: "tools/list" }),
      )) ?? "",
    );
    const tool = listed.result.tools.find(
      (item: { name: string }) => item.name === "MemHub_conversations",
    );

    // The host's model judges whether to call it from this text alone. Wording that
    // is too vague means (do you remember that task I discussed in Cursor) won't trigger it.
    expect(tool.description).toContain("another app");
    expect(tool.description).toContain("MemHub_recall");
    expect(tool.annotations.readOnlyHint).toBe(true);
  });

  it("read-only, and it rejects parameters that carry write intent", async () => {
    const { search, instance } = session();

    const response = await call(instance, {
      query: "x",
      scope: "*",
    });

    expect(response.error.message).toContain("unknown arguments");
    expect(search).not.toHaveBeenCalled();
  });

  it("an out-of-range limit is blocked and never reaches Core", async () => {
    const { search, instance } = session();

    const response = await call(instance, { query: "x", limit: 500 });

    expect(response.error.message).toContain("1 to 50");
    expect(search).not.toHaveBeenCalled();
  });

  it("an in-range limit is passed through", async () => {
    // The positive counterpart to the case above; otherwise (never call the tool) would pass too.
    const { search, instance } = session();

    await call(instance, { query: "x", limit: 50 });

    expect(search).toHaveBeenCalledWith("/fixture/discovery.json", {
      query: "x",
      limit: 50,
    });
  });

  it("it says so when MemHub isn't running instead of returning an empty result", async () => {
    const { instance } = session(
      vi.fn().mockRejectedValue(new Error("MemHub is not running")),
    );

    const response = await call(instance, { query: "x" });

    expect(response.result.isError).toBe(true);
    expect(response.result.content[0].text).toContain("not running");
  });
});
