import { mkdtemp, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import { runRecallCli } from "./recall-cli.js";

const { recallThroughBroker } = vi.hoisted(() => ({
  recallThroughBroker: vi.fn(),
}));

vi.mock("./broker-client.js", () => ({ recallThroughBroker }));

afterEach(() => {
  recallThroughBroker.mockReset();
});

async function discovery(status: "running" | "stopped") {
  const root = await mkdtemp(path.join(os.tmpdir(), "recall-cli-"));
  const file = path.join(root, "broker.json");
  await writeFile(
    file,
    JSON.stringify({
      schema_version: 1,
      service: "one-memory-recall-broker",
      status,
      host: "127.0.0.1",
      port: status === "running" ? 18765 : null,
      instance_id: "instance",
      pid: 1234,
      updated_at: new Date().toISOString(),
    }),
    { encoding: "utf8" },
  );
  return { ONE_MEMORY_BROKER_DISCOVERY: file };
}

const HIT = {
  memory_id: "m1",
  content:
    "The user asked to wrap requests with axios instead of calling fetch directly",
  kind: "chat" as const,
  scopes: ["workspace:v1:abc"],
  evidence_ids: ["e1"],
  updated_at: "2026-08-08T03:16:40.000Z",
  score: 0.8,
};

describe("recall cli", () => {
  it("writes the memories as prose a model can just read", async () => {
    recallThroughBroker.mockResolvedValue({
      state: "grounded",
      query: "code style preference",
      scope: "all-apps",
      hits: [HIT],
      used_characters: 40,
      truncated: false,
    });

    const result = await runRecallCli(
      ["code style preference"],
      await discovery("running"),
      "/Users/someone",
      "/Users/someone/work/api-service",
    );

    expect(result.exitCode).toBe(0);
    // Says which project was included; the receipt's own `scope` is the one
    // that was asked for, which would read "all-apps" over project memories.
    expect(result.stdout).toContain(
      "including the current project api-service",
    );
    expect(result.stdout).toContain(
      "1. The user asked to wrap requests with axios instead of calling fetch directly",
    );
    // Recalled memory arrives in the same channel as instructions, so the text
    // states the boundary rather than leaving the model to assume it.
    expect(result.stdout).toContain(
      "The current user request takes precedence",
    );
  });

  it("does not append governance updated_at to SDK-formatted memory", async () => {
    const formatted =
      "Memory: The user asked to wrap requests with axios instead of calling fetch directly\n" +
      "When: 2026-08-08 03:16:40 UTC";
    recallThroughBroker.mockResolvedValue({
      state: "grounded",
      query: "code style preference",
      scope: "all-apps",
      hits: [{ ...HIT, content: formatted, content_format: "hy-sdk" }],
      used_characters: formatted.length,
      truncated: false,
    });

    const result = await runRecallCli(
      ["code style preference"],
      await discovery("running"),
      "/Users/someone",
      "/Users/someone/work/api-service",
    );

    expect(result.stdout).toContain(formatted);
    expect(result.stdout).not.toContain("Updated on 2026-08-08");
  });

  it("takes an unquoted sentence as one query", async () => {
    recallThroughBroker.mockResolvedValue({
      state: "empty",
      query: "",
      scope: "all-apps",
      hits: [],
      used_characters: 0,
      truncated: false,
    });

    await runRecallCli(
      ["my", "preference", "for", "request", "libraries"],
      await discovery("running"),
      "/Users/someone",
      "/Users/someone",
    );

    // A shell-command skill interpolates the user's sentence, so dropping
    // everything after the first word would silently narrow the query.
    expect(recallThroughBroker.mock.calls[0][1]).toEqual({
      query: "my preference for request libraries",
    });
  });

  it("reports no memory as an answer, not as an error", async () => {
    recallThroughBroker.mockResolvedValue({
      state: "empty",
      query: "x",
      scope: "all-apps",
      hits: [],
      used_characters: 0,
      truncated: false,
    });

    const result = await runRecallCli(["x"], await discovery("running"));

    expect(result.exitCode).toBe(0);
    // The product's committed wording, shared with the context path: an empty
    // lookup is a completed lookup. "No related memories" reads to a model as
    // evidence about the user — that they have no preferences — which is
    // exactly what it must not invent.
    expect(result.stdout).toContain("MemHub memory search completed.");
    // Saying so is not enough; a model told only "nothing found" will often
    // fill the gap itself.
    expect(result.stdout).toContain("don't invent");
    // And it must not fall back to the older "found nothing" phrasing.
    expect(result.stdout).not.toContain(
      "There are no confirmed memories related to this question",
    );
  });

  it("treats a closed app as ordinary and never calls the broker", async () => {
    const result = await runRecallCli(["x"], await discovery("stopped"));

    // The app is not always open. A host that abandons the turn on a nonzero
    // exit would turn that into a broken conversation.
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("isn't running");
    expect(recallThroughBroker).not.toHaveBeenCalled();
  });

  it("does fail when the broker claims to be running and is not", async () => {
    recallThroughBroker.mockRejectedValue(new Error("connect ECONNREFUSED"));

    const result = await runRecallCli(["x"], await discovery("running"));

    // This one is a real fault: the discovery file says the app is up.
    expect(result.exitCode).toBe(4);
    expect(result.stdout).toContain("Can't connect");
  });

  it("treats a missing discovery file as a closed app", async () => {
    const result = await runRecallCli(["x"], {
      ONE_MEMORY_BROKER_DISCOVERY: path.join(
        os.tmpdir(),
        "absent",
        "broker.json",
      ),
    });

    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("isn't running");
  });

  it("gives scripts the fields instead of the prose", async () => {
    const receipt = {
      state: "grounded",
      query: "x",
      scope: "all-apps",
      hits: [HIT],
      used_characters: 40,
      truncated: false,
    };
    recallThroughBroker.mockResolvedValue(receipt);

    const result = await runRecallCli(
      ["x", "--json"],
      await discovery("running"),
    );

    expect(JSON.parse(result.stdout)).toEqual(receipt);
  });

  it("passes the options through as the broker expects them", async () => {
    recallThroughBroker.mockResolvedValue({
      state: "empty",
      query: "x",
      scope: "workspace:v1:abc",
      hits: [],
      used_characters: 0,
      truncated: false,
    });

    await runRecallCli(
      [
        "x",
        "--scope",
        "workspace:v1:abc",
        "--limit",
        "3",
        "--character-budget",
        "500",
      ],
      await discovery("running"),
      "/Users/someone",
      "/Users/someone",
    );

    expect(recallThroughBroker.mock.calls[0][1]).toEqual({
      query: "x",
      scope: "workspace:v1:abc",
      limit: 3,
      character_budget: 500,
    });
  });

  it("refuses a query it would otherwise have to invent", async () => {
    const empty = await runRecallCli([], await discovery("running"));
    expect(empty.exitCode).toBe(2);
    expect(empty.stdout).toContain("query is required");

    const bad = await runRecallCli(
      ["x", "--limit", "0"],
      await discovery("running"),
    );
    expect(bad.exitCode).toBe(2);

    const unknown = await runRecallCli(
      ["x", "--wat"],
      await discovery("running"),
    );
    expect(unknown.exitCode).toBe(2);
    expect(unknown.stdout).toContain("Unknown option");

    expect(recallThroughBroker).not.toHaveBeenCalled();
  });

  it("says when it had more to say, without inventing a budget", async () => {
    recallThroughBroker.mockResolvedValue({
      state: "grounded",
      query: "x",
      scope: "all-apps",
      hits: [HIT],
      used_characters: 500,
      truncated: true,
    });

    const result = await runRecallCli(
      ["x"],
      await discovery("running"),
      "/Users/someone",
      "/Users/someone",
    );

    // Core may omit truncated_by on older receipts; do not invent a cause.
    expect(result.stdout).toContain("More related memories aren't listed");
  });

  it("names the budget when Core says which one fired", async () => {
    recallThroughBroker.mockResolvedValue({
      state: "grounded",
      query: "x",
      scope: "all-apps",
      hits: [HIT],
      used_characters: 500,
      truncated: true,
      truncated_by: "character_budget",
      unexamined_count: 3,
    });

    const result = await runRecallCli(
      ["x"],
      await discovery("running"),
      "/Users/someone",
      "/Users/someone",
    );

    expect(result.stdout).toContain("3 related memories weren't examined");
    expect(result.stdout).toContain("character budget");
  });
});

describe("recall cli and the project it is in", () => {
  it("names the shell's directory so project memories are reachable", async () => {
    recallThroughBroker.mockResolvedValue({
      state: "empty",
      query: "x",
      scope: "all-apps",
      hits: [],
      used_characters: 0,
      truncated: false,
    });

    await runRecallCli(
      ["x"],
      await discovery("running"),
      "/Users/someone",
      "/Users/someone/work/api-service",
    );

    // Without this, a workspace-scoped memory can never be recalled: its
    // scope hashes an identity only Core can produce.
    expect(recallThroughBroker.mock.calls[0][1]).toEqual({
      query: "x",
      workspace_path: "/Users/someone/work/api-service",
    });
  });

  it("lets a skill state the directory itself", async () => {
    recallThroughBroker.mockResolvedValue({
      state: "empty",
      query: "x",
      scope: "all-apps",
      hits: [],
      used_characters: 0,
      truncated: false,
    });

    await runRecallCli(
      ["x", "--workspace-path", "/Users/someone/work/billing"],
      await discovery("running"),
      "/Users/someone",
      "/Users/someone/work/api-service",
    );

    expect(recallThroughBroker.mock.calls[0][1]).toMatchObject({
      workspace_path: "/Users/someone/work/billing",
    });
  });

  it("says nothing about a project when run from home", async () => {
    recallThroughBroker.mockResolvedValue({
      state: "empty",
      query: "x",
      scope: "all-apps",
      hits: [],
      used_characters: 0,
      truncated: false,
    });

    await runRecallCli(
      ["x"],
      await discovery("running"),
      "/Users/someone",
      "/Users/someone",
    );

    expect(recallThroughBroker.mock.calls[0][1]).toEqual({ query: "x" });
  });
});
