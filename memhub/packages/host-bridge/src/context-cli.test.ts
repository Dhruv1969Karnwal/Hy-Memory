import { afterEach, describe, expect, it, vi } from "vitest";
import { BrokerClientError } from "./broker-client.js";
import { runContextCli } from "./context-cli.js";

const { contextThroughBroker } = vi.hoisted(() => ({
  contextThroughBroker: vi.fn(),
}));

vi.mock("./broker-client.js", async () => {
  const actual =
    await vi.importActual<typeof import("./broker-client.js")>(
      "./broker-client.js",
    );
  return { ...actual, contextThroughBroker };
});

afterEach(() => {
  contextThroughBroker.mockReset();
});

const lane = {
  status: "empty",
  reason: null,
  elapsed_ms: 0,
  returned_count: 0,
  truncated: false,
  truncated_by: [],
};

const empty = {
  schema_version: 1,
  request_id: "fixture-request",
  state: "empty" as const,
  paths: { raw: lane, memory: lane },
  items: [] as const,
  used_characters: 0,
  truncated: false,
  total_ms: 1,
  adoption: "unknown" as const,
};

const raw = {
  kind: "raw" as const,
  source_id: "codex",
  conversation_id: "fixture-conversation",
  anchor_evidence_id: "e1",
  messages: [
    {
      evidence_id: "e1",
      revision: 1,
      role: "user" as const,
      written_by: "person",
      created_at: null,
      text: "original",
      truncated: false,
    },
  ],
  truncated: false,
};

const env = {
  ONE_MEMORY_BROKER_DISCOVERY: "/fixture/discovery.json",
  ONE_MEMORY_HOST_CREDENTIAL_FILE: "/fixture/credential",
};

describe("om_context CLI receipts", () => {
  it("announces a complete hit without treating it as adoption", async () => {
    contextThroughBroker.mockResolvedValue({
      ...empty,
      state: "grounded",
      paths: {
        raw: { ...lane, status: "ok", returned_count: 1 },
        memory: lane,
      },
      items: [raw],
      used_characters: 8,
    });

    const result = await runContextCli(["original"], env);

    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain(
      "Successfully recalled relevant memories from MemHub.",
    );
    expect(result.stdout).not.toContain(
      "Partially relevant memories were fetched from MemHub",
    );
    expect(result.stdout).toContain("e1");
    expect(result.stdout).toContain("adoption: unknown");
  });

  it("keeps an ordinary empty distinct from success, partial, and outage", async () => {
    contextThroughBroker.mockResolvedValue(empty);

    const result = await runContextCli(["original"], env);

    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("MemHub memory search completed.");
    expect(result.stdout).not.toContain(
      "Successfully recalled relevant memories",
    );
    expect(result.stdout).not.toContain(
      "Partially relevant memories were fetched from MemHub",
    );
    expect(result.stdout).not.toContain("temporarily unavailable");
    expect(result.stdout).not.toContain("limited coverage");
  });

  it("uses the partial-memory receipt only when items were returned", async () => {
    contextThroughBroker.mockResolvedValue({
      ...empty,
      state: "partial",
      paths: {
        raw: {
          ...lane,
          status: "ok",
          returned_count: 1,
          truncated: true,
          truncated_by: ["character_budget"],
        },
        memory: lane,
      },
      items: [raw],
      truncated: true,
      used_characters: 8,
    });

    const result = await runContextCli(["original"], env);

    expect(result.stdout).toContain(
      "Partially relevant memories were fetched from MemHub; the following continues with this information.",
    );
    expect(result.stdout).not.toContain(
      "Successfully recalled relevant memories",
    );
    expect(result.stdout).toContain("e1");
  });

  it("explains a semantic timeout", async () => {
    contextThroughBroker.mockResolvedValue({
      ...empty,
      state: "partial",
      paths: {
        raw: { ...lane, status: "ok", returned_count: 1 },
        memory: {
          ...lane,
          status: "degraded",
          reason: "timeout_budget",
        },
      },
      items: [raw],
    });

    const result = await runContextCli(["original"], env);

    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain(
      "The semantic memory query timed out this time, so what was available is returned first; it recovers automatically later.",
    );
  });

  it("does not call a zero-hit partial empty or successful", async () => {
    contextThroughBroker.mockResolvedValue({
      ...empty,
      state: "partial",
      paths: {
        raw: { ...lane, status: "degraded", reason: "timeout_budget" },
        memory: {
          ...lane,
          status: "not_authorized",
          reason: "permission_required",
        },
      },
    });

    const result = await runContextCli(["original"], env);

    expect(result.stdout).toContain("MemHub memory search completed.");
    expect(result.stdout).not.toContain(
      "Partially relevant memories were fetched from MemHub",
    );
    expect(result.stdout).not.toContain(
      "Successfully recalled relevant memories",
    );
    expect(result.stdout).not.toContain(
      "Successfully recalled relevant memories from MemHub.",
    );
  });

  it("says unavailable for a returned outage, not a permission denial", async () => {
    contextThroughBroker.mockResolvedValue({
      ...empty,
      state: "unavailable",
    });

    const result = await runContextCli(["original"], env);

    expect(result.exitCode).toBe(4);
    expect(result.stdout).toContain(
      "MemHub is temporarily unavailable; I will continue based on the current conversation.",
    );
    expect(result.stdout).not.toContain(
      "outside the current authorization scope",
    );
    expect(result.stdout).not.toContain(
      "Successfully recalled relevant memories",
    );
  });

  it("uses the denial receipt for host_scope_denied and does not invite a bypass", async () => {
    contextThroughBroker.mockRejectedValue(
      new BrokerClientError("host_scope_denied"),
    );

    const result = await runContextCli(["original"], env);

    expect(result.exitCode).toBe(4);
    expect(result.stdout).toContain(
      "This memory search is outside the current authorization scope; I will continue with the information available.",
    );
    expect(result.stdout).toContain("Do not retry through other tools");
    expect(result.stdout).not.toContain("temporarily unavailable");
    expect(result.stdout).not.toContain(
      "Successfully recalled relevant memories",
    );
  });

  it("does not describe an upgrade requirement as a permission denial", async () => {
    contextThroughBroker.mockRejectedValue(
      new BrokerClientError("context_upgrade_required"),
    );

    const result = await runContextCli(["original"], env);

    expect(result.exitCode).toBe(5);
    expect(result.stdout).toContain(
      "MemHub is temporarily unavailable; I will continue based on the current conversation.",
    );
    expect(result.stdout).toContain("Do not retry through other tools");
    expect(result.stdout).not.toContain(
      "outside the current authorization scope",
    );
  });

  it("leaves --json as the structured envelope, without rewriting receipts", async () => {
    const payload = {
      ...empty,
      state: "grounded",
      items: [raw],
      used_characters: 8,
    };
    contextThroughBroker.mockResolvedValue(payload);

    const result = await runContextCli(["original", "--json"], env);

    expect(JSON.parse(result.stdout)).toEqual(payload);
    expect(result.stdout).not.toContain(
      "Successfully recalled relevant memories",
    );
  });
});
