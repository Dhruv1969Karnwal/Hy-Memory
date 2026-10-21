import type { GovernedMemory } from "@one-memory/contracts";
import { type CoreReader, RESERVED_MEMORY_SEGMENTS } from "./core-client.js";
import { validateMemoryDetail, validateMemoryPage } from "./core-responses.js";
import { counted, countLabel } from "./count-label.js";
import { PRODUCT } from "./version.js";

/** Output budget: a long memory is shortened visibly, never dropped silently. */
const CONTENT_BUDGET = 240;

const MEMORY_ID_SHAPE = /^[A-Za-z0-9][\w.-]{0,127}$/;

/** Reserved segments are rejected here too, before a request is even built. */
export function isMemoryId(value: string): boolean {
  return (
    MEMORY_ID_SHAPE.test(value) &&
    !(RESERVED_MEMORY_SEGMENTS as readonly string[]).includes(value)
  );
}

export interface MemoryQuery {
  query?: string;
  status?: string;
  source?: string;
  offset: number;
  limit: number;
}

export interface MemoryPageDocument {
  schema_version: 1;
  product: string;
  page: Record<string, unknown>;
  memories: Record<string, unknown>[];
}

export interface MemoryDetailDocument {
  schema_version: 1;
  product: string;
  memory: Record<string, unknown>;
  evidence: Record<string, unknown>[];
  history: Record<string, unknown>[];
}

export async function readMemoryPage(
  reader: CoreReader,
  query: MemoryQuery,
): Promise<MemoryPageDocument> {
  const parameters: Record<string, string | number> = {
    offset: query.offset,
    limit: query.limit,
  };
  if (query.query !== undefined) parameters.query = query.query;
  if (query.status !== undefined) parameters.status = query.status;
  if (query.source !== undefined) parameters.source = query.source;
  const outcome = await reader<unknown>("/api/v1/memory-library", parameters);
  if (outcome.state !== "ok") {
    return {
      schema_version: 1,
      product: PRODUCT,
      page: { state: outcome.state, code: outcome.code },
      memories: [],
    };
  }
  const page = validateMemoryPage(outcome.data, query);
  if (page === null) {
    return {
      schema_version: 1,
      product: PRODUCT,
      page: { state: "invalid", code: "response_invalid" },
      memories: [],
    };
  }
  const items = page.items;
  return {
    schema_version: 1,
    product: PRODUCT,
    page: {
      state: "ok",
      total: page.total,
      offset: page.offset,
      limit: page.limit,
      returned: items.length,
      has_more: page.offset + items.length < page.total,
    },
    memories: items.map((item) => describeMemory(item)),
  };
}

export async function readMemoryDetail(
  reader: CoreReader,
  memoryId: string,
): Promise<MemoryDetailDocument> {
  const outcome = await reader<unknown>(`/api/v1/memories/${memoryId}`);
  if (outcome.state !== "ok") {
    return {
      schema_version: 1,
      product: PRODUCT,
      memory: { state: outcome.state, code: outcome.code },
      evidence: [],
      history: [],
    };
  }
  const detail = validateMemoryDetail(outcome.data, memoryId);
  if (detail === null) {
    return {
      schema_version: 1,
      product: PRODUCT,
      memory: { state: "invalid", code: "response_invalid" },
      evidence: [],
      history: [],
    };
  }
  return {
    schema_version: 1,
    product: PRODUCT,
    memory: { state: "ok", ...describeMemory(detail.memory) },
    evidence: (detail.evidence ?? []).map((item) => {
      const budgeted = budget(item.verbatim ?? "");
      return {
        evidence_id: item.evidence_id,
        available: item.available,
        source_id: item.source_id,
        written_by: item.written_by,
        created_at: item.created_at,
        excerpt: budgeted.text,
        excerpt_truncated: budgeted.truncated,
        excerpt_characters: budgeted.characters,
      };
    }),
    history: (detail.history ?? []).map((item) => {
      const content = budget(item.content);
      return {
        revision: item.revision,
        action: item.action,
        status: item.status,
        scopes: item.scopes,
        created_at: item.created_at,
        content: content.text,
        content_truncated: content.truncated,
        content_characters: content.characters,
      };
    }),
  };
}

export function renderMemoryPageText(document: MemoryPageDocument): string {
  const page = document.page;
  if (page.state !== "ok") {
    if (page.state === "unavailable") {
      return "The backend is busy, so the memory list did not finish this time. That does not mean there are no memories. Try again later.\n";
    }
    return `Memory query incomplete: ${page.state} (${page.code})\n`;
  }
  const lines = [
    `Total ${page.total}, this page ${page.returned} (offset=${page.offset} limit=${page.limit})${
      page.has_more ? ", more remain, continue with --offset" : ""
    }`,
  ];
  for (const memory of document.memories) {
    lines.push(
      `- ${memory.memory_id}  ${memory.status}/${memory.kind}  rev ${memory.revision}  scopes=${
        (memory.scopes as string[]).join(",") || "none"
      }`,
      `  ${memory.content}${memory.content_truncated ? ` ... (${countLabel(counted(memory.content_characters), "character")} total)` : ""}`,
    );
  }
  if (document.memories.length === 0)
    lines.push("(no memories match the filters)");
  lines.push("");
  return lines.join("\n");
}

export function renderMemoryDetailText(document: MemoryDetailDocument): string {
  const memory = document.memory;
  if (memory.state !== "ok") {
    if (memory.state === "unavailable") {
      return "The backend is busy, so this memory's detail did not finish this time. That does not mean the memory is missing. Try again later.\n";
    }
    return `Memory read incomplete: ${memory.state} (${memory.code})\n`;
  }
  const lines = [
    `Memory ${memory.memory_id}`,
    `status: ${memory.status}  kind: ${memory.kind}  origin: ${memory.origin}  revision: ${memory.revision}`,
    `scopes: ${(memory.scopes as string[]).join(",") || "none"}`,
    `evidence: ${(memory.evidence_ids as string[]).join(",") || "none"}`,
    `updated: ${memory.updated_at}`,
    "",
    String(memory.content),
    memory.content_truncated
      ? `... (${countLabel(counted(memory.content_characters), "character")} total)`
      : "",
    "",
    "Source excerpt:",
  ];
  for (const item of document.evidence) {
    lines.push(
      `- ${item.evidence_id}  source=${item.source_id ?? "unknown"}  author=${item.written_by ?? "unknown"}  readable=${item.available}`,
      `  ${item.excerpt}${item.excerpt_truncated ? ` ... (${countLabel(counted(item.excerpt_characters), "character")} total)` : ""}`,
    );
  }
  if (document.evidence.length === 0) lines.push("(no readable source text)");
  lines.push("", "Revision history:");
  for (const item of document.history) {
    lines.push(
      `- rev ${item.revision}  ${item.action}  ${item.status}  scopes=${(item.scopes as string[]).join(",")}`,
      `  ${item.content}${item.content_truncated ? ` ... (${countLabel(counted(item.content_characters), "character")} total)` : ""}`,
    );
  }
  if (document.history.length === 0) lines.push("(none)");
  lines.push("");
  return lines.join("\n");
}

function describeMemory(memory: GovernedMemory): Record<string, unknown> {
  const budgeted = budget(memory.content ?? "");
  return {
    memory_id: memory.memory_id,
    status: memory.status,
    kind: memory.kind,
    origin: memory.origin,
    revision: memory.revision,
    scopes: memory.scopes ?? [],
    evidence_ids: memory.evidence_ids ?? [],
    tags: memory.tags ?? [],
    created_at: memory.created_at,
    updated_at: memory.updated_at,
    content: budgeted.text,
    content_truncated: budgeted.truncated,
    content_characters: budgeted.characters,
  };
}

function budget(value: string): {
  text: string;
  truncated: boolean;
  characters: number;
} {
  const points = [...value];
  return points.length <= CONTENT_BUDGET
    ? { text: value, truncated: false, characters: points.length }
    : {
        text: points.slice(0, CONTENT_BUDGET).join(""),
        truncated: true,
        characters: points.length,
      };
}
