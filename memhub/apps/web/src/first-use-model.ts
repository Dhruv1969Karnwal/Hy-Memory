import type { MemoryDetail, SourceResponse } from "@one-memory/contracts";
import type { RecallHostStatus } from "./global";
import { promptFrom } from "./HandoffBar";

export type GuideScenario = "same" | "switch" | "new";
export type GuideRoute = "context" | "query";

export const GUIDE_SCENARIOS = [
  {
    id: "same",
    label: "Continue in a new chat",
    description:
      "Same app, a different conversation, and your earlier decisions are still found.",
  },
  {
    id: "switch",
    label: "Continue in another app",
    description:
      "Same work, just a different AI. Connect the relevant background first.",
  },
  {
    id: "new",
    label: "Let a new tool get to know me",
    description:
      "Start with one piece of background you are willing to share; no self-introduction from scratch.",
  },
] as const;

export function guideDetailAvailable(
  detail: MemoryDetail,
  sources: SourceResponse[],
): boolean {
  const memory = detail.memory;
  if (
    memory.status !== "confirmed" ||
    memory.deleted_at ||
    !memory.evidence_ids.length
  )
    return false;
  return memory.evidence_ids.every((id) => {
    const evidence = detail.evidence.find((item) => item.evidence_id === id);
    if (!evidence?.available || !evidence.verbatim || !evidence.source_id)
      return false;
    const source = sources.find(
      (item) => item.source_id === evidence.source_id,
    );
    return !source || source.authorized;
  });
}

export function sameGuideMemory(
  previous: MemoryDetail,
  next: MemoryDetail,
): boolean {
  const a = previous.memory,
    b = next.memory;
  return (
    a.memory_id === b.memory_id &&
    a.revision === b.revision &&
    a.status === b.status &&
    a.content === b.content &&
    a.deleted_at === b.deleted_at &&
    JSON.stringify(a.scopes) === JSON.stringify(b.scopes) &&
    JSON.stringify(a.evidence_ids) === JSON.stringify(b.evidence_ids) &&
    JSON.stringify(previous.evidence) === JSON.stringify(next.evidence)
  );
}

export function guideHosts(
  scene: GuideScenario,
  detail: MemoryDetail | null,
  hosts: RecallHostStatus[],
): RecallHostStatus[] {
  const sourceIds = new Set(
    detail?.evidence.map((evidence) => evidence.source_id),
  );
  return hosts.filter(
    (host) =>
      host.present &&
      (scene === "new" ||
        (scene === "same"
          ? sourceIds.has(host.hostId)
          : !sourceIds.has(host.hostId))),
  );
}

export function draftForScenario(
  scene: GuideScenario,
  route: GuideRoute,
  topic: string,
  detail: MemoryDetail,
  sourceLabel: string,
): string {
  if (route === "context") {
    const introduction =
      scene === "new"
        ? "This is background I chose to share; it is not a complete personal profile; do not generalize project constraints into long-term personal preferences."
        : scene === "switch"
          ? `I want to continue the work discussed in ${sourceLabel}.`
          : "I am continuing the earlier work in a new conversation.";
    return `${introduction}\n\n${promptFrom([detail.memory], topic.trim() || "this work")}\n\nApplicable scopes (not widened): ${detail.memory.scopes.join(", ")}\nPlease verify the applicable scopes against the current task; do not act on this background alone.`;
  }
  const subject = topic.trim() || "the work discussed earlier";
  const opening =
    scene === "same"
      ? "This is a new conversation."
      : scene === "switch"
        ? `I want to continue the work discussed in ${sourceLabel}.`
        : "This is my first time starting this work here.";
  return `${opening}Please first use MemHub to query existing decisions, constraints and open questions related to "${subject}", list the sources and citations actually returned, then state the background as you understand it. Distinguish project requirements from personal preferences and do not infer a complete personal profile. If nothing is found or the connection fails, say so plainly: do not guess, do not widen authorization, and do not start new tasks yet.`;
}
