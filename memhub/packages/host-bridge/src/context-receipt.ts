import type { ContextSearchResponse } from "@one-memory/contracts";
import type { BrokerClientErrorCode } from "./broker-client.js";

/** v9 visible receipts. Structured path dumps stay after this line. */
export const CONTEXT_RECEIPTS = {
  grounded: "Successfully recalled relevant memories from MemHub.",
  empty: "MemHub memory search completed.",
  partialHits:
    "Partially relevant memories were fetched from MemHub; the following continues with this information.",
  partialZero: "MemHub memory search completed.",
  semanticTimeout:
    "The semantic memory query timed out this time, so what was available is returned first; it recovers automatically later.",
  unavailable:
    "MemHub is temporarily unavailable; I will continue based on the current conversation.",
  denied:
    "This memory search is outside the current authorization scope; I will continue with the information available.",
} as const;

const NO_BYPASS = "Do not retry through other tools or expand permissions.";

export function contextResultReceipt(result: ContextSearchResponse): string {
  if (result.state === "unavailable") return CONTEXT_RECEIPTS.unavailable;
  if (result.state === "partial") {
    if (result.paths.memory.reason === "timeout_budget") {
      return CONTEXT_RECEIPTS.semanticTimeout;
    }
    return result.items.length === 0
      ? CONTEXT_RECEIPTS.partialZero
      : CONTEXT_RECEIPTS.partialHits;
  }
  if (result.state === "grounded" && result.items.length > 0) {
    return CONTEXT_RECEIPTS.grounded;
  }
  return CONTEXT_RECEIPTS.empty;
}

export function contextErrorReceipt(
  code: BrokerClientErrorCode | string,
): string {
  const receipt =
    code === "host_scope_denied"
      ? CONTEXT_RECEIPTS.denied
      : CONTEXT_RECEIPTS.unavailable;
  return `${receipt} ${NO_BYPASS}`;
}
