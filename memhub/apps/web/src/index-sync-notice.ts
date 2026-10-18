import type { MemoryIndexSync } from "@one-memory/contracts";

export type IndexSyncAction = "retry" | "configure" | "refresh" | "none";

export interface IndexSyncPresentation {
  attemptedLine: string;
  explanation: string;
  action: IndexSyncAction;
}

/** What a pending index sync can honestly offer, from the fields Core sends.
 *
 * `attempts === 0` means Core has not tried yet. A `last_error_code` is the
 * only evidence that the last attempt failed; an unknown code gets a neutral
 * retry plus refresh path rather than being mistaken for success. */
export function indexSyncPresentation(
  sync: MemoryIndexSync,
  updatedAt: string,
): IndexSyncPresentation {
  const attempted = sync.attempts > 0;
  const attemptedLine = `${attempted ? `Sync attempted ${sync.attempts} times` : "Sync has not been attempted"} - Status updated at ${updatedAt}`;
  const reason = sync.last_error_code;

  if (reason === "hy_memory_update_rejected") {
    return {
      attemptedLine,
      explanation:
        "HY Memory does not have this memory right now; retrying the sync alone cannot fill it in.",
      action: "none",
    };
  }
  if (
    reason === "model_profile_not_active" ||
    reason === "model_api_key_missing" ||
    reason === "model_secret_unavailable" ||
    reason === "correction_embedding_consent_required"
  ) {
    return {
      attemptedLine,
      explanation:
        "Sync needs an available on-device organization model or local embedding authorization.",
      action: "configure",
    };
  }
  if (
    reason === "context_changed" ||
    reason === "memory_revision_conflict" ||
    reason === "model_content_profile_changed"
  ) {
    return {
      attemptedLine,
      explanation:
        "This memory or the model version has changed, so refresh and verify before the next step.",
      action: "refresh",
    };
  }
  if (!attempted) {
    return {
      attemptedLine,
      explanation:
        "This memory changed and the search index has not started syncing yet.",
      action: "retry",
    };
  }
  if (reason) {
    return {
      attemptedLine,
      explanation:
        "The last sync did not succeed; retry first, then refresh to verify if the result is still unknown.",
      action: "retry",
    };
  }
  return {
    attemptedLine,
    explanation: "This memory changed and the search index has not caught up.",
    action: "retry",
  };
}
