/** Chinese wording for the memory pipeline's own status codes.
 *
 * Lives on its own because three screens now read the same summary — the
 * memory space, the onboarding completion page, and the health panel — and a
 * second copy of this table is how the same blocked pipeline ends up
 * described two different ways on two screens.
 */

import type { MemoryProcessingBlock } from "@one-memory/contracts";

export function taskKindLabel(kind: string): string {
  const labels: Record<string, string> = {
    extract: "Memory extraction",
    system2: "Relation summary",
    ultra: "Deep organizing",
    sweeper: "Cross-domain organizing",
  };
  return labels[kind] ?? kind;
}

/** What a stall means, in the reader's own language.
 *
 * The fallback is deliberately not the code. `hy_memory_write_verification_
 * clause_incomplete` in a red line told a reader nothing and read as a crash;
 * the code stays on the task row in Core, which is where it is actually
 * useful. Anything this build does not recognise is described by what it does
 * to the reader, not by what it is called. */
export function blockReasonLabel(reason: string): string {
  const known = BLOCK_REASON_LABELS[reason];
  if (known) return known;
  if (reason.startsWith("hy_memory_write_verification_")) return RETRYING;
  return "Organizing stalled for now; it will retry automatically later";
}

const RETRYING = "Failed to organize this round; retrying";

const BLOCK_REASON_LABELS: Record<string, string> = {
  // Waiting on the reader, or on something they can go and check.
  waiting_for_model: "Waiting for an available generation model",
  model_profile_not_active: "No generation model is enabled yet",
  model_api_key_missing: "Generation model API key missing",
  model_secret_unavailable: "Model credentials are temporarily unavailable",
  secure_storage_unavailable:
    "Local credential storage is temporarily unavailable",
  // Keychain failures are not all the same ask. macOS asking for permission is
  // answered by authorizing, not by reconfiguring a model, so each code says
  // the one thing the reader can actually go and do.
  secure_storage_authorization_required:
    "Access to the local keychain needs authorization. Reopen the app and approve once",
  secure_storage_read_timeout:
    "Reading the local keychain timed out. Save the model config again to recover",
  secure_storage_write_timeout:
    "Writing to the local keychain timed out. Save the model config again to recover",
  hy_memory_model_authentication_failed:
    "Model authentication failed. Update the key, then run the check in settings once and processing resumes automatically",
  hy_memory_model_payment_required:
    "Model account balance is low. Top up or swap the key, then run the check in settings once and processing resumes automatically",
  hy_memory_model_access_denied:
    "The model service refused access. Update permissions, then run the check in settings once and processing resumes automatically",
  hy_memory_model_rate_limited: "Model service is rate-limited",
  hy_memory_model_service_unavailable: "Model service temporarily unavailable",
  hy_memory_model_timeout: "Model service response timed out",
  hy_memory_model_request_incompatible:
    "Model request is incompatible; check the configuration",
  hy_memory_model_request_failed: "The model request did not succeed",
  hy_memory_unavailable: "HY Memory is unavailable",
  hy_memory_init_failed: "HY Memory failed to start",
  model_content_profile_changed: "The current model has changed",
  model_content_content_scope_changed: "Local conversation content has changed",
  // Named rather than summed up as a retry: Core never picks this one up
  // again (`_guard_task`), so calling it "retrying" would be a lie.
  hy_memory_input_exceeds_window:
    "This conversation contains oversized source text; not organized yet, source text kept",
  // The pipeline's own retries, kept here so a health panel can say what it
  // is doing without a code.
  failed: RETRYING,
  hy_memory_update_rejected: RETRYING,
  model_profile_superseded:
    "The generation model was swapped; this batch will be organized again",
  ingest_policy_superseded:
    "The organizing policy was swapped; this batch will be organized again",
};

/** The stalls worth naming to the reader.
 *
 * Everything Core reports here is picked up and retried on the next pass —
 * `_guard_task` excludes only `hy_memory_input_exceeds_window`. So a stage
 * that is merely being retried is not news, and listing those by name turned
 * a working library into a red line of internal guard codes. Only a stall the
 * reader can do something about, or one Core will never retry, earns a name.
 */
const NAMED_BLOCK_REASONS = new Set([
  "waiting_for_model",
  "model_profile_not_active",
  "model_api_key_missing",
  "model_secret_unavailable",
  "secure_storage_unavailable",
  "hy_memory_model_authentication_failed",
  "hy_memory_model_payment_required",
  "hy_memory_model_access_denied",
  "hy_memory_model_rate_limited",
  "hy_memory_model_service_unavailable",
  "hy_memory_model_timeout",
  "hy_memory_model_request_incompatible",
  "hy_memory_model_request_failed",
  "hy_memory_unavailable",
  "hy_memory_init_failed",
  "model_content_profile_changed",
  "model_content_content_scope_changed",
  // Never retried, and the one stall whose answer is "nothing to do, the
  // original is still there" — worth saying out loud.
  "hy_memory_input_exceeds_window",
]);

export function worthNamingBlock(reason: string): boolean {
  return NAMED_BLOCK_REASONS.has(reason);
}

/** What a blocked stage actually costs the reader. */
export interface BlockedStages {
  /** Stalls worth naming, in the order Core reported them. */
  named: MemoryProcessingBlock[];
  /** How many conversations the pipeline is retrying on its own. */
  retrying: number;
}

export function splitBlockedStages(
  blocks: readonly MemoryProcessingBlock[],
): BlockedStages {
  const named: MemoryProcessingBlock[] = [];
  let retrying = 0;
  for (const block of blocks) {
    if (block.count <= 0) continue;
    if (worthNamingBlock(block.reason)) named.push(block);
    else retrying += block.count;
  }
  return { named, retrying };
}

/** The one line a reader gets for work the pipeline is handling by itself. */
export function retryingBlockLabel(count: number): string {
  return `${count} items: ${RETRYING}`;
}

/** The reasons that mean "no usable model", as the pipeline reports them.
 *
 * `blocked_stages` carries every stall reason, most of which say nothing
 * about the model. Screens that ask "is a model configured" have to match on
 * this set rather than on the list being non-empty.
 */
const MODEL_BLOCK_REASONS = new Set([
  "waiting_for_model",
  "model_profile_not_active",
  "model_api_key_missing",
  "model_secret_unavailable",
  "secure_storage_unavailable",
  "hy_memory_model_authentication_failed",
  "hy_memory_model_payment_required",
  "hy_memory_model_access_denied",
  "hy_memory_model_request_incompatible",
]);

export function modelBlock(
  blocks: readonly MemoryProcessingBlock[],
): MemoryProcessingBlock | null {
  return (
    blocks.find(
      (block) => block.count > 0 && MODEL_BLOCK_REASONS.has(block.reason),
    ) ?? null
  );
}

/** What went wrong reading a source, said without the code.
 *
 * These are `ScanIssueCode` plus the one access-level refusal, and they land
 * in the import receipts where a reader is deciding whether anything was lost.
 * `invalid_json` told them nothing; "format damaged, this record was not read"
 * tells them what it cost. An unrecognised code still gets a sentence rather
 * than itself, for the same reason the block reasons do. */
export function importIssueLabel(reason: string): string {
  return SCAN_ISSUE_LABELS[reason] ?? "This record could not be read";
}

const SCAN_ISSUE_LABELS: Record<string, string> = {
  invalid_json: "The content is corrupted; this record could not be read",
  record_too_large: "The content is too long for the per-record limit",
  unknown_record: "Unrecognized record type; this record could not be read",
  no_text: "This record has no text to organize",
  tool_content_excluded: "Tool output does not count as source text",
  internal_reasoning_excluded:
    "The model's internal reasoning does not count as source text",
  not_written_by_person: "Not written by the user; not counted",
  source_busy: "The source is busy; it will be re-read automatically later",
  unsupported_schema: "This source's file format is not supported yet",
  budget_exceeded:
    "The per-run processing limit was exceeded; the rest was not read",
  source_permission_changed:
    "The read authorization for this source changed and needs re-verification",
};

export function extractionStateLabel(state: string): string {
  const labels: Record<string, string> = {
    extracted: "Extracted",
    outside_window: "Not extracted · outside the window",
    queued: "Not extracted · queued",
  };
  return labels[state] ?? state;
}

// The history API names the step and nothing else, so a delete carried over
// from an older version's rejection reads as a plain delete.
export function memoryHistoryActionLabel(action: string): string {
  const labels: Record<string, string> = {
    legacy: "Legacy confirmation record",
    create: "Create",
    correct: "Correct",
    set_scopes: "Adjust usage scopes",
    undo: "Undo edit",
    hy_reconcile: "Model update",
    hy_supersede: "Replaced by a newer memory",
    delete: "Delete",
  };
  return labels[action] ?? "Other action";
}

export function extractionWindowPresetLabel(preset: string): string {
  const labels: Record<string, string> = {
    "3m": "Last 3 months",
    "6m": "Last 6 months",
    "1y": "Last year",
    all: "All history",
  };
  return labels[preset] ?? preset;
}
