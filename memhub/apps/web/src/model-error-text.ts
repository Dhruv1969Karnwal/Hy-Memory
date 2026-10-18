/** Backend error codes → one sentence you can act on.
 *
 * This layer did not exist before: `use-model-profiles` returned the raw code straight to the UI, so after pasting the key
 * and clicking "Set as current model", the screen showed `model_profile_probe_required`.
 * The codes are for our debugging, not for the person reading the screen.
 *
 * Unknown codes are passed through as-is, not replaced by a generic "something went wrong": that would keep the screen free of
 * underscores, but it would also erase the debugging trail, leaving the next person to reproduce it once more.
 */
const TEXT: Record<string, string> = {
  model_profile_probe_required:
    'This card has not been tested yet. Clicking "Set as current model" runs the test first; if it still fails, the endpoint, key, or model name is wrong somewhere.',
  model_provider_not_supported:
    "This kind of model endpoint cannot be used right now; switch to another card.",
  embedding_model_required:
    "This card is missing an embedding model; add one to enable it.",
  model_profile_not_found:
    "This card no longer exists, it may have just been deleted. Refresh.",
  model_profile_conflict:
    "This card was just modified. Refresh and try again so you do not overwrite changes made elsewhere.",
};

export function modelErrorText(code: string): string {
  return TEXT[code] ?? `Operation failed: ${code}`;
}

export function memoryWriteErrorText(
  code: string | null | undefined,
): string | null {
  if (code?.startsWith("model_content_")) {
    return "The original text is saved on this device; the current content, model recipient, or authorization version has not passed verification, so organizing did not continue.";
  }
  if (code === "hy_memory_import_no_readable_output") {
    return "HY returned no verifiable existing memory, and no substitute content was created; the original record is kept and processing can be retried.";
  }
  if (code === "hy_memory_write_verification_historical_support_only") {
    return "History references are not yet bound to their original sources; the current batch is incomplete, and the original text is kept.";
  }
  if (code?.startsWith("hy_memory_write_verification_")) {
    return "The memory content did not pass source and author verification; the current batch is incomplete, and the original text is kept.";
  }
  return null;
}
