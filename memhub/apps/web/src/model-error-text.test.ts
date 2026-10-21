import { describe, expect, it } from "vitest";

import { modelErrorText } from "./model-error-text";

describe("Model config errors must speak human language", () => {
  it("translates backend codes into one actionable sentence", () => {
    // Really happened: after pasting the key and clicking "Set as current model", this exact English string showed in the UI.
    const text = modelErrorText("model_profile_probe_required");
    expect(text).not.toContain("model_profile_probe_required");
    expect(text).toContain("test");
  });

  it("every known code has a sentence, and none is the raw code", () => {
    const codes = [
      "model_profile_probe_required",
      "model_provider_not_supported",
      "embedding_model_required",
      "model_profile_not_found",
      "model_profile_conflict",
    ];
    for (const code of codes) {
      const text = modelErrorText(code);
      expect(text, code).not.toBe(code);
      expect(text, code).not.toContain("_");
    }
  });

  it("unknown codes pass through untouched, never swallowed", () => {
    // Positive control: without this line, the previous test could be fooled by "always return one 'something went wrong'" —
    // that way no code equals the original text, all tests go green, and the debugging trail is gone.
    const text = modelErrorText("some_code_nobody_mapped_yet");
    expect(text).toContain("some_code_nobody_mapped_yet");
  });
});
