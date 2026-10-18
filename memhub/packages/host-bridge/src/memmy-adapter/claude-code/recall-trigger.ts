import { createHash } from "node:crypto";

/** Openers and acknowledgements that carry no question for history to answer.
 *
 * Deliberately a closed list rather than a rule. Recall costs a round trip and
 * a line of user-visible text; not recalling costs an answer the user wanted.
 * The two are not symmetric, so anything outside this list still recalls. */
const PLEASANTRIES = new Set([
  "hi",
  "hey",
  "hello",
  "yo",
  "thanks",
  "thank you",
  "thx",
  "ty",
  "ok",
  "okay",
  "k",
  "cool",
  "nice",
  "got it",
  "sounds good",
  "bye",
  "goodbye",
  "good morning",
  "good afternoon",
  "good evening",
  "\u4f60\u597d",
  "\u60a8\u597d",
  "\u54c8\u55bd",
  "\u55e8",
  "\u5728\u5417",
  "\u5728\u4e48",
  "\u8c22\u8c22",
  "\u591a\u8c22",
  "\u611f\u8c22",
  "\u8f9b\u82e6\u4e86",
  "\u597d",
  "\u597d\u7684",
  "\u884c",
  "\u884c\u5427",
  "\u53ef\u4ee5",
  "\u6536\u5230",
  "\u660e\u767d",
  "\u61c2\u4e86",
  "\u77e5\u9053\u4e86",
  "\u6ca1\u95ee\u9898",
  "\u518d\u89c1",
  "\u62dc\u62dc",
  "\u65e9\u4e0a\u597d",
  "\u4e0b\u5348\u597d",
  "\u665a\u4e0a\u597d",
]);

/** Punctuation and emphasis that change nothing about what was asked. */
const DECORATION =
  /^[\s!?.,;:~\u3001\u3002\uff0c\uff01\uff1f\uff1b\uff1a…·\-—"'“”‘’()\uff08\uff09]+|[\s!?.,;:~\u3001\u3002\uff0c\uff01\uff1f\uff1b\uff1a…·\-—"'“”‘’()\uff08\uff09]+$/gu;

function bare(prompt: string): string {
  return prompt.trim().replace(DECORATION, "").trim().toLowerCase();
}

/** Same question asked twice is the same question, whatever the spacing. */
export function queryFingerprint(prompt: string): string {
  return createHash("sha256")
    .update(prompt.trim().replace(/\s+/gu, " ").toLowerCase())
    .digest("hex");
}

export function isPleasantry(prompt: string): boolean {
  return PLEASANTRIES.has(bare(prompt));
}
