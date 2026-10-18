import { describe, expect, it } from "vitest";

import { canHandOff, HANDOFF_WITHOUT_PROMPT, handoffUrl } from "./handoff";

const HOME = "/Users/someone";

describe("handoffUrl", () => {
  it("never launches Claude desktop under the Claude Code identity", () => {
    expect(handoffUrl("claude-code", "context", HOME)).toBeNull();
  });

  it("refuses an oversized URL instead of silently losing context", () => {
    expect(handoffUrl("cursor", "prompt ".repeat(5000), HOME)).toBeNull();
  });
  it("encodes the prompt into each app's own scheme", () => {
    const prompt = "Remember that axios thing we talked about? &continue";

    expect(handoffUrl("cursor", prompt, HOME)).toBe(
      `cursor://anysphere.cursor-deeplink/prompt?text=${encodeURIComponent(prompt)}&mode=ask`,
    );
    expect(handoffUrl("claude-code", prompt, HOME)).toBeNull();
  });

  it("does not offer a guessed link for hosts whose input box is unverified", () => {
    expect(handoffUrl("codex", "question", HOME)).toBeNull();
    expect(handoffUrl("opencode", "question", HOME)).toBeNull();
  });

  it("& and ? do not cut the link in half", () => {
    const url = handoffUrl("cursor", "a&b=c?d", HOME);

    expect(url).not.toContain("a&b=c?d");
    expect(new URL(url ?? "").searchParams.get("text")).toBe("a&b=c?d");
  });

  it("an app it does not know never gets a link", () => {
    // This guards the allowlist: the main process builds URLs only from this table,
    // so anything outside it never reaches shell.openExternal.
    expect(handoffUrl("evil", "x", HOME)).toBeNull();
    expect(handoffUrl("file", "x", HOME)).toBeNull();
    expect(canHandOff("evil")).toBe(false);
  });

  it("only apps it knows get one", () => {
    // Positive control for the case above: with the same function, an entry on the
    // table does get a link; otherwise "always returns null" would pass it too.
    expect(canHandOff("cursor")).toBe(true);
    expect(handoffUrl("cursor", "x", HOME)).not.toBeNull();
  });

  it("an empty prompt opens no window", () => {
    expect(handoffUrl("cursor", "   ", HOME)).toBeNull();
  });

  it("apps that connect but cannot be filled in are listed separately instead of faked", () => {
    expect(HANDOFF_WITHOUT_PROMPT.has("workbuddy")).toBe(true);
    expect(canHandOff("workbuddy")).toBe(false);
  });
});
