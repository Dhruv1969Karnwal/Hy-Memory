/** How a host's link and call-rule states are said to the user.
 *
 * Two screens now report the same host — the source list and the health
 * panel — and a second copy of these tables is how one screen ends up
 * calling a host Connected while the other calls it Not connected.
 */

/** The three verdicts, said as what the user would do about them.
 *
 * "Installed" on its own is the word that hid the recorded failure — three hosts
 * were installed, and installed was all it said. So none of these three is
 * that word.
 */
export const LINK_WORDING: Record<string, string> = {
  wired: "Connected",
  reachable_but_blind: "Installed, but no memories come back",
  unreachable: "Not connected",
};

/** Call-rule skill drift, said as what the user would do about it (R-33).
 *
 * Kept beside the link wording on purpose: a host can be wired and still be
 * running on yesterday's instructions, and the reverse. Different failure
 * modes, different actions.
 */
export const SKILL_WORDING: Record<string, string> = {
  in_sync: "Up to date",
  stale: "Needs update",
  modified: "Manually edited by you",
};
