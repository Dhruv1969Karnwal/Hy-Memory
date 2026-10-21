/**
 * One clock format for every user-facing timestamp.
 *
 * Three surfaces grew their own copy of this: `CallHistory` (07), where the
 * "Today HH:MM" wording was settled against the source, `SelectedMemoryDetail`
 * (06E), which agreed on today but differed on older dates, and `HealthPanel`
 * (Diagnostics), which had no today branch at all and printed a full
 * `toLocaleString()` -- so the same moment read as "Today 14:31" on one page and
 * "2026/9/20 14:31:07" on another.
 *
 * Absolute seconds are not information the user asked for; they were an
 * artefact of the default formatter. The source (139:7984) writes "Today 02:10".
 *
 * Unparseable input returns "Time unknown" rather than a fabricated date. That string
 * is already asserted across several suites, so it stays exactly as-is.
 */
export function clockStamp(value: string | null | undefined): string {
  if (!value || !Number.isFinite(Date.parse(value))) return "Time unknown";
  const date = new Date(value);
  const now = new Date();
  const clock = new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
  if (
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate()
  )
    return `Today ${clock}`;
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}
