/** Counts and the nouns they count.
 *
 * The wording this CLI was translated from counts in a language with no plural
 * morphology, so every counter here used to print the noun in its plural form
 * whatever the number was: "Found 1 apps". The count decides the ending.
 */

export function countLabel(
  count: number,
  singular: string,
  plural?: string,
): string {
  return `${count} ${nounFor(count, singular, plural)}`;
}

/** A counter read back out of a rendered document, which carries its fields
 * untyped; a field the document does not carry counts as nothing counted. */
export function counted(value: unknown): number {
  return typeof value === "number" ? value : 0;
}

/** The noun on its own, for copy that prints the things counted instead of
 * the number: "bridge, sync issue" when one section is degraded. */
export function nounLabel(
  count: number,
  singular: string,
  plural?: string,
): string {
  return nounFor(count, singular, plural);
}

function nounFor(count: number, singular: string, plural?: string): string {
  if (count === 1) return singular;
  return plural ?? pluralOf(singular);
}

function pluralOf(word: string): string {
  if (/(?:s|x|z|ch|sh)$/.test(word)) return `${word}es`;
  if (/[^aeiou]y$/.test(word)) return `${word.slice(0, -1)}ies`;
  return `${word}s`;
}
