/** A count and the noun it counts, in one string.
 *
 * The copy this UI was translated from counts in a language with no plural
 * morphology, so every counter here used to print the noun in its plural form
 * whatever the number was: "1 apps connected". The count decides the ending.
 */

export function countLabel(
  count: number,
  singular: string,
  plural?: string,
): string {
  return `${count} ${nounFor(count, singular, plural)}`;
}

/** A count on its own, grouped the one way this UI groups counts. Copy that
 * counts things without naming them ("3 already existed") still has to pick a
 * locale, and it has to pick the same one `groupedCountLabel` picked. */
export function groupedNumber(count: number): string {
  return count.toLocaleString("en-US");
}

/** The same, with thousands separators, for the counters that had them. */
export function groupedCountLabel(
  count: number,
  singular: string,
  plural?: string,
): string {
  return `${groupedNumber(count)} ${nounFor(count, singular, plural)}`;
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
