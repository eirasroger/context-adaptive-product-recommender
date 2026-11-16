import type { FormCategory } from "./api/client";

type Context = FormCategory["contexts"][number];

export type Group = { theme: string | null; contexts: Context[] };

/** The contexts a person can switch on, grouped by theme in the registry's order; the baseline is the absence of all. */
export function switchable(category: FormCategory): Group[] {
  const groups: Group[] = [];
  for (const context of category.contexts) {
    if (context.key === category.default_context) continue;
    const last = groups.at(-1);
    if (last && last.theme === context.theme) last.contexts.push(context);
    else groups.push({ theme: context.theme, contexts: [context] });
  }
  return groups;
}

export const baselineOf = (category: FormCategory): Context | undefined =>
  category.contexts.find((c) => c.key === category.default_context);

/** Switch one context on or off; switching on past the limit is refused. */
export function toggle(chosen: string[], key: string, limit: number): string[] {
  if (chosen.includes(key)) return chosen.filter((k) => k !== key);
  return chosen.length < limit ? [...chosen, key] : chosen;
}

/** Keep the offered, non-baseline contexts once each, up to the limit, and name what was left out. */
export function keepOffered(category: FormCategory, keys: string[]): { contexts: string[]; dropped: string[] } {
  const offered = new Set(category.contexts.map((c) => c.key));
  const contexts: string[] = [];
  const dropped: string[] = [];
  for (const key of keys) {
    if (key === category.default_context || contexts.includes(key)) continue;
    if (!offered.has(key) || contexts.length >= category.max_contexts) dropped.push(key);
    else contexts.push(key);
  }
  return { contexts, dropped };
}
