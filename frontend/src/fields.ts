import type { FormField } from "./api/client";

/** Every family's fields together, families in order of first appearance, fields in their given order. */
export function groupedByFamily(fields: readonly FormField[]): FormField[] {
  const families = new Map<string, FormField[]>();
  for (const field of fields) {
    const members = families.get(field.family);
    if (members) members.push(field);
    else families.set(field.family, [field]);
  }
  return [...families.values()].flat();
}
