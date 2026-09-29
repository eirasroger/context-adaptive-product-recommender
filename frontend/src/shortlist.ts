import type { components } from "./api/schema";

export const MAX_COLUMNS = 5;
export const MIN_COLUMNS = 2;

export type Column = {
  id: number;
  name: string;
  /** Series colour slot, kept for the option's life and freed when it is removed. */
  colour: number;
  values: Record<string, string>;
  levels: Record<string, string>;
};

export type Shortlist = {
  category: string;
  context: string;
  stakeholders: string[];
  columns: Column[];
};

let nextId = 0;

export const letter = (index: number) => String.fromCharCode(65 + index);

export const column = (
  name: string,
  colour: number,
  values: Record<string, string> = {},
  levels: Record<string, string> = {},
): Column => ({ id: nextId++, name, colour, values, levels });

/** The first of Option A to Option E that no name in use already takes. */
export function freeName(taken: Iterable<string>): string {
  const used = new Set(taken);
  for (let index = 0; ; index++) {
    const name = `Option ${letter(index)}`;
    if (!used.has(name)) return name;
  }
}

/** The lowest colour slot no option in use has. */
export function freeColour(existing: readonly Column[]): number {
  const used = new Set(existing.map((c) => c.colour));
  let slot = 0;
  while (used.has(slot)) slot++;
  return slot;
}

export const blankColumn = (existing: readonly Column[]) =>
  column(freeName(existing.map((c) => c.name)), freeColour(existing));

export function blankColumns(count = MIN_COLUMNS): Column[] {
  const columns: Column[] = [];
  while (columns.length < count) columns.push(blankColumn(columns));
  return columns;
}

export const isFilled = (column: Column) =>
  Object.keys(column.values).length > 0 || Object.keys(column.levels).length > 0;

export const isScorable = (shortlist: Shortlist) =>
  shortlist.columns.filter(isFilled).length >= MIN_COLUMNS;

export function withEntry(
  entries: Record<string, string>,
  key: string,
  value: string,
): Record<string, string> {
  const { [key]: _, ...rest } = entries;
  return value === "" ? rest : { ...rest, [key]: value };
}

export function scoreRequest(shortlist: Shortlist): components["schemas"]["ShortlistRequest"] {
  const taken = shortlist.columns.map((c) => c.name);
  return {
    category: shortlist.category,
    context: [shortlist.context],
    stakeholders: shortlist.stakeholders,
    alternatives: shortlist.columns.map((column) => {
      const id = column.name || freeName(taken);
      taken.push(id);
      return {
        id,
        values: Object.fromEntries(
          Object.entries(column.values).map(([key, value]) => [key, Number(value)]),
        ),
        levels: column.levels,
      };
    }),
  };
}
