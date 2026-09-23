import type { components } from "./api/schema";

export const MAX_COLUMNS = 5;
export const MIN_COLUMNS = 2;

export type Column = {
  id: number;
  name: string;
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
  values: Record<string, string> = {},
  levels: Record<string, string> = {},
): Column => ({ id: nextId++, name, values, levels });

export const blankColumn = (index: number) => column(`Option ${letter(index)}`);

export const blankColumns = (count = MIN_COLUMNS) =>
  Array.from({ length: count }, (_, index) => blankColumn(index));

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

export function scoreRequest(shortlist: Shortlist): components["schemas"]["ScoreRequest"] {
  return {
    category: shortlist.category,
    context: [shortlist.context],
    stakeholders: shortlist.stakeholders,
    alternatives: shortlist.columns.map((column, index) => ({
      id: column.name || `Option ${letter(index)}`,
      values: Object.fromEntries(
        Object.entries(column.values).map(([key, value]) => [key, Number(value)]),
      ),
      levels: column.levels,
    })),
  };
}
