import type { components } from "./api/schema";

export const MAX_COLUMNS = 5;

export type Column = {
  id: number;
  name: string;
  values: Record<string, string>;
  levels: Record<string, string>;
};

let nextId = 0;

export const letter = (index: number) => String.fromCharCode(65 + index);

export const blankColumn = (index: number): Column => ({
  id: nextId++,
  name: `Option ${letter(index)}`,
  values: {},
  levels: {},
});

export const blankShortlist = () => [blankColumn(0), blankColumn(1)];

export const isFilled = (column: Column) =>
  Object.keys(column.values).length > 0 || Object.keys(column.levels).length > 0;

export function withEntry(
  entries: Record<string, string>,
  key: string,
  value: string,
): Record<string, string> {
  const { [key]: _, ...rest } = entries;
  return value === "" ? rest : { ...rest, [key]: value };
}

export function scoreRequest(
  category: string,
  context: string,
  stakeholders: string[],
  columns: Column[],
): components["schemas"]["ScoreRequest"] {
  return {
    category,
    context: [context],
    stakeholders,
    alternatives: columns.map((column, index) => ({
      id: column.name || `Option ${letter(index)}`,
      values: Object.fromEntries(
        Object.entries(column.values).map(([key, value]) => [key, Number(value)]),
      ),
      levels: column.levels,
    })),
  };
}
