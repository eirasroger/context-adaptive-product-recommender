import { Fragment } from "react";

import type { FormCategory, FormField } from "../api/client";
import { rangeBound, seriesColour, stagger } from "../format";
import { blankColumn, type Column, MAX_COLUMNS, withEntry } from "../shortlist";
import type { TipContent } from "./Tip";

type Props = {
  category: FormCategory;
  families: Record<string, string>;
  columns: Column[];
  onColumns: (columns: Column[]) => void;
  onTip: (tip: TipContent | null) => void;
};

export function Grid({ category, families, columns, onColumns, onTip }: Props) {
  const replace = (index: number, column: Column) =>
    onColumns(columns.map((c, i) => (i === index ? column : c)));
  const full = columns.length >= MAX_COLUMNS;

  return (
    <table className="grid">
      <thead>
        <tr>
          <th className="rowhead" />
          {columns.map((column, index) => (
            <th key={column.id} className="colhead">
              <div className="bar" style={{ background: seriesColour(index) }} />
              <input
                value={column.name}
                aria-label="Alternative name"
                onChange={(e) => replace(index, { ...column, name: e.target.value })}
              />
              <div className="tools">
                {columns.length > 2 && (
                  <button
                    className="tiny"
                    type="button"
                    onClick={() => onColumns(columns.filter((_, i) => i !== index))}
                  >
                    Remove
                  </button>
                )}
              </div>
            </th>
          ))}
          <th className="addcol">
            <button
              type="button"
              className="addcol-button"
              disabled={full}
              title={
                full
                  ? `${MAX_COLUMNS} alternatives is the widest shortlist the model was trained on`
                  : "Add another alternative to compare"
              }
              onClick={() => onColumns([...columns, blankColumn(columns.length)])}
            >
              <span aria-hidden="true">+</span>
              <span>Add</span>
            </button>
          </th>
        </tr>
      </thead>
      <tbody key={category.key}>
        {category.fields.map((field, row) => (
          <Fragment key={field.key}>
            {field.family !== category.fields[row - 1]?.family && (
              <tr className="familyrow enter" style={stagger(row)}>
                <th className="rowhead">{families[field.family] ?? field.family}</th>
                {columns.map((column) => (
                  <td key={column.id} className="line" />
                ))}
                <td />
              </tr>
            )}
            <tr className="enter" style={stagger(row)}>
              <th className="rowhead">
                <abbr
                  onPointerMove={(e) =>
                    onTip({
                      x: e.clientX,
                      y: e.clientY,
                      title: field.display_name,
                      body: field.definition,
                    })
                  }
                  onPointerLeave={() => onTip(null)}
                >
                  {field.display_name}
                </abbr>
                {field.unit && <div className="unit">{field.unit}</div>}
              </th>
              {columns.map((column, index) => (
                <td key={column.id} className="cell">
                  <Cell
                    field={field}
                    column={column}
                    onChange={(next) => replace(index, next)}
                  />
                </td>
              ))}
              <td />
            </tr>
          </Fragment>
        ))}
      </tbody>
    </table>
  );
}

type CellProps = {
  field: FormField;
  column: Column;
  onChange: (column: Column) => void;
};

function Cell({ field, column, onChange }: CellProps) {
  if (field.levels.length > 0) {
    return (
      <select
        value={column.levels[field.key] ?? ""}
        onChange={(e) =>
          onChange({ ...column, levels: withEntry(column.levels, field.key, e.target.value) })
        }
      >
        <option value="">unknown</option>
        {field.levels.map((level) => (
          <option key={level.key} value={level.key}>
            {level.disqualifying ? `${level.display_name} (never selectable)` : level.display_name}
          </option>
        ))}
      </select>
    );
  }

  return (
    <input
      type="number"
      step="any"
      inputMode="decimal"
      placeholder={
        field.range ? `${rangeBound(field.range.low)} to ${rangeBound(field.range.high)}` : undefined
      }
      value={column.values[field.key] ?? ""}
      onChange={(e) =>
        onChange({ ...column, values: withEntry(column.values, field.key, e.target.value) })
      }
    />
  );
}
