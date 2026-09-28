import { type CSSProperties, Fragment, useLayoutEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";

import type { FormCategory, FormField } from "../api/client";
import { groupedByFamily } from "../fields";
import { rangeBound, seriesColour, stagger, unitText } from "../format";
import { blankColumn, type Column, MAX_COLUMNS, withEntry } from "../shortlist";
import type { TipContent } from "./Tip";
import { GROW_MS, prefersStill, SHRINK_MS, useColumnMotion } from "./useColumnMotion";

type Props = {
  category: FormCategory;
  families: Record<string, string>;
  columns: Column[];
  onColumns: (columns: Column[]) => void;
  onTip: (tip: TipContent | null) => void;
};

/** A removed column's content fades for this long before its width starts to close. */
const FADE_MS = 110;

export function Grid({ category, families, columns, onColumns, onTip }: Props) {
  const table = useRef<HTMLTableElement>(null);
  const addButton = useRef<HTMLButtonElement>(null);
  const names = useRef(new Map<number, HTMLTextAreaElement>());
  const current = useRef({ columns, onColumns });
  const { colRef, resize } = useColumnMotion(table);
  const [arriving, setArriving] = useState<number | null>(null);
  const [leaving, setLeaving] = useState<number | null>(null);

  useLayoutEffect(() => {
    current.current = { columns, onColumns };
  });

  useLayoutEffect(() => {
    if (arriving === null) return;
    const name = names.current.get(arriving);
    name?.focus({ preventScroll: true });
    name?.select();
    if (prefersStill()) return setArriving(null);
    resize({ id: arriving, from: 0, to: 1 }, GROW_MS, () => {
      setArriving(null);
      name?.closest("th")?.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" });
    });
  }, [arriving, resize]);

  const replace = (index: number, column: Column) =>
    onColumns(columns.map((c, i) => (i === index ? column : c)));
  const full = columns.length >= MAX_COLUMNS;
  const moving = arriving !== null || leaving !== null;
  const fields = groupedByFamily(category.fields);

  const add = () => {
    if (moving) return;
    const column = blankColumn(columns);
    setArriving(column.id);
    onColumns([...columns, column]);
  };

  const remove = (id: number) => {
    if (moving) return;
    const drop = () =>
      flushSync(() => {
        const latest = current.current;
        latest.onColumns(latest.columns.filter((c) => c.id !== id));
        setLeaving(null);
      });
    addButton.current?.focus({ preventScroll: true });
    if (prefersStill()) return drop();
    setLeaving(id);
    setTimeout(() => resize({ id, from: 1, to: 0 }, SHRINK_MS, drop), FADE_MS);
  };

  const motion = (id: number) =>
    id === leaving ? " col-leave" : id === arriving ? " col-enter" : "";

  return (
    <table className="grid" ref={table} style={{ "--n": columns.length } as CSSProperties}>
      <colgroup>
        <col className="col-rowhead" />
        {columns.map((column) => (
          <col key={column.id} ref={colRef(column.id)} />
        ))}
        <col className="col-add" />
      </colgroup>
      <thead>
        <tr>
          <th className="rowhead" />
          {columns.map((column, index) => (
            <th key={column.id} className={`colhead${motion(column.id)}`}>
              <div className="bar" style={{ background: seriesColour(column.colour) }} />
              <div className="colname">
                <NameField
                  ref={(element) => {
                    if (!element) return;
                    names.current.set(column.id, element);
                    return () => {
                      names.current.delete(column.id);
                    };
                  }}
                  value={column.name}
                  onChange={(name) => replace(index, { ...column, name })}
                />
                {columns.length > 2 && (
                  <button
                    className="colremove"
                    type="button"
                    aria-label={`Remove ${column.name || "this alternative"}`}
                    title="Remove"
                    onClick={() => remove(column.id)}
                  >
                    <svg viewBox="0 0 16 16" aria-hidden="true">
                      <path d="M4.5 4.5l7 7m0-7l-7 7" stroke="currentColor" strokeWidth="1.6" />
                    </svg>
                  </button>
                )}
              </div>
            </th>
          ))}
          <th className="addcol">
            <button
              ref={addButton}
              type="button"
              className="addcol-button"
              disabled={full}
              title={
                full
                  ? `${MAX_COLUMNS} alternatives is the widest shortlist the model was trained on`
                  : "Add another alternative to compare"
              }
              onClick={add}
            >
              <svg viewBox="0 0 16 16" aria-hidden="true">
                <path d="M8 3.5v9M3.5 8h9" stroke="currentColor" strokeWidth="1.6" />
              </svg>
              <span>Add option</span>
            </button>
          </th>
        </tr>
      </thead>
      <tbody key={category.key}>
        {fields.map((field, row) => (
          <Fragment key={field.key}>
            {field.family !== fields[row - 1]?.family && (
              <tr className="familyrow enter" style={stagger(row)}>
                <th className="rowhead">{families[field.family] ?? field.family}</th>
                {columns.map((column) => (
                  <td key={column.id} className={`line${motion(column.id)}`} />
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
                {field.unit && <div className="unit">{unitText(field.unit)}</div>}
              </th>
              {columns.map((column, index) => (
                <td key={column.id} className={`cell${motion(column.id)}`}>
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

type NameFieldProps = {
  ref: (element: HTMLTextAreaElement | null) => void | (() => void);
  value: string;
  onChange: (value: string) => void;
};

/** A name that wraps onto further lines as it grows; Enter finishes editing instead of breaking the line. */
function NameField({ ref, value, onChange }: NameFieldProps) {
  const own = useRef<HTMLTextAreaElement>(null);

  useLayoutEffect(() => {
    const field = own.current;
    if (!field || CSS.supports("field-sizing", "content")) return;
    field.style.height = "auto";
    field.style.height = `${field.scrollHeight}px`;
  }, [value]);

  return (
    <textarea
      ref={(element) => {
        own.current = element;
        return ref(element) ?? undefined;
      }}
      rows={1}
      value={value}
      aria-label="Alternative name"
      spellCheck={false}
      onChange={(e) => onChange(e.target.value.replace(/\s*\n\s*/g, " "))}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          e.currentTarget.blur();
        }
      }}
    />
  );
}

type CellProps = {
  field: FormField;
  column: Column;
  onChange: (column: Column) => void;
};

function Cell({ field, column, onChange }: CellProps) {
  if (field.levels.length > 0) {
    const level = column.levels[field.key] ?? "";
    return (
      <select
        className={level ? undefined : "empty"}
        value={level}
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
