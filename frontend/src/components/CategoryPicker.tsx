import { type KeyboardEvent, useEffect, useId, useRef, useState } from "react";

import type { FormCategory } from "../api/client";

type Props = {
  categories: FormCategory[];
  active: FormCategory;
  onChoose: (key: string) => void;
};

export function CategoryPicker({ categories, active, onChoose }: Props) {
  const [open, setOpen] = useState(false);
  const [cursor, setCursor] = useState(0);
  const root = useRef<HTMLDivElement>(null);
  const list = useId();

  useEffect(() => {
    if (!open) return;
    const closeOutside = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    addEventListener("pointerdown", closeOutside);
    return () => removeEventListener("pointerdown", closeOutside);
  }, [open]);

  const show = () => {
    setCursor(Math.max(0, categories.findIndex((c) => c.key === active.key)));
    setOpen(true);
  };
  const choose = (key: string) => {
    setOpen(false);
    onChoose(key);
  };

  const onKey = (event: KeyboardEvent) => {
    const last = categories.length - 1;
    if (!open) {
      if (["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) {
        event.preventDefault();
        show();
      }
      return;
    }
    if (event.key === "Escape") setOpen(false);
    else if (event.key === "ArrowDown") setCursor((i) => Math.min(last, i + 1));
    else if (event.key === "ArrowUp") setCursor((i) => Math.max(0, i - 1));
    else if (event.key === "Home") setCursor(0);
    else if (event.key === "End") setCursor(last);
    else if (event.key === "Enter" || event.key === " ") {
      const target = categories[cursor];
      if (target) choose(target.key);
    }
    else if (event.key === "Tab") setOpen(false);
    else return;
    if (event.key !== "Tab") event.preventDefault();
  };

  return (
    <div className="picker" ref={root}>
      <button
        type="button"
        className="picker-button"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={list}
        aria-activedescendant={open ? `${list}-${cursor}` : undefined}
        aria-labelledby="category-label"
        onClick={() => (open ? setOpen(false) : show())}
        onKeyDown={onKey}
      >
        <Summary category={active} />
        <svg className="chevron" viewBox="0 0 16 16" aria-hidden="true">
          <path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.6" />
        </svg>
      </button>
      {open && (
        <ul className="picker-list" role="listbox" id={list} aria-labelledby="category-label">
          {categories.map((c, index) => (
            <li
              key={c.key}
              id={`${list}-${index}`}
              role="option"
              aria-selected={c.key === active.key}
              className={index === cursor ? "option cursor" : "option"}
              onPointerEnter={() => setCursor(index)}
              onClick={() => choose(c.key)}
            >
              <Summary category={c} />
              {c.key === active.key && (
                <svg className="check" viewBox="0 0 16 16" aria-hidden="true">
                  <path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" strokeWidth="1.8" />
                </svg>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Summary({ category }: { category: FormCategory }) {
  return (
    <span className="picker-summary">
      <span className="picker-name">
        {category.display_name}
        {category.preview && <span className="badge">Preview</span>}
      </span>
      <span className="picker-unit">Compared per {category.functional_unit.toLowerCase()}</span>
    </span>
  );
}
