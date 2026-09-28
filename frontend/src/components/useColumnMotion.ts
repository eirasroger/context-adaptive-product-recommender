import { type RefObject, useCallback, useEffect, useRef } from "react";

export const GROW_MS = 300;
export const SHRINK_MS = 260;

const easeOut = (t: number) => 1 - (1 - t) ** 3;

const px = (style: CSSStyleDeclaration, name: string) =>
  Number.parseFloat(style.getPropertyValue(name)) || 0;

export const prefersStill = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

/** One option column's share of the width, moving between 0 (absent) and 1 (a full share). */
export type Resize = { id: number; from: number; to: number };

/** Animates real column widths in a fixed-layout table, so cells and their inputs resize without stretching. */
export function useColumnMotion(table: RefObject<HTMLTableElement | null>) {
  const cols = useRef(new Map<number, HTMLTableColElement>());
  const frame = useRef(0);

  useEffect(() => () => cancelAnimationFrame(frame.current), []);

  const colRef = useCallback(
    (id: number) => (element: HTMLTableColElement | null) => {
      if (!element) return;
      cols.current.set(id, element);
      return () => {
        cols.current.delete(id);
      };
    },
    [],
  );

  const resize = useCallback(
    (change: Resize, duration: number, settle: () => void) => {
      const grid = table.current;
      const wrap = grid?.parentElement;
      if (!grid || !wrap) return settle();
      cancelAnimationFrame(frame.current);

      const tokens = getComputedStyle(grid);
      const box = getComputedStyle(wrap);
      const fixed = px(tokens, "--rowhead-w") + px(tokens, "--add-w");
      const narrowest = px(tokens, "--col-min");
      const room =
        wrap.clientWidth - Number.parseFloat(box.paddingLeft) - Number.parseFloat(box.paddingRight) - fixed;
      const others = cols.current.size - 1;
      const start = performance.now();

      const paint = (now: number) => {
        const t = Math.min(1, (now - start) / duration);
        const share = change.from + (change.to - change.from) * easeOut(t);
        const total = Math.max(room, (others + share) * narrowest);
        const unit = total / (others + share);
        for (const [id, col] of cols.current) {
          col.style.width = `${id === change.id ? unit * share : unit}px`;
        }
        grid.style.width = `${fixed + total}px`;
        if (t < 1) {
          frame.current = requestAnimationFrame(paint);
          return;
        }
        settle();
        for (const col of cols.current.values()) col.style.width = "";
        grid.style.width = "";
        grid.style.minWidth = "";
        grid.classList.remove("resizing");
      };

      grid.classList.add("resizing");
      grid.style.minWidth = "0";
      paint(start);
    },
    [table],
  );

  return { colRef, resize };
}
