import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

import { prefersStill } from "./useColumnMotion";

const EASE = "cubic-bezier(0.2, 0.7, 0.2, 1)";
const easeOut = (t: number) => 1 - (1 - t) ** 3;

/** A number that glides to each new target, starting from zero on first show. */
export function useTween(target: number, duration = 700): number {
  const [shown, setShown] = useState(() => (prefersStill() ? target : 0));
  const current = useRef(shown);

  useEffect(() => {
    if (prefersStill()) {
      current.current = target;
      setShown(target);
      return;
    }
    const origin = current.current;
    const start = performance.now();
    let frame = requestAnimationFrame(function step(now) {
      const t = Math.min(1, (now - start) / duration);
      current.current = origin + (target - origin) * easeOut(t);
      setShown(current.current);
      if (t < 1) frame = requestAnimationFrame(step);
    });
    return () => cancelAnimationFrame(frame);
  }, [target, duration]);

  return shown;
}

/** Slides keyed rows from their old positions to their new ones after a reorder. */
export function useReorder<K>() {
  const rows = useRef(new Map<K, HTMLElement>());
  const tops = useRef(new Map<K, number>());

  useLayoutEffect(() => {
    const still = prefersStill();
    for (const [key, element] of rows.current) {
      const top = element.offsetTop;
      const before = tops.current.get(key);
      if (!still && before !== undefined && before !== top) {
        element.animate(
          [{ transform: `translateY(${before - top}px)` }, { transform: "none" }],
          { duration: 460, easing: EASE },
        );
      }
      tops.current.set(key, top);
    }
  });

  return useCallback(
    (key: K) => (element: HTMLElement | null) => {
      if (!element) return;
      rows.current.set(key, element);
      return () => {
        rows.current.delete(key);
        tops.current.delete(key);
      };
    },
    [],
  );
}
