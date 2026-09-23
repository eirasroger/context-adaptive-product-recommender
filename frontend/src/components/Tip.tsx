import { useLayoutEffect, useRef } from "react";
import { createPortal } from "react-dom";

export type TipContent = { x: number; y: number; title: string; body: string };

const GAP = 14;
const EDGE = 8;

export function Tip({ tip }: { tip: TipContent | null }) {
  const ref = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    const element = ref.current;
    if (!tip || !element) return;
    const { width, height } = element.getBoundingClientRect();
    let x = tip.x + GAP;
    let y = tip.y + GAP;
    if (x + width > innerWidth - EDGE) x = tip.x - width - GAP;
    if (y + height > innerHeight - EDGE) y = tip.y - height - GAP;
    element.style.left = `${Math.max(EDGE, x)}px`;
    element.style.top = `${Math.max(EDGE, y)}px`;
  }, [tip]);

  if (!tip) return null;
  return createPortal(
    <div className="tip" ref={ref}>
      <div>
        <strong>{tip.title}</strong>
      </div>
      <div className="t-label">{tip.body}</div>
    </div>,
    document.body,
  );
}
