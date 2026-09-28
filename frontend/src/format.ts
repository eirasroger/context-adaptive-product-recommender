import type { CSSProperties } from "react";

export const fixed = (value: number, digits = 3) => value.toFixed(digits);

export const seriesColour = (index: number) => `var(--series-${(index % 8) + 1})`;

export function rangeBound(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude === 0) return "0";
  if (magnitude < 0.001) return value.toExponential(0);
  if (magnitude < 1) return String(Number(value.toFixed(4)));
  return String(Number(value.toFixed(magnitude < 100 ? 1 : 0)));
}

const SUPERSCRIPT: Record<string, string> = { "2": "²", "3": "³" };

/** A unit as typeset: m2 to m², m3 to m³, CO2 to CO₂. */
export const unitText = (unit: string) =>
  unit
    .replace(/CO2(?![0-9])/g, "CO₂")
    .replace(/([a-z])([23])(?![0-9])/g, (_, letter: string, power: string) => letter + SUPERSCRIPT[power]);

/** Position in a staggered entrance; the stylesheet turns it into a delay. */
export const stagger = (index: number) => ({ "--row": index }) as CSSProperties;
