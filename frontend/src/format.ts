export const fixed = (value: number, digits = 3) => value.toFixed(digits);

export const seriesColour = (index: number) => `var(--series-${(index % 8) + 1})`;

export function rangeBound(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude === 0) return "0";
  if (magnitude < 0.001) return value.toExponential(0);
  if (magnitude < 1) return String(Number(value.toFixed(4)));
  return String(Number(value.toFixed(magnitude < 100 ? 1 : 0)));
}
