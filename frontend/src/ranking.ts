export type Standing<T> = { entry: T; place: number; group: number; tied: boolean };

/**
 * Orders entries by score. An entry less than `epsilon` below its group's first entry
 * joins that group and shares its place, so no two members of a group differ by `epsilon` or more.
 */
export function standings<T extends { score: number }>(
  entries: readonly T[],
  epsilon: number,
): Standing<T>[] {
  const ordered = [...entries].sort((a, b) => b.score - a.score);
  const placed: Standing<T>[] = [];
  let head: T | undefined;
  let place = 0;
  let group = -1;
  ordered.forEach((entry, index) => {
    if (!head || head.score - entry.score >= epsilon) {
      head = entry;
      place = index + 1;
      group += 1;
    }
    placed.push({ entry, place, group, tied: false });
  });
  for (const standing of placed) {
    standing.tied = placed.some((other) => other !== standing && other.group === standing.group);
  }
  return placed;
}

/** Positions of every score less than `epsilon` below the best. */
export function jointLeaders(scores: readonly number[], epsilon: number): Set<number> {
  const best = Math.max(...scores);
  return new Set(scores.flatMap((score, index) => (best - score < epsilon ? [index] : [])));
}
