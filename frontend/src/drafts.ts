import type { ScoreResponse, Sensitivity } from "./api/client";
import type { Shortlist } from "./shortlist";

export type Draft = {
  shortlist: Shortlist;
  /** The shortlist the result was computed for; the result is stale once they differ. */
  scored: Shortlist | null;
  result: ScoreResponse | null;
  wins: Map<number, string>;
  byContext: Sensitivity | null;
  byStakeholder: Sensitivity | null;
};

export const unscored = (shortlist: Shortlist): Draft => ({
  shortlist,
  scored: null,
  result: null,
  wins: new Map(),
  byContext: null,
  byStakeholder: null,
});

export const isStale = (draft: Draft) => draft.scored !== null && draft.scored !== draft.shortlist;

/** Park the current draft and resume the target category's, or start one that keeps the chosen priorities. */
export function switchCategory(
  parked: ReadonlyMap<string, Draft>,
  current: Draft,
  target: string,
  start: (category: string) => Shortlist,
): { parked: Map<string, Draft>; next: Draft } {
  const kept = new Map(parked);
  kept.set(current.shortlist.category, current);
  const resumed = kept.get(target);
  if (resumed) return { parked: kept, next: resumed };
  return {
    parked: kept,
    next: unscored({ ...start(target), stakeholders: current.shortlist.stakeholders }),
  };
}
