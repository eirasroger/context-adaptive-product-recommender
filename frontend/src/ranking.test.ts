import { describe, expect, it } from "vitest";

import { jointLeaders, standings } from "./ranking";

const scored = (...scores: number[]) => scores.map((score, id) => ({ id, score }));
const placesOf = (scores: number[], epsilon = 0.03) =>
  standings(scored(...scores), epsilon).map((s) => s.place);

describe("standings", () => {
  it("orders by score, best first", () => {
    const order = standings(scored(0.4, 0.9, 0.6), 0.03).map((s) => s.entry.id);
    expect(order).toEqual([1, 2, 0]);
  });

  it("gives an option within the tolerance of the one above it the same place", () => {
    expect(placesOf([0.82, 0.75, 0.494, 0.45, 0.444])).toEqual([1, 2, 3, 4, 4]);
  });

  it("skips as many places after a tie as the tie took", () => {
    expect(placesOf([0.8, 0.79, 0.5])).toEqual([1, 1, 3]);
  });

  it("never lets a group span the tolerance", () => {
    expect(placesOf([0.8, 0.78, 0.765])).toEqual([1, 1, 3]);
  });

  it("treats a gap of exactly the tolerance as decisive", () => {
    expect(placesOf([0.5, 0.25], 0.25)).toEqual([1, 2]);
  });

  it("marks only the members of a shared place as tied", () => {
    const tied = standings(scored(0.9, 0.5, 0.49), 0.03).map((s) => s.tied);
    expect(tied).toEqual([false, true, true]);
  });
});

describe("jointLeaders", () => {
  it("names every option within the tolerance of the best", () => {
    expect([...jointLeaders([0.6, 0.82, 0.8, 0.7], 0.03)].sort()).toEqual([1, 2]);
  });

  it("names a clear winner alone", () => {
    expect([...jointLeaders([0.4, 0.9], 0.03)]).toEqual([1]);
  });
});
