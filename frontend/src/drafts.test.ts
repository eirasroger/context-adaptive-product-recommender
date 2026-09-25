import { describe, expect, it } from "vitest";

import { isStale, switchCategory, unscored } from "./drafts";
import { blankColumns, column, type Shortlist } from "./shortlist";

const start = (category: string): Shortlist => ({
  category,
  context: `${category}-default`,
  stakeholders: ["someone"],
  columns: blankColumns(),
});

describe("switchCategory", () => {
  it("resumes a category exactly as it was left", () => {
    const filled = unscored({ ...start("first"), columns: [column("X", { k: "1" }), column("Y")] });
    const away = switchCategory(new Map(), filled, "second", start);
    const back = switchCategory(away.parked, away.next, "first", start);
    expect(back.next).toBe(filled);
  });

  it("opens an unvisited category blank, keeping the chosen priorities", () => {
    const current = unscored({ ...start("first"), stakeholders: ["advocate", "developer"] });
    const { next } = switchCategory(new Map(), current, "second", start);
    expect(next.shortlist.category).toBe("second");
    expect(next.shortlist.context).toBe("second-default");
    expect(next.shortlist.stakeholders).toEqual(["advocate", "developer"]);
    expect(next.result).toBeNull();
  });
});

describe("isStale", () => {
  it("holds once the shortlist moves on from the one that was scored", () => {
    const shortlist = start("first");
    const scored = { ...unscored(shortlist), scored: shortlist };
    expect(isStale(scored)).toBe(false);
    expect(isStale({ ...scored, shortlist: { ...shortlist, context: "other" } })).toBe(true);
    expect(isStale(unscored(shortlist))).toBe(false);
  });
});
