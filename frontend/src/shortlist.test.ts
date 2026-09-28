import { describe, expect, it } from "vitest";

import { blankColumn, blankColumns, column, freeColour, freeName, scoreRequest } from "./shortlist";

const names = (columns: { name: string }[]) => columns.map((c) => c.name);

describe("new options", () => {
  it("start as Option A and Option B", () => {
    expect(names(blankColumns())).toEqual(["Option A", "Option B"]);
  });

  it("take the letter a removed option freed, never repeating one in use", () => {
    const remaining = ["Option B", "Option C", "Option D", "Option E"].map((name, i) => column(name, i + 1));
    expect(blankColumn(remaining).name).toBe("Option A");
  });

  it("leave custom names alone and skip letters still in use", () => {
    const shortlist = [column("Product B", 0), column("Option A", 1), column("Option C", 2)];
    const added = blankColumn(shortlist);
    expect(added.name).toBe("Option B");
    expect(names(shortlist)).toEqual(["Product B", "Option A", "Option C"]);
  });

  it("never duplicate a name across a full shortlist built by adding and removing", () => {
    let shortlist = blankColumns(5);
    for (const removed of [0, 2, 4, 1]) {
      shortlist = shortlist.filter((_, index) => index !== removed % shortlist.length);
      shortlist = [...shortlist, blankColumn(shortlist)];
      expect(new Set(names(shortlist)).size).toBe(shortlist.length);
    }
  });
});

describe("colours", () => {
  it("stay with their option when another is removed", () => {
    const shortlist = blankColumns(4);
    const kept = shortlist.filter((_, index) => index !== 1);
    expect(kept.map((c) => c.colour)).toEqual([0, 2, 3]);
  });

  it("go to a new option from the pool a removal refilled", () => {
    const kept = blankColumns(5).filter((_, index) => index !== 0);
    expect(blankColumn(kept).colour).toBe(0);
  });

  it("never repeat across a shortlist built by adding and removing", () => {
    let shortlist = blankColumns(5);
    for (const removed of [3, 0, 2, 1]) {
      shortlist = shortlist.filter((_, index) => index !== removed);
      shortlist = [...shortlist, blankColumn(shortlist)];
      expect(new Set(shortlist.map((c) => c.colour)).size).toBe(shortlist.length);
    }
  });

  it("take the lowest free slot", () => {
    expect(freeColour([column("x", 0), column("y", 2)])).toBe(1);
  });
});

describe("freeName", () => {
  it("returns the first free letter", () => {
    expect(freeName(["Option A", "Option C"])).toBe("Option B");
  });
});

describe("scoreRequest", () => {
  it("gives a blank-named option a free letter, never one another option uses", () => {
    const request = scoreRequest({
      category: "widget",
      context: "everyday",
      stakeholders: ["anyone"],
      columns: [column("Option B", 0), column("", 1), column("Option A", 2)],
    });
    expect(request.alternatives.map((a) => a.id)).toEqual(["Option B", "Option C", "Option A"]);
  });
});
