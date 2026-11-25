import { describe, expect, it } from "vitest";

import type { FormCategory } from "./api/client";
import { keepOffered, switchable, toggle } from "./contexts";

const category = {
  default_context: "base",
  max_contexts: 2,
  max_stakeholders: 3,
  contexts: [
    { key: "base", display_name: "Base", definition: "", theme: "Baseline" },
    { key: "cold", display_name: "Cold", definition: "", theme: "Climate" },
    { key: "hot", display_name: "Hot", definition: "", theme: "Climate" },
    { key: "loud", display_name: "Loud", definition: "", theme: "Site" },
  ],
} as FormCategory;

describe("switchable contexts", () => {
  it("groups by theme in the registry's order and leaves the baseline out", () => {
    expect(switchable(category).map((g) => [g.theme, g.contexts.map((c) => c.key)])).toEqual([
      ["Climate", ["cold", "hot"]],
      ["Site", ["loud"]],
    ]);
  });
});

describe("toggling a context", () => {
  it("switches on below the limit and off at any time", () => {
    expect(toggle([], "cold", 2)).toEqual(["cold"]);
    expect(toggle(["cold"], "cold", 2)).toEqual([]);
  });

  it("refuses to pass the limit", () => {
    expect(toggle(["cold", "hot"], "loud", 2)).toEqual(["cold", "hot"]);
  });
});

describe("keeping what the category offers", () => {
  it("drops the baseline, repeats, unknown keys and anything past the limit", () => {
    expect(keepOffered(category, ["base", "cold", "cold", "gone", "hot", "loud"])).toEqual({
      contexts: ["cold", "hot"],
      dropped: ["gone", "loud"],
    });
  });
});
