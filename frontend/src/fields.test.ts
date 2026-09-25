import { describe, expect, it } from "vitest";

import type { FormField } from "./api/client";
import { groupedByFamily } from "./fields";

const field = (key: string, family: string): FormField => ({
  key,
  display_name: key,
  family,
  unit: null,
  value_type: "continuous",
  definition: "",
  direction: 1,
  range: null,
  levels: [],
});

describe("groupedByFamily", () => {
  it("never splits a family, whatever order the fields arrive in", () => {
    const fields = [field("x1", "first"), field("y1", "second"), field("x2", "first"), field("z1", "third")];
    expect(groupedByFamily(fields).map((f) => f.key)).toEqual(["x1", "x2", "y1", "z1"]);
  });

  it("leaves fields that are already grouped as they are", () => {
    const fields = [field("x1", "first"), field("x2", "first"), field("y1", "second")];
    expect(groupedByFamily(fields)).toEqual(fields);
  });
});
