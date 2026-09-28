import { describe, expect, it } from "vitest";

import { unitText } from "./format";

describe("unitText", () => {
  it("raises square and cubic powers", () => {
    expect(unitText("m2")).toBe("m²");
    expect(unitText("m3-eq per m2")).toBe("m³-eq per m²");
    expect(unitText("W/m2K")).toBe("W/m²K");
  });

  it("lowers the digit in CO2", () => {
    expect(unitText("kg CO2-eq per kg")).toBe("kg CO₂-eq per kg");
  });

  it("leaves plain units alone", () => {
    expect(unitText("MPa")).toBe("MPa");
    expect(unitText("%")).toBe("%");
  });
});
