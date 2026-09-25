import { describe, expect, it } from "vitest";

import type { Form } from "./api/client";
import { decodeLink, encodeLink, restore, type Snapshot, snapshotOf } from "./link";

const form: Form = {
  registry_version: "1.0",
  snapshot: "abc",
  families: { measure: "measure" },
  stakeholders: [
    { key: "thrifty", display_name: "Thrifty", definition: "" },
    { key: "green", display_name: "Green", definition: "" },
  ],
  categories: [
    {
      key: "widget",
      display_name: "Widget",
      preview: false,
      functional_unit: "one widget",
      eligibility_precondition: "",
      default_context: "plain",
      contexts: [
        { key: "plain", display_name: "Plain", definition: "" },
        { key: "quiet", display_name: "Quiet", definition: "" },
      ],
      fields: [
        {
          key: "alpha",
          display_name: "Alpha",
          family: "measure",
          unit: null,
          value_type: "continuous",
          definition: "",
          direction: 1,
          range: { low: 0, high: 1 },
          levels: [],
        },
        {
          key: "grade",
          display_name: "Grade",
          family: "measure",
          unit: null,
          value_type: "ordinal",
          definition: "",
          direction: 1,
          range: null,
          levels: [
            { key: "g1", display_name: "G1", disqualifying: false },
            { key: "g2", display_name: "G2", disqualifying: false },
          ],
        },
      ],
    },
  ],
};

const snapshot: Snapshot = {
  registry: "1.0",
  category: "widget",
  context: "quiet",
  stakeholders: ["green", "thrifty"],
  alternatives: [
    { name: "First", values: { alpha: "0.10" }, levels: { grade: "g2" } },
    { name: "Second ünïcode & <b>", values: { alpha: "1e-4" }, levels: {} },
  ],
};

const alternative = (name: string) => ({ name, values: {}, levels: {} });

async function linkFor(value: unknown): Promise<string> {
  return encodeLink(value as Snapshot);
}

describe("a link", () => {
  it("round-trips exactly what went in", async () => {
    expect(await decodeLink(`#${await encodeLink(snapshot)}`)).toEqual(snapshot);
  });

  it("is safe to put in a URL fragment, whatever its length", async () => {
    for (let extra = 0; extra < 6; extra++) {
      const link = await encodeLink({ ...snapshot, context: "q".repeat(extra) });
      expect(link).toMatch(/^v1\.[A-Za-z0-9_-]+$/);
    }
  });

  it("restores the shortlist it was made from, with nothing to report", async () => {
    const restored = restore((await decodeLink(await encodeLink(snapshot)))!, form);
    expect(restored?.notes).toEqual([]);
    expect(snapshotOf(restored!.shortlist, form.registry_version)).toEqual(snapshot);
  });
});

describe("a damaged link", () => {
  it.each([
    ["an empty fragment", ""],
    ["a bare format marker", "#v1."],
    ["characters outside base64url", "#v1.!!!"],
    ["a format this page does not read", "#v2.AAAA"],
    ["bytes that are not compressed", `#v1.${btoa("plain text")}`],
  ])("is refused: %s", async (_, fragment) => {
    expect(await decodeLink(fragment)).toBeNull();
  });

  it("is refused when truncated", async () => {
    const link = await encodeLink(snapshot);
    expect(await decodeLink(link.slice(0, link.length - 6))).toBeNull();
  });

  it("is refused when its content has the wrong shape", async () => {
    expect(await decodeLink(await linkFor({ ...snapshot, stakeholders: "green" }))).toBeNull();
    expect(await decodeLink(await linkFor({ ...snapshot, alternatives: [{ name: 1 }] }))).toBeNull();
    expect(
      await decodeLink(
        await linkFor({ ...snapshot, alternatives: [{ ...alternative("a"), values: { alpha: 1 } }] }),
      ),
    ).toBeNull();
  });

  it("is refused when it expands far past any real comparison", async () => {
    const bomb = await linkFor({ ...snapshot, alternatives: [alternative("x".repeat(200_000))] });
    expect(bomb.length).toBeLessThan(8_000);
    expect(await decodeLink(bomb)).toBeNull();
  });
});

describe("restoring against the current registry", () => {
  it("drops and reports what the registry no longer holds, keeping the rest", () => {
    const restored = restore(
      {
        ...snapshot,
        context: "retired",
        stakeholders: ["green", "gone"],
        alternatives: [
          {
            name: "First",
            values: { alpha: "0.5", beta: "2", grade: "3" },
            levels: { grade: "g9", alpha: "g1" },
          },
          { name: "Second", values: { alpha: "twelve" }, levels: { grade: "g1" } },
        ],
      },
      form,
    )!;

    expect(restored.shortlist.context).toBe("plain");
    expect(restored.shortlist.stakeholders).toEqual(["green"]);
    expect(restored.shortlist.columns.map((c) => [c.values, c.levels])).toEqual([
      [{ alpha: "0.5" }, {}],
      [{}, { grade: "g1" }],
    ]);
    expect(restored.notes).toHaveLength(1);
    for (const mention of ['"retired"', '"gone"', '"beta"', '"g9"', '"twelve"']) {
      expect(restored.notes[0]).toContain(mention);
    }
  });

  it("refuses a category the page does not offer", () => {
    expect(restore({ ...snapshot, category: "gizmo" }, form)).toBeNull();
  });

  it("falls back to a stakeholder when none survive", () => {
    const restored = restore({ ...snapshot, stakeholders: ["gone"] }, form)!;
    expect(restored.shortlist.stakeholders).toEqual(["thrifty"]);
  });

  it("keeps the widest shortlist the model accepts and pads a short one", () => {
    const wide = restore(
      { ...snapshot, alternatives: Array.from({ length: 7 }, (_, i) => alternative(`n${i}`)) },
      form,
    )!;
    expect(wide.shortlist.columns).toHaveLength(5);
    expect(wide.notes[0]).toContain("2 alternatives");

    const short = restore({ ...snapshot, alternatives: [alternative("only")] }, form)!;
    expect(short.shortlist.columns.map((c) => c.name)).toEqual(["only", "Option B"]);
  });

  it("says when the link was made under another registry", () => {
    const restored = restore({ ...snapshot, registry: "0.9" }, form)!;
    expect(restored.notes).toEqual(["This link was made under registry 0.9; the page runs 1.0."]);
  });
});
