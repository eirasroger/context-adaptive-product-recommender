import { describe, expect, it } from "vitest";

import { withoutFragment } from "./analytics";

describe("an analytics event", () => {
  it("never carries the shared comparison in the fragment", () => {
    const event = { type: "pageview" as const, url: "https://example.org/?ref=mail#v1.abcDEF-_" };
    expect(withoutFragment(event).url).toBe("https://example.org/?ref=mail");
  });

  it("is otherwise sent as it was", () => {
    const event = { type: "pageview" as const, url: "https://example.org/" };
    expect(withoutFragment(event)).toEqual(event);
  });
});
