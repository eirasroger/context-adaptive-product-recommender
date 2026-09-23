import type { BeforeSendEvent } from "@vercel/analytics/react";

/** Drops the fragment, which holds a whole shared comparison. */
export const withoutFragment = (event: BeforeSendEvent): BeforeSendEvent => ({
  ...event,
  url: event.url.split("#")[0] ?? event.url,
});
