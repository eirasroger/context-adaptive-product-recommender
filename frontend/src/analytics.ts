import type { BeforeSendEvent } from "@vercel/analytics/react";

/** Vercel's script reports location.href, and the fragment holds a whole shared comparison. */
export const withoutFragment = (event: BeforeSendEvent): BeforeSendEvent => ({
  ...event,
  url: event.url.split("#")[0] ?? event.url,
});
