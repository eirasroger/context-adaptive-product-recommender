import createClient from "openapi-fetch";

import type { components, paths } from "./schema";

type Schemas = components["schemas"];

export type Form = Schemas["Form"];
export type FormCategory = Schemas["FormCategory"];
export type FormField = Schemas["FormField"];
export type ShortlistRequest = Schemas["ShortlistRequest"];
export type ScoreResponse = Schemas["ScoreResponse"];
export type Comparison = Schemas["Comparison"];
export type Sensitivity = Schemas["Sensitivity"];

export const api = createClient<paths>();

type Outcome<T> = { data?: T; error?: unknown; response: Response };

export async function unwrap<T>(pending: Promise<Outcome<T>>): Promise<T> {
  const { data, error, response } = await pending;
  if (data === undefined) throw new Error(detailOf(error) ?? response.statusText);
  return data;
}

function detailOf(error: unknown): string | undefined {
  if (typeof error !== "object" || error === null || !("detail" in error)) return undefined;
  return typeof error.detail === "string" ? error.detail : undefined;
}
