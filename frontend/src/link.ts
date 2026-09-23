import type { Form } from "./api/client";
import { blankColumn, column, MAX_COLUMNS, MIN_COLUMNS, type Shortlist } from "./shortlist";

const FORMAT = "v1";
const MAX_LINK_LENGTH = 8_000;
const MAX_DECODED_BYTES = 64_000;
const MAX_ALTERNATIVES_READ = 20;

type Entries = Record<string, string>;

export type Snapshot = {
  registry: string | null;
  category: string;
  context: string;
  stakeholders: string[];
  alternatives: { name: string; values: Entries; levels: Entries }[];
};

export type Restored = { shortlist: Shortlist; notes: string[] };

export const snapshotOf = (shortlist: Shortlist, registry: string | null): Snapshot => ({
  registry,
  category: shortlist.category,
  context: shortlist.context,
  stakeholders: shortlist.stakeholders,
  alternatives: shortlist.columns.map(({ name, values, levels }) => ({ name, values, levels })),
});

export async function encodeLink(snapshot: Snapshot): Promise<string> {
  const json = new TextEncoder().encode(JSON.stringify(snapshot));
  const packed = await transform(json, new CompressionStream("deflate-raw"));
  return `${FORMAT}.${toBase64Url(packed)}`;
}

export async function decodeLink(fragment: string): Promise<Snapshot | null> {
  const code = fragment.replace(/^#/, "");
  if (!code.startsWith(`${FORMAT}.`) || code.length > MAX_LINK_LENGTH) return null;
  try {
    const packed = fromBase64Url(code.slice(FORMAT.length + 1));
    const json = await transform(packed, new DecompressionStream("deflate-raw"), MAX_DECODED_BYTES);
    const parsed: unknown = JSON.parse(new TextDecoder().decode(json));
    return isSnapshot(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

export function restore(snapshot: Snapshot, form: Form): Restored | null {
  const category = form.categories.find((c) => c.key === snapshot.category);
  const [fallbackStakeholder] = form.stakeholders;
  if (!category || !fallbackStakeholder) return null;

  const dropped: string[] = [];
  const quoted = JSON.stringify;

  let context = snapshot.context;
  if (!category.contexts.some((c) => c.key === context)) {
    dropped.push(`context ${quoted(context)}`);
    context = category.default_context;
  }

  const known = new Set(form.stakeholders.map((s) => s.key));
  let stakeholders = [...new Set(snapshot.stakeholders)].filter((key) => {
    if (!known.has(key)) dropped.push(`stakeholder ${quoted(key)}`);
    return known.has(key);
  });
  if (stakeholders.length === 0) stakeholders = [fallbackStakeholder.key];

  const fields = new Map(category.fields.map((field) => [field.key, field]));
  const columns = snapshot.alternatives.slice(0, MAX_COLUMNS).map((alternative) => {
    const values: Entries = {};
    for (const [key, value] of Object.entries(alternative.values)) {
      const field = fields.get(key);
      if (field && field.levels.length === 0 && isNumber(value)) values[key] = value;
      else dropped.push(`${quoted(key)} = ${quoted(value)} in ${alternative.name}`);
    }
    const levels: Entries = {};
    for (const [key, level] of Object.entries(alternative.levels)) {
      if (fields.get(key)?.levels.some((l) => l.key === level)) levels[key] = level;
      else dropped.push(`${quoted(key)} = ${quoted(level)} in ${alternative.name}`);
    }
    return column(alternative.name, values, levels);
  });
  const extra = snapshot.alternatives.length - MAX_COLUMNS;
  if (extra > 0) dropped.push(`${extra} alternative${extra === 1 ? "" : "s"} past the ${MAX_COLUMNS}th`);
  while (columns.length < MIN_COLUMNS) columns.push(blankColumn(columns.length));

  const notes: string[] = [];
  if (snapshot.registry !== form.registry_version) {
    notes.push(
      `This link was made under registry ${snapshot.registry ?? "unknown"}; ` +
        `the page runs ${form.registry_version ?? "unknown"}.`,
    );
  }
  if (dropped.length > 0) notes.push(`Left out as unrecognised: ${dropped.join(", ")}.`);

  return { shortlist: { category: category.key, context, stakeholders, columns }, notes };
}

const isNumber = (text: string) => text.trim() !== "" && Number.isFinite(Number(text));

const isEntries = (value: unknown): value is Entries =>
  typeof value === "object" &&
  value !== null &&
  !Array.isArray(value) &&
  Object.values(value).every((v) => typeof v === "string");

function isSnapshot(value: unknown): value is Snapshot {
  if (typeof value !== "object" || value === null) return false;
  const s = value as Record<string, unknown>;
  return (
    (s.registry === null || typeof s.registry === "string") &&
    typeof s.category === "string" &&
    typeof s.context === "string" &&
    Array.isArray(s.stakeholders) &&
    s.stakeholders.every((key) => typeof key === "string") &&
    Array.isArray(s.alternatives) &&
    s.alternatives.length <= MAX_ALTERNATIVES_READ &&
    s.alternatives.every(
      (a: unknown) =>
        typeof a === "object" &&
        a !== null &&
        typeof (a as Record<string, unknown>).name === "string" &&
        isEntries((a as Record<string, unknown>).values) &&
        isEntries((a as Record<string, unknown>).levels),
    )
  );
}

async function transform(
  input: Uint8Array<ArrayBuffer>,
  stream: CompressionStream | DecompressionStream,
  limit = Number.POSITIVE_INFINITY,
): Promise<Uint8Array<ArrayBuffer>> {
  const reader = new Blob([input]).stream().pipeThrough(stream).getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.length;
    if (size > limit) {
      await reader.cancel();
      throw new Error("the link expands past the size any comparison needs");
    }
    chunks.push(value);
  }
  const joined = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    joined.set(chunk, offset);
    offset += chunk.length;
  }
  return joined;
}

function toBase64Url(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function fromBase64Url(text: string): Uint8Array<ArrayBuffer> {
  const binary = atob(text.replace(/-/g, "+").replace(/_/g, "/"));
  return Uint8Array.from(binary, (char) => char.charCodeAt(0));
}
