import { useCallback, useEffect, useRef, useState } from "react";

import { api, type Comparison, type Form, type ScoreResponse, unwrap } from "./api/client";
import { Grid } from "./components/Grid";
import { Result } from "./components/Result";
import { Setup } from "./components/Setup";
import { Tip, type TipContent } from "./components/Tip";
import { decodeLink, encodeLink, restore, snapshotOf } from "./link";
import {
  blankColumns,
  type Column,
  isFilled,
  isScorable,
  scoreRequest,
  type Shortlist,
} from "./shortlist";
import { toggleTheme } from "./theme";

type Status = { text: string; error: boolean };

const quiet: Status = { text: "", error: false };
const note = (text: string): Status => ({ text, error: false });
const failure = (text: string): Status => ({ text, error: true });
const messageOf = (err: unknown) => (err instanceof Error ? err.message : String(err));

// Safari refuses more than 100 history updates in 30 seconds, and typing makes one per key.
const URL_UPDATE_DELAY_MS = 400;

export function App() {
  const [form, setForm] = useState<Form | null>(null);
  const [shortlist, setShortlist] = useState<Shortlist | null>(null);
  const [result, setResult] = useState<ScoreResponse | null>(null);
  const [wins, setWins] = useState(new Map<number, string>());
  const [scoring, setScoring] = useState(false);
  const [status, setStatus] = useState(quiet);
  const [linkNotice, setLinkNotice] = useState<Status>(quiet);
  const [tip, setTip] = useState<TipContent | null>(null);
  const latest = useRef(0);

  const score = useCallback(async (target: Shortlist) => {
    const body = scoreRequest(target);
    const request = ++latest.current;
    setStatus(note("Scoring."));
    setScoring(true);
    try {
      setResult(await unwrap(api.POST("/api/score", { body })));
      setWins(new Map());
      setStatus(quiet);
    } catch (err) {
      setStatus(failure(messageOf(err)));
      return;
    } finally {
      setScoring(false);
    }
    try {
      const comparison = await unwrap(api.POST("/api/explore/compare", { body }));
      if (request === latest.current) setWins(annotations(comparison));
    } catch {}
  }, []);

  const open = useCallback(
    async (loaded: Form, fragment: string) => {
      const fresh = defaultShortlist(loaded);
      if (!fragment) {
        setShortlist(fresh);
        return;
      }
      const snapshot = await decodeLink(fragment);
      const restored = snapshot && restore(snapshot, loaded);
      setResult(null);
      if (!restored) {
        setShortlist(fresh);
        setLinkNotice(failure("This link could not be read, so the page opened empty."));
        return;
      }
      setShortlist(restored.shortlist);
      setLinkNotice(note(restored.notes.join(" ")));
      if (isScorable(restored.shortlist)) await score(restored.shortlist);
    },
    [score],
  );

  useEffect(() => {
    let current = true;
    unwrap(api.GET("/api/explore/form"))
      .then((loaded) => {
        if (!current) return;
        setForm(loaded);
        return open(loaded, location.hash);
      })
      .catch((err) => current && setStatus(failure(`Could not reach the model: ${messageOf(err)}`)));
    return () => {
      current = false;
    };
  }, [open]);

  useEffect(() => {
    if (!form) return;
    const reopen = () => open(form, location.hash);
    addEventListener("hashchange", reopen);
    return () => removeEventListener("hashchange", reopen);
  }, [form, open]);

  useEffect(() => {
    if (!form || !shortlist) return;
    const timer = setTimeout(async () => {
      const fragment = shortlist.columns.some(isFilled)
        ? `#${await encodeLink(snapshotOf(shortlist, form.registry_version))}`
        : "";
      if (fragment !== location.hash) {
        history.replaceState(null, "", `${location.pathname}${location.search}${fragment}`);
      }
    }, URL_UPDATE_DELAY_MS);
    return () => clearTimeout(timer);
  }, [form, shortlist]);

  useEffect(() => {
    const hide = () => setTip(null);
    addEventListener("scroll", hide, { passive: true });
    return () => removeEventListener("scroll", hide);
  }, []);

  const category = form?.categories.find((c) => c.key === shortlist?.category);
  if (!form || !shortlist || !category) {
    return (
      <div className="wrap">
        <Header form={form} />
        <div className={status.error ? "msg error" : "msg"}>{status.text}</div>
      </div>
    );
  }

  const update = (change: Partial<Shortlist>) => setShortlist({ ...shortlist, ...change });

  const chooseCategory = (key: string) => {
    const chosen = form.categories.find((c) => c.key === key);
    if (!chosen) return;
    update({ category: key, context: chosen.default_context, columns: blankColumns() });
    setResult(null);
  };

  const clear = () => {
    update({ columns: blankColumns(shortlist.columns.length) });
    setResult(null);
    setLinkNotice(quiet);
  };

  const runScore = () => {
    if (!isScorable(shortlist)) {
      setStatus(failure("Fill in at least two alternatives before scoring."));
      return;
    }
    void score(shortlist);
  };

  const copyLink = async () => {
    const url = `${location.origin}${location.pathname}#${await encodeLink(
      snapshotOf(shortlist, form.registry_version),
    )}`;
    history.replaceState(null, "", url);
    try {
      await navigator.clipboard.writeText(url);
      setStatus(note("Link copied. It holds the whole comparison."));
    } catch {
      setStatus(failure("The browser would not copy. The address bar holds the link."));
    }
  };

  return (
    <div className="wrap">
      <Header form={form} />
      {linkNotice.text && (
        <div className={linkNotice.error ? "msg error" : "msg"}>{linkNotice.text}</div>
      )}

      <Setup
        form={form}
        category={category}
        context={shortlist.context}
        stakeholders={shortlist.stakeholders}
        onCategory={chooseCategory}
        onContext={(context) => update({ context })}
        onStakeholders={(stakeholders) => update({ stakeholders })}
      />

      <div className="card">
        <h2>Alternatives</h2>
        <div className="gridwrap">
          <Grid
            category={category}
            families={form.families}
            columns={shortlist.columns}
            onColumns={(columns: Column[]) => update({ columns })}
            onTip={setTip}
          />
        </div>
        <div className="actions">
          <button type="button" onClick={clear}>
            Clear all
          </button>
          <button type="button" onClick={copyLink}>
            Copy link
          </button>
          <span className="spacer" />
          <span className="hint">Leave a field blank to say the value is unknown.</span>
          <button className="primary" type="button" disabled={scoring} onClick={runScore}>
            Score
          </button>
        </div>
      </div>

      {result && <Result result={result} wins={wins} />}

      <div className={status.error ? "msg error" : "msg"}>{status.text}</div>
      <Tip tip={tip} />
    </div>
  );
}

function Header({ form }: { form: Form | null }) {
  return (
    <header>
      <h1>Product Comparison</h1>
      {form && (
        <span className="build">
          registry {form.registry_version ?? "?"} · snapshot {(form.snapshot ?? "").slice(0, 12)}
        </span>
      )}
      <span className="spacer" />
      <button type="button" onClick={toggleTheme}>
        Theme
      </button>
    </header>
  );
}

function defaultShortlist(form: Form): Shortlist {
  const [category] = form.categories;
  const [stakeholder] = form.stakeholders;
  if (!category || !stakeholder) throw new Error("the registry holds nothing to compare");
  return {
    category: category.key,
    context: category.default_context,
    stakeholders: [stakeholder.key],
    columns: blankColumns(),
  };
}

function annotations(comparison: Comparison): Map<number, string> {
  const notes = new Map<number, string>();
  comparison.scores.forEach((_, index) => {
    const named = (comparison.wins[String(index)] ?? []).slice(0, 2);
    if (named.length > 0) {
      notes.set(index, `wins on ${named.map((n) => n.toLowerCase()).join(", ")}`);
    } else if (index === comparison.leader_index) {
      notes.set(index, "best balance overall");
    }
  });
  return notes;
}
