import { useCallback, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";

import { api, type Comparison, type Form, unwrap } from "./api/client";
import { Grid } from "./components/Grid";
import { Result } from "./components/Result";
import { type Axis, Sensitivity } from "./components/Sensitivity";
import { Setup } from "./components/Setup";
import { Tip, type TipContent } from "./components/Tip";
import { type Draft, isStale, switchCategory, unscored } from "./drafts";
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

// Safari allows 100 history updates per 30 seconds, and typing makes one per key.
const URL_UPDATE_DELAY_MS = 400;

export function App() {
  const [form, setForm] = useState<Form | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [parked, setParked] = useState(new Map<string, Draft>());
  const [axis, setAxis] = useState<Axis>("context");
  const [scoring, setScoring] = useState(false);
  const [status, setStatus] = useState(quiet);
  const [linkNotice, setLinkNotice] = useState<Status>(quiet);
  const [tip, setTip] = useState<TipContent | null>(null);
  const latest = useRef(0);

  const patch = useCallback((scored: Shortlist, change: Partial<Draft>) => {
    setDraft((d) => (d && d.scored === scored ? { ...d, ...change } : d));
  }, []);

  const score = useCallback(
    async (target: Shortlist) => {
      const body = scoreRequest(target);
      const request = ++latest.current;
      setStatus(note("Scoring."));
      setScoring(true);
      try {
        const result = await unwrap(api.POST("/api/score", { body }));
        if (request !== latest.current) return;
        setDraft((d) =>
          d && d.shortlist.category === target.category
            ? { ...unscored(d.shortlist), scored: target, result }
            : d,
        );
        setStatus(quiet);
      } catch (err) {
        setStatus(failure(messageOf(err)));
        return;
      } finally {
        setScoring(false);
      }
      const [comparison, byContext] = await Promise.allSettled([
        unwrap(api.POST("/api/explore/compare", { body })),
        unwrap(api.POST("/api/explore/context-sensitivity", { body })),
      ]);
      patch(target, {
        ...(comparison.status === "fulfilled" && { wins: annotations(comparison.value) }),
        ...(byContext.status === "fulfilled" && { byContext: byContext.value }),
      });
    },
    [patch],
  );

  const open = useCallback(
    async (loaded: Form, fragment: string) => {
      const fresh = unscored(defaultShortlist(loaded));
      setParked(new Map());
      if (!fragment) {
        setDraft(fresh);
        return;
      }
      const snapshot = await decodeLink(fragment);
      const restored = snapshot && restore(snapshot, loaded);
      if (!restored) {
        setDraft(fresh);
        setLinkNotice(failure("This link could not be read, so the page opened empty."));
        return;
      }
      setDraft(unscored(restored.shortlist));
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

  const shortlist = draft?.shortlist;
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

  const scored = draft?.scored;
  const needsStakeholders = axis === "stakeholder" && !!scored && !draft?.byStakeholder;
  useEffect(() => {
    if (!needsStakeholders || !scored) return;
    unwrap(api.POST("/api/explore/stakeholder-sensitivity", { body: scoreRequest(scored) }))
      .then((byStakeholder) => patch(scored, { byStakeholder }))
      .catch((err) => setStatus(failure(messageOf(err))));
  }, [needsStakeholders, scored, patch]);

  const category = form?.categories.find((c) => c.key === draft?.shortlist.category);
  if (!form || !draft || !category) {
    return (
      <div className="wrap">
        <Header />
        <div className={status.error ? "msg error" : "msg"}>{status.text}</div>
      </div>
    );
  }

  const edit = (change: Partial<Shortlist>) =>
    setDraft({ ...draft, shortlist: { ...draft.shortlist, ...change } });

  const chooseCategory = (key: string) => {
    if (key === category.key) return;
    latest.current++;
    const switched = switchCategory(parked, draft, key, (target) =>
      shortlistFor(form, target, draft.shortlist.stakeholders),
    );
    withTransition(() => {
      setParked(switched.parked);
      setDraft(switched.next);
      setStatus(quiet);
      setLinkNotice(quiet);
    });
  };

  const clear = () => {
    edit({ columns: blankColumns(draft.shortlist.columns.length) });
    setLinkNotice(quiet);
  };

  const runScore = () => {
    if (!isScorable(draft.shortlist)) {
      setStatus(failure("Fill in at least two alternatives before scoring."));
      return;
    }
    void score(draft.shortlist);
  };

  const copyLink = async () => {
    const url = `${location.origin}${location.pathname}#${await encodeLink(
      snapshotOf(draft.shortlist, form.registry_version),
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
      <Header />
      {linkNotice.text && (
        <div className={linkNotice.error ? "msg error" : "msg"}>{linkNotice.text}</div>
      )}

      <Setup
        form={form}
        category={category}
        context={draft.shortlist.context}
        stakeholders={draft.shortlist.stakeholders}
        onCategory={chooseCategory}
        onContext={(context) => edit({ context })}
        onStakeholders={(stakeholders) => edit({ stakeholders })}
      />

      <section className="card">
        <h2>Alternatives</h2>
        <div className="gridwrap">
          <Grid
            category={category}
            families={form.families}
            columns={draft.shortlist.columns}
            onColumns={(columns: Column[]) => edit({ columns })}
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
            {scoring ? "Scoring" : "Score"}
          </button>
        </div>
      </section>

      <div className={status.error ? "msg error" : "msg"} aria-live="polite">
        {status.text}
      </div>

      {draft.result && draft.scored && (
        <div className={isStale(draft) ? "outcome enter stale" : "outcome enter"} key={category.key}>
          {isStale(draft) && (
            <p className="stale-note">The inputs have changed since this score. Score again to update it.</p>
          )}
          <Result result={draft.result} wins={draft.wins} preview={category.preview} />
          <Sensitivity
            axis={axis}
            onAxis={setAxis}
            form={form}
            category={category}
            scored={draft.scored}
            byContext={draft.byContext}
            byStakeholder={draft.byStakeholder}
          />
        </div>
      )}

      <Footer form={form} />
      <Tip tip={tip} />
    </div>
  );
}

/** Cross-fades the page between states where the browser can, and applies the change at once elsewhere. */
function withTransition(change: () => void) {
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (still || typeof document.startViewTransition !== "function") {
    change();
    return;
  }
  document.startViewTransition(() => flushSync(change));
}

function Header() {
  return (
    <header className="masthead">
      <div>
        <h1>Product Comparison</h1>
        <p className="lede">
          Score a shortlist of building products for one application and one set of priorities.
        </p>
      </div>
      <button type="button" className="icon" aria-label="Switch colour theme" onClick={toggleTheme}>
        <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
          <circle cx="10" cy="10" r="7" fill="none" stroke="currentColor" strokeWidth="1.6" />
          <path d="M10 3a7 7 0 0 1 0 14z" fill="currentColor" />
        </svg>
      </button>
    </header>
  );
}

function Footer({ form }: { form: Form }) {
  return (
    <footer className="footer">
      <span>
        Registry {form.registry_version ?? "unknown"} · snapshot{" "}
        {(form.snapshot ?? "").slice(0, 12) || "unknown"}
      </span>
      <span className="spacer" />
      <a href="https://doi.org/10.1016/j.spc.2026.06.011" target="_blank" rel="noreferrer">
        Paper
      </a>
      <a href="https://doi.org/10.34810/DATA3164" target="_blank" rel="noreferrer">
        Dataset
      </a>
    </footer>
  );
}

function shortlistFor(form: Form, key: string, stakeholders: string[]): Shortlist {
  const category = form.categories.find((c) => c.key === key);
  if (!category) throw new Error(`the registry holds no category ${key}`);
  return { category: key, context: category.default_context, stakeholders, columns: blankColumns() };
}

function defaultShortlist(form: Form): Shortlist {
  const [category] = form.categories;
  const [stakeholder] = form.stakeholders;
  if (!category || !stakeholder) throw new Error("the registry holds nothing to compare");
  return shortlistFor(form, category.key, [stakeholder.key]);
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
