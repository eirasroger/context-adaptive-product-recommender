import { useCallback, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";

import { api, type Comparison, type Example, type Form, unwrap } from "./api/client";
import { Grid } from "./components/Grid";
import { Result } from "./components/Result";
import { type Axis, Sensitivity } from "./components/Sensitivity";
import { Setup } from "./components/Setup";
import { Tip, type TipContent } from "./components/Tip";
import { type Draft, isStale, switchCategory, unscored } from "./drafts";
import { unitText } from "./format";
import { decodeLink, encodeLink, restore, snapshotOf, snapshotOfRequest } from "./link";
import {
  blankColumns,
  type Column,
  isFilled,
  isScorable,
  scoreRequest,
  type Shortlist,
} from "./shortlist";

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
  const [introducing, setIntroducing] = useState(false);
  const [filling, setFilling] = useState(false);
  const latest = useRef(0);
  const outcome = useRef<HTMLDivElement>(null);
  const alternatives = useRef<HTMLElement>(null);

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

  const runScore = useRef(() => {});
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        runScore.current();
      }
    };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  }, []);

  const category = form?.categories.find((c) => c.key === draft?.shortlist.category);
  if (!form || !draft || !category) {
    return (
      <div className="wrap">
        <Header form={null} />
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

  runScore.current = () => {
    if (scoring) return;
    if (!isScorable(draft.shortlist)) {
      setStatus(failure("Fill in at least two alternatives before scoring."));
      return;
    }
    void score(draft.shortlist);
  };

  const loadExample = async (example: Example) => {
    const restored = restore(snapshotOfRequest(example.request, form.registry_version), form);
    if (!restored) {
      setStatus(failure("The example no longer matches the registry."));
      return;
    }
    const still = stillMotion();
    setIntroducing(true);
    latest.current++;
    await withTransition(() => {
      setParked(new Map());
      setDraft(unscored(restored.shortlist));
      setLinkNotice(quiet);
      setStatus(quiet);
      setFilling(!still);
    });
    glideTo(alternatives.current, still);
    await Promise.all([score(restored.shortlist), still || delay(FILL_MS)]);
    setFilling(false);
    setIntroducing(false);
    requestAnimationFrame(() => glideTo(outcome.current, still));
  };
  const [example] = form.examples;

  const copyLink = async () => {
    const url = `${location.origin}${location.pathname}#${await encodeLink(
      snapshotOf(draft.shortlist, form.registry_version),
    )}`;
    history.replaceState(null, "", url);
    try {
      await navigator.clipboard.writeText(url);
      setStatus(note("Link copied. Opening it restores this comparison with every value you entered."));
    } catch {
      setStatus(failure("The browser would not copy. Copy the link from the address bar instead."));
    }
  };

  return (
    <div className="wrap">
      <Header form={form} />
      {linkNotice.text && (
        <div className={linkNotice.error ? "msg error" : "msg"}>{linkNotice.text}</div>
      )}

      {example && (
        <button
          type="button"
          className={introducing ? "example-strip busy" : "example-strip enter"}
          disabled={introducing}
          onClick={() => void loadExample(example)}
        >
          <span className="spark" aria-hidden="true">
            <svg viewBox="0 0 16 16">
              <path d="M5 3.5v9l7.5-4.5z" fill="currentColor" />
            </svg>
          </span>
          <span className="example-text">
            <strong>Try an example</strong>
            <span>{example.title}</span>
          </span>
          <span className="go" aria-hidden="true">
            <span className="spinner" />
            {introducing ? "Filling in" : "Fill in"}
            <svg viewBox="0 0 16 16">
              <path d="M3 8h9M8.5 4.5L12 8l-3.5 3.5" fill="none" stroke="currentColor" strokeWidth="1.6" />
            </svg>
          </span>
        </button>
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

      <section className="card" ref={alternatives}>
        <div className="card-head">
          <div>
            <h2>Alternatives</h2>
            <p className="card-sub">
              Values per {unitText(category.functional_unit_symbol)}. Leave a field blank when unknown.
            </p>
          </div>
        </div>
        <div className="gridwrap">
          <Grid
            category={category}
            families={form.families}
            columns={draft.shortlist.columns}
            filling={filling}
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
          <button
            className={scoring ? "primary busy" : "primary"}
            type="button"
            disabled={scoring}
            aria-keyshortcuts="Control+Enter Meta+Enter"
            onClick={() => runScore.current()}
          >
            <span className="spinner" aria-hidden="true" />
            <span>{scoring ? "Scoring" : "Score"}</span>
            <kbd>{shortcutKey()} ↵</kbd>
          </button>
        </div>
      </section>

      <div className={status.error ? "msg error" : "msg"} aria-live="polite">
        {status.text}
      </div>

      {draft.result && draft.scored && (
        <div
          ref={outcome}
          className={isStale(draft) ? "outcome enter stale" : "outcome enter"}
          key={category.key}
        >
          {isStale(draft) && (
            <p className="stale-note">The inputs have changed since this score. Score again to update it.</p>
          )}
          <Result
            result={draft.result}
            colours={draft.scored.columns.map((c) => c.colour)}
            wins={draft.wins}
            preview={category.preview}
            epsilon={form.tie_epsilon}
          />
          <Sensitivity
            axis={axis}
            onAxis={setAxis}
            form={form}
            category={category}
            scored={draft.scored}
            byContext={draft.byContext}
            byStakeholder={draft.byStakeholder}
            epsilon={form.tie_epsilon}
          />
        </div>
      )}

      <Footer form={form} />
      <Tip tip={tip} />
    </div>
  );
}

/** Long enough for the fill wave in the stylesheet to reach the last cell. */
const FILL_MS = 1800;

const delay = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

const glideTo = (element: Element | null, still: boolean) =>
  element?.scrollIntoView({ behavior: still ? "auto" : "smooth", block: "start" });

const stillMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Cross-fades the page between states where the browser can, and applies the change at once elsewhere; settles once the change is applied. */
function withTransition(change: () => void): Promise<void> {
  if (stillMotion() || typeof document.startViewTransition !== "function") {
    change();
    return Promise.resolve();
  }
  return document.startViewTransition(() => flushSync(change)).updateCallbackDone;
}

function Header({ form }: { form: Form | null }) {
  return (
    <header className="masthead">
      <div className="brand">
        <svg className="mark" viewBox="0 0 32 32" aria-hidden="true">
          <rect width="32" height="32" rx="9" fill="currentColor" />
          <path d="M9 21.5h4.2V17H9zM14 21.5h4.2v-8H14zM19 21.5h4.2V10.5H19z" fill="var(--ink-contrast)" />
        </svg>
        <div className="brand-text">
          <h1>Product Comparison</h1>
          <span className="tagline">Context-adaptive ranking of building products</span>
        </div>
      </div>
      <span className="spacer" />
      {form && (
        <span className="status-pill" title={`Snapshot ${form.snapshot ?? "unknown"}`}>
          <span className="live" aria-hidden="true" />
          Model live · registry {form.registry_version ?? "unknown"}
        </span>
      )}
    </header>
  );
}

const shortcutKey = () => (/Mac|iPhone|iPad/.test(navigator.platform) ? "⌘" : "Ctrl");

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
  if (!category) throw new Error(`the registry has no category ${key}`);
  return { category: key, context: category.default_context, stakeholders, columns: blankColumns() };
}

function defaultShortlist(form: Form): Shortlist {
  const [category] = form.categories;
  const [stakeholder] = form.stakeholders;
  if (!category || !stakeholder) throw new Error("the registry has nothing to compare");
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
