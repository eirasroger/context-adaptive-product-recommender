import { useEffect, useRef, useState } from "react";

import { api, type Comparison, type Form, type ScoreResponse, unwrap } from "./api/client";
import { Grid } from "./components/Grid";
import { Result } from "./components/Result";
import { Setup } from "./components/Setup";
import { Tip, type TipContent } from "./components/Tip";
import { blankColumn, blankShortlist, type Column, isFilled, scoreRequest } from "./shortlist";
import { toggleTheme } from "./theme";

type Status = { text: string; error: boolean };

const quiet: Status = { text: "", error: false };
const failure = (text: string): Status => ({ text, error: true });
const messageOf = (err: unknown) => (err instanceof Error ? err.message : String(err));

export function App() {
  const [form, setForm] = useState<Form | null>(null);
  const [categoryKey, setCategoryKey] = useState("");
  const [context, setContext] = useState("");
  const [stakeholders, setStakeholders] = useState<string[]>([]);
  const [columns, setColumns] = useState<Column[]>(blankShortlist);
  const [result, setResult] = useState<ScoreResponse | null>(null);
  const [wins, setWins] = useState(new Map<number, string>());
  const [scoring, setScoring] = useState(false);
  const [status, setStatus] = useState(quiet);
  const [tip, setTip] = useState<TipContent | null>(null);
  const latest = useRef(0);

  useEffect(() => {
    let current = true;
    unwrap(api.GET("/api/explore/form"))
      .then((loaded) => {
        if (!current) return;
        const [first] = loaded.categories;
        const [stakeholder] = loaded.stakeholders;
        if (!first || !stakeholder) throw new Error("the registry holds nothing to compare");
        setForm(loaded);
        setCategoryKey(first.key);
        setContext(first.default_context);
        setStakeholders([stakeholder.key]);
      })
      .catch((err) => current && setStatus(failure(`Could not reach the model: ${messageOf(err)}`)));
    return () => {
      current = false;
    };
  }, []);

  useEffect(() => {
    const hide = () => setTip(null);
    addEventListener("scroll", hide, { passive: true });
    return () => removeEventListener("scroll", hide);
  }, []);

  const category = form?.categories.find((c) => c.key === categoryKey);

  const chooseCategory = (key: string) => {
    const chosen = form?.categories.find((c) => c.key === key);
    if (!chosen) return;
    setCategoryKey(key);
    setContext(chosen.default_context);
    setColumns(blankShortlist());
    setResult(null);
  };

  const clear = () => {
    setColumns(columns.map((_, index) => blankColumn(index)));
    setResult(null);
  };

  const score = async () => {
    if (columns.filter(isFilled).length < 2) {
      setStatus(failure("Fill in at least two alternatives before scoring."));
      return;
    }
    const body = scoreRequest(categoryKey, context, stakeholders, columns);
    const request = ++latest.current;
    setStatus({ text: "Scoring.", error: false });
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
  };

  return (
    <div className="wrap">
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

      {form && category && (
        <>
          <Setup
            form={form}
            category={category}
            context={context}
            stakeholders={stakeholders}
            onCategory={chooseCategory}
            onContext={setContext}
            onStakeholders={setStakeholders}
          />

          <div className="card">
            <h2>Alternatives</h2>
            <div className="gridwrap">
              <Grid
                category={category}
                families={form.families}
                columns={columns}
                onColumns={setColumns}
                onTip={setTip}
              />
            </div>
            <div className="actions">
              <button type="button" onClick={clear}>
                Clear all
              </button>
              <span className="spacer" />
              <span className="hint">Leave a field blank to say the value is unknown.</span>
              <button className="primary" type="button" disabled={scoring} onClick={score}>
                Score
              </button>
            </div>
          </div>

          {result && <Result result={result} wins={wins} />}
        </>
      )}

      <div className={status.error ? "msg error" : "msg"}>{status.text}</div>
      <Tip tip={tip} />
    </div>
  );
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
