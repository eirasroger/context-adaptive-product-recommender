import { useState } from "react";

import type { FormCategory } from "../api/client";
import { baselineOf, switchable, toggle } from "../contexts";
import { stagger } from "../format";

type Props = {
  category: FormCategory;
  contexts: string[];
  onContexts: (keys: string[]) => void;
};

const Tick = () => (
  <svg className="tick" viewBox="0 0 16 16" aria-hidden="true">
    <path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" strokeWidth="2" />
  </svg>
);

export function ContextPicker({ category, contexts, onContexts }: Props) {
  const [baselineOn, setBaselineOn] = useState(contexts.length === 0);
  const baseline = baselineOf(category);
  const chosen = category.contexts.filter((c) => contexts.includes(c.key));
  const full = contexts.length >= category.max_contexts;
  let index = 1;

  const switchBaseline = () => {
    if (!baselineOn) onContexts([]);
    setBaselineOn(!baselineOn);
  };

  return (
    <>
      <div className="context-groups" role="group" aria-labelledby="context-label">
        <div className="chips">
          <button
            type="button"
            className="chip enter"
            style={stagger(0)}
            title={baseline?.definition}
            aria-pressed={baselineOn}
            onClick={switchBaseline}
          >
            <Tick />
            Baseline
          </button>
        </div>
        {baselineOn && baseline && <p className="definition">{baseline.definition}</p>}
        {switchable(category).map((group) => (
          <div className="context-group" key={group.theme ?? "all"}>
            {group.theme && <span className="context-theme">{group.theme}</span>}
            <div className="chips">
              {group.contexts.map((c) => {
                const on = contexts.includes(c.key);
                return (
                  <button
                    key={c.key}
                    type="button"
                    className="chip enter"
                    style={stagger(index++)}
                    title={c.definition}
                    aria-pressed={on}
                    disabled={baselineOn || (!on && full)}
                    onClick={() => onContexts(toggle(contexts, c.key, category.max_contexts))}
                  >
                    <Tick />
                    {c.display_name}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      {chosen.map((c) => (
        <p className="definition" key={c.key}>
          <strong>{c.display_name}.</strong> {c.definition}
        </p>
      ))}
    </>
  );
}
