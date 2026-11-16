import type { Form, FormCategory } from "../api/client";
import { baselineOf, switchable, toggle as toggleContext } from "../contexts";
import { stagger, unitText } from "../format";
import { CategoryPicker } from "./CategoryPicker";

type Props = {
  form: Form;
  category: FormCategory;
  contexts: string[];
  stakeholders: string[];
  onCategory: (key: string) => void;
  onContexts: (keys: string[]) => void;
  onStakeholders: (keys: string[]) => void;
};

export function Setup({
  form,
  category,
  contexts,
  stakeholders,
  onCategory,
  onContexts,
  onStakeholders,
}: Props) {
  const toggle = (key: string) => {
    if (!stakeholders.includes(key)) onStakeholders([...stakeholders, key]);
    else if (stakeholders.length > 1) onStakeholders(stakeholders.filter((s) => s !== key));
  };
  const groups = switchable(category);
  const baseline = baselineOf(category);
  const chosen = category.contexts.filter((c) => contexts.includes(c.key));
  const full = contexts.length >= category.max_contexts;
  let index = 0;

  return (
    <section className="card">
      <h2>Setup</h2>
      <div className="setup-layout">
        <div className="setup">
          <div className="field">
            <span className="label" id="category-label">
              Product category
            </span>
            <CategoryPicker categories={form.categories} active={category} onChoose={onCategory} />
            {category.preview && (
              <div className="notice enter" role="note" key={category.key}>
                <span className="badge">Preview</span>
                <p>
                  {category.display_name} is trained on a synthetic working dataset with rule-based
                  labels while real products are collected. Its scores show how the model applies the
                  declared rules; experts have yet to validate them.
                </p>
              </div>
            )}
          </div>

          <div className="field">
            <span className="label" id="context-label">
              Situation
              <span className="label-hint">
                {" "}
                · up to {category.max_contexts}; none means {baseline?.display_name.toLowerCase() ?? "the baseline"}
              </span>
            </span>
            <div className="context-groups" role="group" aria-labelledby="context-label" key={category.key}>
              {groups.map((group) => (
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
                          disabled={!on && full}
                          onClick={() => onContexts(toggleContext(contexts, c.key, category.max_contexts))}
                        >
                          <svg className="tick" viewBox="0 0 16 16" aria-hidden="true">
                            <path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" strokeWidth="2" />
                          </svg>
                          {c.display_name}
                        </button>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
            {chosen.length === 0 && baseline && <p className="definition">{baseline.definition}</p>}
            {chosen.map((c) => (
              <p className="definition" key={c.key}>
                <strong>{c.display_name}.</strong> {c.definition}
              </p>
            ))}
          </div>

          <div className="field">
            <span className="label" id="priorities-label">
              Whose priorities count
            </span>
            <div className="chips" role="group" aria-labelledby="priorities-label">
              {form.stakeholders.map((s) => (
                <button
                  key={s.key}
                  type="button"
                  className="chip"
                  title={s.definition}
                  aria-pressed={stakeholders.includes(s.key)}
                  onClick={() => toggle(s.key)}
                >
                  <svg className="tick" viewBox="0 0 16 16" aria-hidden="true">
                    <path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" strokeWidth="2" />
                  </svg>
                  {s.display_name}
                </button>
              ))}
            </div>
          </div>
        </div>
        <aside className="reference-unit enter" key={category.key} aria-label="Reference unit">
          <span className="reference-unit-label">Reference unit</span>
          <span className="reference-unit-value">{unitText(category.functional_unit_symbol)}</span>
        </aside>
      </div>
    </section>
  );
}
