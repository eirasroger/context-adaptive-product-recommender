import type { Form, FormCategory } from "../api/client";
import { stagger } from "../format";
import { CategoryPicker } from "./CategoryPicker";

type Props = {
  form: Form;
  category: FormCategory;
  context: string;
  stakeholders: string[];
  onCategory: (key: string) => void;
  onContext: (key: string) => void;
  onStakeholders: (keys: string[]) => void;
};

export function Setup({
  form,
  category,
  context,
  stakeholders,
  onCategory,
  onContext,
  onStakeholders,
}: Props) {
  const toggle = (key: string) => {
    if (!stakeholders.includes(key)) onStakeholders([...stakeholders, key]);
    else if (stakeholders.length > 1) onStakeholders(stakeholders.filter((s) => s !== key));
  };
  const chosen = category.contexts.find((c) => c.key === context);

  return (
    <section className="card">
      <h2>Setup</h2>
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
            Application
          </span>
          <div className="chips" role="radiogroup" aria-labelledby="context-label" key={category.key}>
            {category.contexts.map((c, index) => (
              <button
                key={c.key}
                type="button"
                role="radio"
                className="chip enter"
                style={stagger(index)}
                aria-checked={c.key === context}
                onClick={() => onContext(c.key)}
              >
                {c.display_name}
              </button>
            ))}
          </div>
          {chosen && <p className="definition">{chosen.definition}</p>}
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
                {s.display_name}
              </button>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
