import type { Form, FormCategory } from "../api/client";

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

  return (
    <div className="card">
      <h2>Setup</h2>
      <div className="setup">
        <div className="field">
          <label htmlFor="category">Category</label>
          <select id="category" value={category.key} onChange={(e) => onCategory(e.target.value)}>
            {form.categories.map((c) => (
              <option key={c.key} value={c.key}>
                {c.display_name}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="context">Context</label>
          <select id="context" value={context} onChange={(e) => onContext(e.target.value)}>
            {category.contexts.map((c) => (
              <option key={c.key} value={c.key}>
                {c.display_name}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>Priorities</label>
          <div className="chips">
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
    </div>
  );
}
