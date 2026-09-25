import type { Form, FormCategory, Sensitivity as Matrix } from "../api/client";
import { fixed, seriesColour } from "../format";
import type { Shortlist } from "../shortlist";

export type Axis = "context" | "stakeholder";

type Props = {
  axis: Axis;
  onAxis: (axis: Axis) => void;
  form: Form;
  category: FormCategory;
  scored: Shortlist;
  byContext: Matrix | null;
  byStakeholder: Matrix | null;
};

const AXES: { key: Axis; label: string }[] = [
  { key: "context", label: "By application" },
  { key: "stakeholder", label: "By priorities" },
];

export function Sensitivity({ axis, onAxis, form, category, scored, byContext, byStakeholder }: Props) {
  const matrix = axis === "context" ? byContext : byStakeholder;
  const names = new Map<string, string>(
    axis === "context"
      ? category.contexts.map((c) => [c.key, c.display_name])
      : form.stakeholders.map((s) => [s.key, s.display_name]),
  );
  const isCurrent = (label: string) =>
    axis === "context"
      ? label === scored.context
      : scored.stakeholders.length === 1 && scored.stakeholders[0] === label;

  return (
    <section className="card">
      <div className="card-head">
        <h2>How the ranking moves</h2>
        <div className="segmented" role="tablist" aria-label="Vary">
          {AXES.map((a) => (
            <button
              key={a.key}
              type="button"
              role="tab"
              aria-selected={axis === a.key}
              onClick={() => onAxis(a.key)}
            >
              {a.label}
            </button>
          ))}
        </div>
      </div>

      {!matrix ? (
        <p className="msg">Scoring the shortlist under every {axis === "context" ? "application" : "archetype"}.</p>
      ) : (
        <>
          <p className="summary">{summarise(axis, matrix)}</p>
          <div className="gridwrap">
            <table className="matrix">
              <thead>
                <tr>
                  <th />
                  {matrix.alternatives.map((name, index) => (
                    <th key={index} scope="col">
                      <span className="swatch" style={{ background: seriesColour(index) }} />
                      {name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {matrix.rows.map((row) => (
                  <tr key={row.label} className={isCurrent(row.label) ? "current" : undefined}>
                    <th scope="row">
                      {names.get(row.label) ?? row.label}
                      {isCurrent(row.label) && <span className="tag">chosen</span>}
                    </th>
                    {row.scores.map((value, index) => (
                      <td key={index} className={index === row.winner ? "win" : undefined}>
                        <span className="meter">
                          <span
                            style={{ width: `${Math.max(2, value * 100)}%`, background: seriesColour(index) }}
                          />
                        </span>
                        <span className="num">{fixed(value, 2)}</span>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="footnote">
            {axis === "context"
              ? "Each row scores the same shortlist in another application, with the chosen priorities."
              : "Each row scores the same shortlist for one archetype alone, in the chosen application."}
          </p>
        </>
      )}
    </section>
  );
}

function summarise(axis: Axis, matrix: Matrix): string {
  if (matrix.changes_winner) {
    return axis === "context"
      ? "The best option depends on the application."
      : "The best option depends on whose priorities count.";
  }
  const leader = matrix.alternatives[matrix.rows[0]?.winner ?? 0] ?? "The same option";
  return axis === "context"
    ? `${leader} leads in every application.`
    : `${leader} leads for every set of priorities.`;
}
