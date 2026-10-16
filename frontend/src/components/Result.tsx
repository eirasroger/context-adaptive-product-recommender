import type { ScoreResponse } from "../api/client";
import { fixed, seriesColour, stagger } from "../format";
import { type Standing, standings } from "../ranking";
import { useReorder, useTween } from "./motion";

type Scored = ScoreResponse["results"][number];

type Props = {
  result: ScoreResponse;
  /** Colour slot of each scored option, in the order they were sent. */
  colours: readonly number[];
  wins: Map<number, string>;
  preview: boolean;
  epsilon: number;
};

export function Result({ result, colours, wins, preview, epsilon }: Props) {
  const placed = standings(result.results, epsilon);
  const slide = useReorder<number>();
  const indexOf = (r: Scored) => result.results.indexOf(r);

  return (
    <section className="card">
      <div className="card-head">
        <h2>
          Result {preview && <span className="badge">Preview</span>}
        </h2>
      </div>
      <Lead placed={placed} />
      <div className="result">
        {placed.map((standing, position) => {
          const r = standing.entry;
          const index = indexOf(r);
          const annotation = wins.get(index);
          const classes = ["rank", "enter"];
          if (standing.place === 1) classes.push("lead");
          if (r.disqualifying_levels.length > 0) classes.push("disq");
          if (standing.tied) {
            classes.push("tied");
            if (placed[position - 1]?.group !== standing.group) classes.push("tie-start");
            if (placed[position + 1]?.group !== standing.group) classes.push("tie-end");
          }
          return (
            <div key={index} ref={slide(index)} className={classes.join(" ")} style={stagger(position)}>
              <span className="pos">{standing.place}</span>
              <span className="name">
                {r.id}
                {standing.tied && <span className="pill">Tied</span>}
              </span>
              <span className="track">
                <span
                  className="fill"
                  style={{ width: `${r.score * 100}%`, background: seriesColour(colours[index] ?? index) }}
                />
              </span>
              <Figure className="val" value={r.score} />
              {annotation && <span className="wins">{annotation}</span>}
            </div>
          );
        })}
      </div>
      <div className="flags">
        {placed.flatMap(({ entry: r }) => [
          ...r.disqualifying_levels.map((level) => (
            <div key={`${indexOf(r)}-${level}`} className="flag">
              <span className="marker">!</span>
              <span>
                <strong>{r.id}</strong> sits at <code>{level}</code>, which the registry marks as
                never selectable.
              </span>
            </div>
          )),
          ...(r.missing_indicators.length > 0
            ? [
                <div key={`${indexOf(r)}-missing`} className="flag">
                  <span className="marker info">i</span>
                  <span>
                    <strong>{r.id}</strong> has {r.missing_indicators.length} indicator
                    {r.missing_indicators.length === 1 ? "" : "s"} left unknown.
                  </span>
                </div>,
              ]
            : []),
        ])}
      </div>
    </section>
  );
}

function Lead({ placed }: { placed: Standing<Scored>[] }) {
  const leaders = placed.filter((s) => s.group === 0);
  const next = placed.find((s) => s.group === 1);
  const lowestLeader = leaders[leaders.length - 1]!.entry.score;
  const names = leaders.map((s) => s.entry.id);

  return (
    <div className="lead-panel enter">
      <div className="lead-text">
        <span className="eyebrow">{leaders.length > 1 ? "Tied for first" : "Recommended"}</span>
        <span className="lead-name">{listed(names)}</span>
        <span className="lead-margin">
          {next ? (
            <>
              <span className="delta">+{fixed(lowestLeader - next.entry.score)}</span> ahead of{" "}
              {next.entry.id}
            </>
          ) : (
            "Every option is tied"
          )}
        </span>
      </div>
      <Figure className="lead-score" value={leaders[0]!.entry.score} />
    </div>
  );
}

function Figure({ value, className }: { value: number; className: string }) {
  const shown = useTween(value);
  return (
    <span className={className} aria-label={fixed(value)}>
      {fixed(shown)}
    </span>
  );
}

const listed = (names: string[]) =>
  names.length < 2 ? (names[0] ?? "") : `${names.slice(0, -1).join(", ")} and ${names.at(-1)}`;
