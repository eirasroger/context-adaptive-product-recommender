import type { ScoreResponse } from "../api/client";
import { fixed, seriesColour } from "../format";

type Props = {
  result: ScoreResponse;
  wins: Map<number, string>;
  preview: boolean;
};

export function Result({ result, wins, preview }: Props) {
  const ranked = [...result.results].sort((a, b) => a.rank - b.rank);
  const peak = Math.max(...ranked.map((r) => r.score), 0.0001);

  return (
    <section className="card">
      <h2>
        Result {preview && <span className="badge">Preview</span>}
      </h2>
      <div className="result">
        {ranked.map((r) => {
          const index = result.results.indexOf(r);
          const annotation = wins.get(index);
          const classes = ["rank"];
          if (r.rank === 1) classes.push("lead");
          if (r.disqualifying_levels.length > 0) classes.push("disq");
          return (
            <div key={index} className={classes.join(" ")}>
              <span className="pos">{r.rank}</span>
              <span className="name">{r.id}</span>
              <span className="track">
                <span
                  className="fill"
                  style={{ width: `${(r.score / peak) * 100}%`, background: seriesColour(index) }}
                />
              </span>
              <span className="val">{fixed(r.score)}</span>
              {annotation && <span className="wins">{annotation}</span>}
            </div>
          );
        })}
      </div>
      <div className="flags">
        {ranked.flatMap((r) => [
          ...r.disqualifying_levels.map((level) => (
            <div key={`${r.rank}-${level}`} className="flag">
              <span className="marker">!</span>
              <span>
                <strong>{r.id}</strong> sits at <code>{level}</code>, which the registry marks as
                never selectable.
              </span>
            </div>
          )),
          ...(r.missing_indicators.length > 0
            ? [
                <div key={`${r.rank}-missing`} className="flag">
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
