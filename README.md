# Context-adaptive product recommender

Scores product alternatives against each other, given who is choosing and what
they are choosing it for.

You supply a shortlist, say which stakeholder priorities apply and which
application context the product is going into, and the model returns a
preference score per alternative. Scores are relative: they order the shortlist,
and they sit higher or lower as a group according to how good the shortlist is
overall.

## What it is for

The system is category-general. A product category is a set of rows in a
registry. Adding one means adding rows and data, with the schema, the model and
the scripts left alone. The working boundary is any product with an
environmental product declaration behind it, which today means mostly
construction, while the architecture stays agnostic about that.

One category is implemented. It serves as the validation case, chosen because an
open labelled dataset and a published result already exist for it. The whole
stack is built on that one so the next category becomes a configuration
exercise.

## How it works

**The registry is the specification.** It holds the indicators, what they mean,
which way is better, and the range each is measured against. It holds the
categories, what each is measured per, and which indicators apply. It holds the
contexts and the stakeholder archetypes. Changing a row here changes how the
model behaves, so registry changes are reviewed and version-stamped like code.

**Indicators are shared.** An impact indicator is the same entity wherever it
appears, so a category that reuses existing indicators costs nothing extra to
support. Each belongs to a family, and a new indicator starts from what the
model has learned about its family.

**Contexts declare which indicators they care about.** Whether a context applies
to a category follows from whether that category holds those indicators. A
requirement that depends on an indicator only some categories have switches
itself off for the rest, and no list is maintained by hand.

**Each alternative is a bag of tokens, one per indicator.** A token carries what
the indicator is, how the value compares to its declared range, how it compares
to the other alternatives on the table, whether it is present, whether it counts
in this context, and which way the context pulls it. Adding an indicator adds a
token, which is why nothing else has to move. The representation also keeps
"this category has no such indicator" distinct from "this product is missing a
value for it": the first has no token, the second has a token that says so.

**The network has two levels of attention.** The first looks within an
alternative across its indicators and produces one embedding for it. The second
looks across alternatives and compares them. The second stage is blind to
indicators, which makes comparing alternatives in one category and in another
the same operation, and is where transfer between categories comes from.

**Training targets the ordering and the gaps.** The distance between two options
is what someone acts on, so the loss penalises getting a gap wrong as well as
getting an order wrong. Every decision carries equal weight regardless of how
many alternatives it had. Near-ties are treated as near-ties.

**Evaluation has a tier that is pass or fail.** Control cases are generated from
the registry with a known answer, so they double as a test suite: the score
moves the way the registry says it should, it inverts when two contexts disagree
about an indicator, and a level marked never-selectable scores last. Metrics say
how close the model is. These say whether it learned the right thing.

**Nothing trains off the live database.** Each run freezes the data into a
content-hashed snapshot and records that hash plus the registry version with the
weights, so a result can be traced back to what produced it. A checkpoint also
carries its own registry, which keeps it interpretable on its own.

## Layout

```
registry/   the registry exported as readable files, so semantic changes diff
db/         the database: schema, migrations, seed data, integrity checks
ingest/     adapters for outside datasets; the control-case generator
snapshot/   database to frozen, hashed training data
core/       registry loading and token encoding, the shared vocabulary
model/      tokens, encoder, comparator, scoring head, loss, checkpoints
train/      training loop and run configurations
eval/       metrics, the behavioural suite, reporting and the release gate
serve/      inference API and the explorer
tests/      including the test that keeps the two encoding paths honest
```

Category names stay out of `model/`, `core/` and `train/`. A test fails if one
appears.

## Running it

Install, then build the database and load the registry:

```bash
pip install -r requirements.txt

python -m alembic upgrade head                 # create the schema
python -m db.seed                              # load the registry from db/seeds
python -m db.validate                          # check it holds together
python -m db.release 0.1.0 --notes "initial"   # freeze and export it
```

Bring in a dataset, assign folds, and freeze a snapshot:

```bash
python -m ingest.adapters.concrete_v1 <scenarios.json> <labels.json>
python -m ingest.split
python -m snapshot.build 0.1.0
```

Generate control cases for any indicator the registry declares sweepable and the
data never varies on its own:

```bash
python -m ingest.generators.run <category> --only-uncovered --dry-run
python -m ingest.generators.run <category> --indicator <key> --count 1200
```

Train, evaluate and gate in one go:

```bash
python -m train.run --config train/configs/default.yaml
```

That writes a run directory holding the config used, the epoch history, the
metrics, the behavioural results and the checkpoint. Re-evaluate an existing
checkpoint without retraining:

```bash
python -m eval.run runs/<run>/model.pt --device cuda
```

Tests:

```bash
python -m pytest tests
```

Most commands accept `--db` for a database elsewhere, or set `RECOMMENDER_DB`.
Training picks up a CUDA device automatically when one is available.

## Serving

```bash
RECOMMENDER_CHECKPOINT=runs/<run>/model.pt uvicorn serve.api:app
```

The inference API scores a shortlist and echoes the category's declared
eligibility precondition in every response, so a score is never mistaken for a
compliance statement. It also exposes the registry: what categories exist, what
each is measured per, which contexts apply, and what every indicator means.

| Route | Purpose |
|---|---|
| `POST /score` | Score a shortlist under a context and stakeholders |
| `GET /categories` | What can be compared and what each category assumes |
| `GET /categories/{key}/indicators` | Definitions, ranges, units, levels |
| `GET /stakeholders` | The archetypes and what each prioritises |

## The explorer

`GET /explore/` serves an interactive page for inspecting what the model
learned. Four views, all generated from the registry, so a new category becomes
explorable as soon as its rows exist:

- **Indicator response.** Sweep one indicator with everything else held at its
  ideal value and watch the score move, one curve per stakeholder, with the
  label the registry implies drawn alongside. This is the behavioural gate made
  visible.
- **Context sensitivity.** One shortlist scored under every context the category
  allows, which shows directly whether the top choice depends on the
  application.
- **Stakeholder sensitivity.** The same shortlist scored for every archetype.
- **Attribution.** What each indicator contributed to one alternative's score,
  measured by withholding the indicator and scoring again. The token
  representation makes this exact, because a withheld indicator is a state the
  model already understands.

Every view has a table alongside the chart, works in light and dark, and reads
its data from JSON endpoints under `/explore/` that can be called directly from
a notebook or a figure script.

## Adding a category

1. Write a seed file describing it: what it is measured per, which indicators
   apply, the declared range for each, whether a control case may sweep each one,
   and what regulatory filtering it assumes has already happened. Add rows for
   any indicator it needs that does not exist yet, with its definition and which
   way is better.
2. Seed, validate, and cut a new registry version.
3. Bring in its data through an adapter, or generate control cases from the
   registry.
4. Retrain across all categories together. Transfer is meant to run both ways,
   and the evaluation gates catch it if adding one category damages another.

No step involves writing model code. If one appears, treat it as a bug in the
design.

## Status

The stack runs end to end on one category, with the behavioural gate passing.
Outstanding:

- **Gate thresholds.** The behavioural tier gates today. The numeric pass marks
  for gap fidelity and band placement stay unset until there is a reason to
  choose particular values.
- **The reproduction result.** Matching the published baseline under this
  architecture is the experiment that establishes the generalisation, and it has
  yet to be run.
- **A second category.** It amounts to data and rows, and stays unproven until
  it has been done once.

## Background

The first category's dataset and the earlier single-category model it is
validated against are both published:

- Dataset: <https://doi.org/10.34810/DATA3164>
- Paper: <https://doi.org/10.1016/j.spc.2026.06.011>

This repository supersedes that implementation. It reuses the data and the
problem definition, and reproducing the published results under the general
architecture is the check that the generalisation holds.
