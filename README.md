# Context-adaptive product recommender

Scores product alternatives against each other, given who is choosing and what
they are choosing it for.

You hand it a shortlist, say which stakeholder priorities apply and which
application context the product is going into, and it returns a preference score
per alternative. The scores are relative: they order the shortlist, and they sit
higher or lower as a group depending on how good the shortlist is overall.

## The point of it

The system is **category-general**. A product category is a set of rows in a
registry, never code. Adding one means adding rows and data — no schema change,
no model change, no new script. The working boundary is any product with an
environmental product declaration behind it, which is mostly construction, but
nothing in the architecture assumes that.

One category is implemented today. It is the validation case, not the subject:
it exists because there is an open labelled dataset and a published result to
reproduce. The whole stack is built on it so that the next category is a
configuration exercise.

## How it works

**The registry is the specification.** It holds the indicators, what they mean,
which way is better, and the range each is measured against. It holds the
categories, what each is measured per, and which indicators apply. It holds the
contexts and the stakeholder archetypes. Changing a row here changes how the
model behaves, so registry changes are reviewed and version-stamped like code.

**Indicators are shared.** An impact indicator is the same entity wherever it
appears, so a new category that reuses existing indicators costs nothing extra
to support. Each belongs to a family, and a brand-new indicator starts from what
the model has learned about its family rather than from nothing.

**Contexts declare, they don't get listed.** A context says which indicators it
cares about and which way it pulls them. Whether it applies to a category is
worked out from whether that category has those indicators. A requirement that
depends on an indicator only some categories have switches itself off for the
rest. Nobody maintains a list.

**Each alternative is a bag of tokens, one per indicator.** A token carries what
the indicator is, how the value compares to its declared range, how it compares
to the other alternatives on the table, whether it is present, whether it counts
in this context, and which way the context pulls it. This is why an indicator
can be added without disturbing anything: it is another token, not another
column. It is also how "this category doesn't have that indicator" stays
different from "this product is missing a value for it" — the first has no
token, the second has a token that says so.

**The network has two levels of attention.** The first looks within an
alternative, across its indicators, and produces one embedding for it. The
second looks across alternatives and compares them. The second stage never sees
an indicator — which is what makes comparing alternatives in one category and in
another literally the same operation, and is where transfer between categories
would come from.

**Training targets both the ordering and the gaps.** Getting the order right is
not enough; the distance between two options is what someone actually acts on.
Every decision counts the same regardless of how many alternatives it had.
Near-ties are treated as near-ties rather than as errors to be forced apart.

**Evaluation has a tier that is pass or fail.** Because control cases are
generated from the registry with a known answer, they double as a test suite:
does the score move the way the registry says it should, does it invert when two
contexts disagree about an indicator, does a level marked never-selectable
actually score last. Metrics tell you how close the model is; these tell you
whether it learned the right thing.

**Nothing trains off the live database.** Each run freezes the data into a
content-hashed snapshot and records that hash plus the registry version with the
weights, so a result can always be traced back to what produced it. A checkpoint
also carries its own registry, so it stays interpretable without the database.

## Layout

```
registry/   the registry exported as readable files, so semantic changes diff
db/         the database: schema, migrations, seed data, integrity checks
ingest/     adapters that bring outside datasets in; the control-case generator
snapshot/   database to frozen, hashed training data
core/       registry loading and token encoding — the shared vocabulary
model/      tokens, encoder, comparator, scoring head, loss, checkpoints
train/      training loop and run configurations
eval/       metrics, the behavioural suite, reporting and the release gate
serve/      inference API
tests/      including the test that keeps the two encoding paths honest
```

Nothing in `model/`, `core/` or `train/` names a category. There is a test that
fails if one creeps in.

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

Train, evaluate and gate in one go:

```bash
python -m train.run --config train/configs/default.yaml
```

That writes a run directory containing the config used, the epoch history, the
metrics, the behavioural results and the checkpoint. Serve it with:

```bash
RECOMMENDER_CHECKPOINT=runs/<run>/model.pt uvicorn serve.api:app
```

Tests:

```bash
python -m pytest tests
```

Most commands take `--db` if you want a database somewhere other than the
default, or you can set `RECOMMENDER_DB`.

## Adding a category

1. Write a seed file describing it: what it is measured per, which indicators
   apply, the declared range for each, and what regulatory filtering it assumes
   has already happened. Add rows for any indicator it needs that does not exist
   yet, with its definition and which way is better.
2. Seed, validate, and cut a new registry version.
3. Bring in its data through an adapter, or generate control cases from the
   registry.
4. Retrain across all categories together — transfer is meant to go both ways,
   and the evaluation gates catch it if adding one damages another.

There is no step where you write model code. If there ever is, the design has
failed and that is worth treating as a bug rather than as work.

## Status

The stack runs end to end on one category. What it does not yet have:

- **Thresholds on the gates.** The metrics are computed and the behavioural
  suite is wired in, but the numeric pass marks are deliberately unset until the
  model is validated. A threshold nobody has justified is a gate that always
  opens.
- **The reproduction result.** Matching the published baseline under this
  architecture is the experiment that says the design is sound, and it has not
  been run to completion yet.
- **A second category.** By design, that is data and rows rather than code, but
  it is not proven until it has been done once.
- **The analytics explorer.** The attribution and sensitivity analyses that
  exist in the predecessor need porting to a different output.

## Background

The first category's dataset and the earlier single-category model it is
validated against are both published:

- Dataset: <https://doi.org/10.34810/DATA3164>
- Paper: <https://doi.org/10.1016/j.spc.2026.06.011>

That earlier implementation is superseded rather than extended. This repository
reuses none of its code; it reuses its data and the problem definition, and
reproducing its published results under the general architecture is the check
that the generalisation is sound.
