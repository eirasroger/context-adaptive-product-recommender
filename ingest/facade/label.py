"""Language-model labels for facade shortlists."""

from pathlib import Path

from ingest.facade.synthesise import CATEGORY_KEY, DATASET_DIR, LLM_LABELS_FILE, SCENARIOS_FILE
from ingest.labelling import LabelRound, main

ROUND = LabelRound(
    category_key=CATEGORY_KEY,
    dataset_dir=DATASET_DIR,
    scenarios_file=SCENARIOS_FILE,
    labels_file=LLM_LABELS_FILE,
    brief_source=Path(__file__).with_name("labelling_brief.md"),
    examples_source=Path(__file__).with_name("labelling_examples.yaml"),
    letters="ABCDEFGHIJ",
)

if __name__ == "__main__":
    main(ROUND)
