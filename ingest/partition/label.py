"""Language-model labels for partition shortlists."""

from pathlib import Path

from ingest.labelling import LabelRound, main
from ingest.partition.synthesise import CATEGORY_KEY, DATASET_DIR, LLM_LABELS_FILE, SCENARIOS_FILE

ROUND = LabelRound(
    category_key=CATEGORY_KEY,
    dataset_dir=DATASET_DIR,
    scenarios_file=SCENARIOS_FILE,
    labels_file=LLM_LABELS_FILE,
    brief_source=Path(__file__).with_name("labelling_brief.md"),
    examples_source=Path(__file__).with_name("labelling_examples.yaml"),
    letters="ABCDE",
)

if __name__ == "__main__":
    main(ROUND)
