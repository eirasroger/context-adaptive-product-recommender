from eval.compare import untrained


def test_a_comparison_skips_sets_the_old_run_trained_or_validated_on():
    external_ids = ["was_test", "was_train", "was_val", "new_set"]
    old_folds = {"was_test": "test", "was_train": "train", "was_val": "val"}

    assert untrained(external_ids, range(4), old_folds) == [0, 3]
