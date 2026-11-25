from ingest.build import control_prefix


def test_facade_control_ids_keep_the_prefix_every_earlier_corpus_used():
    """Folds hash external ids; a new prefix would reshuffle every facade control set between folds."""
    assert control_prefix("facade_system") == "facade"
    assert control_prefix("indoor_partition") == "indoor"
