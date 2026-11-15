import pytest
from types import SimpleNamespace

from src.evaluate import evaluate, ranking_metrics


def test_metrics_measure_rank_and_relevance_separately():
    metrics = ranking_metrics(["wrong", "right", "other"], ["right", "missing"])
    assert metrics["precision_at_3"] == pytest.approx(1 / 3)
    assert metrics["recall_at_3"] == 0.5
    assert metrics["hit_at_3"] == 1
    assert metrics["reciprocal_rank"] == 0.5


def test_no_relevant_results_scores_zero():
    assert all(value == 0 for value in ranking_metrics(["wrong"], ["right"]).values())


@pytest.mark.parametrize("cases", [[], [{"query": "calm", "relevant_ids": ["unknown"]}]])
def test_evaluation_rejects_invalid_labels(cases):
    with pytest.raises(ValueError):
        evaluate(SimpleNamespace(songs=[{"id": "1"}]), cases)
