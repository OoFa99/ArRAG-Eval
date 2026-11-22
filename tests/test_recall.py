import pytest

from .tests import recall_at_k, recall_at_ks


@pytest.mark.parametrize(
    ("retrieved", "relevant", "k", "expected"),
    [
        pytest.param(["a", "a", "b"], {"a", "b"}, 2, 0.5, id="duplicate-counts-once"),
        pytest.param(["x", "a", "b"], {"a", "b", "c"}, 2, 1 / 3, id="cutoff-excludes-later-hits"),
        pytest.param(["a"], {"a"}, 0, 0.0, id="zero-cutoff"),
        pytest.param(["a", "x"], {"a", "b"}, 10, 0.5, id="cutoff-beyond-results"),
        pytest.param([], {"a"}, 5, 0.0, id="empty-results"),
        pytest.param(["x", "y"], {"a"}, 2, 0.0, id="no-matches"),
    ],
)
def test_recall_at_k(retrieved, relevant, k, expected):
    assert recall_at_k(retrieved, relevant, k) == pytest.approx(expected)


def test_recall_at_k_rejects_negative_cutoff():
    with pytest.raises(ValueError, match="k must be non-negative"):
        recall_at_k(["a"], {"a"}, -1)


def test_recall_at_k_rejects_empty_relevance_labels():
    with pytest.raises(ValueError, match="relevant_items cannot be empty"):
        recall_at_k(["a"], set(), 1)


def test_recall_at_ks_uses_default_cutoffs():
    retrieved = [1, 2, 3, 4, 5, 6]
    relevant = {1, 5, 6}

    assert recall_at_ks(retrieved, relevant) == pytest.approx({1: 1 / 3, 5: 2 / 3, 10: 1.0})


def test_recall_at_ks_accepts_explicit_cutoffs():
    retrieved = [1, 2, 3]
    relevant = {2, 3}

    assert recall_at_ks(retrieved, relevant, ks=[0, 2, 3]) == pytest.approx({0: 0.0, 2: 0.5, 3: 1.0})
