"""Offline metrics, verified against hand-computed values."""

from __future__ import annotations

import math

import pytest

from bankai.evaluation.metrics import (
    bootstrap_ci,
    compare,
    dcg,
    evaluate_rankings,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestRankingMetrics:
    def test_dcg_matches_definition(self):
        # 1/log2(1+1) + 1/log2(2+1) = 1 + 0.6309
        assert dcg([1.0, 1.0]) == pytest.approx(1.0 + 1 / math.log2(3))
        assert dcg([]) == 0.0

    def test_ndcg_perfect_ranking(self):
        assert ndcg_at_k([("a", 1.0), ("b", 0.5)], {"a", "b"}, k=2) == pytest.approx(1.0)

    def test_ndcg_relevant_at_position_two(self):
        ranked = [("a", 1.0), ("b", 0.5), ("c", 0.1)]
        value = ndcg_at_k(ranked, {"b"}, k=3)
        expected = (1 / math.log2(3)) / 1.0
        assert value == pytest.approx(expected)

    def test_ndcg_miss_is_zero(self):
        assert ndcg_at_k([("x", 1.0)], {"a"}, k=5) == 0.0

    def test_ndcg_empty_relevant_is_zero(self):
        assert ndcg_at_k([("a", 1.0)], set(), k=5) == 0.0

    def test_hit_and_precision_and_recall(self):
        ranked = [("a", 1.0), ("b", 0.9), ("c", 0.8), ("d", 0.7)]
        relevant = {"b", "d"}
        assert hit_rate_at_k(ranked, relevant, k=1) == 0.0
        assert hit_rate_at_k(ranked, relevant, k=2) == 1.0
        assert hit_rate_at_k(ranked, relevant, k=3) == 1.0
        assert precision_at_k(ranked, relevant, k=2) == pytest.approx(0.5)
        assert precision_at_k(ranked, relevant, k=4) == pytest.approx(0.5)
        assert recall_at_k(ranked, relevant, k=2) == pytest.approx(0.5)
        assert recall_at_k(ranked, relevant, k=4) == pytest.approx(1.0)

    def test_precision_handles_short_lists(self):
        assert precision_at_k([("a", 1.0)], {"a"}, k=5) == pytest.approx(1.0)

    def test_mrr(self):
        ranked = [("a", 1.0), ("b", 0.9), ("c", 0.8)]
        assert mrr(ranked, {"c"}) == pytest.approx(1 / 3)
        assert mrr(ranked, {"zz"}) == 0.0
        assert mrr(ranked, {"b"}) == pytest.approx(0.5)


class TestAggregate:
    def test_evaluate_rankings_shape(self):
        # Same ordering in both groups, but the target differs: first in one,
        # second in the other. This exercises position sensitivity.
        rankings = [[("a", 1.0), ("b", 0.5)], [("a", 1.0), ("b", 0.5)]]
        relevants = [{"a"}, {"b"}]
        out = evaluate_rankings(rankings, relevants)
        assert out["n_groups"] == 2
        for k in (1, 3, 5, 10):
            assert f"ndcg@{k}" in out and f"hit@{k}" in out
        assert out["hit@1"] == pytest.approx(0.5)
        assert out["hit@5"] == pytest.approx(1.0)
        assert out["mrr"] == pytest.approx(0.75)  # (1.0 + 0.5) / 2

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError):
            evaluate_rankings([[("a", 1.0)]], [{"a"}, {"b"}])

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError):
            evaluate_rankings([], [])


class TestBootstrap:
    def test_identical_variants_have_zero_gain(self):
        # Same ordering in both groups, but the target differs: first in one,
        # second in the other. This exercises position sensitivity.
        rankings = [[("a", 1.0), ("b", 0.5)], [("a", 1.0), ("b", 0.5)]]
        relevants = [{"a"}, {"b"}]
        result = compare(rankings, rankings, relevants, k=2)
        assert result["ndcg_gain"] == pytest.approx(0.0)
        assert not result["significant"]

    def test_strictly_better_candidate_is_significant(self):
        relevants = [{"a"}] * 30
        worse = [[("z", 1.0), ("a", 0.5)] for _ in range(30)]
        better = [[("a", 1.0), ("z", 0.5)] for _ in range(30)]
        result = compare(worse, better, relevants, k=2)
        # Rank 2 -> rank 1 on a single relevant item: 1 - 1/log2(3).
        assert result["ndcg_gain"] == pytest.approx(1.0 - 1 / math.log2(3))
        assert result["significant"]
        assert result["wins"] == 30 and result["losses"] == 0

    def test_ci_brackets_the_point_estimate(self):
        rng = [0.1, 0.5, 0.9, 0.3, 0.7]
        point, lo, hi = bootstrap_ci(rng, n_resamples=200, seed=5)
        assert lo <= point <= hi

    def test_bootstrap_is_deterministic(self):
        a = bootstrap_ci([0.1, 0.5, 0.9], seed=11)
        b = bootstrap_ci([0.1, 0.5, 0.9], seed=11)
        assert a == b

    def test_empty_input(self):
        assert bootstrap_ci([]) == (0.0, 0.0, 0.0)


class TestMetricProperties:
    def test_metrics_are_monotonic_in_k(self):
        rankings = [[("a", 1.0), ("b", 0.5), ("c", 0.2)]]
        relevants = [{"c"}]
        hits = [hit_rate_at_k(rankings[0], relevants[0], k=k) for k in (1, 3, 5)]
        assert hits == sorted(hits)

    def test_perfect_system_beats_random(self):
        relevants = [{"a", "b"}] * 50
        perfect = [[("a", 2.0), ("b", 1.0)]] * 50
        random = [[("z", 1.0), ("a", 0.1), ("b", 0.05), ("q", 0.0)]] * 50
        assert compare(random, perfect, relevants, k=2)["ndcg_gain"] > 0
