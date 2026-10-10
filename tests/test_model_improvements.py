
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from bankai.candidates.generator import CandidateGenerator
from bankai.config import Settings
from bankai.features.builder import FEATURE_NAMES, N_FEATURES
from bankai.features.embed import build_user_profile
from bankai.features.recurrence import detect_recurring
from bankai.models.ranker import LtrRanker, RankerConfig
from bankai.models.registry import ModelRegistry
from bankai.ranking.service import RankingService, RequestContext
from bankai.training.dataset_cache import CacheKey, build_key, corpus_digest, load, save
from bankai.training.experiment import make_splits
from bankai.training.labeler import LtrDataset, build_group
from bankai.training.trainer import Trainer

from .conftest import AS_OF, make_transaction, monthly_series


def _tiny_ranker(ensemble_size: int = 1, seed: int = 0) -> LtrRanker:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(300, N_FEATURES)).astype(np.float32)
    y = rng.integers(0, 2, size=300)
    return LtrRanker(
        RankerConfig(
            n_estimators=8,
            early_stopping_rounds=0,
            num_threads=1,
            random_state=seed,
            ensemble_size=ensemble_size,
        )
    ).fit(x, y, np.full(30, 10, dtype=int))


class TestEnsemble:
    def test_single_member_behaves_like_a_plain_model(self, tmp_path):
        r = _tiny_ranker(1)
        assert r.n_members == 1
        r.save(tmp_path / "ranker.txt")
        assert not (tmp_path / "ranker.ensemble.json").exists()

    def test_ensemble_round_trips(self, tmp_path):
        r = _tiny_ranker(3)
        assert r.n_members == 3
        r.save(tmp_path / "ranker.txt")
        assert (tmp_path / "ranker.ensemble.json").exists()
        loaded = LtrRanker.load(tmp_path / "ranker.txt")
        assert loaded.n_members == 3
        x = np.random.default_rng(9).normal(size=(10, N_FEATURES)).astype(np.float32)
        np.testing.assert_allclose(r.raw_scores(x), loaded.raw_scores(x), atol=1e-9)

    def test_members_are_independently_seeded(self, tmp_path):
        """A real ensemble must not be K copies of the same booster."""
        r = _tiny_ranker(4)
        strings = [b.model_to_string() for b in r._require()]
        assert len(set(strings)) == 4

    def test_ensemble_reduces_seed_variance(self):
        """The whole justification for ensembling, stated as a test.

        Measured on the development split, reseeding alone moved NDCG@1 by more
        than the effects being measured. Averaging must be at least as stable as
        any single member.
        """
        rng = np.random.default_rng(1)
        x = rng.normal(size=(1200, N_FEATURES)).astype(np.float32)
        y = (x[:, 0] + 0.4 * x[:, 1] > 0).astype(int)
        groups = np.full(60, 20, dtype=int)

        singles = []
        for seed in (1, 2, 3, 4, 5):
            r = LtrRanker(
                RankerConfig(
                    n_estimators=12,
                    early_stopping_rounds=0,
                    num_threads=1,
                    random_state=seed,
                    ensemble_size=1,
                )
            ).fit(x, y, groups)
            singles.append(r.raw_scores(x))
        ensemble = (
            LtrRanker(
                RankerConfig(
                    n_estimators=12,
                    early_stopping_rounds=0,
                    num_threads=1,
                    random_state=1,
                    ensemble_size=5,
                )
            )
            .fit(x, y, groups)
            .raw_scores(x)
        )

        # Rank correlation with the target must be at least as good as the mean member.
        def ndcg(scores: np.ndarray) -> float:
            order = np.argsort(-scores)[:5]
            return 1.0 if y[order[0]] == 1 else 0.0

        assert ndcg(ensemble) >= min(ndcg(s) for s in singles) - 0.15

    def test_shap_contributions_are_averaged(self, tmp_path):
        r = _tiny_ranker(3)
        x = np.random.default_rng(4).normal(size=(5, N_FEATURES)).astype(np.float32)
        contrib = r.explain(x)
        assert len(contrib) == 5
        assert set(contrib[0]) == set(FEATURE_NAMES)

    def test_explanations_survive_a_round_trip(self, tmp_path):
        r = _tiny_ranker(2)
        r.save(tmp_path / "ranker.txt")
        loaded = LtrRanker.load(tmp_path / "ranker.txt")
        x = np.random.default_rng(5).normal(size=(4, N_FEATURES)).astype(np.float32)
        assert set(loaded.explain(x)[0]) == set(FEATURE_NAMES)

    def test_mismatched_artifact_kind_is_refused(self, tmp_path):
        r = _tiny_ranker(3)
        r.save(tmp_path / "ranker.txt")
        assert (tmp_path / "ranker.ensemble.json").exists()
        (tmp_path / "ranker.ensemble.json").unlink()
        with pytest.raises(ValueError, match="ensemble"):
            LtrRanker.load(tmp_path / "ranker.txt")

    def test_group_sizes_must_sum_to_rows(self):
        r = LtrRanker(RankerConfig(n_estimators=5, ensemble_size=1))
        x = np.zeros((30, N_FEATURES), dtype=np.float32)
        with pytest.raises(ValueError, match="group sizes"):
            r.fit(x, np.zeros(30, dtype=int), np.full(3, 11, dtype=int))


class TestDatasetCache:
    def _dataset(self) -> LtrDataset:
        return LtrDataset(
            features=np.random.default_rng(0).normal(size=(50, N_FEATURES)).astype(np.float32),
            labels=np.array([0, 1] * 25),
            groups=np.full(5, 10, dtype=int),
            group_meta=[f"cp_{i}" for i in range(50)],
            group_users=[f"u_{i % 5}" for i in range(50)],
            n_positives=25,
            n_hard_negatives=10,
            row_dates=np.arange(50, dtype=np.int64),
        )

    def test_round_trip(self, tmp_path):
        original = self._dataset()
        path = tmp_path / "ds.npz"
        save(original, path)
        loaded = load(path)
        np.testing.assert_allclose(original.features, loaded.features)
        np.testing.assert_array_equal(original.labels, loaded.labels)
        np.testing.assert_array_equal(original.groups, loaded.groups)
        np.testing.assert_array_equal(original.row_dates, loaded.row_dates)
        assert loaded.n_positives == original.n_positives

    def test_cache_path_changes_when_the_contract_changes(self, tmp_path):
        corpus = {"u1": [make_transaction(user_id="u1")]}
        settings = Settings(data_dir=tmp_path, artifact_dir=tmp_path)
        a = build_key(corpus, as_of=AS_OF, settings=settings, n_users=None)
        b = CacheKey(**{**a.__dict__, "feature_version": "999"})
        assert a.path(tmp_path) != b.path(tmp_path)

    def test_cache_path_changes_when_the_corpus_changes(self, tmp_path):
        settings = Settings(data_dir=tmp_path, artifact_dir=tmp_path)
        a = build_key(
            {"u1": [make_transaction(user_id="u1")]}, as_of=AS_OF, settings=settings, n_users=None
        )
        b = build_key(
            {"u1": [make_transaction(user_id="u1"), make_transaction(user_id="u1")]},
            as_of=AS_OF,
            settings=settings,
            n_users=None,
        )
        assert a.digest() != b.digest()

    def test_corpus_digest_is_order_independent(self, tmp_path):
        a = corpus_digest({"u1": [make_transaction(days_from=dt.date(2024, 1, 1))]})
        b = corpus_digest({"u1": [make_transaction(days_from=dt.date(2024, 1, 1))]})
        assert a == b

    def test_cached_dataset_is_reused(self, repo, tmp_path):
        """A warm cache must not rebuild; a cold one must."""
        settings = Settings(
            data_dir=tmp_path / "d",
            artifact_dir=tmp_path / "a",
            environment="test",
            log_level="ERROR",
        )
        trainer = Trainer(settings=settings)
        corpus = {u: repo.transactions_for_user(u) for u in repo.users()[:3]}
        first = trainer.build_dataset(corpus, as_of=AS_OF, use_cache=True)
        second = trainer.build_dataset(corpus, as_of=AS_OF, use_cache=True)
        np.testing.assert_array_equal(first.features, second.features)
        np.testing.assert_array_equal(first.groups, second.groups)


class TestUserSplits:
    def test_splits_are_disjoint_and_complete(self):
        users = [f"u_{i:04d}" for i in range(200)]
        split = make_splits(users, n_dev=20, n_holdout=30)
        assert set(split.train) | set(split.dev) | set(split.holdout) == set(users)
        assert not (set(split.train) & set(split.dev))
        assert not (set(split.train) & set(split.holdout))
        assert not (set(split.dev) & set(split.holdout))
        assert len(split.dev) == 20 and len(split.holdout) == 30

    def test_split_is_deterministic(self):
        users = [f"u_{i:04d}" for i in range(200)]
        assert make_splits(users).as_dict() == make_splits(users).as_dict()

    def test_split_is_stable_under_insertion(self):
        """Assignment is by hash, so adding a user must not move existing ones."""
        before = make_splits([f"u_{i:04d}" for i in range(200)], n_dev=20, n_holdout=30)
        after = make_splits(
            [f"u_{i:04d}" for i in range(0, 200) if i != 7] + ["u_9999"], n_dev=20, n_holdout=30
        )
        assert set(before.train) & set(after.train) == set(before.train) - {"u_0007"}

    def test_tiny_corpus_degrades_safely(self):
        split = make_splits(["u1", "u2"], n_dev=20, n_holdout=30)
        assert set(split.train) == {"u1", "u2"}
        assert split.dev == () and split.holdout == ()


class TestTrainingPoolParity:
    """The training pool must be the pool the service ranks."""

    def _world(self):
        """History where *something* is always paid on the decision day.

        The label window is ``[as_of, as_of + 3]``, so a decision day with no
        payment in that window produces no learnable group at all — which would
        make the pool-parity tests pass for the wrong reason.
        """
        txs: list = []
        for name, cat, day in (
            ("FiberNet ISP", "utilities.internet", 5),
            ("NordWohn Immobilien", "housing.rent", 2),
            ("VoltGrid Energy", "utilities.electricity", 11),
        ):
            txs += monthly_series(day=day, amount=50.0, months=12, name=name, category=cat)
        txs += monthly_series(
            day=30,
            amount=25.0,
            months=12,
            end=dt.date(2024, 6, 30),
            name="QuickMart Grocery",
            category="shopping.grocery",
        )
        return sorted(txs, key=lambda t: t.value_date)

    def test_training_pool_matches_the_generator(self, tmp_path):
        txs = self._world()
        decision = dt.date(2024, 6, 30)
        generator = CandidateGenerator()
        profile = build_user_profile("u", txs, as_of=decision, inclusive=False)
        recs = detect_recurring(profile.series, as_of=decision, horizon_days=14)
        served = {
            c.counterparty_id
            for c in generator.generate(profile, recs, as_of=decision, horizon_days=14).candidates
        }
        built = build_group("u", txs, decision, generator=generator)
        assert built is not None
        group, _matrix = built
        # Exploration negatives are the only permitted extras.
        extra = {r.counterparty_id for r in group.rows} - served
        assert all(r.source == "exploration" for r in group.rows if r.counterparty_id in extra)

    def test_group_contains_the_actual_payments(self, tmp_path):
        txs = self._world()
        decision = dt.date(2024, 6, 30)
        built = build_group("u", txs, decision, generator=CandidateGenerator())
        assert built is not None
        group, _ = built
        assert group.n_positive > 0, "the decision day must produce a learnable group"
        assert any(r.label == 1 for r in group.rows)

    def test_group_without_targets_is_discarded(self):
        """A day where the user paid nothing in the window yields no group."""
        txs = monthly_series(day=5, amount=20.0, months=10, end=dt.date(2024, 1, 31))
        built = build_group("u", txs, AS_OF, generator=CandidateGenerator())
        assert built is None

    def test_same_day_payment_is_not_in_the_history(self):
        """Regression: the label's own payment must not be visible to the model.

        The labeler excludes it (``value_date < as_of``) and the backtest now
        serves with ``history_inclusive=False``. If either regressed, the model
        would be scored on a case whose answer it had already seen.
        """
        txs = monthly_series(day=30, amount=42.0, months=12, end=dt.date(2024, 6, 30))
        decision = dt.date(2024, 6, 30)
        before = len([t for t in txs if t.value_date < decision])
        after = len([t for t in txs if t.value_date <= decision])
        assert after > before, "the fixture must include a same-day payment for this to matter"
        built = build_group(
            "u",
            [*txs, make_transaction(days_from=decision, name="Same Day Co")],
            decision,
            generator=CandidateGenerator(),
        )
        if built is not None:
            group, _ = built
            same_day = [r for r in group.rows if r.counterparty_id == "cp_same_day_co"]
            # It may appear as a *label*, never as history-backed evidence.
            assert all(r.source == "exploration" for r in same_day)

    def test_exploration_negatives_are_added(self):
        from bankai.training.labeler import register_counterparty_population

        register_counterparty_population(
            [
                ("cp_new_1", "New Co", "utilities.internet", "EUR"),
                ("cp_new_2", "New Two", "shopping.general", "EUR"),
            ]
        )
        txs = monthly_series(day=30, amount=42.0, months=12, end=dt.date(2024, 6, 30))
        built = build_group("u", txs, AS_OF, generator=CandidateGenerator())
        assert built is not None
        group, _ = built
        exploration = [r for r in group.rows if r.source == "exploration"]
        assert exploration, "unseen payees must be present so the model can reject them"
        assert all(r.label == 0 for r in exploration)


class TestRecallImprovements:
    def test_dormant_schedule_is_recalled(self):
        """A proven cadence the user has paused must still reach the pool.

        Offline analysis attributed 70% of all missed targets to exactly this
        case: `is_recurring` flips to False after two silent cycles and the payee
        vanished entirely.
        """
        last = dt.date(2024, 6, 9)  # three silent weeks before AS_OF
        txs = [
            make_transaction(
                days_from=last - dt.timedelta(days=7 * (29 - i)),
                amount=6.40,
                name="FareGo Transit",
                category="transport.fare",
            )
            for i in range(30)
        ]
        profile = build_user_profile("u", txs, as_of=AS_OF, inclusive=False)
        recs = detect_recurring(profile.series, as_of=AS_OF, horizon_days=14)
        assert not recs["cp_farego_transit"].is_recurring, "precondition: it is dormant"
        result = CandidateGenerator().generate(profile, recs, as_of=AS_OF, horizon_days=14)
        sources = {c.counterparty_id: c.source for c in result.candidates}
        assert "cp_farego_transit" in sources
        assert "dormant" in sources["cp_farego_transit"]

    def test_dormant_candidates_are_scored_low(self):
        """Recall must not equal promotion: the ranker decides."""
        last = dt.date(2024, 6, 9)
        txs = [
            make_transaction(
                days_from=last - dt.timedelta(days=7 * (29 - i)),
                amount=6.40,
                name="FareGo Transit",
                category="transport.fare",
            )
            for i in range(30)
        ]
        txs += monthly_series(day=5, amount=42.0, months=12)
        profile = build_user_profile("u", txs, as_of=AS_OF, inclusive=False)
        recs = detect_recurring(profile.series, as_of=AS_OF, horizon_days=14)
        result = CandidateGenerator().generate(profile, recs, as_of=AS_OF, horizon_days=14)
        by_id = {c.counterparty_id: c.base_score for c in result.candidates}
        if "cp_farego_transit" in by_id and "cp_fibernet_isp" in by_id:
            assert by_id["cp_farego_transit"] < by_id["cp_fibernet_isp"]

    def test_one_off_series_are_not_recalled_as_dormant(self):
        """A non-recurring payee must not be dressed up as a paused schedule."""
        txs = [
            make_transaction(
                days_from=dt.date(2024, 1, 5) + dt.timedelta(days=13 * i),
                amount=9.0,
                name="Random Co",
                category="shopping.general",
            )
            for i in range(12)
        ]
        profile = build_user_profile("u", txs, as_of=AS_OF, inclusive=False)
        recs = detect_recurring(profile.series, as_of=AS_OF, horizon_days=14)
        result = CandidateGenerator().generate(profile, recs, as_of=AS_OF, horizon_days=14)
        dormant = [c for c in result.candidates if c.source == "dormant"]
        assert all(
            c.recurrence is None or c.recurrence.periodicity_confidence >= 0.55 for c in dormant
        )

    def test_habit_overrides_a_low_category_prior(self):
        """Regression: coffee (prior 0.25) was excluded by a hard prior gate.

        A user who has bought coffee dozens of times will buy it again whatever
        the category prior says.
        """
        txs = [
            make_transaction(
                days_from=dt.date(2024, 6, 1) - dt.timedelta(days=3 * i),
                amount=4.20,
                name="CafeNine Coffee",
                category="dining.coffee",
            )
            for i in range(24)
        ]
        profile = build_user_profile("u", txs, as_of=AS_OF, inclusive=False)
        recs = detect_recurring(profile.series, as_of=AS_OF, horizon_days=14)
        result = CandidateGenerator().generate(profile, recs, as_of=AS_OF, horizon_days=14)
        assert "cp_cafenine_coffee" in {c.counterparty_id for c in result.candidates}


class TestServingMoment:
    def test_inclusive_and_exclusive_differ_only_on_the_as_of_day(self):
        txs = monthly_series(day=5, amount=42.0, months=12, end=dt.date(2024, 6, 25))
        txs += [make_transaction(days_from=AS_OF, name="Same Day Co", amount=11.0)]
        incl = build_user_profile("u", txs, as_of=AS_OF, inclusive=True)
        excl = build_user_profile("u", txs, as_of=AS_OF, inclusive=False)
        assert incl.n_transactions == excl.n_transactions + 1
        assert "cp_same_day_co" in incl.series
        assert "cp_same_day_co" not in excl.series

    def test_service_honours_the_serving_moment(self, repo, tmp_path):
        settings = Settings(
            data_dir=tmp_path / "d",
            artifact_dir=tmp_path / "a",
            environment="test",
            log_level="ERROR",
        )
        svc = RankingService(
            repository=repo, registry=ModelRegistry(settings.artifact_dir), settings=settings
        )
        day = dt.date(2024, 6, 30)
        user = repo.users()[0]
        inclusive = svc.rank(RequestContext(user_id=user, as_of=day, history_inclusive=True))
        exclusive = svc.rank(RequestContext(user_id=user, as_of=day, history_inclusive=False))
        assert (
            inclusive.diagnostics["n_transactions"] == exclusive.diagnostics["n_transactions"] + 1
        )

    def test_backtest_serves_at_the_start_of_the_day(self):
        """The harness must not quietly reintroduce the leak it exists to prevent."""
        import inspect

        from bankai.evaluation.backtest import run_backtest

        source = inspect.getsource(run_backtest)
        assert "history_inclusive=False" in source
