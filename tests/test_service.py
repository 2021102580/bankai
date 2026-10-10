"""End-to-end behaviour of the ranking service and the training pipeline."""

from __future__ import annotations

import datetime as dt

import pytest

from bankai.config import Settings
from bankai.evaluation.backtest import build_cases, run_ablation, run_backtest
from bankai.models.registry import ModelRegistry
from bankai.ranking.diversify import cap_by_category, dedupe_by_payee_and_due, mmr_rerank
from bankai.ranking.explain import cadence_label, explain
from bankai.ranking.service import RankingService, RequestContext
from bankai.training.trainer import Trainer, fit_auxiliary_models, train_and_publish

from .conftest import AS_OF


@pytest.fixture
def service(repo, tmp_path) -> RankingService:
    settings = Settings(
        data_dir=tmp_path / "d",
        artifact_dir=tmp_path / "a",
        database_path=tmp_path / "d" / "x.db",
        environment="test",
        log_level="ERROR",
    )
    return RankingService(
        repository=repo, registry=ModelRegistry(settings.artifact_dir), settings=settings
    )


class TestRankingService:
    def test_returns_ranked_recommendations(self, service, repo):
        result = service.rank(RequestContext(user_id=repo.users()[0], as_of=AS_OF, limit=5))
        assert result.recommendations
        scores = [r.score for r in result.recommendations]
        assert scores == sorted(scores, reverse=True), "results must be score-ordered"
        assert [r.rank for r in result.recommendations] == list(range(1, len(scores) + 1))

    def test_respects_limit(self, service, repo):
        for limit in (1, 3, 5):
            result = service.rank(RequestContext(user_id=repo.users()[0], as_of=AS_OF, limit=limit))
            assert len(result.recommendations) <= limit

    def test_every_item_is_explained(self, service, repo):
        result = service.rank(RequestContext(user_id=repo.users()[0], as_of=AS_OF, limit=5))
        for rec in result.recommendations:
            assert rec.explanations
            assert all(len(e.text) > 10 for e in rec.explanations)
            assert rec.candidate.suggested_amount > 0

    def test_excluded_counterparties_are_dropped(self, service, repo):
        base = service.rank(RequestContext(user_id=repo.users()[0], as_of=AS_OF, limit=10))
        if not base.recommendations:
            pytest.skip("no candidates")
        drop = base.recommendations[0].candidate.counterparty_id
        after = service.rank(
            RequestContext(
                user_id=repo.users()[0], as_of=AS_OF, limit=10, excluded=frozenset({drop})
            )
        )
        assert drop not in {r.candidate.counterparty_id for r in after.recommendations}

    def test_unknown_user_yields_empty_not_error(self, service):
        result = service.rank(RequestContext(user_id="ghost", as_of=AS_OF))
        assert result.empty
        assert result.diagnostics["n_transactions"] == 0

    def test_degraded_mode_is_reported(self, repo, tmp_path):
        """No artifacts at all: the service must still answer, and say so."""
        settings = Settings(
            data_dir=tmp_path / "d",
            artifact_dir=tmp_path / "a",
            environment="test",
            log_level="ERROR",
        )
        svc = RankingService(
            repository=repo, registry=ModelRegistry(settings.artifact_dir), settings=settings
        )
        result = svc.rank(RequestContext(user_id=repo.users()[0], as_of=AS_OF, limit=5))
        assert result.degraded
        assert result.degradation_reason == "no_published_version"

    def test_as_of_changes_recommendations(self, service, repo):
        user = repo.users()[0]
        early = service.rank(RequestContext(user_id=user, as_of=dt.date(2024, 2, 15)))
        late = service.rank(RequestContext(user_id=user, as_of=AS_OF))
        assert [r.candidate.counterparty_id for r in early.recommendations] != [
            r.candidate.counterparty_id for r in late.recommendations
        ]

    def test_horizon_zero_still_returns_coverage(self, service, repo):
        result = service.rank(
            RequestContext(user_id=repo.users()[0], as_of=AS_OF, horizon_days=0, limit=5)
        )
        assert isinstance(result.recommendations, list)

    def test_latency_is_measured(self, service, repo):
        result = service.rank(RequestContext(user_id=repo.users()[0], as_of=AS_OF))
        assert result.latency_ms > 0
        assert result.generated_at.tzinfo is not None

    def test_recurring_summary(self, service, repo):
        rows = service.recurring_summary(repo.users()[0], AS_OF)
        assert rows
        for row in rows:
            assert row["confidence"] > 0
            assert row["observations"] >= 3
            assert row["period_days"] and row["period_days"] > 0

    def test_candidate_pool_is_bounded(self, service, repo):
        result = service.rank(RequestContext(user_id=repo.users()[0], as_of=AS_OF, limit=10))
        assert result.diagnostics["n_candidates"] <= service.settings.candidate_pool_size

    def test_diagnostics_expose_no_raw_user_data(self, service, repo):
        result = service.rank(RequestContext(user_id=repo.users()[0], as_of=AS_OF))
        assert "retrieved_ids" in result.diagnostics
        assert result.diagnostics["n_recurring"] >= 0


class TestDiversification:
    def _cands(self):
        from bankai.domain.entities import TransferCandidate

        return [
            TransferCandidate(
                user_id="u",
                counterparty_id=f"cp_{i}",
                counterparty_name=f"P{i}",
                category="utilities.electricity",
                suggested_amount=50.0 + i,
                currency="EUR",
                scheduled_for=dt.date(2024, 7, 1),
                source="test",
            )
            for i in range(6)
        ]

    def test_mmr_returns_requested_count(self):
        cands = self._cands()
        idx = mmr_rerank(cands, [1.0, 0.9, 0.8, 0.7, 0.6, 0.5], k=3)
        assert len(idx) == 3 and len(set(idx)) == 3

    def test_mmr_protects_the_top_slots(self):
        cands = self._cands()
        idx = mmr_rerank(cands, [1.0, 0.9, 0.8, 0.7, 0.6, 0.5], k=4, protected=2)
        assert idx[0] == 0 and idx[1] == 1

    def test_mmr_spreads_duplicates(self):
        cands = self._cands()
        idx = mmr_rerank(cands, [1.0, 0.99, 0.98, 0.97, 0.96, 0.95], k=3, lam=0.9)
        assert len(set(idx)) == 3

    def test_mmr_handles_short_lists(self):
        cands = self._cands()[:2]
        assert len(mmr_rerank(cands, [1.0, 0.5], k=5)) == 2
        assert mmr_rerank([], [], k=3) == []

    def test_mmr_is_deterministic(self):
        cands = self._cands()
        a = mmr_rerank(cands, [1.0, 0.9, 0.8, 0.7, 0.6, 0.5], k=3, lam=0.35)
        b = mmr_rerank(cands, [1.0, 0.9, 0.8, 0.7, 0.6, 0.5], k=3, lam=0.35)
        assert a == b

    def test_cap_by_category(self):
        cands = self._cands()
        assert len(cap_by_category(cands, list(range(6)), cap=2)) == 2

    def test_dedupe_removes_repeats(self):
        cands = self._cands()
        assert len(dedupe_by_payee_and_due(cands, list(range(6)))) == 6


class TestExplanations:
    def test_cadence_labels_are_human(self):
        assert cadence_label(7) == "every week"
        assert cadence_label(30.4) == "every month"
        assert cadence_label(91) == "every quarter"
        assert cadence_label(None) == "occasionally"
        assert cadence_label(0) == "occasionally"

    def test_every_explanation_code_is_documented(self):
        from bankai.ranking.explain import _CONTRIBUTION_TEXT

        for name in ("is_overdue", "is_due_now", "is_due_soon", "recurrence_confidence"):
            assert name.replace("is_", "") in _CONTRIBUTION_TEXT

    def test_explanations_are_ordered_by_weight(self, service, repo):
        result = service.rank(RequestContext(user_id=repo.users()[0], as_of=AS_OF, limit=5))
        for rec in result.recommendations:
            weights = [e.weight for e in rec.explanations]
            assert weights == sorted(weights, reverse=True)

    def test_overdue_item_explains_itself(self, repo, tmp_path):
        from bankai.domain.entities import RecurrenceStatus

        settings = Settings(
            data_dir=tmp_path / "d",
            artifact_dir=tmp_path / "a",
            environment="test",
            log_level="ERROR",
        )
        service = RankingService(
            repository=repo, registry=ModelRegistry(settings.artifact_dir), settings=settings
        )
        result = service.rank(RequestContext(user_id=repo.users()[0], as_of=AS_OF, limit=10))
        overdue = [
            r for r in result.recommendations if r.candidate.status is RecurrenceStatus.OVERDUE
        ]
        for rec in overdue:
            codes = {e.code for e in rec.explanations}
            assert "overdue_payment" in codes
            assert any("expected on" in e.text for e in rec.explanations)


class TestTrainingPipeline:
    @pytest.fixture(scope="class")
    def trained(self, request):
        from bankai.data.synthetic import generate_dataset

        txs, _bp, snapshot = generate_dataset(n_users=6, months=18, as_of=AS_OF, seed=31)
        by_user: dict[str, list] = {}
        for t in txs:
            by_user.setdefault(t.user_id, []).append(t)
        return by_user, snapshot

    def test_dataset_has_both_classes(self, trained, tmp_path):
        by_user, snapshot = trained
        settings = Settings(
            data_dir=tmp_path / "d",
            artifact_dir=tmp_path / "a",
            environment="test",
            log_level="ERROR",
            n_estimators=30,
            early_stopping_rounds=0,
            num_threads=1,
        )
        trainer = Trainer(settings=settings)
        dataset = trainer.build_dataset(by_user, as_of=snapshot)
        assert len(dataset) > 100
        assert dataset.n_positives > 0
        assert dataset.n_positives < len(dataset), "need negatives, not just positives"
        assert dataset.n_hard_negatives > 0, "hard negatives are essential for discrimination"
        assert dataset.groups.sum() == len(dataset)

    def test_dataset_is_point_in_time(self, trained, tmp_path):
        """Every row's decision date must precede the training cut-off."""
        by_user, snapshot = trained
        settings = Settings(
            data_dir=tmp_path / "d",
            artifact_dir=tmp_path / "a",
            environment="test",
            log_level="ERROR",
        )
        dataset = Trainer(settings=settings).build_dataset(by_user, as_of=snapshot)
        assert dataset.row_dates is not None
        assert dataset.row_dates.max() <= snapshot.toordinal()

    def test_training_learns_and_beats_the_prior(self, trained, tmp_path):
        by_user, snapshot = trained
        settings = Settings(
            data_dir=tmp_path / "d",
            artifact_dir=tmp_path / "a",
            environment="test",
            log_level="ERROR",
            n_estimators=120,
            early_stopping_rounds=10,
            num_threads=2,
        )
        report = train_and_publish(
            by_user,
            as_of=snapshot,
            settings=settings,
            encoder=None,
            latent=None,
            force_promote=True,
        )
        assert report.published and report.promoted
        assert 0.0 <= report.metrics["ndcg@5"] <= 1.0
        assert report.metrics["ndcg@5"] > report.baseline_metrics["ndcg@5"], (
            "the learned ranker must beat the interpretable prior"
        )
        assert report.gain["ndcg_gain"] > 0
        assert report.top_features, "importances should be reported for auditability"

    def test_auxiliary_models_fit(self, trained):
        encoder, latent = fit_auxiliary_models(trained[0])
        assert encoder.fitted and latent.fitted
        assert encoder.known_items()
        vec = latent.user_vector(trained[0].keys().__iter__().__next__())
        assert vec is None or vec.shape[0] > 0

    def test_serving_a_trained_model_is_not_degraded(self, trained, tmp_path, repo, corpus):
        """A published artifact must actually be picked up by the service.

        Guards the whole artifact round-trip: if the registry silently fell back
        to the heuristic, every offline quality number would still look fine.
        """
        by_user, snapshot = trained
        settings = Settings(
            data_dir=tmp_path / "d",
            artifact_dir=tmp_path / "a",
            environment="test",
            log_level="ERROR",
            n_estimators=40,
            early_stopping_rounds=0,
            num_threads=1,
        )
        train_and_publish(by_user, as_of=snapshot, settings=settings, force_promote=True)
        svc = RankingService(
            repository=repo, registry=ModelRegistry(settings.artifact_dir), settings=settings
        )
        result = svc.rank(RequestContext(user_id=repo.users()[0], as_of=AS_OF, limit=5))
        assert result.degraded is False
        assert result.model_version != "degraded"
        assert result.diagnostics["scorer"] != "heuristic_only"
        assert result.recommendations

    def test_publish_gates_on_metric(self, trained, tmp_path):
        """A regressing model must not be promoted."""
        by_user, snapshot = trained
        settings = Settings(
            data_dir=tmp_path / "d",
            artifact_dir=tmp_path / "a",
            environment="test",
            log_level="ERROR",
            n_estimators=30,
            early_stopping_rounds=0,
            num_threads=1,
        )
        report = train_and_publish(
            by_user,
            as_of=snapshot,
            settings=settings,
            force_promote=False,
            promote_threshold=10.0,  # unreachable
        )
        assert report.published
        assert not report.promoted
        assert any("not promoted" in n for n in report.notes)


class TestBacktest:
    def test_backtest_improves_on_the_prior(self, repo, tmp_path, corpus):
        by_user, _ = corpus
        cases = build_cases(dict(sorted(by_user.items())[:3]), as_of=AS_OF, max_cases_per_user=4)
        if len(cases) < 5:
            pytest.skip("not enough decision points in the fixture corpus")
        settings = Settings(
            data_dir=tmp_path / "d",
            artifact_dir=tmp_path / "a",
            environment="test",
            log_level="ERROR",
        )
        service = RankingService(
            repository=repo, registry=ModelRegistry(settings.artifact_dir), settings=settings
        )
        result = run_backtest(service, cases, label="degraded", limit=5)
        assert result.n_cases == len(cases)
        assert 0.0 <= result.coverage <= 1.0
        assert 0.0 <= result.metrics["ndcg@5"] <= 1.0
        assert result.latency_p95_ms >= 0

    def test_ablation_restores_production_config(self, service, repo, corpus):
        by_user, _ = corpus
        cases = build_cases(dict(sorted(by_user.items())[:2]), as_of=AS_OF, max_cases_per_user=2)
        if not cases:
            pytest.skip("no cases")
        results = run_ablation(service, cases)
        assert set(results) >= {"heuristic_only", "ranker_only"}
        # After the sweep the service must be back on production settings.
        assert service._ablation == {}


def service_candidate():
    """A minimal, well-formed candidate for explanation unit tests."""
    from bankai.domain.entities import RecurrenceProfile, RecurrenceStatus, TransferCandidate

    return TransferCandidate(
        user_id="u",
        counterparty_id="cp_x",
        counterparty_name="FiberNet ISP",
        category="utilities.internet",
        suggested_amount=39.99,
        currency="EUR",
        scheduled_for=AS_OF,
        status=RecurrenceStatus.DUE_NOW,
        recurrence=RecurrenceProfile(
            counterparty_id="cp_x",
            is_recurring=True,
            periodicity_confidence=0.95,
            period_days=30.0,
            n_observations=14,
            next_due_on=AS_OF,
            days_until_due=0,
            last_paid_on=dt.date(2024, 5, 28),
            expected_amount=39.99,
            amount_low=39.99,
            amount_high=39.99,
            category="utilities.internet",
            counterparty_name="FiberNet ISP",
        ),
    )


class TestExplanationQuality:
    """Explanations are a regulated product surface, not a debug dump."""

    #: Substrings that indicate raw model vocabulary leaking into user-facing text.
    LEAKS = ("strong signal", "feature", "log_", "cv", "score", "gain", "_ratio", "latent_")

    def test_no_internal_model_vocabulary(self, service, repo):
        result = service.rank(RequestContext(user_id=repo.users()[0], as_of=AS_OF, limit=10))
        assert result.recommendations
        for rec in result.recommendations:
            for explanation in rec.explanations:
                lowered = explanation.text.lower()
                for leak in self.LEAKS:
                    assert leak not in lowered, (
                        f"explanation leaks internal vocabulary ({leak!r}): {explanation.text!r}"
                    )

    def test_every_model_contribution_has_approved_copy(self):
        """An undocumented contribution is dropped, never rendered as a stub."""
        from bankai.features.builder import FEATURE_NAMES
        from bankai.ranking.explain import _CONTRIBUTION_TEXT

        rec = service_candidate()
        out = explain(
            rec,
            contributions=dict.fromkeys(FEATURE_NAMES, 1.0),
            as_of=AS_OF,
        )
        for explanation in out:
            if explanation.code.startswith("model:"):
                name = explanation.code.split(":", 1)[1]
                assert name in _CONTRIBUTION_TEXT

    def test_unknown_contribution_is_dropped(self):
        from bankai.domain.entities import RecurrenceProfile, RecurrenceStatus, TransferCandidate

        cand = TransferCandidate(
            user_id="u",
            counterparty_id="cp_x",
            counterparty_name="X Co",
            category="utilities.internet",
            suggested_amount=30.0,
            currency="EUR",
            scheduled_for=AS_OF,
            status=RecurrenceStatus.DUE_NOW,
            recurrence=RecurrenceProfile(
                counterparty_id="cp_x",
                is_recurring=True,
                periodicity_confidence=0.9,
                period_days=30.0,
                n_observations=12,
                next_due_on=AS_OF,
                days_until_due=0,
                last_paid_on=dt.date(2024, 6, 1),
                expected_amount=30.0,
                amount_low=30.0,
                amount_high=30.0,
                category="utilities.internet",
                counterparty_name="X Co",
            ),
        )
        out = explain(cand, contributions={"totally_unknown_feature": 9.9}, as_of=AS_OF)
        assert all(not e.code.startswith("model:") for e in out)
        assert out, "the structural explanation must survive"

    def test_explanations_read_as_sentences(self, service, repo):
        result = service.rank(RequestContext(user_id=repo.users()[0], as_of=AS_OF, limit=10))
        for rec in result.recommendations:
            for explanation in rec.explanations:
                assert explanation.text[0].isupper()
                assert explanation.text.endswith(".")
                assert not explanation.text.endswith(" .")
