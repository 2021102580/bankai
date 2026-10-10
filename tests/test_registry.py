

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest

from bankai.domain.entities import ModelManifest
from bankai.features.builder import FEATURE_NAMES
from bankai.models.heuristic import HeuristicScorer
from bankai.models.ranker import LtrRanker, RankerConfig, temporal_split
from bankai.models.registry import ModelRegistry, build_manifest, make_version


def _fit_tiny_ranker(seed: int = 0, ensemble_size: int = 1) -> LtrRanker:
    rng = np.random.default_rng(seed)
    n, n_features = 300, len(FEATURE_NAMES)
    x = rng.normal(size=(n, n_features)).astype(np.float32)
    y = rng.integers(0, 2, size=n)
    groups = np.full(30, 10, dtype=int)
    return LtrRanker(
        RankerConfig(
            n_estimators=10,
            early_stopping_rounds=0,
            num_threads=1,
            random_state=seed,
            ensemble_size=ensemble_size,
        )
    ).fit(x, y, groups)


def _manifest(version: str = "test-1", **kw) -> ModelManifest:
    return build_manifest(
        model_name="ranker",
        version=version,
        trained_until=dt.date(2024, 6, 30),
        n_users=kw.get("n_users", 10),
        n_decision_points=kw.get("n_decision_points", 100),
        n_training_rows=kw.get("n_rows", 1000),
        metrics=kw.get("metrics", {"ndcg@5": 0.9}),
    )


class TestEmptyRegistry:
    def test_loads_degraded_when_nothing_published(self, tmp_path):
        loaded = ModelRegistry(tmp_path).load()
        assert loaded.healthy is False
        assert loaded.degradation_reason == "no_published_version"
        assert isinstance(loaded.ranker, HeuristicScorer)

    def test_degraded_model_still_scores(self, tmp_path):
        """The endpoint must work with zero artifacts on disk."""
        loaded = ModelRegistry(tmp_path).load()
        scores = loaded.ranker.score(np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32))
        assert scores.shape == (3,)
        assert np.isfinite(scores).all()


class TestPublishAndLoad:
    def test_round_trip(self, tmp_path):
        registry = ModelRegistry(tmp_path)
        ranker = _fit_tiny_ranker()
        registry.publish(
            version="v1",
            ranker=ranker,
            encoder=None,
            latent=None,
            fallback=HeuristicScorer(),
            manifest=_manifest("v1"),
        )
        loaded = registry.load()
        assert loaded.healthy
        assert loaded.version == "v1"
        assert loaded.manifest.feature_names == list(FEATURE_NAMES)

    def test_predictions_survive_a_save_load_cycle(self, tmp_path):
        ranker = _fit_tiny_ranker(1)
        x = np.random.default_rng(9).normal(size=(5, len(FEATURE_NAMES))).astype(np.float32)
        before = ranker.raw_scores(x)

        path = tmp_path / "ranker.txt"
        ranker.save(path)
        after = LtrRanker.load(path).raw_scores(x)
        np.testing.assert_allclose(before, after, rtol=1e-9, atol=1e-9)

    def test_current_pointer_advances(self, tmp_path):
        registry = ModelRegistry(tmp_path)
        for version in ("v1", "v2"):
            registry.publish(
                version=version,
                ranker=_fit_tiny_ranker(),
                encoder=None,
                latent=None,
                fallback=HeuristicScorer(),
                manifest=_manifest(version),
            )
        assert registry.current_version() == "v2"
        assert registry.load().version == "v2"

    def test_load_specific_version(self, tmp_path):
        registry = ModelRegistry(tmp_path)
        for version in ("v1", "v2"):
            registry.publish(
                version=version,
                ranker=_fit_tiny_ranker(),
                encoder=None,
                latent=None,
                fallback=HeuristicScorer(),
                manifest=_manifest(version),
            )
        assert registry.load("v1").version == "v1"

    def test_corrupt_ranker_degrades_instead_of_crashing(self, tmp_path):
        registry = ModelRegistry(tmp_path)
        registry.publish(
            version="v1",
            ranker=_fit_tiny_ranker(),
            encoder=None,
            latent=None,
            fallback=HeuristicScorer(),
            manifest=_manifest("v1"),
        )
        (registry._version_dir("v1") / "ranker.txt").write_text("not a model", encoding="utf-8")
        loaded = registry.load()
        assert loaded.healthy is False
        assert "ranker_unloadable" in (loaded.degradation_reason or "")

    def test_missing_manifest_degrades(self, tmp_path):
        registry = ModelRegistry(tmp_path)
        registry.publish(
            version="v1",
            ranker=_fit_tiny_ranker(),
            encoder=None,
            latent=None,
            fallback=HeuristicScorer(),
            manifest=_manifest("v1"),
        )
        (registry._version_dir("v1") / "manifest.json").unlink()
        assert registry.load().healthy is False

    def test_no_staging_directories_left_behind(self, tmp_path):
        registry = ModelRegistry(tmp_path)
        registry.publish(
            version="v1",
            ranker=_fit_tiny_ranker(),
            encoder=None,
            latent=None,
            fallback=HeuristicScorer(),
            manifest=_manifest("v1"),
        )
        staging = [p for p in (tmp_path / "versions").iterdir() if p.name.startswith(".staging")]
        assert not staging

    def test_republish_over_the_same_version_is_atomic(self, tmp_path):
        registry = ModelRegistry(tmp_path)
        for _ in range(2):
            registry.publish(
                version="v1",
                ranker=_fit_tiny_ranker(),
                encoder=None,
                latent=None,
                fallback=HeuristicScorer(),
                manifest=_manifest("v1"),
            )
        assert registry.load().healthy
        assert len(registry.available_versions()) == 1


class TestContractEnforcement:
    def test_feature_contract_mismatch_refuses_to_serve(self, tmp_path):
        """A model trained against a different feature vector must not be used.

        Serving it would produce confidently meaningless scores, and the failure
        would only show up as a drop in completion rate weeks later.
        """
        registry = ModelRegistry(tmp_path)
        registry.publish(
            version="v1",
            ranker=_fit_tiny_ranker(),
            encoder=None,
            latent=None,
            fallback=HeuristicScorer(),
            manifest=_manifest("v1"),
        )
        path = registry._version_dir("v1") / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["feature_names"] = manifest["feature_names"][:-1]  # simulate drift
        path.write_text(json.dumps(manifest), encoding="utf-8")

        loaded = registry.load()
        assert loaded.healthy is False
        assert "feature_contract_mismatch" in (loaded.degradation_reason or "")

    def test_ranker_metadata_contract_mismatch_raises(self, tmp_path):
        ranker = _fit_tiny_ranker(2)
        path = tmp_path / "ranker.txt"
        ranker.save(path)
        meta_path = tmp_path / "ranker.txt.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["feature_names"] = ["oops"]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="contract mismatch"):
            LtrRanker.load(path)


class TestRollback:
    def test_rollback_moves_pointer_backwards(self, tmp_path):
        registry = ModelRegistry(tmp_path)
        for version in ("2024-01-01-a", "2024-06-30-b"):
            registry.publish(
                version=version,
                ranker=_fit_tiny_ranker(),
                encoder=None,
                latent=None,
                fallback=HeuristicScorer(),
                manifest=_manifest(version),
            )
        assert registry.current_version() == "2024-06-30-b"
        rolled = registry.rollback()
        assert rolled.version == "2024-01-01-a"
        assert registry.current_version() == "2024-01-01-a"

    def test_rollback_without_history_is_a_noop(self, tmp_path):
        registry = ModelRegistry(tmp_path)
        registry.publish(
            version="v1",
            ranker=_fit_tiny_ranker(),
            encoder=None,
            latent=None,
            fallback=HeuristicScorer(),
            manifest=_manifest("v1"),
        )
        assert registry.rollback().version == "v1"

    def test_prune_keeps_the_newest(self, tmp_path):
        registry = ModelRegistry(tmp_path)
        for i in range(5):
            version = f"2024-0{i}-01-x"
            registry.publish(
                version=version,
                ranker=_fit_tiny_ranker(i),
                encoder=None,
                latent=None,
                fallback=HeuristicScorer(),
                manifest=_manifest(version),
            )
        removed = registry.prune(keep=2)
        assert len(removed) == 3
        assert len(registry.available_versions()) == 2


class TestVersioning:
    def test_version_is_deterministic(self):
        d = dt.date(2024, 6, 30)
        assert make_version(d, {"ndcg@5": 0.9}) == make_version(d, {"ndcg@5": 0.9})

    def test_version_changes_with_training_date(self):
        assert make_version(dt.date(2024, 6, 30)) != make_version(dt.date(2024, 7, 1))

    def test_version_changes_with_metrics(self):
        d = dt.date(2024, 6, 30)
        assert make_version(d, {"ndcg@5": 0.9}) != make_version(d, {"ndcg@5": 0.8})


class TestTemporalSplit:
    def test_split_is_ordered_in_time(self):
        dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(100)]
        mask = temporal_split(dates, holdout_fraction=0.2)
        assert mask.dtype == bool and mask.size == 100
        last_train = max(d for d, m in zip(dates, mask, strict=True) if m)
        first_val = min(d for d, m in zip(dates, mask, strict=True) if not m)
        assert last_train < first_val, "holdout must be strictly in the future"

    def test_split_respects_fraction(self):
        """The mask marks *training* groups, so it is 1 - holdout_fraction."""
        dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(100)]
        train_fraction = temporal_split(dates, holdout_fraction=0.2).mean()
        assert 0.75 <= train_fraction <= 0.85

    def test_unshuffled_by_design(self):
        """A shuffled split is the bug this function exists to prevent."""
        dates = [dt.date(2024, 1, 1)] * 50 + [dt.date(2024, 12, 1)] * 50
        mask = temporal_split(dates, holdout_fraction=0.2)
        assert mask[:50].all() and not mask[50:].any()

    def test_rejects_invalid_fraction(self):
        with pytest.raises(ValueError):
            temporal_split([dt.date(2024, 1, 1)], holdout_fraction=0.0)
