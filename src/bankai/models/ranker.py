

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bankai.features.builder import FEATURE_NAMES, N_FEATURES
from bankai.models.base import BaseScorer, normalise

try:
    import lightgbm as lgb

    _LGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LGB_AVAILABLE = False

ENSEMBLE_SUFFIX = ".ensemble.json"


@dataclass
class RankerConfig:
    lambda_lgb: float = 0.55
    num_leaves: int = 31
    min_data_in_leaf: int = 40
    feature_fraction: float = 0.4
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    n_estimators: int = 400
    learning_rate: float = 0.05
    early_stopping_rounds: int = 50
    num_threads: int = 0
    random_state: int = 42
    objective: str = "lambdarank"
    metric: str = "ndcg"
    eval_at: tuple[int, ...] = (1, 3, 5)
    ensemble_size: int = 5

    def to_params(self) -> dict:
        return {
            "objective": self.objective,
            "metric": self.metric,
            "eval_at": list(self.eval_at),
            "lambdarank_truncation_level": 30,
            "label_gain": [0, 1],
            "lambdarank_lambda_position": 0.0,
            "lambda_lgb": self.lambda_lgb,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "colsample_bytree": self.feature_fraction,
            "subsample": self.bagging_fraction,
            "subsample_freq": self.bagging_freq,
            "learning_rate": self.learning_rate,
            "num_threads": self.num_threads,
            "verbosity": -1,
            "force_col_wise": True,
            "deterministic": True,
        }


class LtrRanker(BaseScorer):
    name = "ranker"

    def __init__(self, config: RankerConfig | None = None) -> None:
        if not _LGB_AVAILABLE:  # pragma: no cover
            raise RuntimeError("lightgbm is required for LtrRanker")
        self.config = config or RankerConfig()
        self._boosters: list = []
        self._best_iterations: list[int] = []
        self.n_features = N_FEATURES

    @property
    def trained(self) -> bool:
        return bool(self._boosters)

    @property
    def best_iteration(self) -> int | None:
        return max(self._best_iterations) if self._best_iterations else None

    @property
    def n_members(self) -> int:
        return len(self._boosters)

    # -------------------------------------------------------------- training
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        *,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        groups_val: np.ndarray | None = None,
    ) -> LtrRanker:
        if x.ndim != 2 or x.shape[1] != N_FEATURES:
            raise ValueError(f"expected (n, {N_FEATURES}) features, got {x.shape}")
        if int(groups.sum()) != x.shape[0]:
            raise ValueError(f"group sizes sum to {int(groups.sum())}, expected {x.shape[0]} rows")
        if x_val is not None and groups_val is not None and int(groups_val.sum()) != x_val.shape[0]:
            raise ValueError("validation group sizes do not match the row count")

        boosters = []
        iterations = []
        for member in range(max(1, int(self.config.ensemble_size))):
            seed = self.config.random_state + member
            params = self.config.to_params() | {
                "random_state": seed,
                "bagging_seed": seed + 7,
                "feature_fraction_seed": seed + 13,
            }
            train_set = lgb.Dataset(x, label=y, group=groups, feature_name=list(FEATURE_NAMES))
            valid_sets: list = []
            valid_names: list[str] = []
            callbacks: list = [lgb.log_evaluation(period=0)]
            if x_val is not None and y_val is not None and groups_val is not None:
                valid_sets = [
                    lgb.Dataset(
                        x_val,
                        label=y_val,
                        group=groups_val,
                        feature_name=list(FEATURE_NAMES),
                        reference=train_set,
                    )
                ]
                valid_names = ["valid"]
                if self.config.early_stopping_rounds > 0:
                    callbacks.append(
                        lgb.early_stopping(
                            self.config.early_stopping_rounds,
                            verbose=False,
                            first_metric_only=True,
                        )
                    )
            booster = lgb.train(
                params,
                train_set,
                num_boost_round=self.config.n_estimators,
                valid_sets=valid_sets or None,
                valid_names=valid_names or None,
                callbacks=callbacks,
            )
            boosters.append(booster)
            iterations.append(booster.best_iteration or self.config.n_estimators)
        self._boosters = boosters
        self._best_iterations = iterations
        return self

    # -------------------------------------------------------------- inference
    def _require(self) -> list:
        if not self._boosters:
            raise RuntimeError("LtrRanker is not fitted")
        return self._boosters

    def raw_scores(self, features: np.ndarray) -> np.ndarray:
        if features.size == 0:
            return np.zeros(0, dtype=float)
        parts = [
            normalise(np.asarray(booster.predict(features, num_iteration=best), dtype=float))
            for booster, best in zip(self._require(), self._best_iterations, strict=True)
        ]
        return np.mean(parts, axis=0)

    def score(self, features: np.ndarray) -> np.ndarray:
        return normalise(self.raw_scores(features))

    def explain(self, features: np.ndarray) -> list[dict[str, float]]:
        """Mean SHAP contribution across members (exact tree walk, not sampled)."""
        if not self.trained or features.size == 0:
            return []
        total = np.zeros((len(features), N_FEATURES), dtype=float)
        for booster in self._require():
            # pred_contrib -> [n_rows, n_features + 1]; the last column is bias.
            total += np.asarray(booster.predict(features, pred_contrib=True), dtype=float)[:, :-1]
        total /= len(self._boosters)
        return [
            {FEATURE_NAMES[j]: float(total[i, j]) for j in range(N_FEATURES)}
            for i in range(len(features))
        ]

    # -------------------------------------------------------------- metadata
    def feature_importances(self) -> dict[str, float]:
        if not self.trained:
            return {}
        accum: dict[str, float] = {}
        for booster in self._require():
            gains = booster.feature_importance(importance_type="gain")
            for i, name in enumerate(FEATURE_NAMES):
                if i < gains.size:
                    accum[name] = accum.get(name, 0.0) + float(gains[i])
        n = len(self._boosters)
        return {k: v / n for k, v in accum.items()}

    def top_features(self, k: int = 15) -> list[tuple[str, float]]:
        return sorted(self.feature_importances().items(), key=lambda kv: -kv[1])[:k]

    def eval_history(self) -> dict[str, list[float]]:
        if not self.trained:
            return {}
        out: dict[str, list[float]] = {}
        for booster in self._require():
            for key, values in booster.evals_result_.items():
                out.setdefault(key, []).extend(values)
        return out

    # ----------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        boosters = self._require()
        path = Path(path)
        if len(boosters) == 1:
            boosters[0].save_model(str(path))
        else:
            path.with_suffix(ENSEMBLE_SUFFIX).write_text(
                json.dumps(
                    {
                        "ensemble": [b.model_to_string() for b in boosters],
                        "best_iterations": self._best_iterations,
                    }
                ),
                encoding="utf-8",
            )
        Path(str(path) + ".meta.json").write_text(
            json.dumps(
                {
                    "best_iteration": self.best_iteration,
                    "ensemble_size": len(boosters),
                    "artifact_kind": "ensemble" if len(boosters) > 1 else "booster",
                    "config": self.config.__dict__,
                    "feature_names": list(FEATURE_NAMES),
                    "n_features": N_FEATURES,
                },
                indent=2,
                default=list,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> LtrRanker:
        path = Path(path)
        meta_path = Path(str(path) + ".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        raw = meta.get("config") or {}
        valid = {k: v for k, v in raw.items() if k in RankerConfig.__dataclass_fields__}
        if "eval_at" in valid:
            valid["eval_at"] = tuple(valid["eval_at"])
        obj = cls(RankerConfig(**valid))

        contract = meta.get("feature_names")
        if contract and list(contract) != list(FEATURE_NAMES):
            raise ValueError(
                f"feature contract mismatch: model={len(contract)} runtime={N_FEATURES}"
            )

        ensemble_path = path.with_suffix(ENSEMBLE_SUFFIX)
        kind = meta.get("artifact_kind")
        if kind is None:  # legacy single-model artifact with no marker
            kind = "ensemble" if ensemble_path.exists() else "booster"
        if kind == "ensemble":
            if not ensemble_path.exists():
                raise ValueError("manifest declares an ensemble but the model file is missing")
            payload = json.loads(ensemble_path.read_text(encoding="utf-8"))
            obj._boosters = [lgb.Booster(model_str=m) for m in payload["ensemble"]]
            obj._best_iterations = [int(v) for v in payload.get("best_iterations") or []]
        else:
            if not path.exists():
                raise ValueError("manifest declares a booster but the model file is missing")
            booster = lgb.Booster(model_file=str(path))
            obj._boosters = [booster]
            obj._best_iterations = [booster.best_iteration or obj.config.n_estimators]
        if not obj._boosters:
            raise ValueError("artifact contains no boosters")
        if not obj._best_iterations:
            obj._best_iterations = [obj.config.n_estimators] * len(obj._boosters)
        return obj


def temporal_split(as_of_by_group: list[dt.date], holdout_fraction: float = 0.2) -> np.ndarray:
    if not as_of_by_group:
        return np.zeros(0, dtype=bool)
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")
    ordered = sorted(as_of_by_group)
    cut_index = int(len(ordered) * (1.0 - holdout_fraction))
    cutoff = ordered[min(max(cut_index, 0), len(ordered) - 1)]
    return np.asarray([d < cutoff for d in as_of_by_group], dtype=bool)


__all__ = ["ENSEMBLE_SUFFIX", "LtrRanker", "RankerConfig", "temporal_split"]
