
from __future__ import annotations

import datetime as dt
import shutil
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from bankai.config import Settings
from bankai.data.store import InMemoryRepository
from bankai.domain.entities import Transaction
from bankai.evaluation.backtest import build_cases, run_backtest
from bankai.evaluation.diagnose import diagnose
from bankai.evaluation.metrics import compare
from bankai.features.builder import FEATURE_NAMES
from bankai.features.embed import BehaviourEncoder, LatentCollaborative
from bankai.logging_setup import get_logger
from bankai.models.ranker import LtrRanker, RankerConfig
from bankai.models.registry import ModelRegistry, build_manifest, make_version
from bankai.ranking.service import RankingService
from bankai.training.trainer import Trainer

log = get_logger(__name__)

__all__ = [
    "UserSplit",
    "Variant",
    "VariantResult",
    "evaluate_variant",
    "make_splits",
]


@dataclass(frozen=True)
class UserSplit:
    train: tuple[str, ...]
    dev: tuple[str, ...]
    holdout: tuple[str, ...]

    def as_dict(self) -> dict[str, list[str]]:
        return {"train": list(self.train), "dev": list(self.dev), "holdout": list(self.holdout)}


def make_splits(user_ids: Sequence[str], *, n_dev: int = 20, n_holdout: int = 20) -> UserSplit:
    import hashlib

    def bucket(uid: str) -> int:
        return int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16) % 10_000

    ordered = sorted(user_ids, key=lambda u: (bucket(u), u))
    n = len(ordered)
    if n < n_dev + n_holdout + 4:
        # Too small to split safely: everything trains, nothing is held out.
        return UserSplit(tuple(ordered), (), ())
    holdout = tuple(ordered[-n_holdout:])
    dev = tuple(ordered[-(n_holdout + n_dev) : -n_holdout])
    train = tuple(ordered[: -(n_holdout + n_dev)])
    return UserSplit(train, dev, holdout)


@dataclass(frozen=True)
class Variant:
    name: str
    overrides: dict = field(default_factory=dict)
    ranker: RankerConfig | None = None

    def label(self) -> str:
        if self.overrides:
            inner = ", ".join(f"{k}={v}" for k, v in sorted(self.overrides.items()))
            return f"{self.name} ({inner})"
        return self.name


@dataclass
class VariantResult:
    name: str
    label: str
    metrics: dict[str, float]
    coverage: float
    empty_rate: float
    latency_p95_ms: float
    stage_losses: dict[str, int] = field(default_factory=dict)
    n_rows: int = 0
    n_groups: int = 0
    best_iteration: int | None = None
    top_features: list[tuple[str, float]] = field(default_factory=list)
    rankings: list[list[tuple[str, float]]] = field(default_factory=list)
    relevants: list[set[str]] = field(default_factory=list)
    duration_s: float = 0.0

    def summary(self) -> str:
        m = self.metrics
        return (
            f"{self.name:26s} ndcg@1={m.get('ndcg@1', 0):.4f} ndcg@5={m.get('ndcg@5', 0):.4f} "
            f"hit@1={m.get('hit@1', 0):.4f} hit@3={m.get('hit@3', 0):.4f} "
            f"hit@5={m.get('hit@5', 0):.4f} cov={self.coverage:.4f} empty={self.empty_rate:.4f}"
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "metrics": {k: round(v, 5) for k, v in self.metrics.items()},
            "coverage": round(self.coverage, 5),
            "empty_rate": round(self.empty_rate, 5),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
            "stage_losses": self.stage_losses,
            "n_rows": self.n_rows,
            "n_groups": self.n_groups,
            "best_iteration": self.best_iteration,
            "top_features": [(f, round(g, 1)) for f, g in self.top_features],
            "duration_s": round(self.duration_s, 2),
        }


def _with(settings: Settings, overrides: dict) -> Settings:
    return settings.model_copy(update=overrides) if overrides else settings


def _service_for(
    corpus: dict[str, list[Transaction]],
    *,
    artifact_dir: Path,
    settings: Settings,
) -> RankingService:
    repo = InMemoryRepository()
    for txs in corpus.values():
        repo.add(txs)
    for user_id in corpus:
        repo.set_account(user_id, available_balance=None, currency="EUR")
    return RankingService(repository=repo, registry=ModelRegistry(artifact_dir), settings=settings)


def evaluate_variant(
    variant: Variant,
    *,
    corpus: dict[str, list[Transaction]],
    as_of: dt.date,
    split: UserSplit,
    base_settings: Settings,
    eval_users: Sequence[str],
    encoder: BehaviourEncoder | None = None,
    latent: LatentCollaborative | None = None,
    limit: int = 5,
    horizon_days: int = 14,
    holdout_fraction: float = 0.2,
    diagnose_misses: bool = True,
    dataset: object | None = None,
) -> VariantResult:
    started = time.perf_counter()
    settings = _with(base_settings, variant.overrides)

    scratch = Path(tempfile.mkdtemp(prefix=f"bankai-{variant.name}-"))
    try:
        trainer = Trainer(settings=_with(settings, {"artifact_dir": scratch}))
        train_corpus = {u: corpus[u] for u in split.train if u in corpus}
        if dataset is None or not _dataset_compatible(dataset, settings, variant):
            dataset = trainer.build_dataset(
                train_corpus,
                as_of=as_of,
                encoder=encoder,
                latent=latent,
                use_cache=False,
            )
        ranker_cfg = variant.ranker or RankerConfig(
            lambda_lgb=settings.lambda_lgb,
            num_leaves=settings.num_leaves,
            min_data_in_leaf=settings.min_data_in_leaf,
            feature_fraction=settings.feature_fraction,
            bagging_fraction=settings.bagging_fraction,
            bagging_freq=settings.bagging_freq,
            n_estimators=settings.n_estimators,
            learning_rate=settings.learning_rate,
            early_stopping_rounds=settings.early_stopping_rounds,
            num_threads=settings.num_threads,
            random_state=settings.random_state,
        )
        ranker = LtrRanker(ranker_cfg)
        split_obj = trainer.split(dataset, holdout_fraction=holdout_fraction)
        ranker.fit(
            split_obj.x_train,
            split_obj.y_train,
            split_obj.groups_train,
            x_val=split_obj.x_val,
            y_val=split_obj.y_val,
            groups_val=split_obj.groups_val,
        )

        registry = ModelRegistry(scratch)
        registry.publish(
            version=make_version(as_of),
            ranker=ranker,
            encoder=None,
            latent=None,
            fallback=__import__(
                "bankai.models.heuristic", fromlist=["HeuristicScorer"]
            ).HeuristicScorer(),
            manifest=build_manifest(
                model_name="ranker",
                version="sweep",
                trained_until=as_of,
                n_users=len(train_corpus),
                n_decision_points=int(dataset.groups.size),
                n_training_rows=len(dataset),
                metrics={},
            ),
        )

        eval_corpus = {u: corpus[u] for u in eval_users if u in corpus}
        service = _service_for(eval_corpus, artifact_dir=scratch, settings=settings)
        cases = build_cases(eval_corpus, as_of=as_of, label_window_days=settings.label_window_days)
        result = run_backtest(
            service, cases, label=variant.name, limit=limit, horizon_days=horizon_days
        )

        stage_losses: dict[str, int] = {}
        if diagnose_misses:
            report = diagnose(service, cases, limit=limit, horizon_days=horizon_days)
            stage_losses = dict(report.stage_losses)

        return VariantResult(
            name=variant.name,
            label=variant.label(),
            metrics=result.metrics,
            coverage=result.coverage,
            empty_rate=result.empty_rate,
            latency_p95_ms=result.latency_p95_ms,
            stage_losses=stage_losses,
            n_rows=len(dataset),
            n_groups=int(dataset.groups.size),
            best_iteration=ranker.best_iteration,
            top_features=ranker.top_features(12),
            rankings=result.rankings,
            relevants=result.relevants,
            duration_s=time.perf_counter() - started,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


_DATASET_SETTINGS = (
    "candidate_pool_size",
    "label_window_days",
    "default_horizon_days",
    "exploration_negatives",
)


def _dataset_compatible(dataset, settings: Settings, variant: Variant) -> bool:
    for name in _DATASET_SETTINGS:
        override = variant.overrides.get(name)
        if override is not None and override != getattr(settings, name):
            return False
    expected = getattr(dataset, "n_features", None)
    return expected is None or expected == len(FEATURE_NAMES)


def paired_gain(baseline: VariantResult, candidate: VariantResult) -> dict[str, float]:
    return compare(baseline.rankings, candidate.rankings, baseline.relevants, k=5)


def render_table(results: Sequence[VariantResult]) -> str:
    lines = [
        "",
        "=" * 96,
        "VARIANT COMPARISON",
        "=" * 96,
        f"  {'variant':26s} {'ndcg@1':>8s} {'ndcg@5':>8s} {'hit@1':>8s} "
        f"{'hit@3':>8s} {'hit@5':>8s} {'coverage':>9s} {'rows':>8s}",
    ]
    for r in results:
        m = r.metrics
        lines.append(
            f"  {r.name:26s} {m.get('ndcg@1', 0):8.4f} {m.get('ndcg@5', 0):8.4f} "
            f"{m.get('hit@1', 0):8.4f} {m.get('hit@3', 0):8.4f} {m.get('hit@5', 0):8.4f} "
            f"{r.coverage:9.4f} {r.n_rows:8d}"
        )
    return "\n".join(lines)
