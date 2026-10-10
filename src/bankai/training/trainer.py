

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field

import numpy as np

from bankai.candidates.generator import CandidateGenerator
from bankai.config import Settings, get_settings
from bankai.evaluation.metrics import compare, evaluate_rankings
from bankai.features.embed import BehaviourEncoder, LatentCollaborative
from bankai.logging_setup import get_logger
from bankai.models.heuristic import HeuristicScorer
from bankai.models.ranker import LtrRanker, RankerConfig, temporal_split
from bankai.models.registry import ModelRegistry, build_manifest, make_version
from bankai.training.dataset_cache import build_key
from bankai.training.dataset_cache import load as load_dataset
from bankai.training.dataset_cache import save as save_dataset
from bankai.training.labeler import (
    LtrDataset,
    build_group,
    build_population_from_transactions,
    find_decision_days,
    register_counterparty_population,
)

log = get_logger(__name__)


@dataclass
class TrainingReport:
    version: str
    published: bool
    promoted: bool
    metrics: dict[str, float] = field(default_factory=dict)
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    gain: dict[str, float] = field(default_factory=dict)
    n_rows: int = 0
    n_groups: int = 0
    n_users: int = 0
    trained_until: dt.date | None = None
    duration_s: float = 0.0
    top_features: list[tuple[str, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "published": self.published,
            "promoted": self.promoted,
            "metrics": {k: round(v, 5) for k, v in self.metrics.items()},
            "baseline_metrics": {k: round(v, 5) for k, v in self.baseline_metrics.items()},
            "gain": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in self.gain.items()},
            "n_rows": self.n_rows,
            "n_groups": self.n_groups,
            "n_users": self.n_users,
            "trained_until": self.trained_until.isoformat() if self.trained_until else None,
            "duration_s": round(self.duration_s, 2),
            "top_features": [(f, round(g, 1)) for f, g in self.top_features],
            "notes": self.notes,
        }


@dataclass
class Split:

    x_train: np.ndarray
    y_train: np.ndarray
    groups_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    groups_val: np.ndarray
    val_item_ids: list[list[str]]
    train_item_ids: list[list[str]]


def group_starts(groups: np.ndarray) -> np.ndarray:
    if groups.size == 0:
        return np.zeros(0, dtype=int)
    return np.concatenate([[0], np.cumsum(groups)[:-1]]).astype(int)


class Trainer:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or ModelRegistry(self.settings.artifact_dir)

    # ------------------------------------------------------------- dataset
    def build_dataset(
        self,
        transactions_by_user: dict[str, list],
        *,
        as_of: dt.date,
        encoder: BehaviourEncoder | None = None,
        latent: LatentCollaborative | None = None,
        max_users: int | None = None,
        use_cache: bool = True,
    ) -> LtrDataset:
        s = self.settings
        cache_path = None
        if use_cache:
            key = build_key(transactions_by_user, as_of=as_of, settings=s, n_users=max_users)
            cache_path = key.path(self.settings.artifact_dir / "cache")
            if cache_path.exists():
                try:
                    dataset = load_dataset(cache_path)
                    log.info(
                        "dataset loaded from cache",
                        extra={"path": str(cache_path), **dataset.summary()},
                    )
                    return dataset
                except Exception as exc:  # a stale cache must never be fatal
                    log.warning("dataset cache unusable, rebuilding", extra={"error": str(exc)})
        population = build_population_from_transactions(
            [t for txs in transactions_by_user.values() for t in txs], min_users=3
        )
        register_counterparty_population(population)
        log.info("payee population", extra={"size": len(population)})

        generator = CandidateGenerator(pool_size=s.candidate_pool_size)
        users = sorted(transactions_by_user)
        if max_users:
            users = users[:max_users]

        blocks: list[np.ndarray] = []
        labels: list[int] = []
        group_sizes: list[int] = []
        row_items: list[str] = []
        row_users: list[str] = []
        row_dates: list[int] = []
        n_hard = n_pos = 0
        skipped = 0

        for user_id in users:
            txs = transactions_by_user[user_id]
            for day in find_decision_days(txs, as_of=as_of):
                built = build_group(
                    user_id,
                    txs,
                    day,
                    generator=generator,
                    encoder=encoder,
                    latent=latent,
                    label_window_days=s.label_window_days,
                    horizon_days=s.default_horizon_days,
                    exploration_negatives=s.exploration_negatives,
                )
                if built is None:
                    skipped += 1
                    continue
                group, matrix = built
                if matrix.shape[0] == 0:
                    continue
                blocks.append(matrix)
                labels.extend(r.label for r in group.rows)
                row_items.extend(r.counterparty_id for r in group.rows)
                row_users.extend(user_id for _ in group.rows)
                row_dates.extend([day.toordinal()] * len(group.rows))
                group_sizes.append(len(group.rows))
                n_hard += sum(1 for r in group.rows if r.is_hard_negative)
                n_pos += group.n_positive

        if not blocks:
            raise ValueError("no decision points were built; check history length and date range")

        dataset = LtrDataset(
            features=np.vstack(blocks),
            labels=np.asarray(labels, dtype=int),
            groups=np.asarray(group_sizes, dtype=int),
            group_meta=row_items,
            group_users=row_users,
            n_hard_negatives=n_hard,
            n_positives=n_pos,
        )
        dataset.row_dates = np.asarray(row_dates, dtype=np.int64)  # type: ignore[attr-defined]
        log.info("dataset built", extra={**dataset.summary(), "skipped_days": skipped})
        if cache_path is not None:
            try:
                save_dataset(dataset, cache_path)
            except Exception:  # pragma: no cover - caching is an optimisation
                log.warning("could not write dataset cache", exc_info=True)
        return dataset

    # ---------------------------------------------------------------- split
    def split(self, dataset: LtrDataset, *, holdout_fraction: float = 0.2) -> Split:
        row_dates: np.ndarray = dataset.row_dates
        starts = group_starts(dataset.groups)
        group_dates = [dt.date.fromordinal(int(row_dates[s])) for s in starts]
        train_groups = temporal_split(group_dates, holdout_fraction)
        row_mask = np.repeat(train_groups, dataset.groups)

        groups_train = dataset.groups[train_groups].astype(int)
        groups_val = dataset.groups[~train_groups].astype(int)
        if groups_train.size == 0 or groups_val.size == 0:
            raise ValueError(
                f"temporal split degenerate (train={groups_train.size}, val={groups_val.size}); "
                "reduce holdout_fraction or supply more history"
            )
        return Split(
            x_train=dataset.features[row_mask],
            y_train=dataset.labels[row_mask],
            groups_train=groups_train,
            x_val=dataset.features[~row_mask],
            y_val=dataset.labels[~row_mask],
            groups_val=groups_val,
            train_item_ids=self._slice_items(dataset, row_mask, groups_train),
            val_item_ids=self._slice_items(dataset, ~row_mask, groups_val),
        )

    @staticmethod
    def _slice_items(
        dataset: LtrDataset, row_mask: np.ndarray, groups: np.ndarray
    ) -> list[list[str]]:
        items = np.asarray(dataset.group_meta, dtype=object)
        idx = np.where(row_mask)[0]
        out: list[list[str]] = []
        pos = 0
        for size in groups:
            size = int(size)
            out.append([str(v) for v in items[idx[pos : pos + size]]])
            pos += size
        return out

    # ------------------------------------------------------------- fitting
    def fit(
        self, dataset: LtrDataset, *, holdout_fraction: float = 0.2
    ) -> tuple[LtrRanker, dict, dict, dict]:
        s = self.settings
        split = self.split(dataset, holdout_fraction=holdout_fraction)

        ranker = LtrRanker(
            RankerConfig(
                lambda_lgb=s.lambda_lgb,
                num_leaves=s.num_leaves,
                min_data_in_leaf=s.min_data_in_leaf,
                feature_fraction=s.feature_fraction,
                ensemble_size=s.ensemble_size,
                bagging_fraction=s.bagging_fraction,
                bagging_freq=s.bagging_freq,
                n_estimators=s.n_estimators,
                learning_rate=s.learning_rate,
                early_stopping_rounds=s.early_stopping_rounds,
                num_threads=s.num_threads,
                random_state=s.random_state,
            )
        )
        ranker.fit(
            split.x_train,
            split.y_train,
            split.groups_train,
            x_val=split.x_val,
            y_val=split.y_val,
            groups_val=split.groups_val,
        )

        base_rankings, base_relevants = self._rank(split.x_val, HeuristicScorer(), split)
        rank_rankings, rank_relevants = self._rank(split.x_val, ranker, split)
        baseline_metrics = evaluate_rankings(base_rankings, base_relevants)
        metrics = evaluate_rankings(rank_rankings, rank_relevants)
        gain = compare(base_rankings, rank_rankings, base_relevants, k=5)
        return (
            ranker,
            metrics,
            baseline_metrics,
            {k: float(v) if isinstance(v, (int, float, bool)) else v for k, v in gain.items()},
        )

    @staticmethod
    def _rank(
        x: np.ndarray, scorer, split: Split
    ) -> tuple[list[list[tuple[str, float]]], list[set[str]]]:
        scores = scorer.score(x)
        labels = split.y_val
        rankings: list[list[tuple[str, float]]] = []
        relevants: list[set[str]] = []
        start = 0
        for g, size in enumerate(split.groups_val):
            size = int(size)
            stop = start + size
            order = np.argsort(-scores[start:stop])
            ids = split.val_item_ids[g]
            rankings.append([(ids[int(i)], float(scores[start + int(i)])) for i in order])
            relevants.append({ids[int(i)] for i in np.where(labels[start:stop] == 1)[0]})
            start = stop
        return rankings, relevants

    # --------------------------------------------------------- publication
    def publish(
        self,
        *,
        ranker: LtrRanker,
        encoder: BehaviourEncoder | None,
        latent: LatentCollaborative | None,
        trained_until: dt.date,
        n_users: int,
        n_groups: int,
        n_rows: int,
        metrics: dict[str, float],
        promote: bool,
        fallback: HeuristicScorer | None = None,
    ) -> TrainingReport:
        version = make_version(
            trained_until, {k: v for k, v in metrics.items() if k.startswith("ndcg")}
        )
        manifest = build_manifest(
            model_name=self.settings.model_name,
            version=version,
            trained_until=trained_until,
            n_users=n_users,
            n_decision_points=n_groups,
            n_training_rows=n_rows,
            metrics=metrics,
        )
        self.registry.publish(
            version=version,
            ranker=ranker,
            encoder=encoder,
            latent=latent,
            fallback=fallback or HeuristicScorer(),
            manifest=manifest,
        )
        if not promote:
            log.warning("artifact written but not promoted", extra={"version": version})
        return TrainingReport(
            version=version,
            published=True,
            promoted=promote,
            metrics=metrics,
            n_rows=n_rows,
            n_groups=n_groups,
            n_users=n_users,
            trained_until=trained_until,
            top_features=ranker.top_features(12),
        )


def train_and_publish(
    transactions_by_user: dict[str, list],
    *,
    as_of: dt.date,
    settings: Settings | None = None,
    encoder: BehaviourEncoder | None = None,
    latent: LatentCollaborative | None = None,
    max_users: int | None = None,
    holdout_fraction: float = 0.2,
    promote_threshold: float = 0.0,
    force_promote: bool = False,
) -> TrainingReport:
    started = time.perf_counter()
    s = settings or get_settings()
    trainer = Trainer(settings=s)

    log.info(
        "building LTR dataset",
        extra={"users": len(transactions_by_user), "as_of": as_of.isoformat()},
    )
    dataset = trainer.build_dataset(
        transactions_by_user, as_of=as_of, encoder=encoder, latent=latent, max_users=max_users
    )
    log.info("dataset ready", extra=dataset.summary())

    ranker, metrics, baseline, gain = trainer.fit(dataset, holdout_fraction=holdout_fraction)
    should_promote = force_promote or (gain.get("ndcg_gain", -1.0) > promote_threshold)

    report = trainer.publish(
        ranker=ranker,
        encoder=encoder,
        latent=latent,
        trained_until=as_of,
        n_users=len(transactions_by_user),
        n_groups=int(dataset.groups.size),
        n_rows=len(dataset),
        metrics=metrics,
        promote=should_promote,
    )
    report.baseline_metrics = baseline
    report.gain = gain
    report.duration_s = time.perf_counter() - started
    if not gain.get("significant", False):
        report.notes.append("NDCG@5 gain is not statistically distinguishable from zero")
    if not should_promote:
        report.notes.append("artifact written but not promoted: below promotion threshold")
    return report


def fit_auxiliary_models(
    transactions_by_user: dict[str, list], *, max_users: int = 800
) -> tuple[BehaviourEncoder, LatentCollaborative]:
    corpus: list[tuple[str, str, str]] = []
    meta: dict[str, dict] = {}
    interactions: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    for user_id, txs in list(transactions_by_user.items())[:max_users]:
        for t in txs:
            if t.counterparty_id in seen:
                continue
            seen.add(t.counterparty_id)
            corpus.append((t.counterparty_id, t.counterparty_name, t.counterparty_category))
            meta[t.counterparty_id] = {
                "name": t.counterparty_name,
                "category": t.counterparty_category,
                "currency": t.currency,
            }
        for t in txs:
            if t.direction.value == "outbound" and t.status.value == "posted":
                weight = float(np.log1p(t.amount) / 50.0) + 0.2
                interactions.append((user_id, t.counterparty_id, weight))

    encoder = BehaviourEncoder().fit(corpus)
    latent = LatentCollaborative().fit(interactions)
    latent.set_item_meta(meta)
    log.info(
        "auxiliary models fitted", extra={"payees": len(corpus), "interactions": len(interactions)}
    )
    return encoder, latent


__all__ = [
    "Split",
    "Trainer",
    "TrainingReport",
    "fit_auxiliary_models",
    "group_starts",
    "train_and_publish",
]
