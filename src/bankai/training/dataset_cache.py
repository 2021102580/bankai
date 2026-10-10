
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bankai.features.builder import FEATURE_NAMES
from bankai.training.labeler import LtrDataset


@dataclass(frozen=True)
class CacheKey:
    feature_version: str
    corpus_digest: str
    n_users: int
    as_of: str
    label_window_days: int
    horizon_days: int
    candidate_pool_size: int
    exploration_negatives: int
    min_history_days: int

    def digest(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def path(self, root: Path) -> Path:
        return root / f"dataset-{self.digest()}.npz"


def corpus_digest(transactions_by_user: dict[str, list], *, n_users: int | None = None) -> str:
    hasher = hashlib.sha256()
    for user_id in sorted(transactions_by_user)[: n_users or len(transactions_by_user)]:
        txs = transactions_by_user[user_id]
        hasher.update(user_id.encode())
        hasher.update(str(len(txs)).encode())
        if txs:
            hasher.update(txs[0].value_date.isoformat().encode())
            hasher.update(txs[-1].value_date.isoformat().encode())
            hasher.update(str(round(sum(t.amount for t in txs), 2)).encode())
    return hasher.hexdigest()[:16]


def build_key(
    transactions_by_user: dict[str, list],
    *,
    as_of: dt.date,
    settings: object,
    n_users: int | None,
) -> CacheKey:
    return CacheKey(
        feature_version=str(FEATURE_NAMES.__len__()),
        corpus_digest=corpus_digest(transactions_by_user, n_users=n_users),
        n_users=n_users or len(transactions_by_user),
        as_of=as_of.isoformat(),
        label_window_days=settings.label_window_days,
        horizon_days=settings.default_horizon_days,
        candidate_pool_size=settings.candidate_pool_size,
        exploration_negatives=settings.exploration_negatives,
        min_history_days=45,
    )


def save(dataset: LtrDataset, path: Path) -> None:
    assert dataset.row_dates is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=dataset.features,
        labels=dataset.labels,
        groups=dataset.groups,
        row_dates=dataset.row_dates,
        group_meta=np.asarray(dataset.group_meta, dtype=object),
        group_users=np.asarray(dataset.group_users, dtype=object),
        meta=np.asarray(
            [
                dataset.n_positives,
                dataset.n_hard_negatives,
                len(FEATURE_NAMES),
            ],
            dtype=np.int64,
        ),
    )


def load(path: Path) -> LtrDataset:
    with np.load(path, allow_pickle=True) as data:
        features = data["features"]
        n_pos, n_hard, n_features = (int(v) for v in data["meta"])
        if features.shape[1] != n_features:
            raise ValueError(f"cached feature width {features.shape[1]} != contract {n_features}")
        return LtrDataset(
            features=features,
            labels=data["labels"],
            groups=data["groups"],
            group_meta=[str(v) for v in data["group_meta"]],
            group_users=[str(v) for v in data["group_users"]],
            n_hard_negatives=n_hard,
            n_positives=n_pos,
            row_dates=data["row_dates"],
        )


__all__ = ["CacheKey", "build_key", "corpus_digest", "load", "save"]
