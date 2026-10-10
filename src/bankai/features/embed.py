

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from bankai.domain.entities import Transaction
from bankai.features.series import CounterpartySeries, build_series


@dataclass
class UserProfile:
    user_id: str
    as_of: dt.date
    series: dict[str, CounterpartySeries] = field(default_factory=dict)
    inbound_total: float = 0.0
    outbound_total: float = 0.0
    n_transactions: int = 0
    salary_hint: float | None = None
    last_salary_date: dt.date | None = None
    failed_recent: int = 0
    category_spend: dict[str, float] = field(default_factory=dict)
    active_start: dt.date | None = None
    # --- batch-payment behaviour -----------------------------------------
    salary_day_of_month: int | None = None
    batch_day_ordinals: frozenset[int] = frozenset()
    active_day_ordinals: tuple[int, ...] = ()
    n_paid_on_last_active_day: int = 0
    days_since_last_active_day: int | None = None
    payments_per_active_day: float = 0.0

    # ---------------------------------------------------------------- derived
    @property
    def n_counterparties(self) -> int:
        return len(self.series)

    @property
    def mean_monthly_outbound(self) -> float:
        if self.active_start is None:
            return 0.0
        months = max(1.0, ((self.as_of - self.active_start).days / MONTH_APPROX))
        return self.outbound_total / months

    @property
    def median_tx_amount(self) -> float:
        values = np.concatenate([s.amounts for s in self.series.values() if s.n] or [np.zeros(1)])
        return float(np.median(values)) if values.size else 0.0

    def category_share(self, category: str) -> float:
        if self.outbound_total <= 0:
            return 0.0
        return self.category_spend.get(category, 0.0) / self.outbound_total

    def available_balance(self, fallback: float = 0.0) -> float:
        if self.salary_hint is None or self.last_salary_date is None:
            return fallback
        days_since_salary = (self.as_of - self.last_salary_date).days
        accrued = self.salary_hint * max(0.0, days_since_salary / 30.436875)
        spent = 0.0
        for series in self.series.values():
            since = series.amount_in_window(self.as_of, days=max(0, days_since_salary))
            spent += since
        return max(0.0, min(accrued, self.salary_hint) - spent)


MONTH_APPROX = 30.436875


def build_user_profile(
    user_id: str,
    transactions: list[Transaction],
    *,
    as_of: dt.date,
    inclusive: bool = True,
) -> UserProfile:
    cutoff = as_of.toordinal() + (1 if inclusive else 0)
    history = [t for t in transactions if t.value_date.toordinal() < cutoff]
    profile = UserProfile(user_id=user_id, as_of=as_of, n_transactions=len(history))
    if not history:
        return profile

    profile.series = build_series(history)
    profile.active_start = min(t.value_date for t in history)
    profile.outbound_total = sum(t.amount for t in history if t.direction.value == "outbound")
    inbound = [t for t in history if t.direction.value == "inbound"]
    profile.inbound_total = sum(t.amount for t in inbound)

    if inbound:
        by_cp: dict[str, list[Transaction]] = {}
        for t in inbound:
            by_cp.setdefault(t.counterparty_id, []).append(t)
        best = max(
            by_cp.items(),
            key=lambda kv: (len(kv[1]), sum(t.amount for t in kv[1])),
        )
        txs = sorted(best[1], key=lambda t: t.value_date)
        if len(txs) >= 2:
            gaps = np.diff([t.value_date.toordinal() for t in txs])
            if gaps.size and abs(float(np.mean(gaps)) - 30.4) < 6.0:
                profile.salary_hint = float(np.median([t.amount for t in txs]))
                profile.last_salary_date = txs[-1].value_date

    for tx in history:
        if tx.direction.value == "outbound":
            profile.category_spend[tx.counterparty_category] = (
                profile.category_spend.get(tx.counterparty_category, 0.0) + tx.amount
            )
    # --- batch-payment aggregates (point-in-time) -------------------------
    per_day: dict[int, set[str]] = {}
    for tx in history:
        if tx.direction.value == "outbound" and tx.status.value == "posted":
            per_day.setdefault(tx.value_date.toordinal(), set()).add(tx.counterparty_id)
    batch_days = frozenset(ordn for ordn, cps in per_day.items() if len(cps) >= 2)
    active_days = sorted(per_day)
    profile.batch_day_ordinals = batch_days
    profile.active_day_ordinals = tuple(active_days)
    if active_days:
        total_outbound = sum(len(v) for v in per_day.values())
        profile.payments_per_active_day = total_outbound / len(active_days)
        last_active = active_days[-1]
        profile.n_paid_on_last_active_day = len(per_day[last_active])
        profile.days_since_last_active_day = max(0, as_of.toordinal() - last_active)
    if profile.salary_hint is not None and profile.last_salary_date is not None:
        profile.salary_day_of_month = profile.last_salary_date.day

    recent_cutoff = cutoff - 90
    profile.failed_recent = sum(
        1
        for t in history
        if t.value_date.toordinal() >= recent_cutoff and t.status.value == "failed"
    )
    return profile


class BehaviourEncoder:

    def __init__(self, n_components: int = 32, max_features: int = 20_000) -> None:
        self.n_components = n_components
        self.max_features = max_features
        self._vectorizer = None
        self._reducer = None
        self._item_index: dict[str, int] = {}
        self._item_matrix: np.ndarray | None = None
        self._item_meta: dict[str, dict] = {}
        self.fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, corpus: list[tuple[str, str, str]]) -> BehaviourEncoder:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not corpus:
            self.fitted = False
            return self
        texts = [f"{name} {category}" for _, name, category in corpus]
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=self.max_features,
            sublinear_tf=True,
            lowercase=True,
        )
        x = self._vectorizer.fit_transform(texts)
        n_components = int(min(self.n_components, max(2, min(x.shape) - 1)))
        self._reducer = TruncatedSVD(n_components=n_components, random_state=42)
        reduced = self._reducer.fit_transform(x)
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        reduced = reduced / np.maximum(norms, 1e-9)
        self._item_index = {cp_id: i for i, (cp_id, _, _) in enumerate(corpus)}
        self._item_matrix = reduced
        for cp_id, name, category in corpus:
            self._item_meta[cp_id] = {"name": name, "category": category}
        self.fitted = True
        return self

    # ------------------------------------------------------------- transforms
    def encode_item(self, name: str, category: str) -> np.ndarray:
        if not self.fitted or self._reducer is None or self._vectorizer is None:
            return np.zeros(self.n_components, dtype=float)
        x = self._vectorizer.transform([f"{name} {category}"])
        v = self._reducer.transform(x)[0]
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else v

    def encode_items(self, items: list[tuple[str, str, str]]) -> np.ndarray:
        if not items:
            return np.zeros((0, self.n_components), dtype=float)
        return np.vstack([self.encode_item(n, c) for _, n, c in items])

    def item_vector(self, counterparty_id: str) -> np.ndarray | None:
        if not self.fitted or self._item_matrix is None:
            return None
        idx = self._item_index.get(counterparty_id)
        return None if idx is None else self._item_matrix[idx]

    def known_items(self) -> list[str]:
        return list(self._item_index.keys())

    def describe(self, counterparty_id: str) -> tuple[str, str]:
        meta = self._item_meta.get(counterparty_id)
        if not meta:
            return (counterparty_id, "uncategorised")
        return (str(meta.get("name", counterparty_id)), str(meta.get("category", "uncategorised")))

    # ------------------------------------------------------------- similarity
    def user_centroid(
        self, series: dict[str, CounterpartySeries], *, as_of: dt.date, half_life_days: float = 45.0
    ) -> np.ndarray:
        vectors: list[np.ndarray] = []
        weights: list[float] = []
        for cp_id, s in series.items():
            vec = self.item_vector(cp_id)
            if vec is None or s.n == 0:
                continue
            age = max(0, (as_of - (s.last_date or as_of)).days)
            recency = 0.5 ** (age / max(half_life_days, 1e-6))
            amount_weight = float(np.clip(s.total_amount() / 1000.0, 0.1, 6.0))
            vectors.append(vec)
            weights.append(recency * amount_weight)
        if not vectors:
            return np.zeros(self.n_components, dtype=float)
        w = np.asarray(weights, dtype=float)
        centroid = np.average(np.vstack(vectors), axis=0, weights=w)
        n = float(np.linalg.norm(centroid))
        return centroid / n if n > 1e-9 else centroid

    def similar_to(
        self, vector: np.ndarray, *, exclude: set[str] | None = None, top_k: int = 20
    ) -> list[tuple[str, float]]:
        if not self.fitted or self._item_matrix is None or vector.size == 0:
            return []
        sims = self._item_matrix @ vector
        ids = list(self._item_index.keys())
        exclude = exclude or set()
        out: list[tuple[str, float]] = []
        for idx in np.argsort(-sims):
            cp_id = ids[int(idx)]
            if cp_id in exclude:
                continue
            out.append((cp_id, float(sims[int(idx)])))
            if len(out) >= top_k:
                break
        return out

    # ------------------------------------------------------------ persistence
    def save(self, path) -> None:

        assert self.fitted and self._vectorizer is not None and self._reducer is not None
        payload = {
            "vectorizer": self._vectorizer,
            "reducer": self._reducer,
            "item_index": self._item_index,
            "item_matrix": self._item_matrix,
            "item_meta": self._item_meta,
        }
        import joblib

        joblib.dump(payload, path)

    @classmethod
    def load(cls, path) -> BehaviourEncoder:
        import joblib

        payload = joblib.load(path)
        enc = cls()
        enc._vectorizer = payload["vectorizer"]
        enc._reducer = payload["reducer"]
        enc._item_index = payload["item_index"]
        enc._item_matrix = payload["item_matrix"]
        enc._item_meta = payload["item_meta"]
        enc.n_components = enc._item_matrix.shape[1] if enc._item_matrix is not None else 32
        enc.fitted = True
        return enc


class LatentCollaborative:


    def __init__(self, n_components: int = 24) -> None:
        self.n_components = n_components
        self._svd = None
        self._item_factors: np.ndarray | None = None
        self._item_index: dict[str, int] = {}
        self._user_index: dict[str, int] = {}
        self._interactions: list[tuple[str, str, float]] = []
        self._user_factors: np.ndarray | None = None
        self._item_meta: dict[str, dict] = {}
        self.fitted = False

    def fit(self, interactions: list[tuple[str, str, float]]) -> LatentCollaborative:
        from scipy import sparse
        from sklearn.decomposition import TruncatedSVD

        if not interactions:
            return self
        user_ids = sorted({u for u, _, _ in interactions})
        item_ids = sorted({i for _, i, _ in interactions})
        self._item_index = {cp: i for i, cp in enumerate(item_ids)}
        self._user_index = {u: i for i, u in enumerate(user_ids)}
        self._interactions = list(interactions)
        rows = np.asarray([self._user_index[u] for u, _, _ in interactions])
        cols = np.asarray([self._item_index[i] for _, i, _ in interactions])
        vals = np.sqrt(np.asarray([w for _, _, w in interactions], dtype=float))
        matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(item_ids)))
        k = int(min(self.n_components, max(2, min(matrix.shape) - 1)))
        self._svd = TruncatedSVD(n_components=k, random_state=42)
        self._user_factors = self._svd.fit_transform(matrix)
        self._item_factors = self._svd.components_.T
        norms = np.linalg.norm(self._item_factors, axis=1, keepdims=True)
        self._item_factors = self._item_factors / np.maximum(norms, 1e-9)
        self.fitted = True
        return self

    def set_item_meta(self, meta: dict[str, dict]) -> None:
        self._item_meta = meta

    def describe(self, counterparty_id: str) -> tuple[str, str]:
        meta = self._item_meta.get(counterparty_id)
        if not meta:
            return (counterparty_id, "uncategorised")
        return (str(meta.get("name", counterparty_id)), str(meta.get("category", "uncategorised")))

    def user_vector(self, user_id: str) -> np.ndarray | None:
        if not self.fitted or self._user_factors is None or self._svd is None:
            return None
        idx = self._user_index.get(user_id)
        if idx is not None:
            vec = self._user_factors[idx]
            norm = float(np.linalg.norm(vec))
            return vec / norm if norm > 1e-9 else vec
        rows = [(item_id, w) for u, item_id, w in self._interactions if u == user_id]
        if not rows or self._item_factors is None:
            return None
        vec = np.zeros(self._user_factors.shape[1], dtype=float)
        for item_id, weight in rows:
            item_idx = self._item_index.get(item_id)
            if item_idx is not None:
                vec += self._item_factors[item_idx] * float(np.sqrt(weight))
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-9 else vec

    def score(self, user_vector: np.ndarray, counterparty_id: str) -> float:
        if not self.fitted or self._item_factors is None or user_vector is None:
            return 0.0
        idx = self._item_index.get(counterparty_id)
        if idx is None:
            return 0.0
        return float(np.dot(user_vector, self._item_factors[idx]))

    def top_for_user(
        self, user_vector: np.ndarray, exclude: set[str], top_k: int
    ) -> list[tuple[str, float]]:
        if not self.fitted or self._item_factors is None or user_vector is None:
            return []
        sims = self._item_factors @ user_vector
        ids = list(self._item_index.keys())
        order = np.argsort(-sims)[: top_k + len(exclude)]
        return [(ids[int(i)], float(sims[int(i)])) for i in order if ids[int(i)] not in exclude]

    def save(self, path) -> None:
        import joblib

        joblib.dump(
            {
                "svd": self._svd,
                "factors": self._item_factors,
                "item_index": self._item_index,
                "user_index": self._user_index,
                "user_factors": self._user_factors,
                "interactions": self._interactions,
                "item_meta": self._item_meta,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> LatentCollaborative:
        import joblib

        payload = joblib.load(path)
        obj = cls()
        obj._svd = payload["svd"]
        obj._item_factors = payload["factors"]
        obj._item_index = payload["item_index"]
        obj._user_index = payload.get("user_index", {})
        obj._user_factors = payload.get("user_factors")
        obj._interactions = payload.get("interactions", [])
        obj._item_meta = payload.get("item_meta", {})
        obj.fitted = obj._item_factors is not None
        return obj
