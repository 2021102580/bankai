

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from bankai.domain.entities import (
    RecurrenceProfile,
    RecurrenceStatus,
    TransferCandidate,
)
from bankai.features.embed import (
    BehaviourEncoder,
    LatentCollaborative,
    UserProfile,
)
from bankai.features.recurrence import discover_cycles
from bankai.logging_setup import get_logger

log = get_logger(__name__)

CATEGORY_AFFINITY_PRIOR: dict[str, float] = {
    "housing.rent": 1.00,
    "utilities.internet": 0.98,
    "utilities.phone": 0.95,
    "utilities.electricity": 0.95,
    "utilities.water": 0.92,
    "utilities.heating": 0.93,
    "utilities.gas": 0.93,
    "utilities.security": 0.88,
    "loans.installment": 0.95,
    "loans.credit_card": 0.94,
    "insurance.health": 0.90,
    "insurance.home": 0.90,
    "taxes": 0.92,
    "education.tuition": 0.90,
    "savings.transfer": 0.80,
    "entertainment.tv": 0.85,
    "entertainment.video": 0.80,
    "entertainment.audio": 0.72,
    "services.cloud": 0.70,
    "services.software": 0.72,
    "health.fitness": 0.62,
    "health.pharmacy": 0.70,
    "transport.fuel": 0.55,
    "transport.parking": 0.55,
    "transport.toll": 0.60,
    "shopping.grocery": 0.45,
    "dining.coffee": 0.25,
    "shopping.books": 0.35,
    "shopping.pets": 0.35,
    "travel.lodging": 0.40,
    "p2p.transfer": 0.45,
    "shopping.general": 0.35,
    "misc": 0.30,
}
DEFAULT_CATEGORY_PRIOR = 0.40

CATEGORY_CAP = 3
MIN_CANDIDATE_SCORE = 0.12


def category_prior(category: str) -> float:
    return CATEGORY_AFFINITY_PRIOR.get(category, DEFAULT_CATEGORY_PRIOR)


@dataclass
class GenerationResult:
    candidates: list[TransferCandidate]
    recall_mix: dict[str, int] = field(default_factory=dict)
    coverage_note: str = ""

    @property
    def size(self) -> int:
        return len(self.candidates)


class CandidateGenerator:
    def __init__(
        self,
        *,
        pool_size: int = 60,
        category_cap: int = CATEGORY_CAP,
        min_score: float = MIN_CANDIDATE_SCORE,
        encoder: BehaviourEncoder | None = None,
        latent: LatentCollaborative | None = None,
    ) -> None:
        self.pool_size = pool_size
        self.category_cap = category_cap
        self.min_score = min_score
        self.encoder = encoder
        self.latent = latent

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _scheduled_for(profile: RecurrenceProfile, as_of: dt.date, horizon_days: int) -> dt.date:
        if profile.next_due_on is None:
            return as_of
        if profile.status is RecurrenceStatus.OVERDUE:
            return as_of
        if profile.status is RecurrenceStatus.RECENTLY_PAID:
            return profile.next_due_on
        return max(profile.next_due_on, as_of)

    def _build(
        self,
        profile: UserProfile,
        rec: RecurrenceProfile,
        *,
        as_of: dt.date,
        horizon_days: int,
        source: str,
        base_score: float,
        extra: dict | None = None,
    ) -> TransferCandidate | None:
        series = profile.series.get(rec.counterparty_id)
        amount = rec.expected_amount
        if amount is None and series is not None:
            amount = series.median_amount()
        if amount is None or amount <= 0:
            return None
        currency = series.currency if series is not None else "EUR"
        return TransferCandidate(
            user_id=profile.user_id,
            counterparty_id=rec.counterparty_id,
            counterparty_name=rec.counterparty_name or (series.counterparty_name if series else ""),
            category=rec.category,
            suggested_amount=round(float(amount), 2),
            currency=currency,
            scheduled_for=self._scheduled_for(rec, as_of, horizon_days),
            status=rec.status,
            recurrence=rec,
            source=source,
            base_score=float(np.clip(base_score, 0.0, 1.0)),
            provenance=extra or {},
        )

    # -------------------------------------------------------------- detectors
    def _recurring(
        self,
        profile: UserProfile,
        recs: dict[str, RecurrenceProfile],
        *,
        as_of: dt.date,
        horizon_days: int,
    ) -> list[TransferCandidate]:
        out: list[TransferCandidate] = []
        for rec in recs.values():
            if not rec.is_recurring or rec.n_observations < 3:
                continue
            if rec.status not in (
                RecurrenceStatus.OVERDUE,
                RecurrenceStatus.DUE_NOW,
                RecurrenceStatus.DUE_SOON,
                RecurrenceStatus.RECENTLY_PAID,
                RecurrenceStatus.RECURRING,
            ):
                continue
            urgency = {
                RecurrenceStatus.OVERDUE: 1.00,
                RecurrenceStatus.DUE_NOW: 0.95,
                RecurrenceStatus.DUE_SOON: 0.80,
                RecurrenceStatus.RECENTLY_PAID: 0.30,
                RecurrenceStatus.RECURRING: 0.18,
            }[rec.status]
            score = 0.55 * rec.confidence + 0.30 * urgency + 0.15 * category_prior(rec.category)
            cand = self._build(
                profile,
                rec,
                as_of=as_of,
                horizon_days=horizon_days,
                source="recurring",
                base_score=score,
            )
            if cand:
                out.append(cand)
        return out

    def _inhabitual(
        self,
        profile: UserProfile,
        recs: dict[str, RecurrenceProfile],
        *,
        as_of: dt.date,
        horizon_days: int,
    ) -> list[TransferCandidate]:
        out: list[TransferCandidate] = []
        for cp_id, series in profile.series.items():
            rec = recs.get(cp_id)
            if rec is None or rec.is_recurring or rec.n_observations < 3:
                continue
            recent = series.count_in_window(as_of, 90)
            if recent < 3:
                continue
            total = series.total_amount()
            prior = category_prior(rec.category)
            # Habit strength: how reliably this payee is used at all.
            habit = min(1.0, recent / 8.0) * min(1.0, rec.n_observations / 10.0)
            if habit < 0.15:
                continue
            score = 0.55 * habit + 0.30 * prior + 0.15 * min(1.0, total / 1000.0)
            cand = self._build(
                profile,
                rec,
                as_of=as_of,
                horizon_days=horizon_days,
                source="inhabitual",
                base_score=score * 0.6,
            )
            if cand:
                out.append(cand)
        return out

    def _dormant(
        self,
        profile: UserProfile,
        recs: dict[str, RecurrenceProfile],
        *,
        as_of: dt.date,
        horizon_days: int,
    ) -> list[TransferCandidate]:
        out: list[TransferCandidate] = []
        for cp_id in profile.series:
            rec = recs.get(cp_id)
            if rec is None or rec.is_recurring or rec.n_observations < 3:
                continue
            if not rec.period_days or rec.periodicity_confidence < 0.55:
                continue
            if rec.period_days > 400:
                continue
            staleness = rec.days_until_due or 0
            if staleness < -rec.period_days * 3.0:
                continue
            score = (
                0.45 * rec.periodicity_confidence
                + 0.30 * category_prior(rec.category)
                + 0.25 * min(1.0, rec.amount_stability)
            )
            cand = self._build(
                profile,
                rec,
                as_of=as_of,
                horizon_days=horizon_days,
                source="dormant",
                base_score=score * 0.45,
                extra={"days_behind": -staleness, "period": rec.period_days},
            )
            if cand:
                out.append(cand)
        return out

    def _recent(
        self,
        profile: UserProfile,
        recs: dict[str, RecurrenceProfile],
        *,
        as_of: dt.date,
        horizon_days: int,
    ) -> list[TransferCandidate]:
        out: list[TransferCandidate] = []
        for cp_id, series in profile.series.items():
            rec = recs.get(cp_id)
            if rec is None:
                continue
            age = series.days_since_last(as_of)
            if age is None or age > 10:
                continue
            prior = category_prior(rec.category)
            if prior < 0.45:
                continue
            score = 0.55 * max(0.0, 1.0 - age / 10.0) + 0.45 * prior
            cand = self._build(
                profile,
                rec,
                as_of=as_of,
                horizon_days=horizon_days,
                source="recent",
                base_score=score * 0.75,
            )
            if cand:
                out.append(cand.model_copy(update={"scheduled_for": as_of}))
        return out

    def _affinity(
        self,
        profile: UserProfile,
        recs: dict[str, RecurrenceProfile],
        *,
        as_of: dt.date,
        horizon_days: int,
    ) -> list[TransferCandidate]:
        out: list[TransferCandidate] = []
        if self.latent is None or not self.latent.fitted or profile.n_transactions < 8:
            return out
        vector = self.latent.user_vector(profile.user_id)
        if vector is None:
            return out
        known = set(profile.series)
        for cp_id, sim in self.latent.top_for_user(vector, exclude=known, top_k=8):
            if sim <= 0.05:
                continue
            name, category = self.latent.describe(cp_id)
            rec = recs.get(cp_id) or RecurrenceProfile(
                counterparty_id=cp_id,
                is_recurring=False,
                periodicity_confidence=0.0,
                n_observations=0,
                category=category,
                counterparty_name=name,
            )
            cand = self._build(
                profile,
                rec,
                as_of=as_of,
                horizon_days=horizon_days,
                source="affinity",
                base_score=0.10 + 0.25 * float(sim),
            )
            if cand:
                out.append(cand)
        return out

    def _similar_text(
        self,
        profile: UserProfile,
        recs: dict[str, RecurrenceProfile],
        *,
        as_of: dt.date,
        horizon_days: int,
    ) -> list[TransferCandidate]:
        out: list[TransferCandidate] = []
        if self.encoder is None or not self.encoder.fitted:
            return out
        centroid = self.encoder.user_centroid(profile.series, as_of=as_of)
        if not centroid.any():
            return out
        known = set(profile.series)
        for cp_id, sim in self.encoder.similar_to(centroid, exclude=known, top_k=6):
            if sim < 0.92:  # only near-duplicates
                continue
            name, category = self.encoder.describe(cp_id)
            rec = RecurrenceProfile(
                counterparty_id=cp_id,
                is_recurring=False,
                periodicity_confidence=0.0,
                n_observations=0,
                category=category,
                counterparty_name=name,
            )
            cand = self._build(
                profile,
                rec,
                as_of=as_of,
                horizon_days=horizon_days,
                source="similar_text",
                base_score=0.20,
            )
            if cand:
                cand.provenance = {"text_similarity": round(float(sim), 4)}
                out.append(cand)
        return out

    def _user_vector(self, profile: UserProfile) -> np.ndarray | None:
        if self.latent is None:
            return None
        return self.latent.user_vector(profile.user_id)

    # ------------------------------------------------------------------ driver
    def generate(
        self,
        profile: UserProfile,
        recs: dict[str, RecurrenceProfile],
        *,
        as_of: dt.date,
        horizon_days: int = 14,
        excluded: set[str] | None = None,
    ) -> GenerationResult:
        excluded = excluded or set()
        detectors = (
            self._recurring,
            self._dormant,
            self._inhabitual,
            self._recent,
            self._affinity,
            self._similar_text,
        )
        collected: dict[str, TransferCandidate] = {}
        mix: dict[str, int] = defaultdict(int)

        for detector in detectors:
            try:
                found = detector(profile, recs, as_of=as_of, horizon_days=horizon_days)
            except Exception:  # pragma: no cover - a broken detector must not 500
                log.exception("candidate detector failed", extra={"detector": detector.__name__})
                continue
            for cand in found:
                mix[detector.__name__.lstrip("_")] += 1
                if cand.counterparty_id in excluded:
                    continue
                existing = collected.get(cand.counterparty_id)
                if existing is None:
                    collected[cand.counterparty_id] = cand
                else:
                    # Union the evidence: keep the best score, remember all sources.
                    sources = set(existing.source.split("+")) | {cand.source}
                    collected[cand.counterparty_id] = existing.model_copy(
                        update={
                            "base_score": max(existing.base_score, cand.base_score),
                            "source": "+".join(sorted(sources)),
                            "provenance": {**existing.provenance, **cand.provenance},
                        }
                    )

        bundles = discover_cycles(recs)
        for bundle in bundles:
            for cp_id in bundle:
                if cp_id in collected:
                    continue
                rec = recs[cp_id]
                cand = self._build(
                    profile,
                    rec,
                    as_of=as_of,
                    horizon_days=horizon_days,
                    source="bundle",
                    base_score=0.30,
                    extra={"bundle_size": len(bundle)},
                )
                if cand:
                    collected[cp_id] = cand
                    mix["bundle"] += 1

        ranked = sorted(collected.values(), key=lambda c: -c.base_score)
        kept = self._apply_caps(ranked)

        note = "ok"
        if not kept and ranked:
            note = "all_candidates_below_threshold"
        elif not ranked:
            note = "no_history"
        return GenerationResult(candidates=kept, recall_mix=dict(mix), coverage_note=note)

    def _apply_caps(self, ranked: list[TransferCandidate]) -> list[TransferCandidate]:
        by_category: dict[str, int] = defaultdict(int)
        kept: list[TransferCandidate] = []
        for cand in ranked:
            if cand.base_score < self.min_score:
                break
            if by_category[cand.category] >= self.category_cap:
                continue
            by_category[cand.category] += 1
            kept.append(cand)
            if len(kept) >= self.pool_size:
                break
        return kept


__all__ = [
    "CATEGORY_AFFINITY_PRIOR",
    "CandidateGenerator",
    "GenerationResult",
    "category_prior",
]
