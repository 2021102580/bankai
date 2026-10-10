
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from bankai.domain.entities import ModelManifest
from bankai.features.builder import FEATURE_NAMES, N_FEATURES
from bankai.logging_setup import get_logger
from bankai.models.base import BaseScorer
from bankai.models.heuristic import HeuristicScorer
from bankai.models.ranker import LtrRanker

log = get_logger(__name__)

RANKER_FILE = "ranker.txt"
ENCODER_FILE = "encoder.joblib"
LATENT_FILE = "latent.joblib"
MANIFEST_FILE = "manifest.json"
FALLBACK_FILE = "heuristic.json"


@dataclass
class LoadedModel:
    manifest: ModelManifest
    ranker: BaseScorer
    encoder: object | None
    latent: object | None
    fallback: HeuristicScorer
    version: str
    healthy: bool = True
    degradation_reason: str | None = None

    @property
    def uses_fallback_only(self) -> bool:
        return self.healthy is False


def _library_versions() -> dict[str, str]:
    import lightgbm
    import numpy
    import pandas
    import sklearn

    return {
        "lightgbm": lightgbm.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
    }


def make_version(trained_until: dt.date, metrics: dict[str, float] | None = None) -> str:
    payload = json.dumps(
        {"d": trained_until.isoformat(), "m": {k: round(v, 6) for k, v in (metrics or {}).items()}},
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:8]
    return f"{trained_until.isoformat()}-{digest}"


class ModelRegistry:
    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._loaded: LoadedModel | None = None

    # ------------------------------------------------------------------ paths
    def _version_dir(self, version: str) -> Path:
        return self.artifact_dir / "versions" / version

    def current_version(self) -> str | None:
        pointer = self.artifact_dir / "CURRENT"
        if pointer.exists():
            return pointer.read_text(encoding="utf-8").strip() or None
        versions = self.available_versions()
        return versions[-1] if versions else None

    def available_versions(self) -> list[str]:
        root = self.artifact_dir / "versions"
        if not root.exists():
            return []
        return sorted(p.name for p in root.iterdir() if p.is_dir())

    # -------------------------------------------------------------- publishing
    def publish(
        self,
        *,
        version: str,
        ranker: BaseScorer | None,
        encoder: object | None,
        latent: object | None,
        fallback: HeuristicScorer,
        manifest: ModelManifest,
    ) -> Path:
        target = self._version_dir(version)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".staging-{version}-", dir=target.parent))
        try:
            if ranker is not None and hasattr(ranker, "save"):
                ranker.save(staging / RANKER_FILE)
            if encoder is not None and hasattr(encoder, "save"):
                encoder.save(staging / ENCODER_FILE)
            if latent is not None and hasattr(latent, "save"):
                latent.save(staging / LATENT_FILE)
            fallback.save(staging / FALLBACK_FILE)
            (staging / MANIFEST_FILE).write_text(
                manifest.model_dump_json(indent=2), encoding="utf-8"
            )
            if target.exists():
                shutil.rmtree(target)
            os.replace(staging, target)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        tmp_pointer = self.artifact_dir / ".CURRENT.tmp"
        tmp_pointer.write_text(version, encoding="utf-8")
        os.replace(tmp_pointer, self.artifact_dir / "CURRENT")
        log.info("published model version", extra={"version": version, "path": str(target)})
        return target

    # ------------------------------------------------------------------ loading
    def load(self, version: str | None = None, *, strict: bool = False) -> LoadedModel:
        version = version or self.current_version()
        if not version:
            return self._degraded("no_published_version")
        path = self._version_dir(version)
        manifest_path = path / MANIFEST_FILE
        if not manifest_path.exists():
            return self._degraded(f"manifest_missing:{version}")

        manifest = ModelManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

        if manifest.feature_names and list(manifest.feature_names) != list(FEATURE_NAMES):
            return self._degraded(
                f"feature_contract_mismatch:model={len(manifest.feature_names)}:runtime={N_FEATURES}"
            )

        try:
            ranker: BaseScorer = LtrRanker.load(path / RANKER_FILE)
        except Exception as exc:  # pragma: no cover - corrupt artifact
            log.exception("ranker load failed", extra={"version": version})
            return self._degraded(f"ranker_unloadable:{type(exc).__name__}")

        encoder = latent = None
        try:
            if (path / ENCODER_FILE).exists():
                from bankai.features.embed import BehaviourEncoder

                encoder = BehaviourEncoder.load(path / ENCODER_FILE)
            if (path / LATENT_FILE).exists():
                from bankai.features.embed import LatentCollaborative

                latent = LatentCollaborative.load(path / LATENT_FILE)
        except Exception:  # pragma: no cover - optional artifacts
            log.exception("auxiliary artifact load failed", extra={"version": version})

        fallback = (
            HeuristicScorer.load(path / FALLBACK_FILE)
            if (path / FALLBACK_FILE).exists()
            else HeuristicScorer()
        )
        return LoadedModel(
            manifest=manifest,
            ranker=ranker,
            encoder=encoder,
            latent=latent,
            fallback=fallback,
            version=version,
            healthy=True,
        )

    def _degraded(self, reason: str) -> LoadedModel:
        log.warning("serving in degraded mode", extra={"reason": reason})
        return LoadedModel(
            manifest=ModelManifest(
                model_name="heuristic",
                version="degraded",
                created_at=dt.datetime.now(tz=dt.UTC),
                trained_until=dt.date.today(),
            ),
            ranker=HeuristicScorer(),
            encoder=None,
            latent=None,
            fallback=HeuristicScorer(),
            version="degraded",
            healthy=False,
            degradation_reason=reason,
        )

    def get(self) -> LoadedModel:
        if self._loaded is None:
            self._loaded = self.load()
        return self._loaded

    def reload(self, version: str | None = None) -> LoadedModel:
        self._loaded = self.load(version)
        return self._loaded

    def rollback(self) -> LoadedModel:
        versions = self.available_versions()
        current = self.current_version()
        earlier = [v for v in versions if current is not None and v < current]
        if not earlier:
            log.warning("no earlier version to roll back to", extra={"current": current})
            return self.reload(current)
        target = earlier[-1]
        (self.artifact_dir / "CURRENT").write_text(target, encoding="utf-8")
        return self.reload(target)

    def prune(self, keep: int = 5) -> list[str]:
        versions = self.available_versions()
        removed: list[str] = []
        for version in versions[:-keep] if keep > 0 else versions:
            shutil.rmtree(self._version_dir(version), ignore_errors=True)
            removed.append(version)
        return removed


def build_manifest(
    *,
    model_name: str,
    version: str,
    trained_until: dt.date,
    n_users: int,
    n_decision_points: int,
    n_training_rows: int,
    metrics: dict[str, float],
) -> ModelManifest:
    return ModelManifest(
        model_name=model_name,
        version=version,
        created_at=dt.datetime.now(tz=dt.UTC),
        trained_until=trained_until,
        n_users=n_users,
        n_decision_points=n_decision_points,
        n_training_rows=n_training_rows,
        metrics=metrics,
        feature_names=list(FEATURE_NAMES),
        library_versions=_library_versions(),
    )


__all__ = ["LoadedModel", "ModelRegistry", "build_manifest", "make_version"]
