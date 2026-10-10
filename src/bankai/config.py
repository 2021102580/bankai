

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BANKAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ paths
    data_dir: Path = DEFAULT_DATA_DIR
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR
    database_path: Path | None = None
    database_url: str | None = None

    # ------------------------------------------------------------------ serving
    service_name: str = "bankai-recommender"
    environment: str = "local"  # local | staging | production
    api_version: str = "v1"
    log_level: str = "INFO"
    log_format: str = "json"  # json | console
    expose_debug_payload: bool = True
    request_timeout_ms: int = 1_500
    max_recommendations: int = 10
    default_recommendations: int = 5
    default_horizon_days: int = 14
    max_horizon_days: int = 60
    default_currency: str = "EUR"
    rate_limit_per_minute: int = 240
    cors_origins: list[str] = Field(default_factory=list)

    # ------------------------------------------------------------------ model
    model_name: str = "ranker"
    model_artifact: str = "ranker.txt"
    embedding_artifact: str = "embeddings.npz"
    manifest_artifact: str = "manifest.json"
    fallback_artifact: str = "heuristic.json"
    model_min_version_age_hours: float = 0.0
    model_download_timeout_ms: int = 2_000
    shadow_ranker_weight: float = 0.0
    enable_heuristic_blend: bool = True
    heuristic_blend_weight: float = 0.25

    # ------------------------------------------------------------------ ranking
    candidate_pool_size: int = 60

    exploration_negatives: int = 6
    lambda_lgb: float = 0.55
    num_leaves: int = 31
    min_data_in_leaf: int = 40

    feature_fraction: float = 0.4

    ensemble_size: int = 5
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    n_estimators: int = 400
    learning_rate: float = 0.05
    early_stopping_rounds: int = 50
    num_threads: int = 0  # 0 -> all cores
    random_state: int = 42
    label_window_days: int = 3
    history_months: int = 18
    mmr_lambda: float = 0.35

    # ------------------------------------------------------------------ guards
    insufficient_balance_buffer: float = 50.0
    max_single_transfer: float = 50_000.0
    daily_transfer_limit: float = 20_000.0
    monthly_transfer_limit: float = 100_000.0
    cutoff_hour_local: int = 21
    enable_policy: bool = True
    fail_open_on_budget: bool = False

    # ------------------------------------------------------------------ data gen
    seed: int = 7
    n_users: int = 400
    n_months: int = 18
    as_of: str | None = None  

    @computed_field  
    @property
    def db_path(self) -> Path:
        if self.database_path is not None:
            return Path(self.database_path)
        if self.database_url:
            return self.data_dir / "bankai.db"
        return self.data_dir / "bankai.db"

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"prod", "production"}

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


def reload_settings() -> Settings:
    """Cache-busting reload (used by tests and the CLI)."""
    get_settings.cache_clear()
    return get_settings()


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
