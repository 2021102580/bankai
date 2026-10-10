

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

REGISTRY = CollectorRegistry(auto_describe=True)

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# ------------------------------------------------------------------ traffic
REQUESTS = Counter(
    "bankai_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"],
    registry=REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "bankai_request_duration_seconds",
    "API request latency",
    ["endpoint", "method"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    registry=REGISTRY,
)
INFLIGHT = Gauge(
    "bankai_requests_in_flight",
    "In-flight requests",
    registry=REGISTRY,
)

# ----------------------------------------------------------------- ranking
RECOMMENDATIONS_SERVED = Counter(
    "bankai_recommendations_served_total",
    "Recommendations returned per request, bucketed",
    ["bucket", "status", "degraded"],
    registry=REGISTRY,
)
CANDIDATES_GENERATED = Histogram(
    "bankai_candidates_generated",
    "Candidate pool size produced by the recall stage",
    buckets=(0, 1, 2, 5, 10, 20, 40, 60, 100),
    registry=REGISTRY,
)
POLICY_SUPPRESSED = Counter(
    "bankai_policy_suppressed_total",
    "Candidates removed by the policy engine",
    ["reason"],
    registry=REGISTRY,
)
MODEL_INFO = Gauge(
    "bankai_model_info",
    "Currently served model version (1 = active)",
    ["version", "trained_until"],
    registry=REGISTRY,
)
MODEL_DEGRADED = Gauge(
    "bankai_model_degraded",
    "1 when serving in fallback mode",
    ["reason"],
    registry=REGISTRY,
)
RANK_SCORE = Histogram(
    "bankai_recommendation_score",
    "Distribution of returned recommendation scores",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=REGISTRY,
)
RECURRENCE_STATUS = Counter(
    "bankai_recurrence_status_total",
    "Status assigned to served recommendations",
    ["status"],
    registry=REGISTRY,
)

# ------------------------------------------------------------------- model
FEATURE_DRIFT = Gauge(
    "bankai_feature_mean",
    "Mean value of key features, for drift monitoring",
    ["feature"],
    registry=REGISTRY,
)
TRAINING_ROWS = Gauge(
    "bankai_training_rows",
    "Rows in the most recent training dataset",
    registry=REGISTRY,
)


def score_bucket(score: float) -> str:
    for edge in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        if score < edge:
            return f"lt{edge:.1f}"
    return "gte1.0"


def count_bucket(n: int) -> str:
    for edge in (1, 3, 5, 8):
        if n <= edge:
            return f"le{edge}"
    return "gt8"


def render() -> bytes:
    return generate_latest(REGISTRY)


__all__ = [
    "CANDIDATES_GENERATED",
    "CONTENT_TYPE",
    "INFLIGHT",
    "MODEL_DEGRADED",
    "MODEL_INFO",
    "POLICY_SUPPRESSED",
    "RANK_SCORE",
    "RECOMMENDATIONS_SERVED",
    "RECURRENCE_STATUS",
    "REGISTRY",
    "REQUESTS",
    "REQUEST_LATENCY",
    "render",
    "score_bucket",
]
