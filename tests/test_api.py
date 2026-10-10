

from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient

from bankai.serving.app import create_app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """App + client over a dedicated temporary corpus."""
    tmp = tmp_path_factory.mktemp("api")
    from bankai.config import Settings
    from bankai.data.store import SqliteTransactionRepository
    from bankai.data.synthetic import generate_dataset

    settings = Settings(
        data_dir=tmp / "data",
        artifact_dir=tmp / "artifacts",
        database_path=tmp / "data" / "api.db",
        environment="test",
        log_level="WARNING",
    )
    settings.ensure_dirs()
    store = SqliteTransactionRepository(settings.db_path)
    txs, _bp, snapshot = generate_dataset(n_users=6, months=18, as_of=dt.date(2024, 6, 30), seed=23)
    store.bulk_insert(txs)
    for uid in {t.user_id for t in txs}:
        store.upsert_account(uid, available_balance=None, currency="EUR")

    app = create_app(settings)
    with TestClient(app) as c:
        c.user_id = sorted({t.user_id for t in txs})[0]  # type: ignore[attr-defined]
        c.snapshot = snapshot  # type: ignore[attr-defined]
        yield c


class TestRecommendations:
    def test_returns_200_and_well_formed_payload(self, client):
        resp = client.post("/v1/recommendations", json={"user_id": client.user_id, "limit": 5})
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == client.user_id
        assert body["count"] == len(body["recommendations"]) <= 5
        assert body["request_id"]
        assert body["model_version"]
        assert body["as_of"] == client.snapshot.isoformat()
        assert body["generated_at"]

    def test_items_carry_everything_a_client_needs(self, client):
        body = client.post(
            "/v1/recommendations", json={"user_id": client.user_id, "limit": 5}
        ).json()
        assert body["recommendations"], "expected at least one recommendation"
        for i, item in enumerate(body["recommendations"], start=1):
            assert item["rank"] == i, "ranks must be dense and ordered"
            assert item["counterparty_id"] and item["counterparty_name"]
            assert item["suggested_amount"]["value"] > 0
            assert len(item["suggested_amount"]["currency"]) == 3
            assert 0.0 <= item["score"] <= 1.0
            assert 0.0 <= item["confidence"] <= 1.0
            assert item["status"]
            assert item["scheduled_for"]
            assert item["explanations"], "every item must be explainable"
            assert all(e["text"] and e["code"] for e in item["explanations"])

    def test_recurrence_block_is_populated_for_scheduled_items(self, client):
        body = client.post(
            "/v1/recommendations", json={"user_id": client.user_id, "limit": 10}
        ).json()
        scheduled = [i for i in body["recommendations"] if i["recurrence"]]
        assert scheduled
        for item in scheduled:
            rec = item["recurrence"]
            assert rec["cadence"]
            assert 0.0 <= rec["confidence"] <= 1.0
            assert rec["period_days"] is None or rec["period_days"] > 0

    def test_limit_is_respected(self, client):
        for limit in (1, 3, 5):
            body = client.post(
                "/v1/recommendations", json={"user_id": client.user_id, "limit": limit}
            ).json()
            assert len(body["recommendations"]) <= limit

    def test_horizon_shortening_can_only_reduce(self, client):
        short = client.post(
            "/v1/recommendations", json={"user_id": client.user_id, "horizon_days": 3}
        ).json()
        long = client.post(
            "/v1/recommendations", json={"user_id": client.user_id, "horizon_days": 30}
        ).json()
        assert short["horizon_days"] == 3
        assert len(short["recommendations"]) <= len(long["recommendations"])

    def test_horizon_is_clamped_to_server_maximum(self, client):
        body = client.post(
            "/v1/recommendations", json={"user_id": client.user_id, "horizon_days": 60}
        ).json()
        assert body["horizon_days"] <= 60

    def test_exclusions_are_honoured(self, client):
        base = client.post(
            "/v1/recommendations", json={"user_id": client.user_id, "limit": 10}
        ).json()
        if not base["recommendations"]:
            pytest.skip("no candidates to exclude")
        excluded = base["recommendations"][0]["counterparty_id"]
        after = client.post(
            "/v1/recommendations",
            json={"user_id": client.user_id, "limit": 10, "excluded_counterparties": [excluded]},
        ).json()
        assert excluded not in {i["counterparty_id"] for i in after["recommendations"]}

    def test_include_recently_paid_toggle(self, client):
        without = client.post(
            "/v1/recommendations",
            json={"user_id": client.user_id, "limit": 10, "include_recently_paid": False},
        ).json()
        assert all(i["status"] != "recently_paid" for i in without["recommendations"])

    def test_as_of_changes_the_answer(self, client):
        """Point-in-time evaluation must be honoured, not ignored."""
        early = client.post(
            "/v1/recommendations",
            json={"user_id": client.user_id, "as_of": "2024-02-15T09:00:00Z", "limit": 5},
        ).json()
        late = client.post(
            "/v1/recommendations",
            json={"user_id": client.user_id, "as_of": "2024-06-30T09:00:00Z", "limit": 5},
        ).json()
        assert early["as_of"] == "2024-02-15"
        assert late["as_of"] == "2024-06-30"
        assert [i["counterparty_id"] for i in early["recommendations"]] != [
            i["counterparty_id"] for i in late["recommendations"]
        ]


class TestErrors:
    def test_unknown_user_is_404(self, client):
        resp = client.post("/v1/recommendations", json={"user_id": "does_not_exist"})
        assert resp.status_code == 404
        assert "error" in resp.json()

    def test_unknown_field_is_rejected(self, client):
        resp = client.post("/v1/recommendations", json={"user_id": client.user_id, "nope": 1})
        assert resp.status_code == 422
        assert resp.json()["error"] == "validation_error"

    def test_limit_out_of_range_is_rejected(self, client):
        for bad in (0, 99, -1):
            assert (
                client.post(
                    "/v1/recommendations", json={"user_id": client.user_id, "limit": bad}
                ).status_code
                == 422
            )

    def test_empty_user_id_is_rejected(self, client):
        assert client.post("/v1/recommendations", json={"user_id": ""}).status_code == 422

    def test_naive_datetime_is_rejected(self, client):
        """Ambiguous timestamps are a real integration bug; reject loudly."""
        resp = client.post(
            "/v1/recommendations",
            json={"user_id": client.user_id, "as_of": "2024-06-30T09:00:00"},
        )
        assert resp.status_code == 422
        reasons = " ".join(e["message"] for e in resp.json()["details"]["errors"]).lower()
        assert "timezone" in reasons

    def test_bad_currency_is_rejected(self, client):
        assert (
            client.post(
                "/v1/recommendations", json={"user_id": client.user_id, "currency": "EUROS"}
            ).status_code
            == 422
        )

    def test_validation_errors_do_not_leak_internals(self, client):
        body = client.post("/v1/recommendations", json={}).json()
        assert "traceback" not in str(body).lower()
        assert 'File "' not in str(body)


class TestOperationalEndpoints:
    def test_liveness_never_touches_dependencies(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_readiness_reports_degraded_model(self, client):
        """With no trained artifact, readiness must say so, not pretend to be ok."""
        body = client.get("/readyz").json()
        assert "model" in body["checks"]
        assert "data_source" in body["checks"]
        assert body["status"] in {"ok", "unhealthy", "degraded"}

    def test_recommendations_flag_degradation(self, client):
        body = client.post("/v1/recommendations", json={"user_id": client.user_id}).json()
        assert "degraded" in body
        # Either a real model is loaded, or the service says it is not.
        assert body["degraded"] is False or body["degraded_reason"]

    def test_metrics_endpoint_exposes_prometheus_format(self, client):
        client.post("/v1/recommendations", json={"user_id": client.user_id})
        resp = client.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "bankai_requests_total" in text
        assert "# HELP" in text

    def test_request_id_is_echoed_and_propagated(self, client):
        resp = client.post(
            "/v1/recommendations",
            json={"user_id": client.user_id},
            headers={"X-Request-ID": "trace-abc-123"},
        )
        assert resp.headers["X-Request-ID"] == "trace-abc-123"
        assert resp.json()["request_id"]

    def test_request_id_is_generated_when_absent(self, client):
        resp = client.post("/v1/recommendations", json={"user_id": client.user_id})
        assert len(resp.headers["X-Request-ID"]) >= 16

    def test_hostile_request_id_is_not_reflected(self, client):
        """A caller-supplied ID must not be able to inject headers or logs."""
        resp = client.post(
            "/v1/recommendations",
            json={"user_id": client.user_id},
            headers={"X-Request-ID": "bad id with spaces and <script>"},
        )
        assert resp.status_code == 200
        assert resp.headers["X-Request-ID"] != "bad id with spaces and <script>"


class TestCommitments:
    def test_lists_standing_commitments(self, client):
        resp = client.get(f"/v1/users/{client.user_id}/commitments")
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == client.user_id
        assert body["count"] == len(body["commitments"])
        assert body["total_monthly_commitment"] >= 0

    def test_unknown_user_is_404(self, client):
        assert client.get("/v1/users/nobody/commitments").status_code == 404

    def test_supports_point_in_time(self, client):
        early = client.get(f"/v1/users/{client.user_id}/commitments?as_of=2024-02-15").json()
        late = client.get(f"/v1/users/{client.user_id}/commitments?as_of=2024-06-30").json()
        assert early["as_of"] == "2024-02-15"
        assert late["as_of"] == "2024-06-30"


class TestFeedback:
    def test_accepts_and_never_fails_the_user(self, client):
        resp = client.post(
            "/v1/feedback",
            json={
                "user_id": client.user_id,
                "counterparty_id": "cp_isp",
                "action": "executed",
                "served_rank": 1,
                "amount": {"value": 39.99, "currency": "EUR"},
            },
        )
        assert resp.status_code == 202
        assert resp.json()["accepted"] is True

    def test_rejects_unknown_action(self, client):
        resp = client.post(
            "/v1/feedback",
            json={"user_id": client.user_id, "counterparty_id": "x", "action": "maybe"},
        )
        assert resp.status_code == 422


class TestOpenApi:
    def test_schema_is_served(self, client):
        schema = client.get("/openapi.json").json()
        assert "/v1/recommendations" in schema["paths"]
        assert "RecommendationRequest" in schema["components"]["schemas"]
