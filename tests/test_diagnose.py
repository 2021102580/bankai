

from __future__ import annotations

import pytest

from bankai.config import Settings
from bankai.data.store import InMemoryRepository
from bankai.evaluation.backtest import build_cases
from bankai.evaluation.diagnose import diagnose
from bankai.models.registry import ModelRegistry
from bankai.ranking.service import RankingService, RequestContext

from .conftest import AS_OF


@pytest.fixture
def traced(repo, tmp_path):
    settings = Settings(
        data_dir=tmp_path / "d",
        artifact_dir=tmp_path / "a",
        environment="test",
        log_level="ERROR",
    )
    return RankingService(
        repository=repo, registry=ModelRegistry(settings.artifact_dir), settings=settings
    )


class TestCandidateTrace:
    def test_trace_is_opt_in(self, traced, repo):
        plain = traced.rank(RequestContext(user_id=repo.users()[0], as_of=AS_OF, limit=5))
        assert plain.trace == []
        with_trace = traced.rank(
            RequestContext(user_id=repo.users()[0], as_of=AS_OF, limit=5, collect_trace=True)
        )
        assert with_trace.trace, "tracing must not change the result, only reveal it"

    def test_trace_covers_every_generated_candidate(self, traced, repo):
        result = traced.rank(
            RequestContext(user_id=repo.users()[0], as_of=AS_OF, limit=5, collect_trace=True)
        )
        assert len(result.trace) == result.diagnostics["n_candidates"]
        assert {t.counterparty_id for t in result.trace} == set(result.diagnostics["retrieved_ids"])

    def test_returned_candidates_have_a_rank_and_no_drop_reason(self, traced, repo):
        result = traced.rank(
            RequestContext(user_id=repo.users()[0], as_of=AS_OF, limit=5, collect_trace=True)
        )
        shown = {r.candidate.counterparty_id for r in result.recommendations}
        for t in result.trace:
            if t.counterparty_id in shown:
                assert t.final_rank is not None
                assert t.dropped_at is None
                assert t.drop_reason is None
                assert t.was_returned
            else:
                assert t.final_rank is None
                assert t.dropped_at is not None, "every unshown candidate must say why"

    def test_ranks_are_the_ones_actually_served(self, traced, repo):
        result = traced.rank(
            RequestContext(user_id=repo.users()[0], as_of=AS_OF, limit=5, collect_trace=True)
        )
        served = {r.candidate.counterparty_id: r.rank for r in result.recommendations}
        traced_ranks = {t.counterparty_id: t.final_rank for t in result.trace if t.final_rank}
        assert served == traced_ranks

    def test_drop_stage_is_always_a_known_stage(self, traced, repo):
        result = traced.rank(
            RequestContext(user_id=repo.users()[0], as_of=AS_OF, limit=3, collect_trace=True)
        )
        assert {t.dropped_at for t in result.trace if t.dropped_at} <= {
            "policy",
            "diversification",
            "limit",
        }

    def test_policy_blocks_are_attributed_to_policy(self, repo, tmp_path):
        """Starve the balance so the policy removes candidates, then check the trace."""
        memory = InMemoryRepository()
        for txs in [repo.transactions_for_user(u) for u in repo.users()[:3]]:
            memory.add(txs)
        for uid in memory.users():
            memory.set_account(uid, available_balance=0.0, currency="EUR")
        settings = Settings(
            data_dir=tmp_path / "d",
            artifact_dir=tmp_path / "a",
            environment="test",
            log_level="ERROR",
        )
        svc = RankingService(
            repository=memory, registry=ModelRegistry(settings.artifact_dir), settings=settings
        )
        result = svc.rank(
            RequestContext(user_id=memory.users()[0], as_of=AS_OF, limit=5, collect_trace=True)
        )
        blocked = [t for t in result.trace if t.dropped_at == "policy"]
        assert blocked, "a zero balance must block something"
        for t in blocked:
            assert t.policy_decision != "allow"
            assert (
                "insufficient_balance" in t.policy_flags or "no_estimated_balance" in t.policy_flags
            )
            assert t.final_rank is None

    def test_tracing_does_not_change_the_answer(self, traced, repo):
        plain = traced.rank(RequestContext(user_id=repo.users()[0], as_of=AS_OF, limit=5))
        with_trace = traced.rank(
            RequestContext(user_id=repo.users()[0], as_of=AS_OF, limit=5, collect_trace=True)
        )
        assert [r.candidate.counterparty_id for r in plain.recommendations] == [
            r.candidate.counterparty_id for r in with_trace.recommendations
        ]


class TestDiagnose:
    def _cases(self, repo):
        corpus = {u: repo.transactions_for_user(u) for u in repo.users()[:3]}
        return build_cases(corpus, as_of=AS_OF, max_cases_per_user=4)

    def test_report_is_self_consistent(self, traced, repo):
        cases = self._cases(repo)
        if not cases:
            pytest.skip("no decision points in the fixture corpus")
        report = diagnose(traced, cases, limit=5)
        assert report.n_cases == len(cases)
        assert report.n_returned <= report.n_relevant
        assert report.total_losses == report.n_relevant - report.n_returned
        assert abs(sum(report.stage_share.values()) - 1.0) < 1e-6

    def test_every_loss_is_attributed_to_exactly_one_stage(self, traced, repo):
        cases = self._cases(repo)
        if not cases:
            pytest.skip("no cases")
        report = diagnose(traced, cases, limit=5)
        per_case = sum(len(case.failures) for case in report.cases)
        assert per_case == report.total_losses

    def test_failure_stages_are_from_the_known_set(self, traced, repo):
        cases = self._cases(repo)
        if not cases:
            pytest.skip("no cases")
        report = diagnose(traced, cases, limit=5)
        for case in report.cases:
            for stage in case.failures.values():
                assert stage in {"recall", "ranking", "policy", "diversification"}

    def test_returned_targets_are_never_counted_as_losses(self, traced, repo):
        cases = self._cases(repo)
        if not cases:
            pytest.skip("no cases")
        report = diagnose(traced, cases, limit=5)
        for case in report.cases:
            returned = set(case.ranking)
            assert not (returned & set(case.failures))

    def test_perfect_system_loses_nothing(self, traced, repo):
        """A system that returns every relevant payee must report zero losses."""
        cases = self._cases(repo)
        if not cases:
            pytest.skip("no cases")
        wide = diagnose(traced, cases, limit=10)
        # 10 slots cannot beat the pool, but losses must be recall-only.
        assert "ranking" not in wide.stage_losses or wide.stage_losses["ranking"] == 0

    def test_to_dict_is_serialisable(self, traced, repo):
        cases = self._cases(repo)
        if not cases:
            pytest.skip("no cases")
        payload = diagnose(traced, cases, limit=5).to_dict()
        assert set(payload) >= {"n_cases", "metrics", "stage_losses", "stage_share"}
        assert all(isinstance(v, (int, float)) for v in payload["stage_losses"].values())
