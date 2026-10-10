"""``bankai`` command line interface.

bankai serve        run the API
bankai seed         generate a synthetic corpus
bankai train        fit and publish a model
bankai pipeline     seed + train + backtest (one shot)
bankai backtest     replay historical decision points
bankai demo         print recommendations for a sample user
bankai models       list / roll back published artifacts
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

from bankai.config import get_settings, reload_settings
from bankai.logging_setup import configure_logging, get_logger

log = get_logger("bankai.cli")


def _settings_from_args(args: argparse.Namespace):
    settings = reload_settings()
    for field in (
        "n_users",
        "n_months",
        "seed",
        "as_of",
        "data_dir",
        "artifact_dir",
        "environment",
    ):
        value = getattr(args, field, None)
        if value is not None:
            setattr(settings, field, value)
    settings.ensure_dirs()
    return settings


# --------------------------------------------------------------------- serve
def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = _settings_from_args(args)
    configure_logging(settings.log_level, settings.log_format)
    uvicorn.run(
        "bankai.serving.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_config=None,  # keep our JSON formatter
        access_log=False,
    )
    return 0


# ---------------------------------------------------------------------- seed
def cmd_seed(args: argparse.Namespace) -> int:
    from bankai.data.store import SqliteTransactionRepository
    from bankai.data.synthetic import generate_dataset

    settings = _settings_from_args(args)
    repo = SqliteTransactionRepository(settings.db_path)
    as_of = (
        dt.date.fromisoformat(settings.as_of)
        if settings.as_of
        else dt.date.today() - dt.timedelta(days=1)
    )
    log.info("generating corpus", extra={"users": settings.n_users, "months": settings.n_months})
    transactions, _, snapshot = generate_dataset(
        n_users=settings.n_users, months=settings.n_months, as_of=as_of, seed=settings.seed
    )
    repo.bulk_insert(transactions)
    by_user: dict[str, list] = {}
    for t in transactions:
        by_user.setdefault(t.user_id, []).append(t)
    for user_id, _txs in by_user.items():

        repo.upsert_account(
            user_id,
            available_balance=None,
            currency="EUR",
            daily_limit=settings.daily_transfer_limit,
            monthly_limit=settings.monthly_transfer_limit,
        )
    print(
        json.dumps(
            {
                "database": str(settings.db_path),
                "users": len(by_user),
                "transactions": len(transactions),
                "snapshot_date": snapshot.isoformat(),
            },
            indent=2,
        )
    )
    return 0


# -------------------------------------------------------------------- train
def cmd_train(args: argparse.Namespace) -> int:
    from bankai.training.pipeline import load_corpus
    from bankai.training.trainer import fit_auxiliary_models, train_and_publish

    settings = _settings_from_args(args)
    by_user, _, as_of = load_corpus(settings)
    encoder, latent = fit_auxiliary_models(by_user)
    report = train_and_publish(
        by_user,
        as_of=as_of,
        settings=settings,
        encoder=encoder,
        latent=latent,
        max_users=args.max_users,
        force_promote=not args.no_promote,
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0


# ----------------------------------------------------------------- pipeline
def cmd_pipeline(args: argparse.Namespace) -> int:
    from bankai.training.pipeline import run_pipeline

    settings = _settings_from_args(args)
    result = run_pipeline(
        settings=settings,
        max_users=args.max_users,
        backtest_users=args.backtest_users,
        force_promote=not args.no_promote,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


# ----------------------------------------------------------------- backtest
def cmd_backtest(args: argparse.Namespace) -> int:
    from bankai.evaluation.backtest import build_cases, run_backtest
    from bankai.models.registry import ModelRegistry
    from bankai.ranking.service import RankingService
    from bankai.training.pipeline import load_corpus

    settings = _settings_from_args(args)
    by_user, repo, as_of = load_corpus(settings)
    subset = dict(sorted(by_user.items())[: args.users])
    cases = build_cases(subset, as_of=as_of, label_window_days=settings.label_window_days)
    if not cases:
        print(json.dumps({"error": "no backtest cases built"}, indent=2))
        return 1
    service = RankingService(
        repository=repo, registry=ModelRegistry(settings.artifact_dir), settings=settings
    )
    result = run_backtest(
        service,
        cases,
        label="current",
        limit=args.limit,
        horizon_days=settings.default_horizon_days,
        on_progress=lambda i, n: print(f"  ... {i}/{n} cases", file=sys.stderr),
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


# -------------------------------------------------------------------- demo
def cmd_diagnose(args: argparse.Namespace) -> int:
    """Attribute every missed recommendation to a pipeline stage."""
    from bankai.evaluation.backtest import build_cases
    from bankai.evaluation.diagnose import diagnose, render
    from bankai.models.registry import ModelRegistry
    from bankai.ranking.service import RankingService
    from bankai.training.pipeline import load_corpus

    settings = _settings_from_args(args)
    by_user, repo, as_of = load_corpus(settings)
    subset = dict(sorted(by_user.items())[: args.users])
    cases = build_cases(subset, as_of=as_of, label_window_days=settings.label_window_days)
    if not cases:
        print(json.dumps({"error": "no backtest cases built"}, indent=2))
        return 1
    service = RankingService(
        repository=repo, registry=ModelRegistry(settings.artifact_dir), settings=settings
    )
    report = diagnose(service, cases, limit=args.limit, horizon_days=settings.default_horizon_days)
    print(render(report), file=sys.stderr)
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    from bankai.models.ranker import RankerConfig
    from bankai.training.experiment import (
        Variant,
        evaluate_variant,
        make_splits,
        paired_gain,
        render_table,
    )
    from bankai.training.pipeline import load_corpus
    from bankai.training.trainer import Trainer, fit_auxiliary_models

    settings = _settings_from_args(args)
    by_user, _repo, as_of = load_corpus(settings)
    split = make_splits(sorted(by_user), n_dev=args.dev_users, n_holdout=args.holdout_users)
    print(
        f"train={len(split.train)} dev={len(split.dev)} holdout={len(split.holdout)}",
        file=sys.stderr,
    )
    enc, lat = fit_auxiliary_models({u: by_user[u] for u in split.train})
    dataset = Trainer(settings=settings).build_dataset(
        {u: by_user[u] for u in split.train}, as_of=as_of, encoder=enc, latent=lat
    )

    def cfg(**kw):
        base = {
            "lambda_lgb": settings.lambda_lgb,
            "num_leaves": settings.num_leaves,
            "min_data_in_leaf": settings.min_data_in_leaf,
            "feature_fraction": settings.feature_fraction,
            "bagging_fraction": settings.bagging_fraction,
            "bagging_freq": settings.bagging_freq,
            "n_estimators": settings.n_estimators,
            "learning_rate": settings.learning_rate,
            "early_stopping_rounds": settings.early_stopping_rounds,
            "num_threads": settings.num_threads,
            "random_state": settings.random_state,
            "ensemble_size": settings.ensemble_size,
        }
        base.update(kw)
        return RankerConfig(**base)

    variants = [
        Variant("ff08_single", ranker=cfg(feature_fraction=0.8, ensemble_size=1)),
        Variant("ff04_single", ranker=cfg(feature_fraction=0.4, ensemble_size=1)),
        Variant("ff04_ens5", ranker=cfg(feature_fraction=0.4, ensemble_size=5)),
    ]
    results = []
    for v in variants:
        r = evaluate_variant(
            v,
            corpus=by_user,
            as_of=as_of,
            split=split,
            base_settings=settings,
            eval_users=split.dev,
            encoder=enc,
            latent=lat,
            dataset=dataset,
        )
        results.append(r)
        print(r.summary(), file=sys.stderr)
    print(render_table(results), file=sys.stderr)
    if len(results) > 1:
        gain = paired_gain(results[0], results[-1])
        print(
            f"paired NDCG@5 gain {gain['ndcg_gain']:+.4f} "
            f"[{gain['ci_low']:+.4f}, {gain['ci_high']:+.4f}] "
            f"significant={gain['significant']}",
            file=sys.stderr,
        )
    print(json.dumps([r.to_dict() for r in results], indent=2))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from bankai.models.registry import ModelRegistry
    from bankai.ranking.service import RankingService, RequestContext
    from bankai.training.pipeline import load_corpus

    settings = _settings_from_args(args)
    by_user, repo, as_of = load_corpus(settings)
    service = RankingService(
        repository=repo, registry=ModelRegistry(settings.artifact_dir), settings=settings
    )
    user_id = args.user or sorted(by_user)[0]
    result = service.rank(
        RequestContext(user_id=user_id, as_of=as_of, horizon_days=args.horizon, limit=args.limit)
    )
    print(
        f"\nRecommendations for {user_id} as of {as_of} "
        f"(model {result.model_version}, {result.latency_ms:.1f} ms)\n" + "=" * 78
    )
    if result.degraded:
        print(f"!! DEGRADED: {result.degradation_reason}\n")
    if not result.recommendations:
        print("  (no recommendations)")
    for rec in result.recommendations:
        c = rec.candidate
        print(
            f"\n{rec.rank}. {c.counterparty_name}  [{c.category}]  "
            f"{c.suggested_amount:,.2f} {c.currency}"
        )
        print(
            f"   status={c.status.value}  scheduled={c.scheduled_for}  "
            f"score={rec.score:.3f}  confidence={rec.confidence:.2f}  source={c.source}"
        )
        if c.recurrence and c.recurrence.period_days:
            print(
                f"   cadence={c.recurrence.period_days:.0f}d  "
                f"observations={c.recurrence.n_observations}  "
                f"amount_stability={c.recurrence.amount_stability:.2f}"
            )
        for e in rec.explanations[:3]:
            print(f"   - {e.text}")
    print()
    return 0


# ------------------------------------------------------------------- models
def cmd_models(args: argparse.Namespace) -> int:
    from bankai.models.registry import ModelRegistry

    settings = _settings_from_args(args)
    registry = ModelRegistry(settings.artifact_dir)
    if args.action == "list":
        versions = registry.available_versions()
        current = registry.current_version()
        print(
            json.dumps(
                {
                    "current": current,
                    "versions": [
                        {
                            "version": v,
                            "active": v == current,
                            **(
                                {
                                    "manifest": json.loads(
                                        (registry._version_dir(v) / "manifest.json").read_text()
                                    )
                                }
                                if (registry._version_dir(v) / "manifest.json").exists()
                                else {}
                            ),
                        }
                        for v in versions
                    ],
                },
                indent=2,
                default=str,
            )
        )
        return 0
    if args.action == "rollback":
        loaded = registry.rollback()
        print(json.dumps({"now_serving": loaded.version, "healthy": loaded.healthy}, indent=2))
        return 0
    if args.action == "prune":
        removed = registry.prune(keep=args.keep)
        print(json.dumps({"removed": removed}, indent=2))
        return 0
    if args.action == "reload":
        loaded = registry.reload(args.version)
        print(
            json.dumps(
                {
                    "now_serving": loaded.version,
                    "healthy": loaded.healthy,
                    "reason": loaded.degradation_reason,
                },
                indent=2,
            )
        )
        return 0
    return 1


# ---------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bankai", description="BankAI transfer recommendations")
    p.add_argument("--log-level", default=None)
    p.add_argument("--environment", default=None)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the API server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("seed", help="generate a synthetic transaction corpus")
    s.add_argument("--users", type=int, dest="n_users")
    s.add_argument("--months", type=int, dest="n_months")
    s.add_argument("--seed", type=int)
    s.add_argument("--as-of", dest="as_of")
    s.set_defaults(func=cmd_seed)

    s = sub.add_parser("train", help="train and publish a ranking model")
    s.add_argument("--max-users", type=int, default=None)
    s.add_argument("--no-promote", action="store_true")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("pipeline", help="seed + train + backtest end to end")
    s.add_argument("--users", type=int, dest="n_users")
    s.add_argument("--months", type=int, dest="n_months")
    s.add_argument("--seed", type=int)
    s.add_argument("--as-of", dest="as_of")
    s.add_argument("--max-users", type=int, default=None)
    s.add_argument("--backtest-users", type=int, default=40)
    s.add_argument("--no-promote", action="store_true")
    s.set_defaults(func=cmd_pipeline)

    s = sub.add_parser("backtest", help="replay historical decision points")
    s.add_argument("--users", type=int, default=40)
    s.add_argument("--limit", type=int, default=5)
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("diagnose", help="attribute missed recommendations to a stage")
    s.add_argument("--users", type=int, default=40)
    s.add_argument("--limit", type=int, default=5)
    s.set_defaults(func=cmd_diagnose)

    s = sub.add_parser("tune", help="compare ranker configurations on a dev user split")
    s.add_argument("--dev-users", type=int, default=60)
    s.add_argument("--holdout-users", type=int, default=40)
    s.set_defaults(func=cmd_tune)

    s = sub.add_parser("demo", help="print recommendations for a user")
    s.add_argument("--user", default=None)
    s.add_argument("--limit", type=int, default=5)
    s.add_argument("--horizon", type=int, default=14)
    s.add_argument("--as-of", dest="as_of")
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("models", help="manage published artifacts")
    s.add_argument("action", choices=["list", "rollback", "prune", "reload"])
    s.add_argument("--keep", type=int, default=5)
    s.add_argument("--version", default=None)
    s.set_defaults(func=cmd_models)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(args.log_level or settings.log_level, settings.log_format)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    except Exception as exc:
        log.exception("command failed", extra={"command": args.command})
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
