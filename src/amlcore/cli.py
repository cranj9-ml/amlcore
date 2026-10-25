"""Command line: `python -m amlcore.cli <command> [--config PATH] [--demo]`.

Commands:
  features   raw files → data/processed/<variant>/features.parquet
  train      train, evaluate, explain → reports/<variant>/
  run        both, in order

--demo runs on the bundled synthetic data in data/demo, so the whole pipeline can be
tried without downloading AMLworld.
"""

from __future__ import annotations

import argparse
import time

from amlcore.config import Config, demo_config, load_config
from amlcore.pipeline import MODEL, RULES, run_features, run_train
from amlcore.data import TEST


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="amlcore", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["features", "train", "run"])
    parser.add_argument("--config", help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--demo", action="store_true", help="use the bundled synthetic data")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.demo:
        config = demo_config(config)
    feats = None
    if args.command in ("features", "run"):
        feats = _timed("features", lambda: run_features(config))
        print(f"  {len(feats):,} transactions, {feats.shape[1]} columns")
    if args.command in ("train", "run"):
        result = _timed("train", lambda: run_train(config, feats))
        _print_headline(result.metrics, config)
        print(f"  outputs in {result.out_dir}")
    return 0


def _timed(name: str, step):
    start = time.perf_counter()
    print(f"{name} ...", flush=True)
    out = step()
    print(f"  done in {time.perf_counter() - start:.0f}s")
    return out


def _print_headline(metrics: dict, config: Config) -> None:
    budget = config.evaluation.primary_budget
    for scorer in (RULES, MODEL):
        ev = metrics[TEST][scorer]
        at = next(b for b in ev["at_budget"] if b.budget == budget)
        ci = metrics["test_recall_interval"][scorer]
        print(f"  test {scorer:8s} recall@{budget:.1%} {at.recall:6.1%} "
              f"[{ci.low:.1%}, {ci.high:.1%}]  precision {at.precision:5.1%}  PR-AUC {ev['pr_auc']:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
