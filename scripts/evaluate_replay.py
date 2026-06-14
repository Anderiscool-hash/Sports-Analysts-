"""Evaluate aligned replay rows without writing to the paper ledger.

Examples:
    python scripts/evaluate_replay.py --directory data/cache --pattern "aligned_generated_*.parquet"
    python scripts/evaluate_replay.py --directory data/cache --grid
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportedge.betting.replay import evaluate_directory  # noqa: E402
from sportedge.config import Config, load_config  # noqa: E402


def _clone_config(cfg: Config) -> Config:
    data = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()
    return Config(**data)


def _floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _print_summary(label: str, rows: int, fills: int, staked: float, pnl: float, wins: int) -> None:
    roi = pnl / staked if staked else 0.0
    print(
        f"{label} rows={rows} fills={fills} wins={wins} "
        f"staked={staked:.2f} pnl={pnl:+.2f} roi={roi:+.1%}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate replay strategy without mutating the paper ledger")
    ap.add_argument("--directory", default="data/cache")
    ap.add_argument("--pattern", default="aligned_generated_*.parquet")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--grid", action="store_true", help="Sweep edge/dip/rebound settings")
    ap.add_argument("--min-edges", default="0.04,0.06,0.08,0.10,0.12,0.15")
    ap.add_argument("--dip-thresholds", default="0.05,0.08,0.10,0.15")
    ap.add_argument("--rebound-ticks", default="1,2,3")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not args.grid:
        summary = evaluate_directory(args.directory, cfg, args.pattern)
        _print_summary(
            "default",
            summary.rows_seen,
            summary.fills,
            summary.staked,
            summary.pnl,
            summary.wins,
        )
        return

    results = []
    for min_edge, dip, rebound in itertools.product(
        _floats(args.min_edges),
        _floats(args.dip_thresholds),
        _ints(args.rebound_ticks),
    ):
        trial = _clone_config(cfg)
        trial.edge.min_edge = min_edge
        trial.edge.dip_threshold = dip
        trial.edge.rebound_ticks = rebound
        summary = evaluate_directory(args.directory, trial, args.pattern)
        results.append(
            (
                summary.pnl,
                summary.fills,
                summary.staked,
                summary.wins,
                summary.rows_seen,
                min_edge,
                dip,
                rebound,
            )
        )

    for pnl, fills, staked, wins, rows, min_edge, dip, rebound in sorted(results, reverse=True)[: args.top]:
        roi = pnl / staked if staked else 0.0
        print(
            f"pnl={pnl:+.2f} roi={roi:+.1%} fills={fills} wins={wins} "
            f"staked={staked:.2f} rows={rows} "
            f"min_edge={min_edge:.2f} dip={dip:.2f} rebound={rebound}"
        )


if __name__ == "__main__":
    main()
