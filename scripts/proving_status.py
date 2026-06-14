"""Show paper-trading proving-ground readiness.

    python scripts/proving_status.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportedge.config import load_config  # noqa: E402
from sportedge.proving import build_proving_ground_status, next_action  # noqa: E402


def _yn(value: bool) -> str:
    return "READY" if value else "BLOCK"


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper-trading proving-ground status")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--ledger", default="data/cache/paper_ledger.parquet")
    ap.add_argument("--training-cache", default="data/cache/training.parquet")
    ap.add_argument("--min-training-games", type=int, default=3)
    ap.add_argument("--no-scan", action="store_true", help="Skip live Kalshi/ESPN market scan")
    ap.add_argument("--sport", default="basketball", choices=["basketball", "soccer", "all"])
    ap.add_argument("--status", default="in,pre", help="Comma-separated ESPN states, e.g. in,pre")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sports = None if args.sport == "all" else {args.sport}
    statuses = {item.strip() for item in args.status.split(",") if item.strip()}
    status = build_proving_ground_status(
        cfg,
        ledger_path=args.ledger,
        training_cache=args.training_cache,
        min_training_games=args.min_training_games,
        scan_markets=not args.no_scan,
        sports=sports,
        statuses=statuses,
    )

    print("SportEdge proving-ground status")
    print(f"markets: {_yn(status.markets.ready)} - {status.markets.ready_games}/{status.markets.games_scanned} games ready")
    print(
        f"paper: {_yn(status.paper.gate_ok)} - fills={status.paper.fills}, "
        f"settled={status.paper.settled_fills}, open={status.paper.open_positions}, "
        f"exposure={status.paper.open_exposure:.2f}, "
        f"realized_pnl={status.paper.realized_pnl:+.2f}, "
        f"realized_roi={status.paper.realized_roi:+.1%}, "
        f"total_pnl={status.paper.total_pnl:+.2f}"
    )
    print(f"paper gate: {'PASS' if status.paper.gate_ok else 'BLOCK'} - {status.paper.gate_reason}")
    print(
        f"training: {_yn(status.training.ready)} - {status.training.reason} "
        f"({status.training.path})"
    )
    print(f"overall: {_yn(status.live_ready)}")
    print(f"next: {next_action(status, cfg)}")


if __name__ == "__main__":
    main()
