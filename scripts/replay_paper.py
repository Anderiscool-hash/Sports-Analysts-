"""Replay historical aligned model/market rows into the paper ledger.

    python scripts/replay_paper.py --aligned data/cache/aligned_2024_finals_g5_celtics.parquet
    python scripts/replay_paper.py --directory data/cache --pattern "aligned*.parquet"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportedge.betting.replay import replay_directory, replay_file  # noqa: E402
from sportedge.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay aligned model/market rows as paper trades")
    ap.add_argument("--aligned", default="", help="Parquet with timestamp/model_p/price columns")
    ap.add_argument("--directory", default="", help="Directory containing aligned parquet files")
    ap.add_argument("--pattern", default="aligned*.parquet")
    ap.add_argument("--ledger", default="data/cache/paper_ledger.parquet")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--token-id", default="REPLAY")
    args = ap.parse_args()

    if not args.aligned and not args.directory:
        ap.error("pass --aligned or --directory")
    cfg = load_config(args.config)

    summaries = (
        replay_directory(args.directory, cfg, args.ledger, pattern=args.pattern)
        if args.directory
        else [replay_file(args.aligned, cfg, args.ledger, token_id=args.token_id)]
    )
    rows_seen = sum(item.rows_seen for item in summaries)
    fills = sum(item.fills for item in summaries)
    staked = sum(item.staked for item in summaries)
    skipped = [item for item in summaries if item.skipped]
    for item in summaries:
        status = "SKIP" if item.skipped else "OK"
        detail = item.reason if item.skipped else f"rows={item.rows_seen} fills={item.fills}"
        print(f"{status} {item.path} {detail}")
    print(f"replayed files: {len(summaries) - len(skipped)}")
    print(f"skipped files: {len(skipped)}")
    print(f"replayed rows: {rows_seen}")
    print(f"paper fills: {fills}")
    print(f"staked: {staked:.2f}")
    print(f"ledger: {args.ledger}")


if __name__ == "__main__":
    main()
