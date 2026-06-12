"""Fetch historical Polymarket token prices into a local parquet cache.

    python scripts/fetch_polymarket_prices.py --token-id <clob-token-id> \
      --start 2024-06-06T00:00:00Z --end 2024-06-07T06:00:00Z
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportedge.config import load_secrets  # noqa: E402
from sportedge.data.storage import save_parquet  # noqa: E402
from sportedge.market.polymarket import PolymarketClient, unix_ts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Polymarket historical token prices")
    ap.add_argument("--token-id", required=True, help="CLOB asset/token id")
    ap.add_argument("--start", default=None, help="Unix seconds or ISO datetime")
    ap.add_argument("--end", default=None, help="Unix seconds or ISO datetime")
    ap.add_argument(
        "--interval",
        default="all",
        choices=["max", "all", "1m", "1w", "1d", "6h", "1h"],
    )
    ap.add_argument("--fidelity", type=int, default=1, help="Accuracy in minutes")
    ap.add_argument("--cache", default=None, help="Cache name under data/cache")
    ap.add_argument("--clob-host", default=None)
    args = ap.parse_args()

    secrets = load_secrets()
    client = PolymarketClient(
        clob_host=args.clob_host or secrets.clob_host,
        secrets=secrets,
    )
    df = client.prices_history_frame(
        args.token_id,
        start_ts=unix_ts(args.start),
        end_ts=unix_ts(args.end),
        interval=args.interval,
        fidelity=args.fidelity,
    )
    if df.empty:
        print("No price history returned for token/time range.")
        return
    cache = args.cache or f"polymarket_prices_{args.token_id[:12]}"
    path = save_parquet(df, cache)
    print(
        f"saved {len(df):,} price points for token {args.token_id} "
        f"from {df['timestamp'].min()} to {df['timestamp'].max()} -> {path}"
    )


if __name__ == "__main__":
    main()
