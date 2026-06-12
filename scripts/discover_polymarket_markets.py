"""Discover Polymarket events/markets and CLOB token IDs.

    python scripts/discover_polymarket_markets.py --query "NBA Celtics Mavericks" --closed true
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportedge.config import load_secrets  # noqa: E402
from sportedge.market.polymarket import PolymarketClient  # noqa: E402


def _json_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(json.loads(value))


def _bool(value: str) -> bool | None:
    if value.lower() == "any":
        return None
    return value.lower() == "true"


def _print_market(market: dict, indent: str = "") -> None:
    outcomes = _json_list(market.get("outcomes"))
    tokens = _json_list(market.get("clobTokenIds"))
    print(f"{indent}market: {market.get('question')}")
    print(f"{indent}  slug: {market.get('slug')}")
    print(f"{indent}  id: {market.get('id')}  closed={market.get('closed')}  end={market.get('endDate')}")
    for outcome, token in zip(outcomes, tokens, strict=False):
        print(f"{indent}  token[{outcome}]: {token}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Discover Polymarket markets and token IDs")
    ap.add_argument("--query", required=True)
    ap.add_argument("--closed", default="false", choices=["true", "false", "any"])
    ap.add_argument("--active", default="any", choices=["true", "false", "any"])
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument(
        "--raw",
        action="store_true",
        help="Also run raw /markets and /events searches after public-search",
    )
    args = ap.parse_args()

    secrets = load_secrets()
    client = PolymarketClient(secrets=secrets)
    active = _bool(args.active)
    closed = _bool(args.closed)

    results = client.public_search(args.query, limit=args.limit)
    events = results.get("events") or []
    markets = results.get("markets") or []
    print(f"Public search events: {len(events)}")
    for event in events:
        print(f"event: {event.get('title') or event.get('ticker')}")
        print(f"  slug: {event.get('slug')}")
        print(f"  id: {event.get('id')}  closed={event.get('closed')}  end={event.get('endDate')}")
        for market in event.get("markets") or []:
            _print_market(market, indent="  ")

    print(f"\nPublic search markets: {len(markets)}")
    for market in markets:
        _print_market(market)

    if args.raw:
        raw_markets = client.list_markets(
            search=args.query,
            active=active,
            closed=closed,
            limit=args.limit,
        )
        print(f"\nRaw markets: {len(raw_markets)}")
        for market in raw_markets:
            _print_market(market)

        events = client.list_events(
            search=args.query,
            active=active,
            closed=closed,
            limit=args.limit,
        )
        print(f"\nRaw events: {len(events)}")
        for event in events:
            print(f"event: {event.get('title') or event.get('ticker')}")
            print(f"  slug: {event.get('slug')}")
            print(f"  id: {event.get('id')}  closed={event.get('closed')}  end={event.get('endDate')}")
            for market in event.get("markets") or []:
                _print_market(market, indent="  ")


if __name__ == "__main__":
    main()
