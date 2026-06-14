"""Scan ESPN live/upcoming games for direct quoted Kalshi winner markets.

    python scripts/scan_kalshi_games.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportedge.data.espn_live import list_live_games  # noqa: E402
from sportedge.market.kalshi import KalshiClient  # noqa: E402
from sportedge.market.scanner import scan_game_markets  # noqa: E402


def _price(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Find live/upcoming games with Kalshi markets")
    ap.add_argument("--sport", default="basketball", choices=["basketball", "soccer", "all"])
    ap.add_argument("--status", default="in,pre", help="Comma-separated ESPN states, e.g. in,pre")
    ap.add_argument("--only-markets", action="store_true", help="Hide games without a match")
    ap.add_argument("--debug-rejections", action="store_true", help="Show why Kalshi search hits were rejected")
    ap.add_argument("--debug-limit", type=int, default=3, help="Rejected markets to show per side")
    args = ap.parse_args()

    sports = None if args.sport == "all" else {args.sport}
    statuses = {item.strip() for item in args.status.split(",") if item.strip()}
    client = KalshiClient()
    games = list_live_games()
    rows = scan_game_markets(games, client=client, sports=sports, statuses=statuses)
    scanned_count = len(rows)
    if args.only_markets:
        rows = [row for row in rows if row.has_market]

    if not rows:
        if scanned_count:
            print(f"No valid Kalshi markets found for {scanned_count} matching ESPN game(s).")
            print("Use --debug-rejections without --only-markets to inspect rejected Kalshi search hits.")
        else:
            print("No matching ESPN games found.")
        return

    for row in rows:
        game = row.game
        tag = "READY" if row.has_market else "NO MARKET"
        print(
            f"[{tag}] {game.status} {game.sport}/{game.league} "
            f"{game.away_team} @ {game.home_team} {game.short_detail}"
        )
        for side, market in [(game.home_team, row.home_market), (game.away_team, row.away_market)]:
            if market is None:
                print(f"  {side}: -")
                if args.debug_rejections:
                    opponent = game.away_team if side == game.home_team else game.home_team
                    rejections = client.explain_team_win_market_search(side, opponent)[: args.debug_limit]
                    for rejected in rejections:
                        ticker = rejected.ticker or "(search)"
                        print(f"    reject {ticker}: {rejected.reason}")
                        if rejected.title:
                            print(f"      {rejected.title[:180]}")
                continue
            print(
                f"  {side}: {market.ticker} "
                f"bid={_price(market.yes_bid)} ask={_price(market.yes_ask)} "
                f"last={_price(market.last_price)}"
            )
            print(f"    {market.title}")


if __name__ == "__main__":
    main()
