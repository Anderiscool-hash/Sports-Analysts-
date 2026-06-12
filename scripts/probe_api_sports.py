"""Probe API-SPORTS access without dumping secrets.

    python scripts/probe_api_sports.py --sport basketball
"""

from __future__ import annotations

import argparse
from datetime import date
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportedge.data import api_sports  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe API-SPORTS access")
    ap.add_argument("--sport", default="basketball", choices=sorted(api_sports.BASE_URLS))
    ap.add_argument("--date", default=date.today().isoformat())
    args = ap.parse_args()

    status = api_sports.status(args.sport)
    subscription = status.get("subscription", {})
    requests = status.get("requests", {})
    print(
        "subscription: "
        f"plan={subscription.get('plan')} active={subscription.get('active')} "
        f"requests={requests.get('current')}/{requests.get('limit_day')}"
    )

    try:
        if args.sport == "basketball":
            rows = api_sports.basketball_games(date.fromisoformat(args.date))
            print(f"basketball games on {args.date}: {len(rows)}")
        elif args.sport == "football":
            rows = api_sports.football_fixtures(date.fromisoformat(args.date))
            print(f"football fixtures on {args.date}: {len(rows)}")
        else:
            print(f"{args.sport} status probe only")
    except api_sports.ApiSportsError as exc:
        print(f"provider error: {exc}")


if __name__ == "__main__":
    main()
