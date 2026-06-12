"""ESPN hidden API data access for historical NBA training rows.

Uses the public-but-undocumented site API endpoints documented by the community:
scoreboard by date, then summary by event id for play-by-play.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from sportedge.data.nba_scraper import HOME_PRIOR, TRAINING_COLUMNS
from sportedge.types import regulation_seconds_remaining

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"


def _date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _clock_to_seconds(clock: dict | str | None) -> float:
    if isinstance(clock, dict):
        value = str(clock.get("displayValue") or "")
    else:
        value = str(clock or "")
    if not value:
        return 0.0
    parts = value.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except ValueError:
        return 0.0


def _period_number(period: dict | int | None) -> int:
    if isinstance(period, dict):
        return int(period.get("number") or 1)
    return int(period or 1)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8))
def list_events_for_date(game_date: date) -> list[dict]:
    response = requests.get(
        f"{BASE_URL}/scoreboard",
        params={"dates": game_date.strftime("%Y%m%d"), "limit": 100},
        timeout=20,
    )
    response.raise_for_status()
    return response.json().get("events", [])


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8))
def _summary(event_id: str) -> dict:
    response = requests.get(f"{BASE_URL}/summary", params={"event": event_id}, timeout=20)
    response.raise_for_status()
    return response.json()


def _home_won(event: dict) -> bool | None:
    competitions = event.get("competitions") or []
    if not competitions:
        return None
    for competitor in competitions[0].get("competitors") or []:
        if competitor.get("homeAway") == "home":
            winner = competitor.get("winner")
            if winner is not None:
                return bool(winner)
    return None


def _is_completed(event: dict) -> bool:
    status = event.get("status", {}).get("type", {})
    return bool(status.get("completed")) or str(status.get("state", "")).lower() == "post"


def event_training_rows(event: dict) -> pd.DataFrame:
    event_id = str(event.get("id") or "")
    home_won = _home_won(event)
    if not event_id or home_won is None or not _is_completed(event):
        return pd.DataFrame(columns=TRAINING_COLUMNS)

    plays = _summary(event_id).get("plays") or []
    rows: list[dict] = []
    for play in plays:
        try:
            home_score = int(play.get("homeScore") or 0)
            away_score = int(play.get("awayScore") or 0)
        except (TypeError, ValueError):
            continue
        period = _period_number(play.get("period"))
        clock_seconds = _clock_to_seconds(play.get("clock"))
        rows.append(
            {
                "game_id": event_id,
                "home_score": home_score,
                "away_score": away_score,
                "period": period,
                "seconds_remaining": regulation_seconds_remaining(period, clock_seconds),
                "pre_game_home_prob": HOME_PRIOR,
                "home_win": int(home_won),
            }
        )

    if not rows:
        return pd.DataFrame(columns=TRAINING_COLUMNS)
    return pd.DataFrame(rows, columns=TRAINING_COLUMNS)


def build_training_set_by_dates(start: date, end: date) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for game_date in _date_range(start, end):
        for event in list_events_for_date(game_date):
            try:
                rows = event_training_rows(event)
            except Exception:
                continue
            if not rows.empty:
                frames.append(rows)
    if not frames:
        return pd.DataFrame(columns=TRAINING_COLUMNS)
    return pd.concat(frames, ignore_index=True)
