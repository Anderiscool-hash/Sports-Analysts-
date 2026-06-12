"""SportsBlaze API access for schedule and boxscore data.

Reads SPORTSBLAZE_API_KEY from the environment by default. SportsBlaze is useful
for game schedules, boxscores, team stats, and roster stats; it does not expose
play-by-play snapshots in the documented NBA core endpoints.
"""

from __future__ import annotations

import os
from datetime import date

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

API_KEY_ENV = "SPORTSBLAZE_API_KEY"
DEFAULT_BASE_URL = "https://api.sportsblaze.com"


def _api_key(api_key: str | None = None) -> str:
    key = api_key or os.getenv(API_KEY_ENV)
    if not key:
        raise ValueError(f"{API_KEY_ENV} is not set")
    return key


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8), reraise=True)
def request(
    path: str,
    params: dict | None = None,
    *,
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> dict:
    query = {**(params or {}), "key": _api_key(api_key)}
    response = requests.get(f"{base_url}{path}", params=query, timeout=30)
    response.raise_for_status()
    return response.json()


def daily_schedule(
    game_date: date,
    *,
    league: str = "nba",
    params: dict | None = None,
    api_key: str | None = None,
) -> list[dict]:
    payload = request(
        f"/{league}/v1/schedule/daily/{game_date.isoformat()}.json",
        params=params,
        api_key=api_key,
    )
    return payload.get("games", [])


def daily_boxscores(
    game_date: date,
    *,
    league: str = "nba",
    params: dict | None = None,
    api_key: str | None = None,
) -> list[dict]:
    payload = request(
        f"/{league}/v1/boxscores/daily/{game_date.isoformat()}.json",
        params=params,
        api_key=api_key,
    )
    return payload.get("games", [])


def game_boxscore(
    game_id: str,
    *,
    league: str = "nba",
    api_key: str | None = None,
) -> dict:
    return request(f"/{league}/v1/boxscores/game/{game_id}.json", api_key=api_key)


def season_schedule(
    season: int,
    *,
    league: str = "nba",
    params: dict | None = None,
    api_key: str | None = None,
) -> list[dict]:
    payload = request(
        f"/{league}/v1/schedule/season/{season}.json",
        params=params,
        api_key=api_key,
    )
    return payload.get("games", [])


def final_home_win(game: dict) -> int | None:
    if game.get("status") != "Final":
        return None
    total = game.get("scores", {}).get("total", {})
    home = total.get("home", {}).get("points")
    away = total.get("away", {}).get("points")
    if home is None or away is None:
        return None
    return int(home > away)
