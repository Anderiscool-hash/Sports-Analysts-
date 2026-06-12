"""iSports API live basketball access.

Reads ISPORTS_API_KEY from the environment by default. The backup host is the
default because it succeeded in validation while the primary host rejected the
same key.
"""

from __future__ import annotations

import os

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from sportedge.types import GameState, regulation_seconds_remaining

API_KEY_ENV = "ISPORTS_API_KEY"
DEFAULT_BASE_URL = "http://api2.isportsapi.com"


def _api_key(api_key: str | None = None) -> str:
    key = api_key or os.getenv(API_KEY_ENV)
    if not key:
        raise ValueError(f"{API_KEY_ENV} is not set")
    return key


def _match_team(match: dict, side: str, name: str) -> bool:
    wanted = name.lower()
    if not wanted:
        return False
    return wanted in str(match.get(f"{side}Name", "")).lower()


def _quarter_clock_to_seconds(value: str | None) -> float:
    if not value:
        return 0.0
    parts = str(value).split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except ValueError:
        return 0.0


def _period_from_status(status: int) -> int:
    # iSports status: 1-4 regulation quarters, 5+ overtime, 50 halftime.
    if status == 50:
        return 2
    if status >= 1:
        return status
    return 1


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8), reraise=True)
def request(
    path: str,
    params: dict | None = None,
    *,
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> dict:
    query = {**(params or {}), "api_key": _api_key(api_key)}
    response = requests.get(f"{base_url}{path}", params=query, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != 0:
        raise RuntimeError(payload.get("message") or f"iSports error code {payload.get('code')}")
    return payload


def list_livescores(api_key: str | None = None, base_url: str = DEFAULT_BASE_URL) -> list[dict]:
    return request(
        "/sport/basketball/livescores",
        api_key=api_key,
        base_url=base_url,
    ).get("data", [])


def match_to_state(match: dict, pre_game_home_prob: float = 0.60) -> GameState:
    period = _period_from_status(int(match.get("status") or 1))
    clock_seconds = _quarter_clock_to_seconds(match.get("quarterRemainTime"))
    return GameState(
        home_team=str(match.get("homeName", "")),
        away_team=str(match.get("awayName", "")),
        home_score=int(match.get("homeScore") or 0),
        away_score=int(match.get("awayScore") or 0),
        period=period,
        seconds_remaining=regulation_seconds_remaining(period, clock_seconds),
        pre_game_home_prob=pre_game_home_prob,
    )


def get_live_state(
    home_team: str,
    away_team: str,
    pre_game_home_prob: float = 0.60,
    *,
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> GameState | None:
    for match in list_livescores(api_key=api_key, base_url=base_url):
        if _match_team(match, "home", home_team) and _match_team(match, "away", away_team):
            return match_to_state(match, pre_game_home_prob)
    return None
