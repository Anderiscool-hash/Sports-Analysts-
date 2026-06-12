"""API-SPORTS multi-sport client.

Reads APISPORTS_API_KEY from the environment by default. The free plan can call
status/metadata endpoints, but historical game endpoints may be date-limited.
"""

from __future__ import annotations

import os
from datetime import date

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

API_KEY_ENV = "APISPORTS_API_KEY"

BASE_URLS = {
    "basketball": "https://v1.basketball.api-sports.io",
    "football": "https://v3.football.api-sports.io",
    "baseball": "https://v1.baseball.api-sports.io",
    "hockey": "https://v1.hockey.api-sports.io",
    "nba": "https://v2.nba.api-sports.io",
    "nfl": "https://v1.american-football.api-sports.io",
}


class ApiSportsError(RuntimeError):
    pass


def _api_key(api_key: str | None = None) -> str:
    key = api_key or os.getenv(API_KEY_ENV)
    if not key:
        raise ValueError(f"{API_KEY_ENV} is not set")
    return key


def base_url(sport: str) -> str:
    try:
        return BASE_URLS[sport]
    except KeyError as exc:
        supported = ", ".join(sorted(BASE_URLS))
        raise ValueError(f"unsupported sport '{sport}'. Supported: {supported}") from exc


def _errors(payload: dict) -> list[str]:
    errors = payload.get("errors")
    if not errors:
        return []
    if isinstance(errors, list):
        return [str(error) for error in errors]
    if isinstance(errors, dict):
        return [str(value) for value in errors.values()]
    return [str(errors)]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8), reraise=True)
def request(
    sport: str,
    path: str,
    params: dict | None = None,
    *,
    api_key: str | None = None,
) -> dict:
    response = requests.get(
        f"{base_url(sport)}{path}",
        params=params or {},
        headers={"x-apisports-key": _api_key(api_key)},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    errors = _errors(payload)
    if errors:
        raise ApiSportsError("; ".join(errors))
    return payload


def status(sport: str, api_key: str | None = None) -> dict:
    return request(sport, "/status", api_key=api_key).get("response", {})


def seasons(sport: str, api_key: str | None = None) -> list:
    return request(sport, "/seasons", api_key=api_key).get("response", [])


def basketball_games(
    game_date: date,
    *,
    league: int | None = None,
    season: str | int | None = None,
    api_key: str | None = None,
) -> list[dict]:
    params: dict[str, str | int] = {"date": game_date.isoformat()}
    if league is not None:
        params["league"] = league
    if season is not None:
        params["season"] = season
    return request("basketball", "/games", params, api_key=api_key).get("response", [])


def football_fixtures(
    game_date: date,
    *,
    league: int | None = None,
    season: int | None = None,
    api_key: str | None = None,
) -> list[dict]:
    params: dict[str, str | int] = {"date": game_date.isoformat()}
    if league is not None:
        params["league"] = league
    if season is not None:
        params["season"] = season
    return request("football", "/fixtures", params, api_key=api_key).get("response", [])
