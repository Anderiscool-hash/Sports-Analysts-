"""Live World Cup match state from ESPN's public soccer site API.

Free, no key. Mirrors the role of the NBA ``nba_scraper.get_live_state`` but returns a
``SoccerGameState``. The loop uses this as one link in a provider chain; on any failure
it returns ``None`` so the caller can fall back or degrade to model-only display.

Endpoint: ``site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard`` where
``league`` is e.g. ``fifa.world`` for the World Cup.
"""

from __future__ import annotations

import re
import time

import requests

from sportedge.types import SoccerGameState

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"


def _scoreboard(league: str, attempts: int = 3) -> dict:
    """Fetch the league scoreboard with a small exponential-backoff retry.

    Kept dependency-free (no tenacity) so the live pipeline runs with only requests.
    """
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(f"{BASE_URL}/{league}/scoreboard", timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001 - retry then re-raise on exhaustion
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(min(8.0, 2.0**attempt))
    raise last_exc  # type: ignore[misc]


def parse_minute(display_clock: str | None, status_state: str | None) -> float:
    """ESPN soccer clocks look like ``"67'"`` or ``"90'+3'"``. Halftime / pre /
    post are mapped to the obvious match minute so the model scales time correctly."""
    state = (status_state or "").lower()
    if state == "pre":
        return 0.0
    if state == "post":
        return 90.0
    text = display_clock or ""
    nums = [int(n) for n in re.findall(r"\d+", text)]
    if not nums:
        return 0.0
    minute = float(nums[0])
    if "+" in text and len(nums) > 1:  # stoppage time, e.g. 90'+3'
        minute += float(nums[1])
    return minute


def _team_matches(team: dict, wanted: str) -> bool:
    wanted_l = wanted.lower()
    for key in ("displayName", "shortDisplayName", "name", "abbreviation", "location"):
        value = str(team.get(key) or "").lower()
        if value and (wanted_l in value or value in wanted_l):
            return True
    return False


def _red_cards(competitor: dict) -> int:
    """Best-effort red-card count from competitor statistics; 0 if unavailable."""
    for stat in competitor.get("statistics") or []:
        name = str(stat.get("name") or stat.get("abbreviation") or "").lower()
        if "redcard" in name.replace(" ", "") or name == "rc":
            try:
                return int(float(stat.get("displayValue") or stat.get("value") or 0))
            except (TypeError, ValueError):
                return 0
    return 0


def state_from_event(
    event: dict,
    home_team: str,
    away_team: str,
    lambda_home: float,
    lambda_away: float,
) -> SoccerGameState | None:
    """Build a ``SoccerGameState`` if this event is the wanted matchup, else ``None``."""
    competitions = event.get("competitions") or []
    if not competitions:
        return None
    competition = competitions[0]
    competitors = competition.get("competitors") or []

    home = away = None
    for competitor in competitors:
        if competitor.get("homeAway") == "home":
            home = competitor
        elif competitor.get("homeAway") == "away":
            away = competitor
    if home is None or away is None:
        return None

    home_team_obj = home.get("team") or {}
    away_team_obj = away.get("team") or {}
    # The configured "home_team" must match ESPN's home side (and likewise away).
    if not (_team_matches(home_team_obj, home_team) and _team_matches(away_team_obj, away_team)):
        return None

    try:
        home_goals = int(home.get("score") or 0)
        away_goals = int(away.get("score") or 0)
    except (TypeError, ValueError):
        home_goals = away_goals = 0

    status = event.get("status") or {}
    minute = parse_minute(
        status.get("displayClock"),
        (status.get("type") or {}).get("state"),
    )

    return SoccerGameState(
        home_team=home_team,
        away_team=away_team,
        home_goals=home_goals,
        away_goals=away_goals,
        minute=minute,
        home_red_cards=_red_cards(home),
        away_red_cards=_red_cards(away),
        lambda_home=lambda_home,
        lambda_away=lambda_away,
    )


def get_live_state(
    home_team: str,
    away_team: str,
    lambda_home: float = 1.45,
    lambda_away: float = 1.15,
    league: str = "fifa.world",
) -> SoccerGameState | None:
    """Return the live state for the wanted matchup, or ``None`` if not found/in error."""
    try:
        events = _scoreboard(league).get("events") or []
    except Exception:  # noqa: BLE001 - network/parse failure -> let caller fall back
        return None
    for event in events:
        state = state_from_event(event, home_team, away_team, lambda_home, lambda_away)
        if state is not None:
            return state
    return None
