"""Unified ESPN live source for the dashboard (basketball + soccer).

Free, no key. Two endpoints off ``site.api.espn.com``:

- ``/sports/{sport}/{league}/scoreboard`` — list games (for the picker).
- ``/sports/{sport}/{league}/summary?event={id}`` — rich per-game detail
  (score, fouls, cards, possession, last play).

The parsing functions (``parse_candidates`` / ``parse_detail``) are pure: they
take already-decoded JSON and return our types, so they're unit-tested without
network. The ``list_live_games`` / ``get_game_detail`` wrappers add the HTTP call
and degrade to ``[]`` / ``None`` on any failure so the dashboard never crashes.
"""

from __future__ import annotations

import requests

from sportedge.data.espn_soccer import parse_minute
from sportedge.types import GameCandidate, LiveDetail, detect_free_throw, detect_set_piece

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports"

# Leagues scanned for the picker. NBA plus a handful of common soccer comps.
NBA = ("basketball", "nba")
DEFAULT_SOCCER_LEAGUES = ("fifa.world", "eng.1", "esp.1", "ita.1", "ger.1", "usa.1")


# ----- HTTP -----
def _get(sport: str, league: str, path: str, params: dict | None = None) -> dict:
    resp = requests.get(f"{BASE_URL}/{sport}/{league}/{path}", params=params or {}, timeout=20)
    resp.raise_for_status()
    return resp.json()


# ----- pure parsing -----
def _competitors(competition: dict) -> tuple[dict | None, dict | None]:
    home = away = None
    for c in competition.get("competitors") or []:
        if c.get("homeAway") == "home":
            home = c
        elif c.get("homeAway") == "away":
            away = c
    return home, away


def _team_name(competitor: dict | None) -> str:
    return str(((competitor or {}).get("team") or {}).get("displayName") or "")


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def parse_candidates(events: list[dict], sport: str, league: str) -> list[GameCandidate]:
    """Scoreboard ``events`` -> pickable candidates (skips malformed entries)."""
    out: list[GameCandidate] = []
    for event in events:
        competition = (event.get("competitions") or [{}])[0]
        home, away = _competitors(competition)
        if home is None or away is None:
            continue
        status_type = (event.get("status") or competition.get("status") or {}).get("type") or {}
        out.append(
            GameCandidate(
                sport=sport,
                league=league,
                event_id=str(event.get("id") or ""),
                home_team=_team_name(home),
                away_team=_team_name(away),
                status=str(status_type.get("state") or ""),
                short_detail=str(status_type.get("shortDetail") or ""),
            )
        )
    return out


def _box_stats_by_homeaway(summary: dict) -> dict[str, dict[str, str]]:
    """``boxscore.teams[]`` -> {"home": {stat_name: value}, "away": {...}}."""
    out: dict[str, dict[str, str]] = {}
    for team in (summary.get("boxscore") or {}).get("teams") or []:
        side = team.get("homeAway")
        if side in ("home", "away"):
            out[side] = {
                str(s.get("name") or "").lower(): str(s.get("displayValue") or s.get("value") or "")
                for s in (team.get("statistics") or [])
            }
    return out


def _foul_count(stats: dict[str, str]) -> int:
    """Live NBA team fouls. ESPN's stat key varies; match the plain foul count
    and ignore technical/flagrant breakdowns."""
    for key in ("fouls", "totalfouls", "personalfouls"):
        if key in stats:
            return _int(stats[key])
    for key, value in stats.items():
        if "foul" in key and "technical" not in key and "flagrant" not in key:
            return _int(value)
    return 0


def _latest_play_text(competition: dict, summary: dict) -> str:
    """Most recent play text: prefer live ``situation.lastPlay``, else the newest
    commentary entry (highest ``sequence``)."""
    situation = competition.get("situation") or {}
    last_play = situation.get("lastPlay") or {}
    text = str(last_play.get("text") or "")
    if text:
        return text
    commentary = summary.get("commentary") or []
    if commentary:
        latest = max(commentary, key=lambda c: c.get("sequence") or 0)
        return str(latest.get("text") or "")
    return ""


def _possession(home: dict | None, away: dict | None, competition: dict) -> str:
    """Which side has the ball, from competitor ``possession`` flags or the live
    situation's possession team id."""
    if home and home.get("possession") is True:
        return "home"
    if away and away.get("possession") is True:
        return "away"
    poss_id = (competition.get("situation") or {}).get("possession")
    if poss_id is not None:
        if str(poss_id) == str(((home or {}).get("team") or {}).get("id")):
            return "home"
        if str(poss_id) == str(((away or {}).get("team") or {}).get("id")):
            return "away"
    return ""


def parse_detail(summary: dict, sport: str, league: str) -> LiveDetail:
    """Summary JSON -> ``LiveDetail`` (defensive: missing fields default sanely)."""
    competition = ((summary.get("header") or {}).get("competitions") or [{}])[0]
    home, away = _competitors(competition)
    status = competition.get("status") or {}
    status_type = status.get("type") or {}
    state = str(status_type.get("state") or "pre")
    clock = str(status.get("displayClock") or status_type.get("shortDetail") or "")
    period = _int(status.get("period"))

    box = _box_stats_by_homeaway(summary)
    home_box = box.get("home", {})
    away_box = box.get("away", {})

    last_play = _latest_play_text(competition, summary)

    detail = LiveDetail(
        sport=sport,
        league=league,
        home_team=_team_name(home),
        away_team=_team_name(away),
        home_score=_int((home or {}).get("score")),
        away_score=_int((away or {}).get("score")),
        status=state,
        clock=clock,
        period=period if sport == "basketball" else 0,
        minute=parse_minute(clock, state) if sport == "soccer" else 0.0,
        possession=_possession(home, away, competition),
        last_play_text=last_play,
    )

    if sport == "basketball":
        detail.home_fouls = _foul_count(home_box)
        detail.away_fouls = _foul_count(away_box)
        detail.free_throw_active = detect_free_throw(last_play)
    else:
        detail.home_yellow = _int(home_box.get("yellowcards"))
        detail.away_yellow = _int(away_box.get("yellowcards"))
        detail.home_red = _int(home_box.get("redcards"))
        detail.away_red = _int(away_box.get("redcards"))
        detail.set_piece = detect_set_piece(last_play)

    return detail


# ----- network wrappers (degrade to empty / None) -----
def _safe_candidates(sport: str, league: str) -> list[GameCandidate]:
    try:
        events = _get(sport, league, "scoreboard").get("events") or []
    except Exception:  # noqa: BLE001 - skip a flaky league rather than abort
        return []
    return parse_candidates(events, sport, league)


def list_live_games(soccer_leagues: tuple[str, ...] = DEFAULT_SOCCER_LEAGUES) -> list[GameCandidate]:
    """All games found across NBA + the given soccer leagues. Caller filters by
    ``status`` (the picker shows in-progress and upcoming, hides finished)."""
    candidates = _safe_candidates(*NBA)
    for league in soccer_leagues:
        candidates.extend(_safe_candidates("soccer", league))
    return candidates


def get_game_detail(sport: str, league: str, event_id: str) -> LiveDetail | None:
    """Fetch + parse one game's summary, or ``None`` on any network/parse error."""
    try:
        summary = _get(sport, league, "summary", {"event": event_id})
    except Exception:  # noqa: BLE001 - let the dashboard keep its last good frame
        return None
    try:
        return parse_detail(summary, sport, league)
    except Exception:  # noqa: BLE001 - malformed payload -> degrade
        return None
