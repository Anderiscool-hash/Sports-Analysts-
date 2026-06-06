"""NBA data access via nba_api.

Historical: list games for a season, pull play-by-play, turn each in-game moment
into a (state -> final outcome) training row.
Live: read the current game's score / clock / period into a GameState.

nba_api is imported lazily inside functions so this module imports fine even before
the dependency is installed. Endpoints can be flaky / rate-limited, hence tenacity
retries. SCOREMARGIN orientation is validated against the known final result.
"""

from __future__ import annotations

import re

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from sportedge.types import GameState, regulation_seconds_remaining

HOME_PRIOR = 0.60  # simple home-court prior used for historical rows

TRAINING_COLUMNS = [
    "game_id",
    "home_score",
    "away_score",
    "period",
    "seconds_remaining",
    "pre_game_home_prob",
    "home_win",
]


def _pctime_to_seconds(pctime: str) -> float:
    """'MM:SS' game clock string -> seconds left in the period."""
    if not pctime or ":" not in pctime:
        return 0.0
    mm, ss = pctime.split(":")
    return int(mm) * 60 + float(ss)


def _iso_clock_to_seconds(clock: str) -> float:
    """ISO-8601 'PT05M30.00S' -> seconds left in the period."""
    if not clock:
        return 0.0
    m = re.match(r"PT(?:(\d+)M)?(?:([\d.]+)S)?", clock)
    if not m:
        return 0.0
    minutes = int(m.group(1) or 0)
    seconds = float(m.group(2) or 0.0)
    return minutes * 60 + seconds


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def list_games(season: str, season_type: str = "Playoffs") -> pd.DataFrame:
    """One row per team-game. MATCHUP 'vs.' = home, '@' = away."""
    from nba_api.stats.endpoints import leaguegamefinder

    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season, season_type_nullable=season_type
    )
    return finder.get_data_frames()[0]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def _playbyplay(game_id: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import playbyplayv2

    return playbyplayv2.PlayByPlayV2(game_id=game_id).get_data_frames()[0]


def game_training_rows(game_id: str, home_won: bool) -> pd.DataFrame:
    """Convert one game's play-by-play into labeled state rows.

    SCOREMARGIN is the home margin; its final sign is checked against `home_won`
    and flipped if the source orientation disagrees.
    """
    pbp = _playbyplay(game_id)
    rows: list[dict] = []
    margins: list[int] = []
    parsed: list[tuple[int, float, int]] = []  # (period, secs_remaining, margin)

    for _, ev in pbp.iterrows():
        margin = ev.get("SCOREMARGIN")
        if margin in (None, "", "TIE"):
            margin = 0
        try:
            margin = int(margin)
        except (TypeError, ValueError):
            continue
        period = int(ev.get("PERIOD", 1) or 1)
        secs = regulation_seconds_remaining(period, _pctime_to_seconds(ev.get("PCTIMESTRING", "")))
        margins.append(margin)
        parsed.append((period, secs, margin))

    if not parsed:
        return pd.DataFrame(columns=TRAINING_COLUMNS)

    # Orientation check: last non-zero margin should agree with who won.
    last_nonzero = next((m for m in reversed(margins) if m != 0), 0)
    flip = (last_nonzero > 0) != home_won and last_nonzero != 0

    for period, secs, margin in parsed:
        m = -margin if flip else margin
        rows.append(
            {
                "game_id": game_id,
                "home_score": max(m, 0),  # only the diff matters to the model
                "away_score": max(-m, 0),
                "period": period,
                "seconds_remaining": secs,
                "pre_game_home_prob": HOME_PRIOR,
                "home_win": int(home_won),
            }
        )
    return pd.DataFrame(rows, columns=TRAINING_COLUMNS)


def build_training_set(seasons: list[str], season_type: str = "Playoffs") -> pd.DataFrame:
    """Pull many games across seasons into one labeled training frame."""
    frames: list[pd.DataFrame] = []
    for season in seasons:
        games = list_games(season, season_type)
        # Home rows only (MATCHUP contains 'vs.'); WL gives the label.
        home_rows = games[games["MATCHUP"].str.contains("vs.", regex=False, na=False)]
        for _, g in home_rows.iterrows():
            home_won = str(g.get("WL", "")).upper().startswith("W")
            try:
                frames.append(game_training_rows(str(g["GAME_ID"]), home_won))
            except Exception:
                continue  # skip a single bad game rather than abort the whole pull
    if not frames:
        return pd.DataFrame(columns=TRAINING_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _match_team(game: dict, name: str) -> bool:
    name = name.lower()
    for key in ("teamName", "teamTricode", "teamCity"):
        if name and name in str(game.get(key, "")).lower():
            return True
    return False


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8))
def get_live_state(
    home_team: str, away_team: str, pre_game_home_prob: float = HOME_PRIOR
) -> GameState | None:
    """Current GameState for the live game matching the given teams, or None."""
    from nba_api.live.nba.endpoints import scoreboard

    games = scoreboard.ScoreBoard().get_dict()["scoreboard"]["games"]
    for g in games:
        home = g["homeTeam"]
        away = g["awayTeam"]
        if _match_team(home, home_team) and _match_team(away, away_team):
            period = int(g.get("period", 1) or 1)
            secs = regulation_seconds_remaining(period, _iso_clock_to_seconds(g.get("gameClock", "")))
            return GameState(
                home_team=home_team,
                away_team=away_team,
                home_score=int(home.get("score", 0)),
                away_score=int(away.get("score", 0)),
                period=period,
                seconds_remaining=secs,
                pre_game_home_prob=pre_game_home_prob,
            )
    return None
