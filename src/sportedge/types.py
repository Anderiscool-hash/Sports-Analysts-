"""Core shared types. Kept dependency-free so every layer can import it."""

from __future__ import annotations

from dataclasses import dataclass

REGULATION_SECONDS = 4 * 12 * 60  # 2880


def regulation_seconds_remaining(period: int, clock_seconds: float) -> float:
    """Total seconds left in regulation given the current period and the clock
    (seconds left in that period). Overtime is treated as ~0 remaining — at that
    point score differential dominates the model anyway."""
    if period >= 5:  # overtime
        return 0.0
    return max(0.0, clock_seconds + (4 - period) * 12 * 60)


@dataclass(frozen=True)
class GameState:
    """A single in-game snapshot, from the home team's perspective."""

    home_team: str
    away_team: str
    home_score: int
    away_score: int
    period: int  # 1-4 regulation, 5+ overtime
    seconds_remaining: float  # seconds left in regulation (see helper above)
    pre_game_home_prob: float = 0.5
    home_has_possession: bool | None = None
    home_recent_net_rating: float = 0.0
    away_recent_net_rating: float = 0.0

    @property
    def score_diff(self) -> int:
        """Home minus away. Positive = home leading."""
        return self.home_score - self.away_score

    @property
    def recent_net_rating_diff(self) -> float:
        """Home team recent net rating minus away team recent net rating."""
        return self.home_recent_net_rating - self.away_recent_net_rating

    @property
    def is_final(self) -> bool:
        return self.seconds_remaining <= 0 and self.period >= 4


REGULATION_MINUTES = 90


@dataclass(frozen=True)
class WinProb3:
    """A 1X2 match-result distribution from the home team's perspective.

    The three values are probabilities of the 90-minute regulation result and
    always sum to 1: home win, draw, away win."""

    home: float
    draw: float
    away: float


@dataclass(frozen=True)
class SoccerGameState:
    """A single in-game soccer snapshot, from the home team's perspective.

    ``lambda_home`` / ``lambda_away`` are full-match expected goals (xG) priors for
    this matchup — supplied by pre-match calibration. The in-game model scales them
    by the fraction of the match remaining. Red cards adjust the effective rates."""

    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    minute: float  # minutes elapsed (0..90, may exceed 90 in stoppage time)
    home_red_cards: int = 0
    away_red_cards: int = 0
    lambda_home: float = 1.45  # full-match expected goals, home
    lambda_away: float = 1.15  # full-match expected goals, away

    @property
    def goal_diff(self) -> int:
        """Home minus away. Positive = home leading."""
        return self.home_goals - self.away_goals

    @property
    def minutes_remaining(self) -> float:
        """Regulation minutes left; clamped to [0, 90]. Stoppage (minute > 90)
        reads as 0 remaining — the scoreline dominates the result there anyway."""
        return max(0.0, REGULATION_MINUTES - float(self.minute))

    @property
    def is_final(self) -> bool:
        return self.minute >= REGULATION_MINUTES


# --------------------------------------------------------------------------- #
# Live dashboard types: a richer, sport-agnostic view of a game in progress.
# These carry display detail (fouls, cards, possession, last play) that the
# trading models don't need, so they're kept separate from GameState.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GameCandidate:
    """A pickable live/upcoming game for the dashboard's selection menu."""

    sport: str  # "basketball" | "soccer"
    league: str  # ESPN league slug, e.g. "nba" or "fifa.world"
    event_id: str
    home_team: str
    away_team: str
    status: str  # "pre" | "in" | "post"
    short_detail: str = ""  # e.g. "Q3 4:21" or "67'"


@dataclass
class LiveDetail:
    """A rich live snapshot of one game, from the home team's perspective.

    Sport-agnostic: basketball-only fields stay 0/False for soccer and vice
    versa. ``last_play_text`` is shown verbatim and is the source of truth; the
    ``free_throw_active`` / ``set_piece`` flags are best-effort derivations from
    it (see ``detect_free_throw`` / ``detect_set_piece``)."""

    sport: str
    league: str
    home_team: str
    away_team: str
    home_score: int = 0
    away_score: int = 0
    status: str = "pre"  # "pre" | "in" | "post"
    clock: str = ""  # raw display clock, e.g. "4:21" or "67'"
    period: int = 0  # basketball quarter (1-4, 5+ OT); 0 for soccer
    minute: float = 0.0  # soccer match minute; 0.0 for basketball
    possession: str = ""  # "home" | "away" | "" if unknown
    last_play_text: str = ""
    # basketball
    home_fouls: int = 0
    away_fouls: int = 0
    free_throw_active: bool = False
    # soccer
    home_yellow: int = 0
    away_yellow: int = 0
    home_red: int = 0
    away_red: int = 0
    set_piece: str = ""  # "" | "free kick" | "penalty" | "corner"


def detect_free_throw(last_play_text: str) -> bool:
    """True if the most recent play describes a free throw (best-effort)."""
    return "free throw" in (last_play_text or "").lower()


def detect_set_piece(last_play_text: str) -> str:
    """Classify a soccer set-piece from the most recent play text (best-effort).

    Returns "penalty", "free kick", "corner", or "" if none is recognized.
    Penalty is checked first since "penalty" text can also mention a kick."""
    text = (last_play_text or "").lower()
    if "penalty" in text:
        return "penalty"
    if "free kick" in text or "free-kick" in text:
        return "free kick"
    if "corner" in text:
        return "corner"
    return ""
