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

    @property
    def score_diff(self) -> int:
        """Home minus away. Positive = home leading."""
        return self.home_score - self.away_score

    @property
    def is_final(self) -> bool:
        return self.seconds_remaining <= 0 and self.period >= 4
