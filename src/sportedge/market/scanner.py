"""Scan live games for direct, quoted Kalshi team-win markets."""

from __future__ import annotations

from dataclasses import dataclass

from sportedge.data.espn_live import list_live_games
from sportedge.market.kalshi import KalshiClient, KalshiDiscoveryResult
from sportedge.types import GameCandidate


@dataclass(frozen=True)
class GameMarketCoverage:
    game: GameCandidate
    home_market: KalshiDiscoveryResult | None
    away_market: KalshiDiscoveryResult | None

    @property
    def has_market(self) -> bool:
        return self.home_market is not None or self.away_market is not None

    @property
    def market_count(self) -> int:
        return int(self.home_market is not None) + int(self.away_market is not None)


def scan_game_markets(
    games: list[GameCandidate],
    client: KalshiClient,
    sports: set[str] | None = None,
    statuses: set[str] | None = None,
) -> list[GameMarketCoverage]:
    """Return Kalshi coverage for the supplied ESPN games."""
    sports = sports or {"basketball", "soccer"}
    statuses = statuses or {"in", "pre"}
    coverage: list[GameMarketCoverage] = []
    for game in games:
        if game.sport not in sports or game.status not in statuses:
            continue
        home = client.discover_team_win_market(game.home_team, game.away_team)
        away = client.discover_team_win_market(game.away_team, game.home_team)
        coverage.append(GameMarketCoverage(game=game, home_market=home, away_market=away))
    coverage.sort(key=lambda row: (not row.has_market, -row.market_count, row.game.status))
    return coverage


def scan_live_game_markets(
    client: KalshiClient | None = None,
    sports: set[str] | None = None,
    statuses: set[str] | None = None,
) -> list[GameMarketCoverage]:
    """Fetch ESPN games, then scan them for Kalshi market coverage."""
    return scan_game_markets(
        list_live_games(),
        client or KalshiClient(),
        sports=sports,
        statuses=statuses,
    )
