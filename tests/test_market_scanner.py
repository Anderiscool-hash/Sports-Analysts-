from sportedge.market.kalshi import KalshiDiscoveryResult
from sportedge.market.scanner import scan_game_markets
from sportedge.types import GameCandidate


class _Client:
    def discover_team_win_market(self, team, opponent):  # noqa: ARG002
        if team == "Lakers":
            return KalshiDiscoveryResult(
                ticker="KXNBA-LAL-WIN",
                title="Will the Lakers win?",
                team=team,
                score=8.0,
                yes_bid=0.50,
                yes_ask=0.54,
            )
        return None


def test_scan_game_markets_prioritizes_games_with_markets():
    games = [
        GameCandidate("basketball", "nba", "1", "Spurs", "Knicks", "in", "Q2"),
        GameCandidate("basketball", "nba", "2", "Lakers", "Celtics", "in", "Q1"),
    ]

    rows = scan_game_markets(games, _Client())

    assert rows[0].game.home_team == "Lakers"
    assert rows[0].has_market is True
    assert rows[0].home_market is not None
    assert rows[1].has_market is False


def test_scan_game_markets_includes_soccer_and_filters_finished_games():
    games = [
        GameCandidate("basketball", "nba", "1", "Lakers", "Celtics", "post", "FT"),
        GameCandidate("soccer", "fifa.world", "2", "France", "Senegal", "in", "23'"),
    ]

    rows = scan_game_markets(games, _Client())

    # Finished (post) game is filtered; the live World Cup game is now scanned.
    assert len(rows) == 1
    assert rows[0].game.sport == "soccer"
    assert rows[0].game.event_id == "2"
