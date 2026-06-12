from datetime import date

from sportedge.data import sportsblaze


def test_daily_boxscores_returns_games(monkeypatch):
    def fake_request(path, params=None, *, api_key=None, base_url=sportsblaze.DEFAULT_BASE_URL):
        assert path == "/nba/v1/boxscores/daily/2025-04-11.json"
        assert api_key == "k"
        return {"games": [{"id": "game-1"}]}

    monkeypatch.setattr(sportsblaze, "request", fake_request)

    assert sportsblaze.daily_boxscores(date(2025, 4, 11), api_key="k") == [{"id": "game-1"}]


def test_game_boxscore_uses_league_and_id(monkeypatch):
    def fake_request(path, params=None, *, api_key=None, base_url=sportsblaze.DEFAULT_BASE_URL):
        assert path == "/nfl/v1/boxscores/game/super-bowl.json"
        return {"id": "super-bowl"}

    monkeypatch.setattr(sportsblaze, "request", fake_request)

    assert sportsblaze.game_boxscore("super-bowl", league="nfl", api_key="k")["id"] == "super-bowl"


def test_final_home_win():
    game = {
        "status": "Final",
        "scores": {
            "total": {
                "away": {"points": 109},
                "home": {"points": 140},
            }
        },
    }

    assert sportsblaze.final_home_win(game) == 1


def test_final_home_win_none_when_not_final():
    assert sportsblaze.final_home_win({"status": "Scheduled"}) is None
