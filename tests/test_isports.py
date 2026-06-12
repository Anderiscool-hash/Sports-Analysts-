import pytest

from sportedge.data import isports


def _match() -> dict:
    return {
        "matchId": "10105073",
        "status": 3,
        "quarterRemainTime": "04:07",
        "homeName": "New York Liberty",
        "awayName": "Indiana Fever",
        "homeScore": 45,
        "awayScore": 55,
    }


def test_match_to_state_converts_clock_and_score():
    state = isports.match_to_state(_match())

    assert state.home_team == "New York Liberty"
    assert state.away_team == "Indiana Fever"
    assert state.home_score == 45
    assert state.away_score == 55
    assert state.period == 3
    assert state.seconds_remaining == 967.0


def test_get_live_state_matches_teams(monkeypatch):
    monkeypatch.setattr(isports, "list_livescores", lambda **_: [_match()])

    state = isports.get_live_state("Liberty", "Fever", api_key="k")

    assert state is not None
    assert state.home_score == 45


def test_get_live_state_returns_none_without_match(monkeypatch):
    monkeypatch.setattr(isports, "list_livescores", lambda **_: [_match()])

    assert isports.get_live_state("Celtics", "Mavericks", api_key="k") is None


def test_request_raises_on_api_error(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"code": 2, "message": "Invalid [api_key], illegal access."}

    monkeypatch.setattr(isports.requests, "get", lambda *_, **__: Response())

    with pytest.raises(RuntimeError, match="Invalid"):
        isports.request("/sport/basketball/livescores", api_key="bad")
