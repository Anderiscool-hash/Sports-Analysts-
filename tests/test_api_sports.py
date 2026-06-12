from datetime import date

import pytest

from sportedge.data import api_sports


def test_base_url_supported_sports():
    assert api_sports.base_url("basketball") == "https://v1.basketball.api-sports.io"
    assert api_sports.base_url("football") == "https://v3.football.api-sports.io"


def test_base_url_rejects_unknown_sport():
    with pytest.raises(ValueError, match="unsupported sport"):
        api_sports.base_url("cricket")


def test_basketball_games_call_shape(monkeypatch):
    def fake_request(sport, path, params=None, *, api_key=None):
        assert sport == "basketball"
        assert path == "/games"
        assert params == {"date": "2026-06-06", "league": 12, "season": "2025-2026"}
        assert api_key == "k"
        return {"response": [{"id": 1}]}

    monkeypatch.setattr(api_sports, "request", fake_request)

    games = api_sports.basketball_games(
        date(2026, 6, 6),
        league=12,
        season="2025-2026",
        api_key="k",
    )

    assert games == [{"id": 1}]


def test_request_raises_provider_errors(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "errors": {
                    "plan": "Free plans do not have access to this date.",
                }
            }

    monkeypatch.setattr(api_sports.requests, "get", lambda *_, **__: Response())

    with pytest.raises(api_sports.ApiSportsError, match="Free plans"):
        api_sports.request("basketball", "/games", api_key="k")
