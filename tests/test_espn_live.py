"""Tests for the live dashboard data layer and pure helpers.

No network: ESPN ``summary`` / scoreboard payloads are represented as fixture
dicts mirroring the real shape, and the dashboard's conversion / render / picker
logic is exercised with stubs.
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from sportedge.config import Config
from sportedge.data.espn_live import parse_candidates, parse_detail
from sportedge.live.dashboard import (
    build_readout,
    clock_to_seconds,
    detail_to_basketball_state,
    detail_to_soccer_state,
    pick_game,
    render,
)
from sportedge.types import (
    GameCandidate,
    WinProb3,
    detect_free_throw,
    detect_set_piece,
)

# --------------------------------------------------------------------------- #
# Fixtures (trimmed to the fields the parser reads)
# --------------------------------------------------------------------------- #

SOCCER_SUMMARY = {
    "header": {
        "competitions": [
            {
                "status": {
                    "displayClock": "67'",
                    "period": 2,
                    "type": {"state": "in", "shortDetail": "67'"},
                },
                "situation": {},
                "competitors": [
                    {
                        "homeAway": "home",
                        "score": "1",
                        "possession": True,
                        "team": {"id": "331", "displayName": "Brighton"},
                    },
                    {
                        "homeAway": "away",
                        "score": "0",
                        "possession": False,
                        "team": {"id": "360", "displayName": "Man United"},
                    },
                ],
            }
        ]
    },
    "boxscore": {
        "teams": [
            {
                "homeAway": "home",
                "team": {"id": "331"},
                "statistics": [
                    {"name": "foulsCommitted", "displayValue": "8"},
                    {"name": "yellowCards", "displayValue": "2"},
                    {"name": "redCards", "displayValue": "0"},
                ],
            },
            {
                "homeAway": "away",
                "team": {"id": "360"},
                "statistics": [
                    {"name": "yellowCards", "displayValue": "1"},
                    {"name": "redCards", "displayValue": "1"},
                ],
            },
        ]
    },
    "commentary": [
        {"sequence": 0, "text": "Kickoff."},
        {"sequence": 50, "text": "Free kick awarded to Brighton."},
    ],
}

NBA_SUMMARY = {
    "header": {
        "competitions": [
            {
                "status": {
                    "displayClock": "4:21",
                    "period": 3,
                    "type": {"state": "in", "shortDetail": "Q3 4:21"},
                },
                "situation": {
                    "lastPlay": {"text": "LeBron James makes free throw 1 of 2"},
                    "possession": "13",
                },
                "competitors": [
                    {"homeAway": "home", "score": "77", "team": {"id": "13", "displayName": "Lakers"}},
                    {"homeAway": "away", "score": "80", "team": {"id": "5", "displayName": "Celtics"}},
                ],
            }
        ]
    },
    "boxscore": {
        "teams": [
            {"homeAway": "home", "team": {"id": "13"}, "statistics": [{"name": "fouls", "displayValue": "5"}]},
            {"homeAway": "away", "team": {"id": "5"}, "statistics": [{"name": "fouls", "displayValue": "7"}]},
        ]
    },
}


# --------------------------------------------------------------------------- #
# Detection helpers
# --------------------------------------------------------------------------- #


def test_detect_free_throw():
    assert detect_free_throw("Player makes FREE THROW 2 of 2") is True
    assert detect_free_throw("Jump shot good") is False
    assert detect_free_throw("") is False


def test_detect_set_piece_priority_and_kinds():
    assert detect_set_piece("Penalty awarded after a foul") == "penalty"
    assert detect_set_piece("Free kick in a dangerous area") == "free kick"
    assert detect_set_piece("Corner taken short") == "corner"
    assert detect_set_piece("Throw-in on the left") == ""


# --------------------------------------------------------------------------- #
# parse_candidates
# --------------------------------------------------------------------------- #


def test_parse_candidates_maps_and_skips_malformed():
    events = [
        {
            "id": "111",
            "status": {"type": {"state": "in", "shortDetail": "67'"}},
            "competitions": [
                {
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "Brazil"}},
                        {"homeAway": "away", "team": {"displayName": "Morocco"}},
                    ]
                }
            ],
        },
        {"id": "222", "competitions": [{"competitors": []}]},  # malformed -> skipped
    ]
    cands = parse_candidates(events, "soccer", "fifa.world")
    assert len(cands) == 1
    c = cands[0]
    assert (c.event_id, c.home_team, c.away_team, c.status) == ("111", "Brazil", "Morocco", "in")


# --------------------------------------------------------------------------- #
# parse_detail
# --------------------------------------------------------------------------- #


def test_parse_detail_soccer():
    d = parse_detail(SOCCER_SUMMARY, "soccer", "eng.1")
    assert (d.home_team, d.away_team) == ("Brighton", "Man United")
    assert (d.home_score, d.away_score) == (1, 0)
    assert d.status == "in"
    assert d.minute == 67.0
    assert d.possession == "home"
    assert (d.home_yellow, d.away_yellow) == (2, 1)
    assert (d.home_red, d.away_red) == (0, 1)
    # situation empty -> falls back to newest commentary -> set piece detected
    assert d.set_piece == "free kick"
    assert "Free kick" in d.last_play_text


def test_parse_detail_basketball():
    d = parse_detail(NBA_SUMMARY, "basketball", "nba")
    assert (d.home_team, d.away_team) == ("Lakers", "Celtics")
    assert (d.home_score, d.away_score) == (77, 80)
    assert d.period == 3
    assert d.clock == "4:21"
    assert (d.home_fouls, d.away_fouls) == (5, 7)
    assert d.free_throw_active is True
    assert d.possession == "home"  # situation.possession id "13" == home team id


# --------------------------------------------------------------------------- #
# dashboard conversions
# --------------------------------------------------------------------------- #


def test_clock_to_seconds():
    assert clock_to_seconds("4:21") == 261
    assert clock_to_seconds("12:00") == 720
    assert clock_to_seconds("67'") == 0.0  # soccer clock, no colon
    assert clock_to_seconds("") == 0.0


def test_detail_to_basketball_state():
    d = parse_detail(NBA_SUMMARY, "basketball", "nba")
    state = detail_to_basketball_state(d)
    assert state.period == 3
    # 4:21 left in Q3 -> 261 + one full 12-min quarter remaining
    assert state.seconds_remaining == 261 + 12 * 60
    assert state.score_diff == 77 - 80


def test_detail_to_soccer_state():
    d = parse_detail(SOCCER_SUMMARY, "soccer", "eng.1")
    state = detail_to_soccer_state(d, lambda_home=1.4, lambda_away=1.1)
    assert state.home_goals == 1 and state.away_goals == 0
    assert state.minute == 67.0
    assert state.away_red_cards == 1
    assert state.lambda_home == 1.4


# --------------------------------------------------------------------------- #
# build_readout (stub model, no market client)
# --------------------------------------------------------------------------- #


class _BballModel:
    def predict(self, state) -> float:  # noqa: ARG002
        return 0.62


class _SoccerModel:
    def predict(self, state) -> WinProb3:  # noqa: ARG002
        return WinProb3(0.50, 0.30, 0.20)


def test_build_readout_basketball_no_market():
    d = parse_detail(NBA_SUMMARY, "basketball", "nba")
    rows = build_readout(Config(), d, _BballModel(), client=None)
    assert len(rows) == 1
    label, prob, price, edge = rows[0]
    assert label == "Lakers win"
    assert prob == 0.62
    assert price is None and edge is None


def test_build_readout_soccer_no_market():
    d = parse_detail(SOCCER_SUMMARY, "soccer", "eng.1")
    rows = build_readout(Config(), d, _SoccerModel(), client=None)
    assert [r[0] for r in rows] == ["Brighton", "Draw", "Man United"]
    assert [r[1] for r in rows] == [0.50, 0.30, 0.20]
    assert all(r[2] is None and r[3] is None for r in rows)


def test_build_readout_uses_market_price_for_edge():
    d = parse_detail(NBA_SUMMARY, "basketball", "nba")

    class _Client:
        def get_price(self, ticker, side):  # noqa: ARG002
            return 0.55

    cfg = Config()
    cfg.market.kalshi_ticker = "NBA-LAL-WIN"
    rows = build_readout(cfg, d, _BballModel(), client=_Client())
    _, prob, price, edge = rows[0]
    assert price == 0.55
    assert abs(edge - (prob - 0.55)) < 1e-9


# --------------------------------------------------------------------------- #
# render + picker
# --------------------------------------------------------------------------- #


def test_render_builds_for_both_sports():
    console = Console(file=StringIO(), record=True, width=100)
    for summary, sport, league in [
        (SOCCER_SUMMARY, "soccer", "eng.1"),
        (NBA_SUMMARY, "basketball", "nba"),
    ]:
        d = parse_detail(summary, sport, league)
        model = _SoccerModel() if sport == "soccer" else _BballModel()
        rows = build_readout(Config(), d, model, None)
        console.print(render(d, rows, stale=False, updated_at="12:00:00"))
    out = console.export_text()
    assert "SportEdge Live" in out
    assert "Brighton" in out and "Lakers" in out


class _FakeConsole:
    def __init__(self, answer: str):
        self.answer = answer
        self.printed: list[str] = []

    def print(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.printed.append(" ".join(str(a) for a in args))

    def input(self, prompt: str = "") -> str:  # noqa: ARG002
        return self.answer


def _cands():
    return [
        GameCandidate("soccer", "eng.1", "1", "Brighton", "Man United", "post", "FT"),
        GameCandidate("basketball", "nba", "2", "Lakers", "Celtics", "in", "Q3 4:21"),
        GameCandidate("soccer", "fifa.world", "3", "Brazil", "Morocco", "pre", "6/14"),
    ]


def test_pick_game_hides_finished_and_picks_live_first():
    console = _FakeConsole("1")
    chosen = pick_game(console, _cands())
    # finished game excluded; live sorted first -> choice 1 is the in-progress NBA game
    assert chosen is not None and chosen.event_id == "2" and chosen.status == "in"


def test_pick_game_quit_returns_none():
    assert pick_game(_FakeConsole("q"), _cands()) is None


def test_pick_game_empty_returns_none():
    only_finished = [GameCandidate("soccer", "eng.1", "1", "A", "B", "post", "FT")]
    assert pick_game(_FakeConsole("1"), only_finished) is None
