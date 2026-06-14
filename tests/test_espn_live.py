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
    LiveTrainingRecorder,
    PaperTradingEngine,
    TrendTracker,
    apply_market_coverage,
    auto_configure_kalshi_market,
    build_market_info,
    build_readout,
    clock_to_seconds,
    detail_to_basketball_state,
    detail_to_soccer_state,
    pick_game,
    pick_ready_game,
    wait_for_ready_game,
    render,
    settlement_marks,
    sparkline,
)
from sportedge.market.kalshi import KalshiDiscoveryResult, KalshiMarketSnapshot, candle_close_prob
from sportedge.market.scanner import GameMarketCoverage
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
    label, prob, price, edge, trend = rows[0]
    assert label == "Lakers win"
    assert prob == 0.62
    assert price is None and edge is None
    assert trend == ""  # no tracker -> no sparkline


def test_build_readout_basketball_home_and_away_markets():
    d = parse_detail(NBA_SUMMARY, "basketball", "nba")

    class _Client:
        def get_price(self, ticker, side):  # noqa: ARG002
            return {"NBA-LAL-WIN": 0.55, "NBA-BOS-WIN": 0.45}[ticker]

    cfg = Config()
    cfg.market.kalshi_ticker = "NBA-LAL-WIN"
    cfg.market.kalshi_away_ticker = "NBA-BOS-WIN"
    rows = build_readout(cfg, d, _BballModel(), _Client())

    assert [row[0] for row in rows] == ["Lakers win", "Celtics win"]
    assert [row[1] for row in rows] == [0.62, 0.38]
    assert [row[2] for row in rows] == [0.55, 0.45]


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
    _, prob, price, edge, _trend = rows[0]
    assert price == 0.55
    assert abs(edge - (prob - 0.55)) < 1e-9


def test_build_market_info_uses_configured_ticker():
    d = parse_detail(NBA_SUMMARY, "basketball", "nba")

    class _Client:
        def get_market_snapshot(self, ticker):
            return KalshiMarketSnapshot(ticker=ticker, status="active", yes_bid=0.40, yes_ask=0.44)

    cfg = Config()
    cfg.market.kalshi_ticker = "NBA-LAL-WIN"
    rows = build_market_info(cfg, d, _Client())

    assert len(rows) == 1
    label, ticker, snap = rows[0]
    assert label == "Lakers win"
    assert ticker == "NBA-LAL-WIN"
    assert snap is not None and snap.yes_ask == 0.44


def test_auto_configure_kalshi_market_sets_missing_basketball_ticker():
    cfg = Config()
    candidate = GameCandidate(
        "basketball",
        "nba",
        "401",
        "San Antonio Spurs",
        "New York Knicks",
        "in",
        "Q2",
    )

    class _Client:
        def discover_team_win_market(self, team, opponent):
            if (team, opponent) == ("New York Knicks", "San Antonio Spurs"):
                return None
            return KalshiDiscoveryResult(
                ticker="KXNBA-SPURS-WIN",
                title="Will the San Antonio Spurs win?",
                team=team,
                score=8.0,
                yes_bid=0.51,
                yes_ask=0.54,
            )

    auto_configure_kalshi_market(cfg, candidate, _Client(), _FakeConsole(""))

    assert cfg.market.home_team == "San Antonio Spurs"
    assert cfg.market.away_team == "New York Knicks"
    assert cfg.market.kalshi_ticker == "KXNBA-SPURS-WIN"


def test_auto_configure_kalshi_market_can_set_away_ticker_only():
    cfg = Config()
    candidate = GameCandidate(
        "basketball",
        "nba",
        "401",
        "San Antonio Spurs",
        "New York Knicks",
        "in",
        "Q2",
    )

    class _Client:
        def discover_team_win_market(self, team, opponent):  # noqa: ARG002
            if team == "San Antonio Spurs":
                return None
            return KalshiDiscoveryResult(
                ticker="KXNBA-KNICKS-WIN",
                title="Will the New York Knicks win?",
                team=team,
                score=8.0,
                yes_bid=0.46,
                yes_ask=0.49,
            )

    auto_configure_kalshi_market(cfg, candidate, _Client(), _FakeConsole(""))

    assert cfg.market.kalshi_ticker == ""
    assert cfg.market.kalshi_away_ticker == "KXNBA-KNICKS-WIN"


def test_apply_market_coverage_sets_home_and_away_tickers():
    cfg = Config()
    game = GameCandidate("basketball", "nba", "401", "Lakers", "Celtics", "in", "Q2")
    coverage = GameMarketCoverage(
        game=game,
        home_market=KalshiDiscoveryResult(
            "KXNBA-LAL-WIN", "Will Lakers win?", "Lakers", 8.0, yes_ask=0.54
        ),
        away_market=KalshiDiscoveryResult(
            "KXNBA-BOS-WIN", "Will Celtics win?", "Celtics", 8.0, yes_ask=0.49
        ),
    )

    apply_market_coverage(cfg, coverage)

    assert cfg.market.home_team == "Lakers"
    assert cfg.market.away_team == "Celtics"
    assert cfg.market.kalshi_ticker == "KXNBA-LAL-WIN"
    assert cfg.market.kalshi_away_ticker == "KXNBA-BOS-WIN"


def test_pick_ready_game_chooses_first_market_covered_game():
    cfg = Config()
    games = [
        GameCandidate("basketball", "nba", "1", "Spurs", "Knicks", "in", "Q2"),
        GameCandidate("basketball", "nba", "2", "Lakers", "Celtics", "in", "Q1"),
    ]

    class _Client:
        def discover_team_win_market(self, team, opponent):  # noqa: ARG002
            if team == "Lakers":
                return KalshiDiscoveryResult(
                    "KXNBA-LAL-WIN",
                    "Will Lakers win?",
                    "Lakers",
                    8.0,
                    yes_ask=0.54,
                )
            return None

    chosen = pick_ready_game(_FakeConsole(""), games, cfg, _Client())

    assert chosen is not None and chosen.event_id == "2"
    assert cfg.market.kalshi_ticker == "KXNBA-LAL-WIN"


def test_wait_for_ready_game_returns_when_market_appears(monkeypatch):
    cfg = Config()
    games = [GameCandidate("basketball", "nba", "2", "Lakers", "Celtics", "in", "Q1")]
    monkeypatch.setattr("sportedge.live.dashboard.list_live_games", lambda: games)

    class _Client:
        def discover_team_win_market(self, team, opponent):  # noqa: ARG002
            if team == "Lakers":
                return KalshiDiscoveryResult(
                    "KXNBA-LAL-WIN",
                    "Will Lakers win?",
                    "Lakers",
                    8.0,
                    yes_ask=0.54,
                )
            return None

    chosen = wait_for_ready_game(_FakeConsole(""), cfg, _Client(), poll_seconds=1, max_wait_seconds=1)

    assert chosen is not None and chosen.event_id == "2"
    assert cfg.market.kalshi_ticker == "KXNBA-LAL-WIN"


def test_wait_for_ready_game_times_out_without_sleeping(monkeypatch):
    cfg = Config()
    monkeypatch.setattr("sportedge.live.dashboard.list_live_games", lambda: [])
    now = {"t": 0.0}
    monkeypatch.setattr("sportedge.live.dashboard.time.monotonic", lambda: now["t"])

    def fake_sleep(seconds):
        now["t"] += seconds

    monkeypatch.setattr("sportedge.live.dashboard.time.sleep", fake_sleep)

    chosen = wait_for_ready_game(
        _FakeConsole(""),
        cfg,
        client=object(),
        poll_seconds=5,
        max_wait_seconds=10,
    )

    assert chosen is None


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
    assert "Game context" in out
    assert "Kalshi market detail" in out
    assert "Brighton" in out and "Lakers" in out


def test_live_training_recorder_writes_labeled_final_rows(tmp_path):
    d = parse_detail(NBA_SUMMARY, "basketball", "nba")
    path = tmp_path / "training.parquet"
    recorder = LiveTrainingRecorder(str(path))

    recorder.capture("401", d)
    recorder.capture("401", d)  # duplicate tick is ignored
    assert recorder.buffered_count == 1

    d.status = "post"
    d.home_score = 101
    d.away_score = 99
    saved = recorder.save_if_final(d)

    assert saved == 1
    df = __import__("pandas").read_parquet(path)
    assert len(df) == 1
    assert df.iloc[0]["game_id"] == "401"
    assert df.iloc[0]["home_win"] == 1


def test_settlement_marks_basketball_final():
    d = parse_detail(NBA_SUMMARY, "basketball", "nba")
    d.status = "post"
    d.home_score = 101
    d.away_score = 99
    cfg = Config()
    cfg.market.kalshi_ticker = "NBA-LAL-WIN"
    cfg.market.kalshi_away_ticker = "NBA-BOS-WIN"

    assert settlement_marks(cfg, d) == {"NBA-LAL-WIN": 1.0, "NBA-BOS-WIN": 0.0}


def test_paper_trading_engine_places_and_summarizes(tmp_path):
    d = parse_detail(NBA_SUMMARY, "basketball", "nba")
    path = tmp_path / "paper.parquet"
    cfg = Config()
    cfg.bankroll = 100
    cfg.max_stake = 5
    cfg.edge.min_edge = 0.04
    cfg.edge.dip_threshold = 0.05
    cfg.edge.rebound_ticks = 1
    cfg.market.kalshi_ticker = "NBA-LAL-WIN"
    engine = PaperTradingEngine(cfg, str(path))

    # Seed a peak, then dip, then rebound with enough edge to paper-buy.
    assert engine.update(d, [("Lakers win", 0.70, 0.60, 0.10, "")])[0][3] == "WAIT"
    assert engine.update(d, [("Lakers win", 0.70, 0.50, 0.20, "")])[0][3] == "WAIT"
    signal = engine.update(d, [("Lakers win", 0.70, 0.52, 0.18, "")])[0]

    assert signal[3] == "PAPER BUY"
    summary = engine.summary(marks={"NBA-LAL-WIN": 0.60})
    assert summary["fills"] == 1
    assert summary["open_exposure"] > 0
    assert summary["unrealized_pnl"] > 0


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


# --------------------------------------------------------------------------- #
# chart: sparkline, candlestick parsing, trend tracker
# --------------------------------------------------------------------------- #


def test_sparkline_levels_and_edges():
    assert sparkline([]) == ""
    assert sparkline([0.5, 0.5, 0.5]) == "▁▁▁"  # flat -> lowest block
    s = sparkline([0.1, 0.5, 0.9])
    assert len(s) == 3 and s[0] == "▁" and s[-1] == "█"  # rising series


def test_candle_close_prob_shapes():
    assert candle_close_prob({"price": {"close": 55}}) == 0.55
    assert candle_close_prob({"yes_ask": {"close": 40}}) == 0.40
    assert candle_close_prob({"yes_bid": 30}) == 0.30
    assert candle_close_prob({"volume": 0}) is None


def test_trend_tracker_seed_then_append():
    t = TrendTracker(maxlen=4)
    t.seed("Brazil", [0.40, 0.42, 0.41])
    t.append("Brazil", 0.45)
    # maxlen=4 keeps the most recent four samples in order
    assert t.series("Brazil") == [0.42, 0.41, 0.45][-3:] or t.series("Brazil") == [0.40, 0.42, 0.41, 0.45]
    t.append("Brazil", None)  # None is ignored
    assert t.series("Brazil")[-1] == 0.45
    assert t.series("unknown") == []


def test_build_readout_populates_trend_with_tracker():
    d = parse_detail(NBA_SUMMARY, "basketball", "nba")

    class _Client:
        def get_price(self, ticker, side):  # noqa: ARG002
            return 0.50

    cfg = Config()
    cfg.market.kalshi_ticker = "NBA-LAL-WIN"
    tracker = TrendTracker()
    tracker.seed("Lakers win", [0.45, 0.48])
    rows = build_readout(cfg, d, _BballModel(), _Client(), tracker)
    trend = rows[0][4]
    assert trend != ""  # seeded history + sampled price -> a sparkline
    assert len(trend) == 3  # two seeded + one appended sample
