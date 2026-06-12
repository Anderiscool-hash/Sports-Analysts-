"""End-to-end (offline) tests for the World Cup 3-token sniping pipeline.

No network: drives a real ``SoccerGameState`` through the real model, edge math,
``BottomDetector``, ``Strategy``, and ``PaperExecutor`` to prove the chain places a
paper fill when a token dips below model fair value and rebounds.
"""

from __future__ import annotations

from sportedge.betting.executor import PaperExecutor
from sportedge.betting.strategy import Strategy
from sportedge.live.soccer_loop import map_outcome_tokens
from sportedge.market.edge import BottomDetector
from sportedge.model.soccer_winprob import SoccerWinProbModel
from sportedge.types import SoccerGameState


def test_map_outcome_tokens_resolves_all_three():
    tokens = map_outcome_tokens(
        outcomes=["Brazil", "Draw", "Croatia"],
        token_ids=["tok_home", "tok_draw", "tok_away"],
        home_label="Brazil",
        draw_label="Draw",
        away_label="Croatia",
    )
    assert tokens == {"home": "tok_home", "draw": "tok_draw", "away": "tok_away"}


def test_map_outcome_tokens_partial_when_label_missing():
    tokens = map_outcome_tokens(
        outcomes=["Brazil", "Draw", "Croatia"],
        token_ids=["tok_home", "tok_draw", "tok_away"],
        home_label="Brazil",
        draw_label="",
        away_label="Croatia",
    )
    assert "draw" not in tokens
    assert tokens["home"] == "tok_home" and tokens["away"] == "tok_away"


def test_end_to_end_snipe_places_paper_fill():
    # Home up 1-0 with 5 minutes left -> model strongly favours home.
    state = SoccerGameState("Brazil", "Croatia", 1, 0, minute=85)
    model = SoccerWinProbModel.load("models/does-not-exist.joblib")  # -> Poisson fallback
    probs = model.predict(state)
    assert probs.home > 0.85

    strategy = Strategy(min_edge=0.04, kelly_fraction=0.25, max_stake=5.0, bankroll=100.0)
    detector = BottomDetector(dip_threshold=0.05, min_edge=0.04, rebound_ticks=1)
    executor = PaperExecutor()

    # Crowd panics on a Croatia near-miss and dumps the home token: peak -> dip -> rebound,
    # all while the model still rates home well above the price (real edge).
    price_stream = [0.80, 0.78, 0.70, 0.72]  # peak 0.80, trough 0.70, ticks back up
    for price in price_stream:
        sig = detector.update(price, probs.home)
        order = strategy.decide(sig)
        if order and executor.staked + order.size <= 100.0:
            executor.place(order, "tok_home")

    assert len(executor.fills) == 1
    fill = executor.fills[0]
    assert fill.side == "BUY"
    assert fill.price == 0.72  # bought on the rebound tick, not at the very bottom
    assert fill.size > 0.0
    assert fill.token_id == "tok_home"


def test_no_fill_when_no_edge():
    # Price sits at/above model fair value the whole time -> never snipe.
    state = SoccerGameState("Brazil", "Croatia", 0, 0, minute=10)
    probs = SoccerWinProbModel().predict(state)

    strategy = Strategy(min_edge=0.04, kelly_fraction=0.25, max_stake=5.0, bankroll=100.0)
    detector = BottomDetector(dip_threshold=0.05, min_edge=0.04, rebound_ticks=1)
    executor = PaperExecutor()

    rich_price = probs.home + 0.10  # market pricier than the model -> negative edge
    for price in [rich_price, rich_price - 0.01, rich_price + 0.01]:
        sig = detector.update(price, probs.home)
        order = strategy.decide(sig)
        if order:
            executor.place(order, "tok_home")

    assert executor.fills == []
