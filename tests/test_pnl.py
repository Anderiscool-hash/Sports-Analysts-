import pandas as pd

from sportedge.market.pnl import align_model_prices, simulate_buy_and_hold, trades_frame


def test_align_model_prices_asof_with_tolerance():
    states = pd.DataFrame(
        {
            "timestamp": [100, 200, 400],
            "model_p": [0.6, 0.7, 0.8],
        }
    )
    prices = pd.DataFrame(
        {
            "timestamp": [90, 150],
            "price": [0.5, 0.55],
        }
    )

    aligned = align_model_prices(states, prices, tolerance_seconds=75)

    assert aligned["timestamp"].tolist() == [100, 200]
    assert aligned["price"].tolist() == [0.5, 0.55]
    assert aligned["edge"].round(2).tolist() == [0.10, 0.15]


def test_simulate_buy_and_hold_winning_token():
    aligned = pd.DataFrame(
        {
            "timestamp": [100, 200],
            "price": [0.5, 0.6],
            "model_p": [0.7, 0.8],
            "edge": [0.2, 0.2],
        }
    )

    trades = simulate_buy_and_hold(
        aligned,
        token_won=True,
        min_edge=0.04,
        bankroll=100,
        kelly_fraction=0.25,
        max_stake=5,
        cooldown_seconds=0,
    )
    df = trades_frame(trades)

    assert len(df) == 2
    assert df["stake"].sum() > 0
    assert df["pnl"].sum() > 0


def test_simulate_buy_and_hold_respects_cooldown():
    aligned = pd.DataFrame(
        {
            "timestamp": [100, 150, 300],
            "price": [0.5, 0.5, 0.5],
            "model_p": [0.7, 0.7, 0.7],
            "edge": [0.2, 0.2, 0.2],
        }
    )

    trades = simulate_buy_and_hold(aligned, True, cooldown_seconds=100)

    assert [trade.timestamp for trade in trades] == [100, 300]
