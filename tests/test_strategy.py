from sportedge.betting.strategy import Order, Strategy, kelly_stake
from sportedge.market.edge import BottomSignal


def test_kelly_positive_when_model_above_price():
    stake = kelly_stake(model_p=0.70, price=0.55, bankroll=100, kelly_fraction=0.25, max_stake=5)
    assert stake > 0
    assert stake <= 5  # capped by max_stake


def test_kelly_zero_when_no_edge():
    assert kelly_stake(0.50, 0.55, 100, 0.25, 5) == 0.0
    assert kelly_stake(0.70, 0.0, 100, 0.25, 5) == 0.0  # invalid price


def test_kelly_capped_by_max_stake():
    stake = kelly_stake(0.95, 0.50, 10_000, 1.0, 7.5)
    assert stake == 7.5


def test_strategy_decides_only_on_bottom():
    s = Strategy(min_edge=0.04, kelly_fraction=0.25, max_stake=5, bankroll=100)
    not_bottom = BottomSignal(False, price=0.55, model_p=0.70, edge=0.15)
    assert s.decide(not_bottom) is None

    bottom = BottomSignal(True, price=0.55, model_p=0.70, edge=0.15)
    order = s.decide(bottom)
    assert isinstance(order, Order)
    assert order.side == "BUY"
    assert 0 < order.size <= 5


def test_strategy_rejects_thin_edge():
    s = Strategy(min_edge=0.10, kelly_fraction=0.25, max_stake=5, bankroll=100)
    thin = BottomSignal(True, price=0.55, model_p=0.58, edge=0.03)
    assert s.decide(thin) is None
