"""Trade-feed whale / contrarian-momentum confirmation (pure)."""

from sportedge.betting.flow import FlowSignal, detect_flow, trade_yes_price
from sportedge.config import FlowConfig

NOW = 1000.0
CFG = FlowConfig(
    mode="confirm",
    lookback_sec=120,
    whale_min_notional=250.0,
    price_move_threshold=0.05,
    cluster_min_trades=3,
    cluster_min_notional=100.0,
)


def _t(price, count, ts):
    return {"yes_price_dollars": f"{price:.4f}", "count": count, "created_ts": ts}


def test_no_trades_does_not_confirm():
    s = detect_flow([], CFG, now=NOW)
    assert isinstance(s, FlowSignal) and not s.confirms_buy


def test_whale_confirms_buy():
    s = detect_flow([_t(0.50, 1000, 990)], CFG, now=NOW)  # $500 notional
    assert s.whale and s.confirms_buy


def test_contrarian_selloff_confirms_buy():
    trades = [_t(0.60, 10, 950), _t(0.55, 10, 970), _t(0.50, 10, 990)]  # -0.10 move
    s = detect_flow(trades, CFG, now=NOW)
    assert s.momentum_down and s.confirms_buy
    assert not s.whale


def test_flat_small_flow_does_not_confirm():
    trades = [_t(0.50, 5, 950), _t(0.50, 5, 990)]
    s = detect_flow(trades, CFG, now=NOW)
    assert not s.confirms_buy
    assert "no confirm" in s.reason


def test_old_trades_outside_window_are_ignored():
    trades = [_t(0.60, 1000, 100), _t(0.59, 1000, 200)]  # both before now-120
    s = detect_flow(trades, CFG, now=NOW)
    assert s.trade_count == 0 and not s.confirms_buy


def test_cluster_flagged_without_being_required_for_confirm():
    # Three small flat trades: a cluster, but no whale and no sell-off.
    trades = [_t(0.50, 40, 950), _t(0.50, 40, 970), _t(0.50, 40, 990)]  # $60 total
    s = detect_flow(trades, CFG, now=NOW)
    assert s.trade_count == 3
    assert not s.confirms_buy  # cluster alone doesn't confirm a buy


def test_trade_yes_price_parses_all_shapes():
    assert trade_yes_price({"yes_price_dollars": "0.42"}) == 0.42
    assert trade_yes_price({"yes_price": 42}) == 0.42            # cents
    assert abs(trade_yes_price({"no_price_dollars": "0.42"}) - 0.58) < 1e-9
    assert trade_yes_price({}) is None
