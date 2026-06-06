from sportedge.market.edge import BottomDetector, edge, implied_prob_from_price


def test_edge_and_implied():
    assert edge(0.70, 0.55) == 0.70 - 0.55
    assert implied_prob_from_price(0.62) == 0.62


def test_bottom_fires_on_dip_then_rebound_with_edge():
    d = BottomDetector(dip_threshold=0.05, min_edge=0.04, rebound_ticks=1)
    assert not d.update(0.60, 0.70).is_bottom  # peak
    assert not d.update(0.50, 0.70).is_bottom  # new trough, no rebound yet
    sig = d.update(0.52, 0.70)  # ticked up off the low; dip 0.10, edge 0.18
    assert sig.is_bottom
    assert "dip" in sig.reason


def test_no_fire_without_rebound():
    d = BottomDetector(dip_threshold=0.05, min_edge=0.04, rebound_ticks=1)
    d.update(0.60, 0.70)
    assert not d.update(0.50, 0.70).is_bottom  # still falling


def test_no_fire_when_edge_too_small():
    d = BottomDetector(dip_threshold=0.05, min_edge=0.04, rebound_ticks=1)
    d.update(0.60, 0.52)
    d.update(0.50, 0.52)
    assert not d.update(0.52, 0.52).is_bottom  # edge 0.00 < 0.04


def test_does_not_refire_same_dip():
    d = BottomDetector(dip_threshold=0.05, min_edge=0.04, rebound_ticks=1)
    d.update(0.60, 0.70)
    d.update(0.50, 0.70)
    assert d.update(0.52, 0.70).is_bottom  # fires once
    assert not d.update(0.54, 0.70).is_bottom  # new peak, rebaselined


def test_requires_two_rebound_ticks_when_configured():
    d = BottomDetector(dip_threshold=0.05, min_edge=0.04, rebound_ticks=2)
    d.update(0.60, 0.70)
    d.update(0.50, 0.70)
    assert not d.update(0.52, 0.70).is_bottom  # only 1 rebound tick
    assert d.update(0.54, 0.70).is_bottom  # 2nd consecutive up-tick
