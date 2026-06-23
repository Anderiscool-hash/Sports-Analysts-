"""Order-book-aware limit pricing (pure)."""

from sportedge.betting.execution_policy import (
    best_yes_bid_ask_cents,
    compute_limit_price_cents,
)

# A YES bid at 40c, and a NO bid at 58c => YES ask = 100 - 58 = 42c.
BOOK = {"yes": [[38, 100], [40, 50]], "no": [[55, 20], [58, 10]]}


def test_best_yes_bid_ask_from_book():
    bid, ask = best_yes_bid_ask_cents(BOOK)
    assert bid == 40
    assert ask == 42


def test_best_yes_bid_ask_unwraps_orderbook_envelope():
    bid, ask = best_yes_bid_ask_cents({"orderbook": BOOK})
    assert (bid, ask) == (40, 42)


def test_best_yes_bid_ask_handles_empty_sides():
    assert best_yes_bid_ask_cents({"yes": [], "no": []}) == (None, None)
    assert best_yes_bid_ask_cents(None) == (None, None)


def test_limit_cross_takes_the_ask():
    assert compute_limit_price_cents(BOOK, 0.40, "limit_cross") == 42


def test_market_style_also_takes_the_ask():
    assert compute_limit_price_cents(BOOK, 0.40, "market") == 42


def test_limit_mid_is_midpoint_of_bid_and_ask():
    # midpoint of 40 and 42 = 41
    assert compute_limit_price_cents(BOOK, 0.40, "limit_mid") == 41


def test_falls_back_to_fair_plus_offset_when_book_missing():
    # No book -> fair 0.50 => 50c + 2c offset = 52c
    assert compute_limit_price_cents({}, 0.50, "limit_cross", fallback_offset_cents=2) == 52
    assert compute_limit_price_cents(None, 0.50, "limit_mid", fallback_offset_cents=3) == 53


def test_price_is_always_clamped_to_1_99():
    assert compute_limit_price_cents({}, 0.999, "limit_cross", fallback_offset_cents=5) == 99
    assert compute_limit_price_cents({}, 0.0, "limit_cross", fallback_offset_cents=0) == 1


# --- current Kalshi shape: orderbook_fp with dollar-string prices ---

# NO bid at $0.3580 (35.8c -> 36c) => YES ask = 100 - 36 = 64c. No YES bids.
FP_BOOK = {"orderbook_fp": {"no_dollars": [["0.3580", "158.00"]], "yes_dollars": []}}


def test_fp_dollars_shape_yes_ask_from_no_side():
    bid, ask = best_yes_bid_ask_cents(FP_BOOK)
    assert bid is None       # yes_dollars empty
    assert ask == 64         # 100 - round(0.3580 * 100)


def test_fp_dollars_both_sides():
    book = {
        "orderbook_fp": {
            "yes_dollars": [["0.40", "10"], ["0.41", "5"]],
            "no_dollars": [["0.55", "20"], ["0.58", "8"]],
        }
    }
    bid, ask = best_yes_bid_ask_cents(book)
    assert bid == 41                 # highest yes bid
    assert ask == 100 - 58           # cross against best no bid
    assert compute_limit_price_cents(book, 0.40, "limit_cross") == 42
    # midpoint of 41 and 42 rounds to 42 (round-half-to-even on 41.5)
    assert compute_limit_price_cents(book, 0.40, "limit_mid") in (41, 42)


def test_fp_empty_book_falls_back():
    empty = {"orderbook_fp": {"yes_dollars": [], "no_dollars": []}}
    assert compute_limit_price_cents(empty, 0.50, "limit_cross", fallback_offset_cents=2) == 52
