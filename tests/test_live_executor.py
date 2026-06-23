"""Live executor place flow: orderbook pricing, fill polling, expiry auto-cancel.

All network is faked; no Kalshi keys or live calls are used.
"""

from sportedge.betting.executor import KalshiLiveExecutor
from sportedge.betting.strategy import Order
from sportedge.config import ExecutionConfig, Secrets

# YES bid 40c / NO bid 58c -> YES ask 42c.
BOOK = {"yes": [[40, 50]], "no": [[58, 10]]}


def _secrets() -> Secrets:
    # Dummy but "complete" so the executor constructs; the real client is replaced.
    return Secrets(kalshi_api_key_id="k", kalshi_private_key_pem="pem")


def _stepping_clock(step: float = 3.0):
    state = {"t": 0.0}

    def clock() -> float:
        value = state["t"]
        state["t"] += step
        return value

    return clock


class _FakeClient:
    def __init__(self, order_responses, get_order_seq):
        self.order_responses = order_responses
        self.get_order_seq = list(get_order_seq)
        self.placed = []
        self.canceled = []

    def get_orderbook(self, ticker, depth=10):
        return BOOK

    def place_limit_order(self, ticker, count, price_cents, client_order_id=""):
        self.placed.append((ticker, count, price_cents, client_order_id))
        return self.order_responses

    def get_order(self, order_id):
        if len(self.get_order_seq) > 1:
            return self.get_order_seq.pop(0)
        return self.get_order_seq[0] if self.get_order_seq else {}

    def cancel_order(self, order_id):
        self.canceled.append(order_id)
        # After cancel, the order reports canceled with whatever had filled.
        self.get_order_seq = [
            {**(self.get_order_seq[-1] if self.get_order_seq else {}), "status": "canceled"}
        ]
        return {}


def _executor(client, **exec_overrides) -> KalshiLiveExecutor:
    ex = KalshiLiveExecutor(
        _secrets(),
        ExecutionConfig(order_style="limit_cross", order_expiration_sec=8, fill_poll_seconds=1.0,
                        **exec_overrides),
        clock=_stepping_clock(),
        sleep=lambda _s: None,
    )
    ex._client = client
    return ex


def test_full_immediate_fill_records_actual_avg_price():
    client = _FakeClient(
        order_responses={"order": {"order_id": "o1", "status": "executed",
                                   "place_count": 11, "fill_count": 11,
                                   "average_fill_price": 42}},
        get_order_seq=[],
    )
    ex = _executor(client)
    fill = ex.place(Order("BUY", size=5.0, price=0.40, model_p=0.6, edge=0.2), token_id="T")

    # Priced at the book ask (42c), not the blind 0.40 signal price.
    assert client.placed[0][2] == 42
    assert fill.status == "filled"
    assert fill.filled_count == 11
    assert fill.price == 0.42
    assert fill.size == round(0.42 * 11, 4)
    assert client.canceled == []  # nothing to cancel


def test_partial_fill_cancels_remainder_at_expiry():
    partial = {"order_id": "o2", "status": "resting", "place_count": 11,
               "remaining_count": 7, "fill_count": 4, "average_fill_price": 41}
    client = _FakeClient(
        order_responses={"order": {"order_id": "o2", "status": "resting",
                                   "place_count": 11, "remaining_count": 11}},
        get_order_seq=[partial],
    )
    ex = _executor(client)
    fill = ex.place(Order("BUY", size=5.0, price=0.40, model_p=0.6, edge=0.2), token_id="T")

    assert client.canceled == ["o2"]      # unfilled remainder was canceled
    assert fill.status == "partial"
    assert fill.filled_count == 4
    assert fill.size == round(0.41 * 4, 4)


def test_unfilled_order_cancels_and_records_zero():
    client = _FakeClient(
        order_responses={"order": {"order_id": "o3", "status": "resting",
                                   "place_count": 11, "remaining_count": 11}},
        get_order_seq=[{"order_id": "o3", "status": "resting", "place_count": 11,
                        "remaining_count": 11, "fill_count": 0}],
    )
    ex = _executor(client)
    fill = ex.place(Order("BUY", size=5.0, price=0.40, model_p=0.6, edge=0.2), token_id="T")

    assert client.canceled == ["o3"]
    assert fill.status == "unfilled"
    assert fill.filled_count == 0
    assert fill.size == 0.0
    assert ex.staked == 0.0


def test_client_order_id_is_passed_for_idempotency():
    client = _FakeClient(
        order_responses={"order": {"order_id": "o4", "status": "executed",
                                   "place_count": 11, "fill_count": 11,
                                   "average_fill_price": 42}},
        get_order_seq=[],
    )
    ex = _executor(client)
    ex.place(Order("BUY", size=5.0, price=0.40, model_p=0.6, edge=0.2), token_id="T")
    client_order_id = client.placed[0][3]
    assert client_order_id.startswith("se-T-")
