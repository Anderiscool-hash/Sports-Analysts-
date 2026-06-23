import pandas as pd

from sportedge.betting.executor import Fill
from sportedge.betting.history import ExecutionHistory
from sportedge.live.web_dashboard import trade_history_payload


def test_execution_history_persists_live_details(tmp_path):
    path = tmp_path / "history.parquet"
    history = ExecutionHistory(str(path))
    history.append(
        Fill(
            ts=20.0,
            side="BUY",
            size=4.2,
            price=0.42,
            model_p=0.60,
            edge=0.18,
            mode="live",
            token_id="KXNBA-SPURS",
            order_id="order-1",
            status="partial",
            requested_count=10,
            filled_count=4,
            avg_price=0.42,
        )
    )

    saved = ExecutionHistory(str(path)).load().iloc[0]

    assert saved["order_id"] == "order-1"
    assert saved["status"] == "partial"
    assert saved["requested_count"] == 10
    assert saved["filled_count"] == 4


def test_trade_history_merges_paper_and_live_newest_first():
    paper = pd.DataFrame(
        [
            {
                "ts": 10.0,
                "side": "BUY",
                "size": 5.0,
                "price": 0.5,
                "model_p": 0.7,
                "edge": 0.2,
                "token_id": "PAPER",
                "shares": 10.0,
                "is_settled": True,
                "pnl": 5.0,
            }
        ]
    )
    live = pd.DataFrame(
        [
            {
                "ts": 20.0,
                "side": "BUY",
                "size": 4.2,
                "price": 0.42,
                "avg_price": 0.42,
                "model_p": 0.6,
                "edge": 0.18,
                "token_id": "LIVE",
                "status": "filled",
                "order_id": "order-1",
                "requested_count": 10,
                "filled_count": 10,
            }
        ]
    )

    rows = trade_history_payload(paper, live)

    assert [row["mode"] for row in rows] == ["demo-live", "paper"]
    assert rows[0]["order_id"] == "order-1"
    assert rows[0]["status"] == "FILLED"
    assert rows[1]["status"] == "SETTLED"
    assert rows[1]["pnl"] == 5.0
