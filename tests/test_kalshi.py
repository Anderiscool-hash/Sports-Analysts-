"""Tests for the Kalshi market venue: price conversion, order payloads, RSA-PSS
signing, and the venue-aware executor selection. All network is mocked; no keys
or live calls are used.
"""

from __future__ import annotations

import base64

import pytest

from sportedge.betting.executor import Fill, KalshiLiveExecutor, PaperExecutor, make_executor, paper_gate_status
from sportedge.betting.paper import PaperLedger
from sportedge.config import Config, Secrets
from sportedge.market.kalshi import (
    KalshiClient,
    cents_to_prob,
    rejection_reason_for_team_market,
    score_market_for_team,
)


def test_cents_to_prob_converts_and_clamps():
    assert cents_to_prob(50) == 0.5
    assert cents_to_prob(1) == 0.01
    assert 0.0 < cents_to_prob(0) < 0.01  # clamped, never 0
    assert cents_to_prob(100) <= 0.999  # clamped, never 1


def test_get_price_buy_uses_yes_ask(monkeypatch):
    client = KalshiClient()
    monkeypatch.setattr(
        client, "get_market", lambda t: {"yes_bid": 40, "yes_ask": 45, "last_price": 42}
    )
    assert client.get_price("WC-BRA", "BUY") == 0.45
    assert client.get_price("WC-BRA", "SELL") == 0.40


def test_get_price_none_when_no_quotes(monkeypatch):
    client = KalshiClient()
    monkeypatch.setattr(client, "get_market", lambda t: {})
    assert client.get_price("WC-BRA", "BUY") is None


def test_get_market_snapshot_maps_display_fields(monkeypatch):
    client = KalshiClient()
    monkeypatch.setattr(
        client,
        "get_market",
        lambda t: {
            "ticker": t,
            "title": "Will the Lakers win?",
            "status": "active",
            "yes_bid": 41,
            "yes_ask": 44,
            "last_price": 43,
            "volume": "1200",
            "volume_24h": 55,
            "liquidity": 9000,
            "open_interest": 88,
            "close_time": "2026-06-14T02:00:00Z",
        },
    )

    snap = client.get_market_snapshot("NBA-LAL-WIN")

    assert snap is not None
    assert snap.title == "Will the Lakers win?"
    assert snap.status == "active"
    assert snap.yes_bid == 0.41
    assert snap.yes_ask == 0.44
    assert snap.last_price == 0.43
    assert snap.volume == 1200
    assert snap.open_interest == 88


def test_score_market_for_team_accepts_direct_quoted_winner():
    market = {
        "ticker": "KXNBA-SPURS-WIN",
        "title": "Will the San Antonio Spurs win against the New York Knicks?",
        "status": "active",
        "yes_bid": 51,
        "yes_ask": 54,
        "volume": 1200,
        "liquidity": 8000,
    }

    result = score_market_for_team(market, "San Antonio Spurs", "New York Knicks")

    assert result is not None
    assert result.ticker == "KXNBA-SPURS-WIN"
    assert result.yes_ask == 0.54


def test_score_market_for_team_rejects_combo_props_and_unquoted():
    combo = {
        "ticker": "KXCOMBO",
        "title": "yes San Antonio Spurs,yes Jalen Brunson: 25+,yes Over 184.5 points scored",
        "status": "active",
        "yes_bid": 51,
        "yes_ask": 54,
    }
    unquoted = {
        "ticker": "KXNBA-SPURS-WIN",
        "title": "Will the San Antonio Spurs win against the New York Knicks?",
        "status": "active",
    }

    assert score_market_for_team(combo, "San Antonio Spurs", "New York Knicks") is None
    assert score_market_for_team(unquoted, "San Antonio Spurs", "New York Knicks") is None
    assert rejection_reason_for_team_market(combo, "San Antonio Spurs", "New York Knicks") == "not a direct team-winner market"
    assert rejection_reason_for_team_market(unquoted, "San Antonio Spurs", "New York Knicks") == "no usable quote"


def test_explain_team_win_market_search_returns_rejection_reasons(monkeypatch):
    client = KalshiClient()
    markets = [
        {
            "ticker": "KXCOMBO",
            "title": "yes San Antonio Spurs,yes Jalen Brunson: 25+",
            "status": "active",
            "yes_ask": 40,
        },
        {
            "ticker": "KXNBA-SPURS-WIN",
            "title": "Will the San Antonio Spurs win against the New York Knicks?",
            "status": "active",
            "yes_bid": 51,
            "yes_ask": 54,
        },
    ]
    monkeypatch.setattr(client, "list_open_markets", lambda **kwargs: markets)

    rejections = client.explain_team_win_market_search("San Antonio Spurs", "New York Knicks")

    assert len(rejections) == 1
    assert rejections[0].ticker == "KXCOMBO"
    assert rejections[0].reason == "not a direct team-winner market"


def test_score_market_for_team_accepts_city_name_for_multiword_team():
    market = {
        "ticker": "KXNBA-SPURS-WIN",
        "title": "Will San Antonio win against the New York Knicks?",
        "status": "active",
        "yes_bid": 51,
        "yes_ask": 54,
    }

    result = score_market_for_team(market, "San Antonio Spurs", "New York Knicks")

    assert result is not None
    assert result.ticker == "KXNBA-SPURS-WIN"


def test_discover_team_win_market_returns_best_candidate(monkeypatch):
    client = KalshiClient()
    markets = [
        {
            "ticker": "KXCOMBO",
            "title": "yes San Antonio,yes Jalen Brunson: 25+",
            "status": "active",
            "yes_ask": 40,
        },
        {
            "ticker": "KXNBA-SPURS-WIN",
            "title": "Will the San Antonio Spurs win against the New York Knicks?",
            "status": "active",
            "yes_bid": 51,
            "yes_ask": 54,
            "liquidity": 8000,
        },
    ]
    monkeypatch.setattr(client, "list_open_markets", lambda **kwargs: markets)

    result = client.discover_team_win_market("San Antonio Spurs", "New York Knicks")

    assert result is not None
    assert result.ticker == "KXNBA-SPURS-WIN"


def test_build_order_payload_shape():
    payload = KalshiClient.build_order_payload("WC-BRA-WIN", stake_usd=5.0, price_prob=0.50)
    assert payload["ticker"] == "WC-BRA-WIN"
    assert payload["action"] == "buy"
    assert payload["side"] == "yes"
    assert payload["type"] == "limit"
    assert payload["yes_price"] == 50  # cents
    assert payload["count"] == 10  # $5 / $0.50 = 10 shares


def _gen_secrets() -> Secrets:
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    return Secrets(kalshi_api_key_id="test-key-id", kalshi_private_key_pem=pem)


def test_auth_headers_produce_a_verifiable_signature():
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    secrets = _gen_secrets()
    client = KalshiClient(secrets=secrets)
    headers = client.auth_headers("POST", "/portfolio/orders", timestamp_ms=1700000000000)

    assert headers["KALSHI-ACCESS-KEY"] == "test-key-id"
    assert headers["KALSHI-ACCESS-TIMESTAMP"] == "1700000000000"

    message = "1700000000000POST/portfolio/orders"
    public_key = client._load_private_key().public_key()
    # Raises on a bad signature; no exception == verified.
    public_key.verify(
        base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"]),
        message.encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


def test_auth_headers_require_keys():
    client = KalshiClient(secrets=Secrets())
    with pytest.raises(ValueError):
        client.auth_headers("POST", "/portfolio/orders")


def test_place_order_signs_full_path_and_posts(monkeypatch):
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    secrets = _gen_secrets()
    client = KalshiClient(secrets=secrets)

    captured = {}

    class _Resp:
        content = b"{}"

        def raise_for_status(self):
            pass

        def json(self):
            return {"order": {"status": "resting"}}

    def fake_request(method, url, params=None, json=None, headers=None, timeout=None):
        captured.update(method=method, url=url, json=json, headers=headers)
        return _Resp()

    monkeypatch.setattr("sportedge.market.kalshi.requests.request", fake_request)
    result = client.place_order("WC-BRA-WIN", stake_usd=5.0, price_prob=0.5)

    assert result == {"order": {"status": "resting"}}
    assert captured["method"] == "POST"
    # v2 create-order endpoint, signed against the FULL path incl. /trade-api/v2.
    assert captured["url"].endswith("/trade-api/v2/portfolio/events/orders")
    # v2 fixed-point payload: buying YES is side "bid"; price/count are dollar strings.
    assert captured["json"]["ticker"] == "WC-BRA-WIN"
    assert captured["json"]["side"] == "bid"
    assert captured["json"]["price"] == "0.5000"
    assert captured["json"]["count"] == "10.00"  # $5 / $0.50
    assert captured["json"]["time_in_force"]
    assert captured["json"]["self_trade_prevention_type"]

    # The signature must verify against the v2 full-path message.
    ts = captured["headers"]["KALSHI-ACCESS-TIMESTAMP"]
    message = f"{ts}POST/trade-api/v2/portfolio/events/orders"
    client._load_private_key().public_key().verify(
        base64.b64decode(captured["headers"]["KALSHI-ACCESS-SIGNATURE"]),
        message.encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


def test_cancel_order_uses_v2_event_path_and_signs_full_path(monkeypatch):
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    client = KalshiClient(secrets=_gen_secrets())
    order_id = "daf09c33-1618-4100-9465-050e33b5420a"
    response = {
        "order_id": order_id,
        "client_order_id": "client-id",
        "reduced_by": "1.00",
        "ts_ms": 1700000000000,
    }
    captured = {}

    class _Resp:
        content = b"{}"

        def raise_for_status(self):
            pass

        def json(self):
            return response

    def fake_request(method, url, params=None, json=None, headers=None, timeout=None):
        captured.update(method=method, url=url, json=json, headers=headers)
        return _Resp()

    monkeypatch.setattr("sportedge.market.kalshi.requests.request", fake_request)

    assert client.cancel_order(order_id) == response
    assert captured["method"] == "DELETE"
    assert captured["url"].endswith(
        f"/trade-api/v2/portfolio/events/orders/{order_id}"
    )
    assert captured["json"] is None

    ts = captured["headers"]["KALSHI-ACCESS-TIMESTAMP"]
    message = f"{ts}DELETE/trade-api/v2/portfolio/events/orders/{order_id}"
    client._load_private_key().public_key().verify(
        base64.b64decode(captured["headers"]["KALSHI-ACCESS-SIGNATURE"]),
        message.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )


def test_get_balance_uses_signed_portfolio_endpoint(monkeypatch):
    client = KalshiClient(secrets=_gen_secrets())
    captured = {}

    def fake_auth(method, path, **kwargs):
        captured.update(method=method, path=path, kwargs=kwargs)
        return {"balance": 10000, "balance_dollars": "100.00", "portfolio_value": 0}

    monkeypatch.setattr(client, "_auth_request", fake_auth)

    result = client.get_balance()

    assert result["balance_dollars"] == "100.00"
    assert captured["method"] == "GET"
    assert captured["path"] == "/portfolio/balance"


def test_get_trades_can_read_global_tape_without_ticker(monkeypatch):
    client = KalshiClient()
    captured = {}

    def fake_get(path, params):
        captured.update(path=path, params=params)
        return {"trades": [{"ticker": "KXTEST"}]}

    monkeypatch.setattr(client, "_get", fake_get)

    assert client.get_trades(limit=1000) == [{"ticker": "KXTEST"}]
    assert captured == {"path": "/markets/trades", "params": {"limit": 1000}}


# ----- venue-aware executor selection -----


def _proof_ledger(path, fills: int = 25, pnl_per_fill: float = 1.0):
    ledger = PaperLedger(str(path))
    settlement = 1.0 if pnl_per_fill >= 0 else 0.0
    price = 0.50
    for i in range(fills):
        ledger.append(
            Fill(
                ts=float(i),
                side="BUY",
                size=1.0,
                price=price,
                model_p=0.70,
                edge=0.20,
                mode="paper",
                token_id=f"tok-{i}",
            )
        )
    if pnl_per_fill >= 0:
        return ledger
    return ledger


def test_make_executor_kalshi_live_when_keys_and_paper_proof_present(tmp_path):
    cfg = Config(mode="live", confirm_live=True)
    cfg.paper_gate.min_fills = 1
    cfg.paper_gate.min_settled_fills = 0
    ledger = tmp_path / "paper.parquet"
    PaperLedger(str(ledger)).append(
        Fill(1.0, "BUY", 5.0, 0.50, 0.70, 0.20, "paper", "tok")
    )
    secrets = _gen_secrets()
    assert isinstance(make_executor(cfg, secrets, paper_ledger_path=str(ledger)), KalshiLiveExecutor)


def test_make_executor_blocks_live_with_only_open_paper_fills(tmp_path):
    cfg = Config(mode="live", confirm_live=True)
    cfg.paper_gate.min_fills = 1
    ledger = tmp_path / "paper.parquet"
    PaperLedger(str(ledger)).append(
        Fill(1.0, "BUY", 5.0, 0.50, 0.70, 0.20, "paper", "tok")
    )

    ex = make_executor(cfg, _gen_secrets(), paper_ledger_path=str(ledger))

    assert type(ex) is PaperExecutor
    ok, reason = paper_gate_status(cfg, str(ledger))
    assert ok is False
    assert "settled fills" in reason


def test_paper_gate_blocks_when_realized_pnl_is_negative(tmp_path, monkeypatch):
    cfg = Config(mode="live", confirm_live=True)
    cfg.paper_gate.min_fills = 2
    cfg.paper_gate.min_settled_fills = 1
    cfg.paper_gate.min_realized_pnl = 0.0
    cfg.paper_gate.min_total_pnl = -999.0
    ledger = tmp_path / "paper.parquet"
    store = PaperLedger(str(ledger))
    store.append(Fill(1.0, "BUY", 5.0, 0.50, 0.70, 0.20, "paper", "settled-loser"))
    store.append(Fill(2.0, "BUY", 5.0, 0.50, 0.70, 0.20, "paper", "open-winner"))
    monkeypatch.setattr(
        "sportedge.betting.report.collect_all_settlements",
        lambda fills: {"settled-loser": 0.0},
    )

    ok, reason = paper_gate_status(cfg, str(ledger))

    assert ok is False
    assert "realized PnL" in reason


def test_paper_gate_counts_espn_final_settlements(tmp_path, monkeypatch):
    cfg = Config(mode="live", confirm_live=True)
    cfg.paper_gate.min_fills = 1
    cfg.paper_gate.min_settled_fills = 1
    cfg.paper_gate.min_realized_pnl = 0.0
    ledger = tmp_path / "paper.parquet"
    PaperLedger(str(ledger)).append(
        Fill(
            ts=1.0,
            side="BUY",
            size=5.0,
            price=0.50,
            model_p=0.70,
            edge=0.20,
            mode="paper",
            token_id="tok",
            event_id="401",
            sport="basketball",
            league="nba",
            home_team="Lakers",
            away_team="Celtics",
            selected_team="Lakers",
        )
    )

    from sportedge.types import LiveDetail

    detail = LiveDetail(
        sport="basketball",
        league="nba",
        home_team="Lakers",
        away_team="Celtics",
        home_score=101,
        away_score=99,
        status="post",
    )
    monkeypatch.setattr("sportedge.betting.report.get_game_detail", lambda *args: detail)

    ok, reason = paper_gate_status(cfg, str(ledger))

    assert ok is True
    assert "1 settled" in reason


def test_paper_gate_blocks_when_realized_roi_is_too_low(tmp_path, monkeypatch):
    cfg = Config(mode="live", confirm_live=True)
    cfg.paper_gate.min_fills = 1
    cfg.paper_gate.min_settled_fills = 1
    cfg.paper_gate.min_realized_pnl = 0.0
    cfg.paper_gate.min_realized_roi = 0.25
    ledger = tmp_path / "paper.parquet"
    PaperLedger(str(ledger)).append(
        Fill(1.0, "BUY", 10.0, 0.95, 0.99, 0.04, "paper", "small-winner")
    )
    monkeypatch.setattr(
        "sportedge.betting.report.collect_all_settlements",
        lambda fills: {"small-winner": 1.0},
    )

    ok, reason = paper_gate_status(cfg, str(ledger))

    assert ok is False
    assert "realized ROI" in reason


def test_make_executor_blocks_live_without_paper_proof(tmp_path):
    cfg = Config(mode="live", confirm_live=True)
    cfg.paper_gate.min_fills = 2
    ledger = tmp_path / "paper.parquet"
    PaperLedger(str(ledger)).append(
        Fill(1.0, "BUY", 5.0, 0.50, 0.70, 0.20, "paper", "tok")
    )

    ex = make_executor(cfg, _gen_secrets(), paper_ledger_path=str(ledger))

    assert type(ex) is PaperExecutor
    ok, reason = paper_gate_status(cfg, str(ledger))
    assert ok is False
    assert "needs 2 fills" in reason


def test_make_executor_live_when_paper_gate_disabled():
    cfg = Config(mode="live", confirm_live=True)
    cfg.paper_gate.enabled = False
    assert isinstance(make_executor(cfg, _gen_secrets()), KalshiLiveExecutor)


def test_make_executor_kalshi_falls_back_to_paper_without_keys():
    cfg = Config(mode="live", confirm_live=True)
    cfg.paper_gate.enabled = False
    ex = make_executor(cfg, Secrets())
    assert type(ex) is PaperExecutor


def test_make_executor_paper_when_not_live():
    cfg = Config(mode="paper")
    assert type(make_executor(cfg, _gen_secrets())) is PaperExecutor
