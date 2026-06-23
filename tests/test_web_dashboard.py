import pandas as pd
import pytest

from sportedge.live.web_dashboard import (
    WebDashboardService,
    aggregate_holdings,
    assess_whale_follow,
    candidate_payload,
    detail_payload,
    portfolio_stats,
    tracking_status,
    whale_flow_payload,
)
from sportedge.types import GameCandidate, LiveDetail


def test_candidate_payload_has_display_and_live_state():
    game = GameCandidate(
        sport="soccer",
        league="fifa.world",
        event_id="42",
        home_team="Brazil",
        away_team="Japan",
        status="in",
        short_detail="67'",
    )

    payload = candidate_payload(game)

    assert payload["display"] == "Japan @ Brazil"
    assert payload["is_live"] is True
    assert payload["event_id"] == "42"


def test_detail_payload_adds_browser_labels():
    detail = LiveDetail(
        sport="basketball",
        league="nba",
        home_team="Knicks",
        away_team="Spurs",
        status="post",
        clock="0:00",
    )

    payload = detail_payload(detail)

    assert payload["status_label"] == "FINAL"
    assert payload["clock_label"] == "0:00"


def test_trade_radar_tracking_states():
    assert tracking_status("", None, None, 0.04, "OFF")[0] == "UNCONFIGURED"
    assert tracking_status("KXNBA", None, None, 0.04, "WAIT")[0] == "SEEKING QUOTE"
    assert tracking_status("KXNBA", 0.40, 0.03, 0.04, "WAIT")[0] == "BELOW EDGE"
    assert tracking_status("KXNBA", 0.40, 0.08, 0.04, "WAIT")[0] == "WATCHING BOTTOM"
    assert tracking_status("KXNBA", 0.40, 0.08, 0.04, "PAPER BUY")[0] == "TRIGGERED"


def test_aggregate_holdings_combines_fills_and_excludes_settled():
    report = pd.DataFrame(
        [
            {"token_id": "KXNBA", "side": "BUY", "size": 5.0, "shares": 10.0,
             "price": 0.50, "mark": 0.60, "is_settled": False, "selected_team": "Spurs"},
            {"token_id": "KXNBA", "side": "BUY", "size": 3.0, "shares": 5.0,
             "price": 0.60, "mark": 0.60, "is_settled": False, "selected_team": "Spurs"},
            {"token_id": "OLD", "side": "BUY", "size": 2.0, "shares": 4.0,
             "price": 0.50, "mark": 1.0, "is_settled": True},
        ]
    )

    positions = aggregate_holdings(report)

    assert len(positions) == 1
    assert positions[0]["ticker"] == "KXNBA"
    assert positions[0]["shares"] == 15.0
    assert positions[0]["cost_basis"] == 8.0
    assert positions[0]["average_entry"] == 8.0 / 15.0
    assert positions[0]["market_value"] == 9.0
    assert positions[0]["unrealized_pnl"] == 1.0


def test_portfolio_stats_separates_realized_and_open_performance():
    report = pd.DataFrame(
        [
            {"size": 5.0, "edge": 0.10, "pnl": 2.0, "is_settled": True},
            {"size": 3.0, "edge": 0.06, "pnl": -1.0, "is_settled": True},
            {"size": 2.0, "edge": 0.08, "pnl": 0.5, "is_settled": False},
        ]
    )
    summary = {
        "staked": 10.0,
        "open_exposure": 2.0,
        "realized_pnl": 1.0,
        "realized_roi": 0.125,
        "unrealized_pnl": 0.5,
        "total_pnl": 1.5,
    }

    stats = portfolio_stats(report, summary, [{"mark": 0.6}])

    assert stats["fills"] == 3
    assert stats["winners"] == 1
    assert stats["losers"] == 1
    assert stats["win_rate"] == 0.5
    assert stats["average_edge"] == 0.08
    assert stats["average_stake"] == 10.0 / 3.0
    assert stats["total_return"] == 0.15


def test_settings_validate_balance_then_store_key_outside_env(tmp_path, monkeypatch):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    env_path = tmp_path / ".env"
    key_path = tmp_path / "kalshi.pem"
    monkeypatch.setenv("KALSHI_API_KEY_ID", "")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", "")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PEM", "")
    monkeypatch.setenv("KALSHI_HOST", "https://api.elections.kalshi.com/trade-api/v2")
    monkeypatch.setattr(
        "sportedge.live.web_dashboard.KalshiClient.get_balance",
        lambda _self: {
            "balance": 12345,
            "balance_dollars": "123.45",
            "portfolio_value": 500,
        },
    )
    service = WebDashboardService(
        config_path=str(tmp_path / "missing.yaml"),
        ledger_path=str(tmp_path / "paper.parquet"),
        env_path=str(env_path),
        private_key_path=str(key_path),
    )

    result = service.save_settings(
        {
            "kalshi_host": "https://api.elections.kalshi.com/trade-api/v2",
            "api_key_id": "key-id-1234",
            "private_key_pem": pem,
        }
    )

    assert result["account"]["connected"] is True
    assert result["account"]["balance"] == 123.45
    assert result["api_key_hint"] == "••••1234"
    assert "private_key_pem" not in result
    assert key_path.read_text().startswith("-----BEGIN PRIVATE KEY-----")
    assert "BEGIN PRIVATE KEY" not in env_path.read_text()


def test_settings_reject_non_kalshi_host(tmp_path):
    service = WebDashboardService(
        config_path=str(tmp_path / "missing.yaml"),
        ledger_path=str(tmp_path / "paper.parquet"),
    )

    with pytest.raises(ValueError, match="Kalshi API host"):
        service.save_settings(
            {
                "kalshi_host": "https://attacker.example/trade-api/v2",
                "api_key_id": "key",
                "private_key_pem": "bad",
            }
        )


def test_mode_switch_allows_demo_and_disarms_strategy(tmp_path, monkeypatch):
    service = WebDashboardService(
        config_path=str(tmp_path / "missing.yaml"),
        ledger_path=str(tmp_path / "paper.parquet"),
    )
    service.secrets = service.secrets.model_copy(
        update={"kalshi_host": "https://external-api.demo.kalshi.co/trade-api/v2"}
    )
    service.strategy_enabled = True
    monkeypatch.setattr(service, "account", lambda force=False: {"connected": True})

    result = service.set_mode("live")

    assert result["trading_mode"] == "live"
    assert result["strategy_enabled"] is False
    assert result["label"] == "Demo Live Trading"


def test_mode_switch_rejects_production_live(tmp_path):
    service = WebDashboardService(
        config_path=str(tmp_path / "missing.yaml"),
        ledger_path=str(tmp_path / "paper.parquet"),
    )
    service.secrets = service.secrets.model_copy(
        update={"kalshi_host": "https://external-api.kalshi.com/trade-api/v2"}
    )

    with pytest.raises(ValueError, match="restricted to Kalshi's demo"):
        service.set_mode("live")


def test_whale_flow_payload_identifies_large_recent_print():
    from sportedge.config import FlowConfig

    config = FlowConfig(lookback_sec=120, whale_min_notional=250.0)
    payload, signal = whale_flow_payload(
        "Spurs",
        "KXNBA-SPURS",
        [
            {
                "yes_price_dollars": "0.5000",
                "count": "600.00",
                "created_ts": 990,
                "taker_side": "yes",
            }
        ],
        config,
        now=1000,
    )

    assert signal.whale and signal.confirms_buy
    assert payload["biggest_notional"] == 300.0
    assert payload["trades"][0]["is_whale"] is True
    assert payload["trades"][0]["side"] == "YES"


def test_flow_confirmation_toggle_applies_to_current_engine(tmp_path):
    service = WebDashboardService(
        config_path=str(tmp_path / "missing.yaml"),
        ledger_path=str(tmp_path / "paper.parquet"),
    )

    enabled = service.set_flow_confirmation(True)

    assert enabled["flow_confirm_enabled"] is True
    assert enabled["trading_mode"] == "paper"


def test_model_assessment_follows_aligned_yes_whale():
    payload = {
        "signal": {"cluster": True},
        "trades": [{"side": "YES", "notional": 500.0, "is_whale": True}],
    }

    result = assess_whale_follow(payload, 0.70, 0.50, 0.04, 250.0)

    assert result["verdict"] == "FOLLOW YES"
    assert result["supports_yes"] is True
    assert result["score"] >= 60


def test_model_assessment_fades_disagreed_no_whale_and_supports_yes():
    payload = {
        "signal": {"cluster": False},
        "trades": [{"side": "NO", "notional": 300.0, "is_whale": True}],
    }

    result = assess_whale_follow(payload, 0.65, 0.50, 0.04, 250.0)

    assert result["verdict"] == "FADE NO"
    assert result["supports_yes"] is True
