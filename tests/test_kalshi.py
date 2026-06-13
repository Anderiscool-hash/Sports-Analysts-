"""Tests for the Kalshi market venue: price conversion, order payloads, RSA-PSS
signing, and the venue-aware executor selection. All network is mocked; no keys
or live calls are used.
"""

from __future__ import annotations

import base64

import pytest

from sportedge.betting.executor import KalshiLiveExecutor, PaperExecutor, make_executor
from sportedge.config import Config, Secrets
from sportedge.market.kalshi import KalshiClient, cents_to_prob


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


def test_place_order_signs_and_posts(monkeypatch):
    pytest.importorskip("cryptography")
    secrets = _gen_secrets()
    client = KalshiClient(secrets=secrets)

    captured = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"order": {"status": "resting"}}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _Resp()

    monkeypatch.setattr("sportedge.market.kalshi.requests.post", fake_post)
    result = client.place_order("WC-BRA-WIN", stake_usd=5.0, price_prob=0.5)

    assert result == {"order": {"status": "resting"}}
    assert captured["url"].endswith("/portfolio/orders")
    assert captured["json"]["ticker"] == "WC-BRA-WIN"
    assert "KALSHI-ACCESS-SIGNATURE" in captured["headers"]


# ----- venue-aware executor selection -----


def test_make_executor_kalshi_live_when_keys_present():
    cfg = Config(mode="live", confirm_live=True)
    secrets = _gen_secrets()
    assert isinstance(make_executor(cfg, secrets), KalshiLiveExecutor)


def test_make_executor_kalshi_falls_back_to_paper_without_keys():
    cfg = Config(mode="live", confirm_live=True)
    ex = make_executor(cfg, Secrets())
    assert type(ex) is PaperExecutor


def test_make_executor_paper_when_not_live():
    cfg = Config(mode="paper")
    assert type(make_executor(cfg, _gen_secrets())) is PaperExecutor
