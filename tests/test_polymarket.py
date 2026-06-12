from sportedge.market.polymarket import PolymarketClient, unix_ts


def test_unix_ts_accepts_epoch_and_iso():
    assert unix_ts("1717632000") == 1717632000
    assert unix_ts(1717632000.9) == 1717632000
    assert unix_ts("2024-06-06T00:00:00Z") == 1717632000


def test_get_prices_history(monkeypatch):
    calls = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"history": [{"t": 1, "p": 0.42}, {"t": 2, "p": "0.43"}]}

    def fake_get(url, params=None, timeout=None):
        calls["url"] = url
        calls["params"] = params
        calls["timeout"] = timeout
        return Response()

    monkeypatch.setattr("sportedge.market.polymarket.requests.get", fake_get)
    client = PolymarketClient(clob_host="https://clob.example")

    points = client.get_prices_history("token", start_ts=10, end_ts=20, interval="1m", fidelity=5)

    assert calls["url"] == "https://clob.example/prices-history"
    assert calls["params"] == {
        "market": "token",
        "interval": "1m",
        "fidelity": 5,
        "startTs": 10,
        "endTs": 20,
    }
    assert points[0].timestamp == 1
    assert points[0].price == 0.42
    assert points[1].price == 0.43


def test_list_markets_builds_gamma_query(monkeypatch):
    calls = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return []

    def fake_get(url, params=None, timeout=None):
        calls["url"] = url
        calls["params"] = params
        calls["timeout"] = timeout
        return Response()

    monkeypatch.setattr("sportedge.market.polymarket.requests.get", fake_get)
    client = PolymarketClient(gamma_host="https://gamma.example")

    client.list_markets(search="NBA", active=True, closed=None, limit=5)

    assert calls["url"] == "https://gamma.example/markets"
    assert calls["params"] == {"limit": 5, "search": "NBA", "active": "true"}


def test_public_search_uses_q_param(monkeypatch):
    calls = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"events": [], "markets": []}

    def fake_get(url, params=None, timeout=None):
        calls["url"] = url
        calls["params"] = params
        return Response()

    monkeypatch.setattr("sportedge.market.polymarket.requests.get", fake_get)
    client = PolymarketClient(gamma_host="https://gamma.example")

    assert client.public_search("Celtics Mavericks", limit=3) == {"events": [], "markets": []}
    assert calls["url"] == "https://gamma.example/public-search"
    assert calls["params"] == {"q": "Celtics Mavericks", "limit": 3}


def test_find_market_decodes_tokens(monkeypatch):
    client = PolymarketClient()
    monkeypatch.setattr(
        client,
        "list_markets",
        lambda **_: [
            {
                "id": "1",
                "conditionId": "condition",
                "slug": "nba-game",
                "question": "Will Boston win?",
                "clobTokenIds": '["yes-token", "no-token"]',
                "outcomes": '["Yes", "No"]',
                "closed": True,
                "endDate": "2024-06-06T00:00:00Z",
            }
        ],
    )

    market = client.find_market(query="Boston", closed=True)

    assert market is not None
    assert market.id == "1"
    assert market.condition_id == "condition"
    assert market.token_ids == ["yes-token", "no-token"]
    assert market.outcomes == ["Yes", "No"]
    assert market.closed is True


def test_prices_history_frame(monkeypatch):
    monkeypatch.setattr(
        PolymarketClient,
        "get_prices_history",
        lambda *_, **__: [],
    )
    client = PolymarketClient()

    df = client.prices_history_frame("token")

    assert list(df.columns) == ["token_id", "timestamp", "price"]
    assert df.empty
