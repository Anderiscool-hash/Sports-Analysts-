from sportedge.config import Config
from sportedge.live.loop import paper_metadata


def test_nba_loop_paper_metadata_supports_espn_settlement():
    cfg = Config()
    cfg.market.market_slug = "KXNBA-LAL-WIN"
    cfg.market.espn_event_id = "401"
    cfg.market.home_team = "Lakers"
    cfg.market.away_team = "Celtics"

    assert paper_metadata(cfg) == {
        "event_id": "401",
        "sport": "basketball",
        "league": "nba",
        "home_team": "Lakers",
        "away_team": "Celtics",
        "selected_team": "Lakers",
    }


def test_nba_loop_paper_metadata_falls_back_to_market_slug():
    cfg = Config()
    cfg.market.market_slug = "401"
    cfg.market.home_team = "Lakers"
    cfg.market.away_team = "Celtics"

    assert paper_metadata(cfg)["event_id"] == "401"
