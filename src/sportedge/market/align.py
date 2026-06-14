"""Build replay-ready model/market rows from raw cached price series."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sportedge.model.live_winprob import WinProbModel
from sportedge.model.train import clean_training_rows
from sportedge.types import REGULATION_SECONDS, GameState

RAW_PRICE_RE = re.compile(r"^polymarket_(?P<game_id>[^_]+)_(?P<side>home|away)_(?P<window>[^.]+)\.parquet$")
REPLAY_COLUMNS = [
    "game_id",
    "timestamp",
    "home_score",
    "away_score",
    "period",
    "seconds_remaining",
    "model_p",
    "token_outcome",
    "token_id",
    "price",
    "edge",
    "token_won",
]


@dataclass(frozen=True)
class RawPriceFile:
    path: Path
    game_id: str
    side: str
    window: str


@dataclass(frozen=True)
class AlignmentSummary:
    source_path: str
    output_path: str
    game_id: str
    side: str
    rows: int
    skipped: bool = False
    reason: str = ""


def parse_raw_price_path(path: str | Path) -> RawPriceFile | None:
    """Parse ``polymarket_{game_id}_{home|away}_{window}.parquet`` filenames."""
    p = Path(path)
    match = RAW_PRICE_RE.match(p.name)
    if not match:
        return None
    return RawPriceFile(
        path=p,
        game_id=match.group("game_id"),
        side=match.group("side"),
        window=match.group("window"),
    )


def _state_from_row(row) -> GameState:
    return GameState(
        home_team="",
        away_team="",
        home_score=int(row.home_score),
        away_score=int(row.away_score),
        period=int(row.period),
        seconds_remaining=float(row.seconds_remaining),
        pre_game_home_prob=float(getattr(row, "pre_game_home_prob", 0.5) or 0.5),
    )


def _timestamp_bounds(prices: pd.DataFrame) -> tuple[float, float]:
    timestamps = pd.to_numeric(prices["timestamp"], errors="coerce").dropna()
    if timestamps.empty:
        raise ValueError("price timestamps are empty")
    return float(timestamps.min()), float(timestamps.max())


def _synthetic_timestamps(training: pd.DataFrame, start_ts: float, end_ts: float) -> pd.Series:
    seconds = pd.to_numeric(training["seconds_remaining"], errors="coerce").fillna(REGULATION_SECONDS)
    elapsed = (REGULATION_SECONDS - seconds).clip(lower=0, upper=REGULATION_SECONDS)
    elapsed_min = float(elapsed.min())
    elapsed_max = float(elapsed.max())
    if elapsed_max <= elapsed_min:
        return pd.Series([start_ts] * len(training), index=training.index, dtype="float64")
    progress = (elapsed - elapsed_min) / (elapsed_max - elapsed_min)
    return start_ts + progress * (end_ts - start_ts)


def align_price_rows(
    training_rows: pd.DataFrame,
    price_rows: pd.DataFrame,
    game_id: str,
    side: str,
    model: WinProbModel,
) -> pd.DataFrame:
    """Align one side's raw prices to model probabilities for a game.

    The cached training rows are game-clock states, while the raw market rows are
    wall-clock timestamps. We approximate the join by mapping regulation elapsed
    time across the observed market timestamp range, then merge to the most recent
    available price at each synthetic timestamp.
    """
    if side not in {"home", "away"}:
        raise ValueError("side must be 'home' or 'away'")
    required_training = {"game_id", "home_score", "away_score", "period", "seconds_remaining", "home_win"}
    required_prices = {"timestamp", "price"}
    if not required_training.issubset(training_rows.columns):
        raise ValueError(f"training rows must include {sorted(required_training)}")
    if not required_prices.issubset(price_rows.columns):
        raise ValueError(f"price rows must include {sorted(required_prices)}")

    cleaned_training = clean_training_rows(training_rows)
    game_rows = cleaned_training.loc[cleaned_training["game_id"].astype(str) == str(game_id)].copy()
    if game_rows.empty:
        raise ValueError(f"no training rows found for game_id={game_id}")

    prices = price_rows.copy()
    prices["timestamp"] = pd.to_numeric(prices["timestamp"], errors="coerce").astype("float64")
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    prices = prices.dropna(subset=["timestamp", "price"]).sort_values("timestamp")
    if prices.empty:
        raise ValueError("no usable price rows")

    start_ts, end_ts = _timestamp_bounds(prices)
    game_rows["timestamp"] = _synthetic_timestamps(game_rows, start_ts, end_ts)
    game_rows = game_rows.sort_values("timestamp")

    home_probs = [model.predict(_state_from_row(row)) for row in game_rows.itertuples(index=False)]
    game_rows["model_p"] = home_probs if side == "home" else [1.0 - prob for prob in home_probs]
    game_rows["token_outcome"] = side
    if "token_id" in prices.columns:
        token_id = str(prices["token_id"].dropna().astype(str).iloc[0]) if prices["token_id"].notna().any() else side
    else:
        token_id = side
    game_rows["token_id"] = token_id

    aligned = pd.merge_asof(
        game_rows.sort_values("timestamp"),
        prices[["timestamp", "price"]].sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    aligned["price"] = aligned["price"].ffill().bfill()
    aligned = aligned.dropna(subset=["price"])
    aligned["edge"] = aligned["model_p"] - aligned["price"]
    home_win = pd.to_numeric(aligned["home_win"], errors="coerce").fillna(0).astype(int)
    aligned["token_won"] = home_win if side == "home" else 1 - home_win
    return aligned[REPLAY_COLUMNS].reset_index(drop=True)


def build_aligned_file(
    raw_path: str | Path,
    training_path: str | Path,
    output_dir: str | Path,
    model_path: str,
    prefix: str = "aligned_generated",
) -> AlignmentSummary:
    """Build one replay parquet from a raw Polymarket price parquet."""
    raw = parse_raw_price_path(raw_path)
    if raw is None:
        return AlignmentSummary(str(raw_path), "", "", "", 0, skipped=True, reason="not a raw Polymarket file")
    try:
        training = pd.read_parquet(training_path)
        prices = pd.read_parquet(raw.path)
        model = WinProbModel.load(model_path)
        aligned = align_price_rows(training, prices, raw.game_id, raw.side, model)
    except Exception as exc:  # noqa: BLE001
        return AlignmentSummary(str(raw.path), "", raw.game_id, raw.side, 0, skipped=True, reason=str(exc))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{prefix}_{raw.game_id}_{raw.side}.parquet"
    aligned.to_parquet(output_path, index=False)
    return AlignmentSummary(str(raw.path), str(output_path), raw.game_id, raw.side, int(len(aligned)))


def build_aligned_directory(
    directory: str | Path,
    training_path: str | Path,
    output_dir: str | Path,
    model_path: str,
    pattern: str = "polymarket_*_10m.parquet",
    prefix: str = "aligned_generated",
) -> list[AlignmentSummary]:
    """Build replay files for every matching raw price file in a directory."""
    root = Path(directory)
    return [
        build_aligned_file(path, training_path, output_dir, model_path, prefix=prefix)
        for path in sorted(root.glob(pattern))
    ]
