"""Durable execution history for non-paper fills."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from sportedge.betting.executor import Fill

HISTORY_COLUMNS = list(Fill.__dataclass_fields__)


class ExecutionHistory:
    """Append-only parquet store preserving live execution details."""

    def __init__(self, path: str = "data/cache/execution_history.parquet") -> None:
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=HISTORY_COLUMNS)
        frame = pd.read_parquet(self.path)
        for column in HISTORY_COLUMNS:
            if column not in frame.columns:
                frame[column] = None
        return frame[HISTORY_COLUMNS].copy()

    def append(self, fill: Fill) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        current = self.load()
        row = pd.DataFrame([asdict(fill)], columns=HISTORY_COLUMNS)
        combined = row if current.empty else pd.concat([current, row], ignore_index=True)
        combined.to_parquet(self.path, index=False)
