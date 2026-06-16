"""Run a self-contained paper-trading session and show the result.

This is the on-demand way to make paper trading *start and work* without waiting
for a live Kalshi-covered game. It replays the cached aligned model/market rows
through the exact same strategy spine the live loop uses (BottomDetector ->
Strategy -> PaperExecutor), writes fills to a dedicated sim ledger, then settles
them from cached final scores and prints the P&L.

    python scripts/paper_sim.py
    python scripts/paper_sim.py --keep          # append instead of resetting
    python scripts/paper_sim.py --ledger data/cache/paper_ledger.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportedge.betting.replay import replay_directory  # noqa: E402
from sportedge.betting.report import build_paper_report  # noqa: E402
from sportedge.betting.executor import paper_gate_status  # noqa: E402
from sportedge.config import load_config  # noqa: E402

DEFAULT_LEDGER = "data/cache/paper_sim_ledger.parquet"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a paper-trading session over cached aligned rows")
    ap.add_argument("--data-dir", default="data/cache")
    ap.add_argument("--pattern", default="aligned*.parquet")
    ap.add_argument("--ledger", default=DEFAULT_LEDGER)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--keep", action="store_true", help="append to the ledger instead of resetting")
    ap.add_argument("--no-settle", action="store_true", help="skip ESPN settlement (offline)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ledger_path = Path(args.ledger)

    if not args.keep and ledger_path.exists():
        ledger_path.unlink()
        print(f"reset sim ledger: {ledger_path}")

    data_dir = Path(args.data_dir)
    aligned_files = sorted(data_dir.glob(args.pattern))
    if not aligned_files:
        print(f"no aligned files match {args.data_dir}/{args.pattern}; nothing to trade.")
        return

    print(f"paper trading over {len(aligned_files)} aligned file(s) in {data_dir}/ ...")
    summaries = replay_directory(data_dir, cfg, str(ledger_path), pattern=args.pattern)
    rows_seen = sum(s.rows_seen for s in summaries if not s.skipped)
    fills = sum(s.fills for s in summaries if not s.skipped)
    for s in summaries:
        if s.skipped:
            print(f"  SKIP {Path(s.path).name}: {s.reason}")
        elif s.fills:
            print(f"  {Path(s.path).name}: {s.fills} fill(s) over {s.rows_seen} rows")
    print(f"-> replayed {rows_seen} rows, generated {fills} paper fill(s)")

    if fills == 0:
        print("no fills triggered (no qualifying dip+edge). Paper engine ran cleanly.")
        return

    report = build_paper_report(str(ledger_path), settle_from_espn=not args.no_settle)
    s = report.summary
    print("\n=== paper P&L ===")
    print(f"fills: {s['fills']}  settled: {s.get('settled_fills', 0)}  open: {s['open_positions']}")
    print(f"staked: {s['staked']:.2f}  open exposure: {s['open_exposure']:.2f}")
    print(
        "PnL: "
        f"realized={s['realized_pnl']:+.2f} "
        f"realized_roi={s.get('realized_roi', 0.0):+.1%} "
        f"unrealized={s['unrealized_pnl']:+.2f} "
        f"total={s['total_pnl']:+.2f}"
    )
    gate_ok, gate_reason = paper_gate_status(cfg, str(ledger_path))
    print(f"paper gate: {'PASS' if gate_ok else 'BLOCK'} - {gate_reason}")
    print(f"ledger: {ledger_path}")


if __name__ == "__main__":
    main()
