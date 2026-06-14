"""Report paper-trading P&L from the persistent ledger.

    python scripts/paper_report.py --ledger data/cache/paper_ledger.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportedge.betting.report import build_paper_report  # noqa: E402
from sportedge.betting.executor import paper_gate_status  # noqa: E402
from sportedge.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper ledger mark-to-market and settlement report")
    ap.add_argument("--ledger", default="data/cache/paper_ledger.parquet")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--out", default="", help="Optional parquet path for fill-level report")
    ap.add_argument("--no-settle", action="store_true", help="Do not settle final ESPN games")
    args = ap.parse_args()

    report = build_paper_report(args.ledger, settle_from_espn=not args.no_settle)
    summary = report.summary
    print(f"ledger: {args.ledger}")
    print(
        f"fills: {summary['fills']}  settled: {summary.get('settled_fills', 0)} "
        f"open: {summary['open_positions']}"
    )
    print(f"staked: {summary['staked']:.2f}  open exposure: {summary['open_exposure']:.2f}")
    print(
        "PnL: "
        f"realized={summary['realized_pnl']:+.2f} "
        f"realized_roi={summary.get('realized_roi', 0.0):+.1%} "
        f"unrealized={summary['unrealized_pnl']:+.2f} "
        f"total={summary['total_pnl']:+.2f}"
    )
    print(f"marked tokens: {len(report.marks)}  settled tokens: {len(report.settlements)}")
    gate_ok, gate_reason = paper_gate_status(load_config(args.config), args.ledger)
    print(f"paper gate: {'PASS' if gate_ok else 'BLOCK'} - {gate_reason}")

    if not report.fills.empty:
        view_cols = [
            "token_id",
            "selected_team",
            "size",
            "price",
            "mark",
            "settlement",
            "pnl",
            "is_settled",
        ]
        print(report.fills[view_cols].tail(10).to_string(index=False))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        report.fills.to_parquet(out, index=False)
        print(f"saved fill report -> {out}")


if __name__ == "__main__":
    main()
