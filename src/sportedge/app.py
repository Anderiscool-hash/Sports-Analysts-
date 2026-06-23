"""Desktop entry point for the packaged, licensed SportEdge build.

This is what the shipped .exe runs: it verifies the customer's license first, then
launches the live picks loop. Keeping a single ``main`` here gives PyInstaller one
clean target and keeps the license gate in front of every run.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sportedge.licensing import LicenseError, check_license


def _anchor_working_dir() -> None:
    """When frozen by PyInstaller, run from the exe's folder so relative
    ``config/`` and ``models/`` paths resolve next to the shipped executable."""
    if getattr(sys, "frozen", False):
        os.chdir(Path(sys.executable).resolve().parent)


def _await_exit(code: int) -> None:
    # When launched by double-clicking the .exe, keep the window up so the user
    # can read the message before it closes.
    try:
        input("\nPress Enter to exit...")
    except EOFError:
        pass
    sys.exit(code)


def main() -> None:
    ap = argparse.ArgumentParser(description="SportEdge desktop")
    ap.add_argument("--mode", choices=["paper", "live"], default=None)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--paper-ledger", default="data/cache/paper_ledger.parquet")
    args = ap.parse_args()
    _anchor_working_dir()

    try:
        info = check_license()
    except LicenseError as exc:
        print(f"[SportEdge] License check failed: {exc}")
        print("Contact the seller to obtain or renew your license key.")
        _await_exit(1)
        return

    print(
        f"[SportEdge] Licensed to {info.subject or 'customer'} "
        f"(tier={info.tier or 'standard'}, expires {info.expires.isoformat()})."
    )

    # Imported lazily so a license failure never spins up the trading stack.
    from sportedge.live.loop import run

    try:
        run(mode=args.mode, config_path=args.config, paper_ledger=args.paper_ledger)
    except KeyboardInterrupt:
        print("\n[SportEdge] Stopped.")
    except Exception as exc:  # noqa: BLE001 - surface a friendly message in the exe
        print(f"[SportEdge] Unexpected error: {exc}")
        _await_exit(1)


if __name__ == "__main__":
    main()
