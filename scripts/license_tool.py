"""Seller-side license tooling. Run this on YOUR machine only -- never ship it.

Usage:
  # One time: create your signing keypair. Keep the private key secret forever.
  python scripts/license_tool.py keygen

  # Put the printed public key into sportedge/licensing.EMBEDDED_PUBLIC_KEY_B64
  # (or set SPORTEDGE_LICENSE_PUBKEY in the build environment).

  # Per customer: mint a license key, then send the customer the printed key.
  python scripts/license_tool.py mint --priv <PRIVATE_B64> --sub "buyer@email" \
      --tier pro --days 30
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportedge.licensing import generate_keypair, generate_license  # noqa: E402


def _keygen(_args: argparse.Namespace) -> None:
    priv, pub = generate_keypair()
    print("PRIVATE KEY (keep secret, never ship):")
    print(f"  {priv}")
    print("PUBLIC KEY (embed in the app / set SPORTEDGE_LICENSE_PUBKEY):")
    print(f"  {pub}")


def _mint(args: argparse.Namespace) -> None:
    exp = (date.today() + timedelta(days=args.days)).isoformat()
    payload = {"sub": args.sub, "tier": args.tier, "exp": exp}
    key = generate_license(payload, args.priv)
    print(f"License for {args.sub} (tier={args.tier}, expires {exp}):")
    print(key)


def main() -> None:
    ap = argparse.ArgumentParser(description="SportEdge seller license tool")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("keygen", help="generate a signing keypair").set_defaults(func=_keygen)

    m = sub.add_parser("mint", help="mint a customer license key")
    m.add_argument("--priv", required=True, help="your base64 private key")
    m.add_argument("--sub", required=True, help="customer identifier (email/handle)")
    m.add_argument("--tier", default="standard")
    m.add_argument("--days", type=int, default=30, help="validity window in days")
    m.set_defaults(func=_mint)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
