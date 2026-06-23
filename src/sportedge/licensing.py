"""Offline, signed license-key gate for the paid desktop build.

The valuable thing you sell is access. This module lets the app refuse to run
without a valid, unexpired license key that *only you* can mint, with no license
server required:

  - You hold an Ed25519 *private* key (kept secret, never shipped).
  - The app embeds the matching *public* key and verifies keys offline.
  - A license key is ``b64url(payload_json).b64url(signature)`` where the payload
    is ``{"sub": <buyer>, "tier": <plan>, "exp": "YYYY-MM-DD"}``.

Because verification is a signature check, customers cannot forge or extend keys
without the private key. (This gate controls *who may run the app*; it is not by
itself source-code protection -- see PACKAGING.md for obfuscation and the
server-side model split.)

Seller tools: ``scripts/make_keypair.py`` (once) and ``scripts/make_license.py``
(per customer). Set the generated public key here or via ``SPORTEDGE_LICENSE_PUBKEY``.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# Replace with YOUR Ed25519 public key (base64 of the 32 raw bytes) before shipping.
# Generate it with scripts/make_keypair.py. Keep the PRIVATE key off the customer's
# machine forever. An env override is supported for development/testing.
EMBEDDED_PUBLIC_KEY_B64 = ""


class LicenseError(Exception):
    """Raised when a license is missing, malformed, forged, or expired."""


@dataclass(frozen=True)
class LicenseInfo:
    subject: str
    tier: str
    expires: date
    raw: dict


def _b64url_decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def generate_keypair() -> tuple[str, str]:
    """Return ``(private_key_b64, public_key_b64)`` for a fresh Ed25519 keypair."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    priv = Ed25519PrivateKey.generate()
    priv_raw = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(priv_raw).decode(), base64.b64encode(pub_raw).decode()


def generate_license(payload: dict, private_key_b64: str) -> str:
    """Mint a signed license key from a payload and your private key (seller-side)."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    if "exp" not in payload:
        raise ValueError("license payload requires an 'exp' date (YYYY-MM-DD)")
    priv = Ed25519PrivateKey.from_private_bytes(base64.b64decode(private_key_b64))
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    sig = priv.sign(body)
    return f"{_b64url_encode(body)}.{_b64url_encode(sig)}"


def _resolve_public_key(public_key_b64: str | None) -> bytes:
    key = public_key_b64 or os.getenv("SPORTEDGE_LICENSE_PUBKEY") or EMBEDDED_PUBLIC_KEY_B64
    if not key:
        raise LicenseError("no license public key configured in this build")
    try:
        return base64.b64decode(key)
    except Exception as exc:  # noqa: BLE001
        raise LicenseError(f"invalid license public key: {exc}") from exc


def verify_license(
    license_key: str,
    public_key_b64: str | None = None,
    today: date | None = None,
) -> LicenseInfo:
    """Verify a license key's signature and expiry. Raises ``LicenseError`` on failure."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    if not license_key or "." not in license_key:
        raise LicenseError("malformed license key")
    pub = Ed25519PublicKey.from_public_bytes(_resolve_public_key(public_key_b64))
    body_b64, sig_b64 = license_key.strip().split(".", 1)
    try:
        body = _b64url_decode(body_b64)
        sig = _b64url_decode(sig_b64)
    except Exception as exc:  # noqa: BLE001
        raise LicenseError(f"license key is not valid base64: {exc}") from exc

    try:
        pub.verify(sig, body)
    except InvalidSignature as exc:
        raise LicenseError("license signature does not verify (forged or tampered)") from exc

    try:
        payload = json.loads(body)
        expires = date.fromisoformat(str(payload["exp"]))
    except (ValueError, KeyError, TypeError) as exc:
        raise LicenseError(f"license payload is invalid: {exc}") from exc

    if (today or date.today()) > expires:
        raise LicenseError(f"license expired on {expires.isoformat()}")

    return LicenseInfo(
        subject=str(payload.get("sub", "")),
        tier=str(payload.get("tier", "")),
        expires=expires,
        raw=payload,
    )


def _license_file() -> Path:
    """Default per-user license location (override with ``SPORTEDGE_LICENSE_FILE``)."""
    override = os.getenv("SPORTEDGE_LICENSE_FILE")
    if override:
        return Path(override)
    base = os.getenv("APPDATA") or os.path.expanduser("~")
    return Path(base) / "SportEdge" / "license.key"


def load_license_key() -> str | None:
    """Read the license key from ``SPORTEDGE_LICENSE`` env or the license file."""
    env = os.getenv("SPORTEDGE_LICENSE")
    if env:
        return env.strip()
    path = _license_file()
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return None


def check_license(public_key_b64: str | None = None) -> LicenseInfo:
    """Load and verify the active license, or raise ``LicenseError`` with guidance."""
    key = load_license_key()
    if not key:
        raise LicenseError(
            "No license found. Set SPORTEDGE_LICENSE or place your key at "
            f"{_license_file()}"
        )
    return verify_license(key, public_key_b64=public_key_b64)
