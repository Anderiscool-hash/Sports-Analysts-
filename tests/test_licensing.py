"""Offline signed license gate: sign/verify, expiry, tamper, and wrong-key paths."""

from datetime import date

import pytest

from sportedge.licensing import (
    LicenseError,
    generate_keypair,
    generate_license,
    verify_license,
)

pytest.importorskip("cryptography")


def _mint(priv, exp="2099-01-01", sub="buyer@example.com", tier="pro"):
    return generate_license({"sub": sub, "tier": tier, "exp": exp}, priv)


def test_valid_license_verifies():
    priv, pub = generate_keypair()
    key = _mint(priv)
    info = verify_license(key, public_key_b64=pub)
    assert info.subject == "buyer@example.com"
    assert info.tier == "pro"
    assert info.expires == date(2099, 1, 1)


def test_expired_license_rejected():
    priv, pub = generate_keypair()
    key = _mint(priv, exp="2020-01-01")
    with pytest.raises(LicenseError, match="expired"):
        verify_license(key, public_key_b64=pub)


def test_expiry_boundary_is_inclusive():
    priv, pub = generate_keypair()
    key = _mint(priv, exp="2026-06-23")
    # Valid on the expiry day, invalid the next day.
    assert verify_license(key, public_key_b64=pub, today=date(2026, 6, 23))
    with pytest.raises(LicenseError):
        verify_license(key, public_key_b64=pub, today=date(2026, 6, 24))


def test_tampered_payload_rejected():
    priv, pub = generate_keypair()
    key = _mint(priv)
    body, sig = key.split(".", 1)
    forged = body[:-1] + ("A" if body[-1] != "A" else "B") + "." + sig
    with pytest.raises(LicenseError, match="base64|verify|invalid"):
        verify_license(forged, public_key_b64=pub)


def test_wrong_public_key_rejected():
    priv, _ = generate_keypair()
    _, other_pub = generate_keypair()
    key = _mint(priv)
    with pytest.raises(LicenseError, match="verify"):
        verify_license(key, public_key_b64=other_pub)


def test_malformed_key_rejected():
    _, pub = generate_keypair()
    with pytest.raises(LicenseError, match="malformed"):
        verify_license("not-a-license", public_key_b64=pub)


def test_missing_public_key_config_errors():
    priv, _ = generate_keypair()
    key = _mint(priv)
    with pytest.raises(LicenseError, match="public key"):
        verify_license(key, public_key_b64="")
