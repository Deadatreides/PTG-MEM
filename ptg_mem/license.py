# -*- coding: utf-8 -*-
"""Pro licence: an offline, signed key. No server, no phone-home.

The model is WinRAR's: everything a single developer needs is free and stays
free; the key unlocks what teams and heavy users need, and it buys the right to
use PTG-MEM inside a company without AGPL obligations (see COMMERCIAL.md).
The check is honest rather than hostile — the source is open, and a key is a
receipt, not a lock.

Key format:  PTGMEM-<base64url(json payload)>.<base64url(ed25519 signature)>
Payload:     {"v":1, "id":..., "name":..., "plan":"pro"|"team", "seats":1,
              "issued":"YYYY-MM-DD", "expires": null | "YYYY-MM-DD"}
"""
from __future__ import annotations

import base64
import datetime as dt
import json
import os
import time

from . import paths

# Public half of the signing key. The private half never enters this repository.
PUBLIC_KEY_B64 = "774cQEXKM0_tgQgdCkIktHd2zzaksFE5xM2fYLFYinE"

PRO_FEATURES = {
    "cross_project": "Recall and search across all your projects at once",
    "decision_export": "Export the decision log (supersedes / contradicts / fixes) as Markdown",
    "packs": "Export a project's memory as a pack and mount a teammate's pack read-only",
    "no_nag": "No 'unregistered' reminder in the GUI",
}
PLANS = {"pro": set(PRO_FEATURES), "team": set(PRO_FEATURES)}
NAG_AFTER_DAYS = 40            # the WinRAR number


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def parse(key: str) -> dict:
    """Verify a key and return its payload. Raises ValueError on any problem."""
    key = (key or "").strip()
    if not key.startswith("PTGMEM-") or "." not in key:
        raise ValueError("not a PTG-MEM key")
    body, sig = key[len("PTGMEM-"):].rsplit(".", 1)
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as ex:
        raise ValueError("the 'cryptography' package is needed to check a key") from ex
    if PUBLIC_KEY_B64.startswith("REPLACE"):
        raise ValueError("this build has no licence public key")
    pub = Ed25519PublicKey.from_public_bytes(_b64d(PUBLIC_KEY_B64))
    try:
        pub.verify(_b64d(sig), body.encode("ascii"))
    except Exception as ex:                          # noqa: BLE001 — InvalidSignature and decode errors
        raise ValueError("signature does not match") from ex
    payload = json.loads(_b64d(body).decode("utf-8"))
    exp = payload.get("expires")
    if exp and dt.date.fromisoformat(exp) < dt.date.today():
        raise ValueError("key expired on %s" % exp)
    if payload.get("plan") not in PLANS:
        raise ValueError("unknown plan %r" % payload.get("plan"))
    return payload


def install(key: str) -> dict:
    payload = parse(key)
    paths.ensure_home()
    with open(paths.LICENSE, "w", encoding="utf-8") as f:
        f.write(key.strip() + "\n")
    return payload


def current() -> dict | None:
    try:
        with open(paths.LICENSE, encoding="utf-8") as f:
            return parse(f.read())
    except (OSError, ValueError):
        return None


def has(feature: str) -> bool:
    lic = current()
    return bool(lic) and feature in PLANS.get(lic.get("plan"), set())


def first_run() -> float:
    """When this installation first started (for the gentle reminder only)."""
    p = os.path.join(paths.HOME, "first_run")
    try:
        with open(p, encoding="utf-8") as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        paths.ensure_home()
        now = time.time()
        with open(p, "w", encoding="utf-8") as f:
            f.write(str(now))
        return now


def status() -> dict:
    lic = current()
    days = int((time.time() - first_run()) // 86400)
    return {
        "registered": bool(lic),
        "plan": lic.get("plan") if lic else "free",
        "name": lic.get("name") if lic else None,
        "expires": lic.get("expires") if lic else None,
        "features": sorted(PLANS.get(lic.get("plan"), set())) if lic else [],
        "all_features": PRO_FEATURES,
        "days_used": days,
        "nag": (not lic) and days >= NAG_AFTER_DAYS,
    }
