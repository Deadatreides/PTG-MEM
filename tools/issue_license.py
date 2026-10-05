# -*- coding: utf-8 -*-
"""Vendor tool: create the signing key once, then issue PTG-MEM licence keys.

    python tools/issue_license.py keygen  <private_key.pem>
    python tools/issue_license.py issue   <private_key.pem> "<licensee>" [pro|team] [seats] [expires YYYY-MM-DD]

The private key must never be committed. ``keygen`` prints the public key to paste
into ``ptg_mem/license.py`` (PUBLIC_KEY_B64).
"""
import base64
import datetime as dt
import json
import sys
import uuid

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def keygen(path):
    k = Ed25519PrivateKey.generate()
    pem = k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                          serialization.NoEncryption())
    with open(path, "xb") as f:                 # never overwrite an existing key
        f.write(pem)
    pub = k.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    print("private key written to", path)
    print("PUBLIC_KEY_B64 =", b64e(pub))


def issue(path, name, plan="pro", seats="1", expires=None):
    with open(path, "rb") as f:
        k = serialization.load_pem_private_key(f.read(), password=None)
    payload = {"v": 1, "id": str(uuid.uuid4()), "name": name, "plan": plan,
               "seats": int(seats), "issued": dt.date.today().isoformat(), "expires": expires}
    body = b64e(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    sig = b64e(k.sign(body.encode("ascii")))
    print("PTGMEM-%s.%s" % (body, sig))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "keygen":
        keygen(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "issue":
        issue(*sys.argv[2:])
    else:
        print(__doc__)
        sys.exit(2)
