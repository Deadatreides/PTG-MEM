# -*- coding: utf-8 -*-
r"""Ревизия PTG-MEM, замер 6: вычистка секретов — готовый detect-secrets против своего списка (ptg_mem/filters.py).

Строки — поддельные ключи по форматам поставщиков (настоящих ключей здесь нет) и обычный текст, который ловить
нельзя. Считается, кто что поймал.

    .venv-rev\Scripts\python bench\secrets_compare.py
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ptg_mem import filters  # noqa: E402

FAKE = "Xq7Lm2Pz9Rt4Vw8Ks3Nd6Fh1Jb5Gc0YaUe"
CASES = {
    "OpenAI / Anthropic sk-": f"key = sk-ant-api03-{FAKE}{FAKE[:10]}",
    "GitHub ghp_": f"token ghp_{FAKE[:36]}",
    "Google AIza": f"AIza{FAKE}{FAKE[:1]}",
    "AWS AKIA": "aws_access_key_id = AKIAZ7Q3EXAMPLE4X2Q7",
    "AWS secret": f"aws_secret_access_key = {FAKE}abcdEFGH12",
    "Slack xoxb-": "xoxb-1234567890-0987654321-" + FAKE[:24],
    "Hugging Face hf_": f"hf_{FAKE[:34]}",
    "JWT": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0." + FAKE[:30],
    "закрытый ключ PEM": "-----BEGIN RSA PRIVATE KEY-----\nMIIEow" + FAKE + "\n-----END RSA PRIVATE KEY-----",
    "пароль в конфиге": 'password = "Tr0ub4dor&3-horse-battery"',
    "строка подключения": "postgres://admin:S3cr3tPassw0rd@db.example.com:5432/app",
    "Stripe sk_live_": "sk_live_" + FAKE[:24],
    "Telegram бот": "123456789:AAH" + FAKE[:32],
    "обычный текст": "Порог 0.47/0.57/0.65 откалиброван на русском корпусе, D: 3",
    "хэш коммита": "commit b7b8b53c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a",
}


def ours(text):
    return filters.scrub_secrets(text)[1] > 0


def theirs(text):
    from detect_secrets.core.scan import scan_line
    from detect_secrets.settings import default_settings
    with default_settings():
        return any(True for line in text.splitlines() for _ in scan_line(line))


def main():
    print(f"{'вид':<24} {'свой':>6} {'detect-secrets':>15}")
    hits = {"свой": 0, "detect-secrets": 0}
    for name, text in CASES.items():
        a, b = ours(text), theirs(text)
        hits["свой"] += a and name not in ("обычный текст", "хэш коммита")
        hits["detect-secrets"] += b and name not in ("обычный текст", "хэш коммита")
        print(f"{name:<24} {'да' if a else '—':>6} {'да' if b else '—':>15}")
    print(f"\nпоймано секретов (из {len(CASES) - 2}): свой {hits['свой']}, detect-secrets {hits['detect-secrets']}")


if __name__ == "__main__":
    main()
