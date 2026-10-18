# -*- coding: utf-8 -*-
r"""Ревизия PTG-MEM, замер 6б: вычистка секретов detect-secrets с точными значениями.

Замер 6 (secrets_compare.py) считал «поймано», если scan_line вернул хоть что-то. В разовом режиме scan_line
(enable_eager_search) энтропийные детекторы Base64/Hex метят почти любое латинское слово — «token», «commit»,
«password» — поэтому «13 из 13» там ничего не значит. Здесь энтропийные детекторы выключены. Каждая находка
расширяется до целого слова, потому что GitHubTokenDetector отдаёт группу регулярки («ghp»), а не ключ целиком.
Считается, что заменено в каждой строке, а обычного текста больше.

    .venv-kit\Scripts\python bench\secrets_exact.py
"""
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
from detect_secrets.core.plugins.util import get_mapping_from_secret_type_to_class  # noqa: E402
from detect_secrets.core.scan import scan_line  # noqa: E402
from detect_secrets.settings import transient_settings  # noqa: E402

OFF = {"HexHighEntropyString", "Base64HighEntropyString", "IPPublicDetector"}
PLUGINS = [{"name": n} for n in sorted(c.__name__ for c in get_mapping_from_secret_type_to_class().values()) if n not in OFF]
WORD = r"[A-Za-z0-9_\-]"  # тело ключа: дальше / . = не тянуть — иначе съест ссылку целиком

FAKE = "Xq7Lm2Pz9Rt4Vw8Ks3Nd6Fh1Jb5Gc0YaUe"
SECRETS = {
    "OpenAI / Anthropic sk-": f"key = sk-ant-api03-{FAKE}{FAKE[:10]}",
    "OpenAI sk-proj-": f"OPENAI_API_KEY=sk-proj-{FAKE}T3BlbkFJ{FAKE[:20]}",
    "GitHub ghp_": f"token ghp_{FAKE}12",
    "Google AIza": f"AIza{FAKE}{FAKE[:1]}",
    "AWS AKIA": "aws_access_key_id = AKIAZ7Q3EXAMPLE4X2Q7",
    "AWS secret": f"aws_secret_access_key = {FAKE}abcdEFGH12",
    "Slack xoxb-": "xoxb-1234567890-0987654321-" + FAKE[:24],
    "Hugging Face hf_": f"hf_{FAKE[:34]}",
    "JWT": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0." + FAKE[:30],
    "закрытый ключ PEM": "-----BEGIN RSA PRIVATE KEY-----",
    "пароль в конфиге": 'password = "Tr0ub4dor&3-horse-battery"',
    "строка подключения": "postgres://admin:S3cr3tPassw0rd@db.example.com:5432/app",
    "Stripe sk_live_": "sk_live_" + FAKE[:24],
    "Telegram бот": "123456789:AAH" + FAKE[:32],
}
PLAIN = {
    "обычный текст": "Порог 0.47/0.57/0.65 откалиброван на русском корпусе, D: 3",
    "хэш коммита": "commit b7b8b53c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a",
    "sha256": "sha256 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    "uuid": "session 2a9ebf22-f8b0-4b20-b905-406f8ef48cc9",
    "английский": "The token budget is passed to the planner and the password field stays empty",
    "код": "def plan_sync(stored: list[Stored], fresh: list[Atom]) -> SyncPlan:",
    "присваивание": "api_key = os.environ['ANTHROPIC_API_KEY']",
    "путь": r"H:\работа\сайты\идеи\память\PTG-MEM\bench\out\hybrid.json",
    "ссылка": "https://github.com/DeusData/codebase-memory-mcp/releases/tag/v0.11.0",
    "слово с sk-": "task-management-pipeline-for-agents и break_something_long_variable_name",
    "модель": "claude-opus-5-5 и Qwen3-Embedding-0.6B, idle_unload_seconds=600",
}


# Чего нет у detect-secrets — форматы поставщиков из своего списка ptg_mem/filters.py (замер 6б: без них 10 из 14).
VENDOR = re.compile(r"\b(?:sk-[A-Za-z0-9_\-]{20,}|AIza[A-Za-z0-9_\-]{30,}|hf_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
                    r"|uak_[A-Za-z0-9_\-]{16,}|ak_[A-Za-z0-9_\-]{20,}|ck_[A-Za-z0-9_\-]{16,})")


def scrub(text, vendor=True):
    with transient_settings({"plugins_used": PLUGINS}):
        values = {s.secret_value for line in text.splitlines() for s in scan_line(line) if s.secret_value}
    if vendor:
        values |= set(VENDOR.findall(text))
    spans = set()
    for v in values:
        for m in re.finditer(re.escape(v), text):
            a, b = m.start(), m.end()
            while a > 0 and re.match(WORD, text[a - 1]):
                a -= 1
            while b < len(text) and re.match(WORD, text[b]):
                b += 1
            spans.add((a, b))
    out, last = [], 0
    for a, b in sorted(spans):
        if a < last:
            continue
        out.append(text[last:a] + "<secret>")
        last = b
    return "".join(out) + text[last:], len(values)


def main():
    caught, alone = 0, 0
    for name, text in SECRETS.items():
        new, n = scrub(text)
        caught += n > 0
        alone += scrub(text, vendor=False)[1] > 0
        print(f"[секрет] {name:<24} {'да' if n else '—'}  {new}")
    false = 0
    for name, text in PLAIN.items():
        new, n = scrub(text)
        false += n > 0
        print(f"[текст]  {name:<24} {'ЛОЖНО' if n else 'чисто'}  {new}")
    print(f"\nпоймано секретов {caught} из {len(SECRETS)} (один detect-secrets — {alone}); "
          f"ложных срабатываний {false} из {len(PLAIN)}")


if __name__ == "__main__":
    main()
