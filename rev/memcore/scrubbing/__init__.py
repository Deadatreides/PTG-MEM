"""Вычистка секретов из текста до записи в память.

Готовые части: detect-secrets 1.5 (Yelp) — детекторы ключей поставщиков и паролей в присваиваниях; свой список
форматов, которых в detect-secrets нет (VENDOR_KEY_RE из memcore.types). Замер bench/secrets_exact.py: вместе
13 секретов из 14 и 0 ложных срабатываний из 11, один detect-secrets — 10 из 14.
Рецепт:
  • detect-secrets — по каждой строке текста: внутри `with transient_settings({"plugins_used": PLUGINS}):` вызвать
    `scan_line(line)` (detect_secrets.core.scan, detect_secrets.settings); значение — secret_value (пустые — мимо).
    PLUGINS — [{"name": имя}] всех детекторов из get_mapping_from_secret_type_to_class()
    (detect_secrets.core.plugins.util; имя — __name__ класса), кроме HexHighEntropyString и Base64HighEntropyString
    (в разовом режиме scan_line они метят любое латинское слово, в том числе хэши коммитов) и IPPublicDetector
    (адрес в заметке — не секрет);
  • свои форматы — re.findall(VENDOR_KEY_RE, text) по всему тексту;
  • каждое найденное значение, кроме самого REDACTION_MARK, — все его вхождения в текст, и каждое вхождение
    расширяется в обе стороны по символам [A-Za-z0-9_-]: GitHubTokenDetector отдаёт только префикс «ghp»;
  • пересекающиеся и смежные куски сливаются; каждый итоговый кусок заменяется на REDACTION_MARK; found — число
    итоговых кусков.

node: composite
"""
import deal

from memcore.scrubbing.extension import extend_span
from memcore.scrubbing.merging import merge_spans
from memcore.scrubbing.provider import provider_values
from memcore.scrubbing.redaction import redact
from memcore.scrubbing.starts import secret_starts
from memcore.scrubbing.vendor import vendor_values
from memcore.types import Scrubbed


@deal.ensure(lambda text, result: result.found >= 0)
@deal.ensure(lambda text, result: (result.found == 0) == (result.text == text))
@deal.has()
def scrub(text: str) -> Scrubbed:
    """Текст с секретами, заменёнными на «<secret>», и число заменённых кусков — по рецепту из описания модуля.
    Хэш коммита и уже вычищенный текст не трогаются.

    D: 8

    >>> scrub("token ghp_Xq7Lm2Pz9Rt4Vw8Ks3Nd6Fh1Jb5Gc0YaUe12")
    Scrubbed(text='token <secret>', found=1)
    >>> scrub("commit b7b8b53c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a")
    Scrubbed(text='commit b7b8b53c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a', found=0)
    >>> scrub('password = "Tr0ub4dor&3-horse-battery"')
    Scrubbed(text='password = "<secret>"', found=1)
    >>> scrub("key = sk-ant-api03-Xq7Lm2Pz9Rt4Vw8Ks3Nd6Fh1Jb5Gc0YaUe")
    Scrubbed(text='key = <secret>', found=1)
    >>> scrub("token <secret>")
    Scrubbed(text='token <secret>', found=0)
    """
    values = provider_values(text) + vendor_values(text)
    spans = tuple(
        extend_span(text, start, start + len(value))
        for value in values
        for start in secret_starts(text, value)
    )
    merged = merge_spans(spans)
    return Scrubbed(redact(text, merged), len(merged))
