"""Значения секретов в форматах поставщиков, которых нет в detect-secrets: Anthropic и OpenAI sk-, Google AIza,
Hugging Face hf_, GitHub github_pat_, Composio uak_/ak_/ck_ и закрытый ключ PEM блоком целиком (VENDOR_KEY_RE из
memcore.types).

Рецепт: re.findall(VENDOR_KEY_RE, text) по всему тексту, в порядке появления.

node: leaf
"""
import re

import deal

from memcore.types import VENDOR_KEY_RE


@deal.ensure(lambda text, result: all(v != '' and v in text for v in result))
@deal.has()
def vendor_values(text: str) -> tuple[str, ...]:
    """Значения секретов в форматах поставщиков (VENDOR_KEY_RE), в порядке появления в тексте. Хэш коммита и обычный
    текст ничего не дают.

    D: 1

    >>> vendor_values("key = sk-ant-api03-Xq7Lm2Pz9Rt4Vw8Ks3Nd6Fh1Jb5Gc0YaUe")
    ('sk-ant-api03-Xq7Lm2Pz9Rt4Vw8Ks3Nd6Fh1Jb5Gc0YaUe',)
    >>> vendor_values("token ghp_Xq7Lm2Pz9Rt4Vw8Ks3Nd6Fh1Jb5Gc0YaUe12")
    ()
    >>> vendor_values("commit b7b8b53c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a")
    ()
    >>> vendor_values("hf_abcdefghijklmnopqrst and hf_ABCDEFGHIJKLMNOPQRST")
    ('hf_abcdefghijklmnopqrst', 'hf_ABCDEFGHIJKLMNOPQRST')
    >>> vendor_values("-----BEGIN PRIVATE KEY----- MIIB -----END PRIVATE KEY-----")
    ('-----BEGIN PRIVATE KEY----- MIIB -----END PRIVATE KEY-----',)
    """
    return tuple(re.findall(VENDOR_KEY_RE, text))
