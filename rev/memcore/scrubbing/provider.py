"""Значения секретов, которые detect-secrets 1.5 (Yelp) находит в тексте построчно: ключи поставщиков и пароли в
присваиваниях.

Рецепт: внутри `with transient_settings({"plugins_used": PLUGINS}):` по каждой строке текста (str.splitlines) вызвать
`scan_line(line)` (detect_secrets.core.scan, detect_secrets.settings); значение — secret_value, пустые (и None) — мимо.
PLUGINS — [{"name": имя}] всех детекторов из get_mapping_from_secret_type_to_class()
(detect_secrets.core.plugins.util; имя — __name__ класса), кроме HexHighEntropyString и Base64HighEntropyString
(в разовом режиме scan_line они метят любое латинское слово, в том числе хэши коммитов) и IPPublicDetector
(адрес в заметке — не секрет). Значения без повторов, по возрастанию.

node: leaf
"""
import deal
from detect_secrets.core.plugins.util import get_mapping_from_secret_type_to_class
from detect_secrets.core.scan import scan_line
from detect_secrets.settings import transient_settings


@deal.ensure(lambda text, result: all(v != '' for v in result))
@deal.ensure(lambda text, result: list(result) == sorted(set(result)))
@deal.has()
def provider_values(text: str) -> tuple[str, ...]:
    """Значения секретов, найденных detect-secrets по строкам текста: без пустых и повторов, по возрастанию.
    Хэш коммита и обычный текст ничего не дают; GitHub-токен detect-secrets отдаёт лишь префиксом «ghp».

    D: 6

    >>> provider_values("token ghp_Xq7Lm2Pz9Rt4Vw8Ks3Nd6Fh1Jb5Gc0YaUe12")
    ('ghp',)
    >>> provider_values('password = "Tr0ub4dor&3-horse-battery"')
    ('Tr0ub4dor&3-horse-battery',)
    >>> provider_values("commit b7b8b53c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a")
    ()
    >>> provider_values("")
    ()
    """
    excluded: set[str] = {"HexHighEntropyString", "Base64HighEntropyString", "IPPublicDetector"}
    plugins: list[dict[str, str]] = []
    cls: type
    for cls in get_mapping_from_secret_type_to_class().values():
        name: str = cls.__name__
        if name not in excluded:
            plugins.append({"name": name})
    found: set[str] = set()
    with transient_settings({"plugins_used": plugins}):
        for line in text.splitlines():
            for secret in scan_line(line):
                value: str | None = secret.secret_value
                if value:
                    found.add(value)
    return tuple(sorted(found))
