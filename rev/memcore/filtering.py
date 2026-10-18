"""Отбор файлов в память: шаблоны в синтаксисе .gitignore (pathspec, GitIgnoreSpec) и предел размера.

node: leaf
"""
import re

import deal
import pathspec

from memcore.types import Admission, Rules


@deal.ensure(lambda path, patterns, result: result is False or len(patterns) > 0)
@deal.has()
def matches(path: str, patterns: tuple[str, ...]) -> bool:
    """Подпадает ли путь под шаблоны .gitignore. Обратные косые черты в пути — прямые. Каждый шаблон сначала
    проверяется отдельно: pathspec.GitIgnoreSpec.from_lines([шаблон]); битый (ValueError или re.error) отбрасывается
    и ни с чем не совпадает. Годные шаблоны — вместе, по порядку, одним GitIgnoreSpec (так работают отрицания «!»).

    D: 3

    >>> matches("docs/plan.md", ("*.md",))
    True
    >>> matches("docs/plan.md", ("*.md", "!plan.md"))
    False
    >>> matches("docs/plan.md", ("[z-a]", "!", "*.txt"))
    False
    >>> matches("docs/plan.md", ())
    False
    """
    norm = path.replace("\\", "/")
    good: list[str] = []
    for pattern in patterns:
        try:
            pathspec.GitIgnoreSpec.from_lines([pattern])
        except (ValueError, re.error):
            continue
        good.append(pattern)
    if not good:
        return False
    spec = pathspec.GitIgnoreSpec.from_lines(good)
    return bool(spec.match_file(norm))


@deal.pre(lambda path, size, rules: size >= 0 and rules.max_bytes >= 0)
@deal.ensure(lambda path, size, rules, result: result.ok == (result.reason == ""))
@deal.ensure(lambda path, size, rules, result: result.reason in ("", "исключено", "больше предела", "не входит в правила"))
@deal.ensure(lambda path, size, rules, result: not result.ok or size <= rules.max_bytes)
@deal.has()
def admit(path: str, size: int, rules: Rules) -> Admission:
    """Брать ли файл в память. Проверки по порядку: путь подпадает под rules.exclude — «исключено»; size больше
    rules.max_bytes — «больше предела»; rules.include не пуст и путь под него не подпадает — «не входит в правила»;
    иначе принят с пустой причиной. Совпадение — matches.

    D: 3

    >>> admit("docs/plan.md", 120, Rules(("*.md",), ("node_modules/",), 1000))
    Admission(ok=True, reason='')
    >>> admit("node_modules/x/readme.md", 120, Rules(("*.md",), ("node_modules/",), 1000))
    Admission(ok=False, reason='исключено')
    >>> admit("docs/big.md", 5000, Rules(("*.md",), (), 1000))
    Admission(ok=False, reason='больше предела')
    >>> admit("src/app.py", 120, Rules(("*.md",), (), 1000))
    Admission(ok=False, reason='не входит в правила')
    >>> admit("src/app.py", 120, Rules((), ("[z-a]",), 1000))
    Admission(ok=True, reason='')
    """
    if matches(path, rules.exclude):
        return Admission(ok=False, reason="исключено")
    if size > rules.max_bytes:
        return Admission(ok=False, reason="больше предела")
    if rules.include and not matches(path, rules.include):
        return Admission(ok=False, reason="не входит в правила")
    return Admission(ok=True, reason="")
