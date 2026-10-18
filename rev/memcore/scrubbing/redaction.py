"""Замена кусков текста на метку REDACTION_MARK.

node: leaf
"""
import deal

from memcore.types import REDACTION_MARK


@deal.pre(lambda text, spans: all(0 <= s < e <= len(text) for s, e in spans) and all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False)))
@deal.ensure(lambda text, spans, result: len(result) == len(text) - sum(e - s for s, e in spans) + len(REDACTION_MARK) * len(spans))
@deal.ensure(lambda text, spans, result: spans != () or result == text)
@deal.ensure(lambda text, spans, result: spans == () or result.startswith(text[:spans[0][0]] + REDACTION_MARK))
@deal.ensure(lambda text, spans, result: spans == () or result.endswith(REDACTION_MARK + text[spans[-1][1]:]))
@deal.has()
def redact(text: str, spans: tuple[tuple[int, int], ...]) -> str:
    """Текст, в котором каждый кусок text[начало:конец] заменён на REDACTION_MARK, остальное без изменений. Куски идут
    по возрастанию и не накладываются; без кусков — тот же текст.

    D: 2

    >>> redact("token ghp_Xq7Lm2Pz9Rt4Vw8Ks3Nd6Fh1Jb5Gc0YaUe12", ((6, 46),))
    'token <secret>'
    >>> redact('password = "Tr0ub4dor&3-horse-battery"', ((12, 37),))
    'password = "<secret>"'
    >>> redact("key = sk-ant-api03-Xq7Lm2Pz9Rt4Vw8Ks3Nd6Fh1Jb5Gc0YaUe", ((6, 53),))
    'key = <secret>'
    >>> redact("a b c", ((0, 1), (4, 5)))
    '<secret> b <secret>'
    >>> redact("commit b7b8b53c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a", ())
    'commit b7b8b53c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a'
    """
    parts: list[str] = []
    pos = 0
    for start, end in spans:
        parts.append(text[pos:start])
        parts.append(REDACTION_MARK)
        pos = end
    parts.append(text[pos:])
    return "".join(parts)
