"""Где в тексте стоит значение-секрет: начала всех его вхождений, в том числе перекрывающихся (значение с «&» или «-»
внутри может перекрыть само себя). Пустое значение и сама метка REDACTION_MARK секретом не считаются.

node: leaf
"""
import deal

from memcore.types import REDACTION_MARK


@deal.ensure(lambda text, value, result: list(result) == sorted(set(result)))
@deal.ensure(lambda text, value, result: value not in ('', REDACTION_MARK) or result == ())
@deal.ensure(lambda text, value, result: value in ('', REDACTION_MARK) or set(result) == {i for i in range(len(text)) if text.startswith(value, i)})
@deal.has()
def secret_starts(text: str, value: str) -> tuple[int, ...]:
    """Номера символов, с которых значение value входит в text (перекрывающиеся вхождения — тоже), по возрастанию.
    Пустое значение и сама метка REDACTION_MARK — не секрет: вхождений нет.

    D: 4

    >>> secret_starts("token ghp_Xq7Lm2Pz9Rt4Vw8Ks3Nd6Fh1Jb5Gc0YaUe12", "ghp")
    (6,)
    >>> secret_starts('password = "Tr0ub4dor&3-horse-battery"', "Tr0ub4dor&3-horse-battery")
    (12,)
    >>> secret_starts("key = sk-ant-api03-Xq7Lm2Pz9Rt4Vw8Ks3Nd6Fh1Jb5Gc0YaUe", "sk-ant-api03-Xq7Lm2Pz9Rt4Vw8Ks3Nd6Fh1Jb5Gc0YaUe")
    (6,)
    >>> secret_starts("token <secret>", "<secret>")
    ()
    >>> secret_starts("x&x&x", "x&x")
    (0, 2)
    """
    if value == '' or value == REDACTION_MARK:
        return ()
    return tuple(i for i in range(len(text)) if text[i:i + len(value)] == value)
