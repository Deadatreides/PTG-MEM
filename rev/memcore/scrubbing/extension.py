"""Расширение найденного куска до границ ключа: символы [A-Za-z0-9_-] вокруг куска входят в секрет целиком
(GitHubTokenDetector отдаёт только префикс «ghp», остальной токен — рядом). Символ слова — латинская буква, цифра,
«_» или «-»; кириллица и прочие буквы — не слово.

node: leaf
"""
import deal


@deal.pre(lambda text, start, end: 0 <= start < end <= len(text))
@deal.ensure(lambda text, start, end, result: 0 <= result[0] <= start and end <= result[1] <= len(text))
@deal.ensure(lambda text, start, end, result: all(c.isascii() and (c.isalnum() or c in '_-') for c in text[result[0]:start] + text[end:result[1]]))
@deal.ensure(lambda text, start, end, result: (result[0] == 0 or not (text[result[0] - 1].isascii() and (text[result[0] - 1].isalnum() or text[result[0] - 1] in '_-'))) and (result[1] == len(text) or not (text[result[1]].isascii() and (text[result[1]].isalnum() or text[result[1]] in '_-'))))
@deal.has()
def extend_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Границы куска text[start:end], расширенного влево и вправо, пока рядом символы слова ([A-Za-z0-9_-]).
    Сам кусок не проверяется и не сжимается.

    D: 4

    >>> extend_span("token ghp_Xq7Lm2Pz9Rt4Vw8Ks3Nd6Fh1Jb5Gc0YaUe12", 6, 9)
    (6, 46)
    >>> extend_span('password = "Tr0ub4dor&3-horse-battery"', 12, 37)
    (12, 37)
    >>> extend_span("key = sk-ant-api03-Xq7Lm2Pz9Rt4Vw8Ks3Nd6Fh1Jb5Gc0YaUe", 6, 53)
    (6, 53)
    >>> extend_span("xx-ab-yy zz", 3, 5)
    (0, 8)
    >>> extend_span("ключabc", 4, 7)
    (4, 7)
    """
    def is_word(c: str) -> bool:
        return c.isascii() and (c.isalnum() or c in '_-')

    left = start
    while left > 0 and is_word(text[left - 1]):
        left -= 1
    right = end
    while right < len(text) and is_word(text[right]):
        right += 1
    return (left, right)
