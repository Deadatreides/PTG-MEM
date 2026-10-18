"""Пространства эмбеддингов: разбор строки пространства, пути запроса, слияние мест RRF, проверочный эмбеддер.

node: leaf
"""
import hashlib
import math
import re

import deal

from ptgfull.types import Hit, Route, Space

SPEC = r"[a-z0-9_]+=[a-z0-9-]+:.+@[1-9][0-9]*"
# Наибольшая размерность среди эмбеддеров: 8192 — скрытый слой Llama-3-70B у эмбеддеров LLM2Vec; у Qwen3-Embedding-8B,
# NV-Embed-v2 и e5-mistral — 4096, у OpenAI text-embedding-3-large — 3072.
MAX_DIM = 8192


@deal.pre(lambda spec: re.fullmatch(SPEC, spec) is not None)
@deal.ensure(lambda spec, result: result.name == spec.split("=", 1)[0] and result.dim == int(spec.rsplit("@", 1)[1])
             and result.provider + ":" + result.model + "@" + str(result.dim) == spec.split("=", 1)[1])
@deal.has()
def parse_space(spec: str) -> Space:
    """ФТ-001. Строка «имя=поставщик:модель@размерность» → Space; поставщик — до первого «:», размерность — после
    последнего «@».

    D: 1

    >>> parse_space("qwen=sentence-transformers:Qwen/Qwen3-Embedding-0.6B@1024")
    Space(name='qwen', provider='sentence-transformers', model='Qwen/Qwen3-Embedding-0.6B', dim=1024)
    >>> parse_space("nomic=ollama:nomic-embed-text:v1.5@768")
    Space(name='nomic', provider='ollama', model='nomic-embed-text:v1.5', dim=768)
    >>> parse_space("h=hash:words@16")
    Space(name='h', provider='hash', model='words', dim=16)
    """
    name, rest = spec.split("=", 1)
    provider, tail = rest.split(":", 1)
    model, dim = tail.rsplit("@", 1)
    return Space(name, provider, model, int(dim))


@deal.ensure(lambda stored, ready, adapters, result: len(result) <= len(stored)
             and all(r.space in {s.name for s in stored} and r.source in ready for r in result)
             and all((r.how == "direct") == (r.source == r.space) and r.how in ("direct", "adapter") for r in result)
             and all(r.how == "direct" or (r.source, r.space) in adapters for r in result))
@deal.has()
def plan_routes(stored: tuple[Space, ...], ready: tuple[str, ...], adapters: tuple[tuple[str, str], ...]) -> tuple[Route, ...]:
    """ФТ-002. Путь запроса в каждое пространство базы, по порядку stored: 'direct', если его имя в ready; иначе
    'adapter' от первого (источник, цель) из adapters, где цель — это пространство, а источник есть в ready; иначе
    пространство пропускается. adapters — пары (источник, цель).

    D: 4

    >>> plan_routes((Space('a', 'hash', 'w', 8), Space('b', 'openai', 'm', 8)), ('a',), (('a', 'b'),))
    (Route(space='a', how='direct', source='a'), Route(space='b', how='adapter', source='a'))
    >>> plan_routes((Space('b', 'openai', 'm', 8),), (), (('a', 'b'),))
    ()
    >>> plan_routes((Space('b', 'openai', 'm', 8),), ('a', 'c'), (('x', 'b'), ('c', 'b'), ('a', 'b')))
    (Route(space='b', how='adapter', source='c'),)
    """
    routes: list[Route] = []
    for space in stored:
        if space.name in ready:
            routes.append(Route(space.name, "direct", space.name))
            continue
        for source, target in adapters:
            if target == space.name and source in ready:
                routes.append(Route(space.name, "adapter", source))
                break
    return tuple(routes)


@deal.pre(lambda rankings, k, limit: k >= 1 and limit >= 0)
@deal.ensure(lambda rankings, k, limit, result: len(result) <= limit
             and len({h.id for h in result}) == len(result)
             and all(a.score >= b.score for a, b in zip(result, result[1:], strict=False))
             and {h.id for h in result} <= {h.id for r in rankings for h in r})
@deal.has()
def fuse(rankings: tuple[tuple[Hit, ...], ...], k: int, limit: int) -> tuple[Hit, ...]:
    """ФТ-003. Слияние RRF: оценка id = сумма 1/(k + место) по спискам, место — с 1 по первому появлению id в
    списке; оценка до 6 знаков; порядок — по убыванию оценки, при равенстве — по id; via — из первого списка, где
    id встретился; не больше limit.

    D: 5

    >>> fuse(((Hit('x', 0.9, 'a'), Hit('y', 0.5, 'a')), (Hit('y', 3.0, 'lexical'),)), 60, 5)
    (Hit(id='y', score=0.032522, via='a'), Hit(id='x', score=0.016393, via='a'))
    >>> fuse(((Hit('x', 0.9, 'a'), Hit('y', 0.5, 'a')),), 60, 1)
    (Hit(id='x', score=0.016393, via='a'),)
    >>> fuse(((Hit('b', 1.0, 'a'),), (Hit('a', 1.0, 'c'),)), 60, 5)
    (Hit(id='a', score=0.016393, via='c'), Hit(id='b', score=0.016393, via='a'))
    >>> fuse((), 60, 3)
    ()
    """
    scores: dict[str, float] = {}
    via: dict[str, str] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for place, hit in enumerate(ranking, 1):
            if hit.id in seen:
                continue
            seen.add(hit.id)
            scores[hit.id] = scores.get(hit.id, 0.0) + 1 / (k + place)
            via.setdefault(hit.id, hit.via)
    ranked = sorted(
        ((round(score, 6), item_id) for item_id, score in scores.items()),
        key=lambda pair: (-pair[0], pair[1]),
    )
    return tuple(Hit(item_id, score, via[item_id]) for score, item_id in ranked[:limit])


@deal.pre(lambda texts, dim: 1 <= dim <= MAX_DIM)
@deal.ensure(lambda texts, dim, result: len(result) == len(texts) and all(len(v) == dim for v in result)
             and all(sum(x * x for x in v) == 0 or abs(sum(x * x for x in v) - 1) < 1e-4 for v in result))
@deal.has()
def hash_embed(texts: tuple[str, ...], dim: int) -> tuple[tuple[float, ...], ...]:
    """ФТ-004. Проверочный эмбеддер: слова re.findall(r"\\w+", text.lower()); корзина слова =
    int.from_bytes(hashlib.blake2b(слово.encode(), digest_size=8).digest(), "big") % dim; в корзине — число слов;
    нормировка L2, round(x, 6) + 0.0; пустой текст — нули.

    D: 4

    >>> hash_embed(("кэш кэш",), 4)
    ((0.0, 0.0, 1.0, 0.0),)
    >>> hash_embed(("LanceDB выбран",), 4)
    ((0.707107, 0.707107, 0.0, 0.0),)
    >>> hash_embed(("a b b", ""), 4)
    ((0.0, 0.894427, 0.0, 0.447214), (0.0, 0.0, 0.0, 0.0))
    """
    vectors: list[tuple[float, ...]] = []
    for text in texts:
        counts = [0.0] * dim
        for word in re.findall(r"\w+", text.lower()):
            bucket = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "big") % dim
            counts[bucket] += 1.0
        norm = math.sqrt(sum(c * c for c in counts))
        if norm == 0:
            vectors.append(tuple(0.0 for _ in counts))
        else:
            vectors.append(tuple(round(c / norm, 6) + 0.0 for c in counts))
    return tuple(vectors)
