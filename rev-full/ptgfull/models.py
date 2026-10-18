"""Любые эмбеддеры: модели берёт реестр LanceDB (lancedb.embeddings.get_registry), 'hash' — проверочный эмбеддер.

Модель поставщика: get_registry().get(space.provider).create(name=space.model); векторы —
compute_source_embeddings(list(texts)). Ключи и адреса серверов — только из окружения.

node: leaf
effects: io
"""
import deal
from lancedb.embeddings import get_registry

from ptgfull.spaces import MAX_DIM, hash_embed
from ptgfull.types import Space


@deal.ensure(lambda space, result: result or space.provider != "hash")
@deal.has("io", "network")
def can_embed(space: Space) -> bool:
    """Может ли модель пространства кодировать сейчас: 'hash' — всегда; иначе — поставщик есть в реестре, модель
    создаётся и её размерность (ndims()) равна space.dim; любая ошибка — False.

    D: 3

    >>> can_embed(Space('h', 'hash', 'words', 4))
    True
    >>> can_embed(Space('x', 'нет-такого-поставщика', 'm', 4))
    False
    """
    if space.provider == "hash":
        return True
    try:
        model = get_registry().get(space.provider).create(name=space.model)  # type: ignore[no-untyped-call]
        return bool(model.ndims() == space.dim)
    except Exception:
        return False


@deal.pre(lambda space, texts: 1 <= space.dim <= MAX_DIM)
@deal.ensure(lambda space, texts, result: len(result) == len(texts) and all(len(v) == space.dim for v in result))
@deal.raises(ValueError)
@deal.has("io", "network")
def embed(space: Space, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
    """Векторы текстов в пространстве: 'hash' — hash_embed(texts, space.dim); иначе — модель реестра, числа float;
    размерность не та — ValueError.

    D: 3

    >>> embed(Space('h', 'hash', 'words', 4), ('кэш кэш', ''))
    ((0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 0.0))
    >>> embed(Space('h', 'hash', 'words', 4), ())
    ()
    """
    if space.provider == "hash":
        return hash_embed(texts, space.dim)
    if not texts:
        return ()
    try:
        model = get_registry().get(space.provider).create(name=space.model)  # type: ignore[no-untyped-call]
        raw = model.compute_source_embeddings(list(texts))
        result = tuple(tuple(float(x) for x in vector) for vector in raw)
    except Exception as error:
        raise ValueError(f"модель {space.provider}:{space.model} не выдала векторы: {error}") from error
    if len(result) != len(texts) or any(len(vector) != space.dim for vector in result):
        raise ValueError(f"векторы не размерности {space.dim} для {len(texts)} текстов")
    return result
