"""Выбор способа поиска по тому, какие модели загружены и какие столбцы векторов не пусты.

node: leaf
"""
import deal

from memcore.types import Engine, SearchPlan


@deal.pre(lambda engines, preferred: len({e.model for e in engines}) == len(engines) and all(e.model and e.rows >= 0 for e in engines))
@deal.ensure(lambda engines, preferred, result: result.mode in ("vector", "translated", "lexical"))
@deal.ensure(lambda engines, preferred, result: (result.mode == "lexical") == (result.column == "" and result.encoder == ""))
@deal.ensure(lambda engines, preferred, result: result.mode != "vector" or result.column == result.encoder)
@deal.ensure(lambda engines, preferred, result: result.mode == "lexical" or any(e.model == result.encoder and e.loaded for e in engines))
@deal.ensure(lambda engines, preferred, result: result.mode == "lexical" or any(e.model == result.column and e.rows > 0 for e in engines))
@deal.has()
def choose_search(engines: list[Engine], preferred: str) -> SearchPlan:
    """Как искать по запросу. Порядок выбора:
    1) модель preferred загружена и её столбец не пуст — vector по столбцу preferred;
    2) столбец preferred не пуст, модель не загружена, а у загруженной модели есть адаптер к preferred — translated:
       столбец preferred, запрос кодирует эта модель (первая такая по порядку engines);
    3) любая загруженная модель с непустым столбцом (первая по порядку) — vector по её столбцу;
    4) иначе — lexical (BM25), column и encoder пустые.

    D: 6

    >>> choose_search([Engine("qwen3-0.6b", True, 120, ())], "qwen3-0.6b")
    SearchPlan(mode='vector', column='qwen3-0.6b', encoder='qwen3-0.6b')
    >>> choose_search([Engine("qwen3-4b", False, 120, ()), Engine("qwen3-0.6b", True, 120, ("qwen3-4b",))], "qwen3-4b")
    SearchPlan(mode='translated', column='qwen3-4b', encoder='qwen3-0.6b')
    >>> choose_search([Engine("qwen3-4b", False, 120, ()), Engine("jasper", True, 80, ())], "qwen3-4b")
    SearchPlan(mode='vector', column='jasper', encoder='jasper')
    >>> choose_search([Engine("qwen3-4b", False, 120, ())], "qwen3-4b")
    SearchPlan(mode='lexical', column='', encoder='')
    >>> choose_search([Engine("qwen3-4b", True, 0, ())], "qwen3-4b")
    SearchPlan(mode='lexical', column='', encoder='')
    """
    own = [e for e in engines if e.model == preferred]
    if any(e.loaded and e.rows > 0 for e in own):
        return SearchPlan(mode="vector", column=preferred, encoder=preferred)
    if any(e.rows > 0 for e in own):
        for e in engines:
            if e.loaded and preferred in e.adapters:
                return SearchPlan(mode="translated", column=preferred, encoder=e.model)
    for e in engines:
        if e.loaded and e.rows > 0:
            return SearchPlan(mode="vector", column=e.model, encoder=e.model)
    return SearchPlan(mode="lexical", column="", encoder="")
