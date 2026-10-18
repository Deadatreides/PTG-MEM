"""memcore — ядро решений памяти PTG-MEM 0.3: что сохранить при правке файла, каким способом искать, что подать
в подсказку, как записать связь «что чем заменено», какие файлы брать и как вычистить секреты.

Ввод-вывод (хранилище LanceDB, эмбеддер, MarkItDown, демон) — тонкая оболочка PTG-MEM вокруг готовых библиотек;
сюда не входит. Спецификация — spec.md.

node: composite
"""
import math

import deal

from memcore import context, filtering, relations, scrubbing, search, sync
from memcore.types import (
    RELATION_KINDS,
    Admission,
    Atom,
    Budget,
    Claim,
    Edge,
    Engine,
    Hit,
    Item,
    Rules,
    Scrubbed,
    SearchPlan,
    Stored,
    SyncPlan,
)


@deal.pre(lambda stored, fresh: len({s.id for s in stored}) == len(stored) and len({a.id for a in fresh}) == len(fresh))
@deal.ensure(lambda stored, fresh, result: len(result.add) + len(result.keep) == len(fresh))
@deal.ensure(lambda stored, fresh, result: set(result.keep) | set(result.retract) == {s.id for s in stored if s.live})
@deal.ensure(lambda stored, fresh, result: not set(result.keep) & set(result.retract))
@deal.has()
def plan_sync(stored: list[Stored], fresh: list[Atom]) -> SyncPlan:
    """ФТ-001, ФТ-002. Файл перечитан: неизменённые атомы остаются, исчезнувшие снимаются, новый на месте
    снятого связан с прежним (revises).

    D: 1

    >>> plan_sync([Stored("s1", "plan.md", 0, "d1", True), Stored("s2", "plan.md", 1, "d2", True)], [Atom("n1", "plan.md", 0, "d1"), Atom("n2", "plan.md", 1, "d3")])
    SyncPlan(add=('n2',), keep=('s1',), retract=('s2',), revises=(('n2', 's2'),))
    >>> plan_sync([Stored("s1", "plan.md", 0, "d1", True)], [])
    SyncPlan(add=(), keep=(), retract=('s1',), revises=())
    """
    return sync.plan_sync(stored, fresh)


@deal.pre(lambda engines, preferred: len({e.model for e in engines}) == len(engines) and all(e.model and e.rows >= 0 for e in engines))
@deal.ensure(lambda engines, preferred, result: (result.mode == "lexical") == (result.column == "" and result.encoder == ""))
@deal.ensure(lambda engines, preferred, result: result.mode == "lexical" or any(e.model == result.encoder and e.loaded for e in engines))
@deal.has()
def choose_search(engines: list[Engine], preferred: str) -> SearchPlan:
    """ФТ-003, ФТ-004. Поиск работает и без эмбеддера: вектор нужного столбца, перевод запроса адаптером,
    другой загруженный столбец или слова (BM25).

    D: 1

    >>> choose_search([Engine("qwen3-0.6b", False, 120, ())], "qwen3-0.6b")
    SearchPlan(mode='lexical', column='', encoder='')
    >>> choose_search([Engine("qwen3-4b", False, 120, ()), Engine("qwen3-0.6b", True, 120, ("qwen3-4b",))], "qwen3-4b")
    SearchPlan(mode='translated', column='qwen3-4b', encoder='qwen3-0.6b')
    """
    return search.choose_search(engines, preferred)


@deal.pre(lambda hits, budget: budget.chars >= 0 and budget.items >= 0 and budget.per_file >= 0)
@deal.ensure(lambda hits, budget, result: len(result) <= budget.items)
@deal.ensure(lambda hits, budget, result: sum(len(i.text) + len(i.note) for i in result) <= budget.chars)
@deal.ensure(lambda hits, budget, result: len({i.id for i in result}) == len(result))
@deal.ensure(lambda hits, budget, result: {i.id for i in result} <= {h.id for h in hits if h.score >= budget.min_score})
@deal.has()
def select_context(hits: list[Hit], budget: Budget) -> list[Item]:
    """ФТ-005, ФТ-006. Подсказка не раздувается и не врёт: в бюджете, без повторов, не ниже порога; у заменённого —
    пометка преемника.

    D: 1

    >>> select_context([Hit("a", "plan.md", "старый план", 0.9, "b"), Hit("b", "plan.md", "новый план", 0.8, "")], Budget(100, 5, 0.5, 3))
    [Item(id='a', text='старый план', note='заменено: b'), Item(id='b', text='новый план', note='')]
    >>> select_context([Hit("a", "plan.md", "старый план", 0.9, "b"), Hit("b", "plan.md", "новый план", 0.8, "")], Budget(25, 5, 0.5, 3))
    [Item(id='a', text='старый план', note='заменено: b')]
    """
    return context.select_context(hits, budget)


@deal.pre(lambda edges, claim: claim.at > 0 and len({(e.src, e.dst) for e in edges if e.invalid_at == 0.0}) == sum(1 for e in edges if e.invalid_at == 0.0))
@deal.pre(lambda edges, claim: math.isfinite(claim.at) and all(math.isfinite(e.valid_at) and math.isfinite(e.invalid_at) for e in edges))
@deal.ensure(lambda edges, claim, result: len({(e.src, e.dst) for e in result if e.invalid_at == 0.0}) == sum(1 for e in result if e.invalid_at == 0.0))
@deal.ensure(lambda edges, claim, result: len(result) >= len(edges))
@deal.has()
def record_relation(edges: list[Edge], claim: Claim) -> list[Edge]:
    """ФТ-007, ФТ-008. Связь другого вида закрывает прежнюю сроком, старое не удаляется; на пару — одна действующая.

    D: 1

    >>> record_relation([Edge("b", "a", "refines", 5.0, 0.0, "agent")], Claim("b", "a", "supersedes", 10.0, "agent"))
    [Edge(src='b', dst='a', kind='refines', valid_at=5.0, invalid_at=10.0, source='agent'), Edge(src='b', dst='a', kind='supersedes', valid_at=10.0, invalid_at=0.0, source='agent')]
    >>> record_relation([], Claim("a", "a", "supersedes", 10.0, "agent"))
    []
    """
    return relations.record_relation(edges, claim)


@deal.pre(lambda atom, edges: all(math.isfinite(e.valid_at) and math.isfinite(e.invalid_at) for e in edges))
@deal.ensure(lambda atom, edges, result: result == atom or any(e.src == result and e.kind == "supersedes" and e.invalid_at == 0.0 for e in edges))
@deal.has()
def current_version(atom: str, edges: list[Edge]) -> str:
    """История P1. Действующая версия атома по цепочке замен; цикл замен не зацикливает.

    D: 1

    >>> current_version("a", [Edge("b", "a", "supersedes", 5.0, 0.0, "agent"), Edge("c", "b", "supersedes", 7.0, 0.0, "agent")])
    'c'
    >>> current_version("a", [Edge("b", "a", "supersedes", 5.0, 0.0, "agent"), Edge("a", "b", "supersedes", 6.0, 0.0, "agent")])
    'b'
    """
    return relations.current_version(atom, edges)


@deal.pre(lambda path, size, rules: size >= 0 and rules.max_bytes >= 0)
@deal.ensure(lambda path, size, rules, result: result.ok == (result.reason == ""))
@deal.ensure(lambda path, size, rules, result: not result.ok or size <= rules.max_bytes)
@deal.has()
def admit(path: str, size: int, rules: Rules) -> Admission:
    """ФТ-009. В память не попадает мусор: исключения, предел размера, включения — с причиной отказа.

    D: 1

    >>> admit("node_modules/lib/readme.md", 300, Rules(("*.md",), ("node_modules/",), 100000))
    Admission(ok=False, reason='исключено')
    >>> admit("docs/plan.md", 300, Rules(("*.md",), ("node_modules/",), 100000))
    Admission(ok=True, reason='')
    """
    return filtering.admit(path, size, rules)


@deal.ensure(lambda text, result: (result.found == 0) == (result.text == text))
@deal.has()
def scrub(text: str) -> Scrubbed:
    """ФТ-010. Секреты вычищаются до записи; хэш коммита секретом не считается.

    D: 1

    >>> scrub('db password = "Tr0ub4dor&3-horse-battery"')
    Scrubbed(text='db password = "<secret>"', found=1)
    >>> scrub("commit b7b8b53c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a")
    Scrubbed(text='commit b7b8b53c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a', found=0)
    """
    return scrubbing.scrub(text)
