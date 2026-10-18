"""Ядро PTG-MEM: полный PTG для демона — все операции движка над папкой базы home.

Текст, сеансы Claude Code и Codex, экспорты чатов, заметки агента; связи со сроком действия; поиск RRF по словам и
любым пространствам эмбеддингов; подсказка и сводка в бюджете; лента решений; граф кода (codebase-memory-mcp) и
метадерево; перенос базы формата 6 без пересчёта. Векторы даёт демон: query_routes говорит, какие нужны.

node: composite
effects: io
"""
import re

import deal
import ptgfull.spaces
from memcore.types import Edge
from ptgfull.types import Route

from ptgcore import codegraph, context, intake, ledger, migration, relations, retrieval
from ptgcore.types import Batch, Card, Decision, Draft, Entry, Graph, Node, Plan, Recall, Report, Settings, Source, Stats

LIMIT = 1200                 # атом — абзац; длиннее режет Chonkie (замер chunk_locality)
WINDOW = 0.12                # окно от лучшей находки в подсказке (PTG-MEM 0.2)
NOTES = "memory://notes"
CHUNK = 2000                 # векторов за запись при переносе
SPEC = ptgfull.spaces.SPEC

Vectors = tuple[tuple[str, tuple[float, ...]], ...]


@deal.pre(lambda home, spec: home != "" and re.fullmatch(SPEC, spec) is not None)
@deal.ensure(lambda home, spec, result: spec.split("=", 1)[0] in result.spaces)
@deal.raises(ValueError)
@deal.has("io")
def open_home(home: str, spec: str) -> Stats:
    """P5, ФТ-015. Открыть базу с пространством spec («имя=поставщик:модель@размерность»); то же имя с другой
    моделью — ValueError.

    D: 1

    >>> open_home('core_open', 'h=hash:words@4')
    Stats(atoms=0, live=0, retracted=0, superseded=0, relations=0, files=0, code=0, edges=0, spaces=('h',))
    """
    retrieval.register(home, ptgfull.spaces.parse_space(spec))
    return stats(home)


@deal.pre(lambda home, source, lang: home != "" and source.kind in intake.KINDS and lang in ("ru", "en"))
@deal.ensure(lambda home, source, lang, result: result.file == source.path)
@deal.has("io")
def prepare(home: str, source: Source, lang: str) -> Batch:
    """P1, ФТ-001…ФТ-005. План приёма источника: черновики против записанного; источника нет — снять все его живые
    атомы.

    D: 2

    >>> prepare('core_prepare', Source('tests/fixtures/session.jsonl', 'claude'), 'ru')
    Batch(file='tests/fixtures/session.jsonl', fresh=(Entry(id='56b7bc2533b494e7', file='tests/fixtures/session.jsonl', position=0, digest='45d7469e15a9ed8c', text='Какой базой храним векторы памяти?\\n\\nХраним в LanceDB: вектор, полнотекст и SQL в одной таблице.\\n\\nTools: [Edit] H:/proj/memcore/filtering.py', question='Какой базой храним векторы памяти?', answer='Храним в LanceDB: вектор, полнотекст и SQL в одной таблице.\\n\\nTools: [Edit] H:/proj/memcore/filtering.py', at=1759996800.0, status='active', retracted_at=0.0, session='session', source='claude'), Entry(id='788f8df1ac20deb7', file='tests/fixtures/session.jsonl', position=1, digest='465aa9f6a0ae35a0', text='А граф кода откуда берём?\\n\\nГраф кода строит codebase-memory-mcp целиком.', question='А граф кода откуда берём?', answer='Граф кода строит codebase-memory-mcp целиком.', at=1759996920.0, status='active', retracted_at=0.0, session='session', source='claude')), retract=(), revises=(), kept=0, secrets=0)
    >>> prepare('core_prepare_gone', Source('нет-такого.md', 'file'), 'ru')
    Batch(file='нет-такого.md', fresh=(), retract=(), revises=(), kept=0, secrets=0)
    """
    try:
        drafts, secrets = intake.drafts(source, lang, LIMIT)
    except FileNotFoundError:
        drafts, secrets = (), 0
    return ledger.plan_batch(source.path, ledger.stored(home, source.path), drafts, secrets)


@deal.pre(lambda home, batch, space, vectors, at: home != "" and at > 0.0 and len(vectors) in (0, len(batch.fresh))
          and (not vectors or space != ""))
@deal.ensure(lambda home, batch, space, vectors, at, result: result.file == batch.file and result.added == len(batch.fresh)
             and result.kept == batch.kept)
@deal.has("io")
def commit(home: str, batch: Batch, space: str, vectors: tuple[tuple[float, ...], ...], at: float) -> Report:
    """P1, ФТ-005. Записать план: новые атомы и их векторы (vectors = () — эмбеддера нет), снять исчезнувшие, связать
    «новый заменяет старый» (supersedes, источник 'edit'), отметить время правки источника.

    D: 1

    >>> commit('core_commit', Batch('n.md', (Entry('a1', 'n.md', 0, '0b00d70ea64ce933', 'LanceDB выбран.', '', 'LanceDB выбран.', 1760000000.0, 'active', 0.0, '', 'file'),), (), (), 0, 0), 'h', ((1.0, 0.0, 0.0, 0.0),), 1760000000.0)
    Report(file='n.md', added=1, retracted=0, kept=0)
    """
    ledger.write(home, batch.fresh)
    retrieval.index(home, batch.fresh, space, vectors)
    gone = ledger.retract(home, batch.retract, at)
    relations.record_pairs(home, batch.revises, "supersedes", at, "edit")
    ledger.mark(home, batch.file, intake.mtime(batch.file))
    return Report(batch.file, len(batch.fresh), gone, batch.kept)


@deal.pre(lambda home, path, at: home != "" and path != "" and at > 0.0)
@deal.ensure(lambda home, path, at, result: result.file == path and result.added == 0 and result.kept == 0)
@deal.has("io")
def retract_file(home: str, path: str, at: float) -> Report:
    """P1. Источник удалён: снять его живые атомы (база их хранит) и забыть время правки.

    D: 1

    >>> retract_file('core_retract', 'n.md', 1760000000.0)
    Report(file='n.md', added=0, retracted=0, kept=0)
    """
    live = tuple(e.id for e in ledger.stored(home, path) if e.status == "active")
    gone = ledger.retract(home, live, at)
    ledger.mark(home, path, 0.0)
    return Report(path, 0, gone, 0)


@deal.ensure(lambda text, result: result == "" or len(result) >= intake.MIN_NOTE)
@deal.has()
def note_text(text: str) -> str:
    """P1. Текст заметки после вычистки секретов; слишком короткая — ''. Демон кодирует ровно этот текст.

    D: 1

    >>> note_text('Решение: векторы храним в LanceDB.')
    'Решение: векторы храним в LanceDB.'
    """
    return intake.note(text)


@deal.pre(lambda home, text, session, source, space, vector, at, root: home != "" and text != "" and note_text(text) == text
          and at > 0.0 and (not vector or space != ""))
@deal.ensure(lambda home, text, session, source, space, vector, at, root, result: result.text == text[:700]
             and result.status == "active" and result.file == NOTES)
@deal.has("io")
def remember(home: str, text: str, session: str, source: str, space: str, vector: tuple[float, ...], at: float, root: str) -> Card:
    """P1. Заметка или решение агента — сразу в базу (источник memory://notes); vector = () — эмбеддера нет.

    D: 1

    >>> remember('core_remember', 'Решение: векторы храним в LanceDB.', '', 'agent', 'h', (1.0, 0.0, 0.0, 0.0), 1760000000.0, '.')
    Card(id='1faee2181895166d', score=None, status='active', date='2025-10-09', file='memory://notes', rel_file='memory://notes', question='Note (agent)', text='Решение: векторы храним в LanceDB.', chars=34, links=(), current='', session='')
    """
    position = len(ledger.stored(home, NOTES))
    entry = ledger.entry(NOTES, Draft(position, text, f"Note ({source})", text, at, session, source))
    ledger.write(home, (entry,))
    retrieval.index(home, (entry,), space, (vector,) if vector else ())
    return retrieval.card(entry, None, (), (), root, 700)


@deal.pre(lambda home, src, dst, kind, source, at: home != "" and src != dst and kind in relations.TYPE_ORDER
          and source != "" and at > 0.0)
@deal.ensure(lambda home, src, dst, kind, source, at, result: all({e.src, e.dst} == {src, dst} for e in result))
@deal.raises(KeyError)
@deal.has("io")
def relate(home: str, src: str, dst: str, kind: str, source: str, at: float) -> tuple[Edge, ...]:
    """P3. Связь, которую заявил агент или модель (не догадка по косинусу); неизвестный атом — KeyError.

    D: 1

    >>> relate('core_relate', 'a2', 'a1', 'supersedes', 'agent', 1760000000.0)
    Traceback (most recent call last):
    ...
    KeyError: 'неизвестный атом'
    """
    if len(ledger.entries(home, (src, dst))) < 2:
        raise KeyError("неизвестный атом")
    return relations.record(home, src, dst, kind, at, source)


@deal.pre(lambda home, settings: home != "")
@deal.ensure(lambda home, settings, result: not {s.path for s in result.todo} & set(result.gone))
@deal.has("io")
def plan(home: str, settings: Settings) -> Plan:
    """P1. Что устарело: изменённые и новые источники проекта, исчезнувшие обработанные пути.

    D: 1

    >>> plan('core_plan', Settings('нет-такой-папки', 'core_plan', True, False, False, (), (), 'ru'))
    Plan(todo=(), gone=())
    """
    todo, unseen = ledger.outdated(intake.sources(settings), ledger.processed(home))
    return Plan(todo, intake.missing(unseen))


@deal.pre(lambda home: home != "")
@deal.has("io")
def save(home: str) -> Stats:
    """P4, ФТ-013. Пересобрать метадерево по живым атомам и графу кода, сжать таблицы.

    D: 1

    >>> save('core_save')
    Stats(atoms=0, live=0, retracted=0, superseded=0, relations=0, files=0, code=0, edges=0, spaces=())
    """
    live = ledger.entries(home, ledger.live_ids(home)) if ledger.live_ids(home) else ()
    codegraph.relink(home, live)
    ledger.compact(home)
    return stats(home)


@deal.pre(lambda home, ready: home != "")
@deal.ensure(lambda home, ready, result: all(r.source in ready for r in result))
@deal.has("io")
def query_routes(home: str, ready: tuple[str, ...]) -> tuple[Route, ...]:
    """P5. Какими моделями закодировать запрос: путь в каждое пространство базы (свой вектор или через адаптер).

    D: 1

    >>> query_routes('core_routes', ('h',))
    ()
    """
    return retrieval.routes(home, ready)


@deal.pre(lambda home, query, vectors, k, exclude_ids, with_retracted, root: home != "" and k >= 1)
@deal.ensure(lambda home, query, vectors, k, exclude_ids, with_retracted, root, result: len(result) <= k)
@deal.has("io")
def search(home: str, query: str, vectors: Vectors, k: int, exclude_ids: tuple[str, ...], with_retracted: bool,
           root: str) -> tuple[Card, ...]:
    """P2, ФТ-007. Поиск: RRF по словам и пространствам, карточки со связями.

    D: 1

    >>> search('core_search', 'кэш', (), 5, (), False, '.')
    ()
    """
    hits = retrieval.ranked(home, query, vectors, max(8 * k, 60))
    found = ledger.entries(home, tuple(h.id for h in hits))
    edges = relations.claims(home)
    others = ledger.entries(home, relations.ends(edges, tuple(e.id for e in found)))
    return retrieval.pick(hits, found, edges, others, k, exclude_ids, with_retracted, root)


@deal.pre(lambda home, prompt, vectors, session_id, transcript_path, budget, max_items, min_score, root:
          home != "" and budget >= 0 and max_items >= 0)
@deal.ensure(lambda home, prompt, vectors, session_id, transcript_path, budget, max_items, min_score, root, result:
             len(result.context) <= budget and len(result.items) <= max_items)
@deal.has("io")
def recall(home: str, prompt: str, vectors: Vectors, session_id: str, transcript_path: str, budget: int, max_items: int,
           min_score: float, root: str) -> Recall:
    """P2, ФТ-008. Подсказка к запросу: порядок — RRF по словам и пространствам, пропуск — по косинусу (context.admit,
    замер 13.10). Короче 12 знаков, нет векторов запроса или ничего не прошло порог — пусто; top — лучший косинус.

    D: 4

    >>> recall('core_recall', 'как устроен кэш эмбеддингов', (), '', '', 3600, 5, 0.5, '.')
    Recall(context='', items=(), top=0.0)
    """
    if len(prompt.strip()) < 12:
        return Recall("", (), 0.0)
    n = max(24 * max_items, 60)
    hits, near, words = retrieval.evidence(home, prompt, vectors, n)
    found = ledger.entries(home, tuple(h.id for h in hits))
    files = context.session_files(tuple(e.file for e in found), session_id, transcript_path)
    allowed = context.eligible(found, files, intake.compact_time(transcript_path) if transcript_path else 0.0)
    edges = relations.claims(home)
    others = ledger.entries(home, relations.ends(edges, tuple(e.id for e in allowed)))
    cards = retrieval.pick(hits, allowed, edges, others, n, (), False, root)
    chosen = context.admit(cards, near, words, min_score, WINDOW, max_items)
    later = ledger.entries(home, tuple(c.current for c in chosen if c.current))
    text, shown = context.render(chosen, later, codegraph.code_links(home, tuple(c.id for c in chosen)), budget)
    return Recall(text, chosen[:shown], near[0][1] if near else 0.0)


@deal.pre(lambda home, now, max_chars: home != "" and max_chars >= 0)
@deal.has("io")
def brief(home: str, now: float, max_chars: int) -> str:
    """P2, ФТ-009. Сводка начала сеанса: сколько живых атомов, заметки агента за 14 дней.

    D: 1

    >>> brief('core_brief', 1760000000.0, 1600)
    '<ptg-memory>\\nPTG-MEM is active for this project: 0 memory atoms. Relevant memory is attached to each prompt automatically. Tools: ptg_search, ptg_node, ptg_decisions, ptg_remember (store a decision the moment it is made).\\n</ptg-memory>'
    """
    _, live, _ = ledger.counts(home)
    notes = tuple(e for e in ledger.stored(home, NOTES) if e.status == "active")
    return context.brief(live, context.recent_notes(notes, now, 14, 5), max_chars)


@deal.pre(lambda home, limit, root: home != "" and limit >= 0)
@deal.ensure(lambda home, limit, root, result: len(result) <= limit)
@deal.has("io")
def decisions(home: str, limit: int, root: str) -> tuple[Decision, ...]:
    """P3, ФТ-010. Лента решений: что заменило что и что чему противоречит, новые сверху.

    D: 1

    >>> decisions('core_decisions', 50, '.')
    ()
    """
    edges = relations.claims(home)
    return relations.decisions(edges, ledger.entries(home, relations.ends(edges, ())), root, limit)


@deal.pre(lambda home, atom, root: home != "")
@deal.ensure(lambda home, atom, root, result: result is None or result.card.id == atom)
@deal.has("io")
def node(home: str, atom: str, root: str) -> Node | None:
    """P3. Атом целиком: полный текст, ответ, связи, соседи по файлу; неизвестный — None.

    D: 2

    >>> node('core_node', 'нет', '.')
    """
    found = ledger.entries(home, (atom,))
    if not found:
        return None
    entry = found[0]
    edges = relations.claims(home)
    others = ledger.entries(home, relations.ends(edges, (atom,)))
    card = retrieval.card(entry, None, edges, others, root, 20000)
    return Node(card, entry.answer, relations.lineage(entry, ledger.stored(home, entry.file), edges, root))


@deal.pre(lambda home, atom, depth, limit, root: home != "" and depth >= 1 and limit >= 1)
@deal.ensure(lambda home, atom, depth, limit, root, result: len(result.nodes) <= limit)
@deal.has("io")
def graph(home: str, atom: str, depth: int, limit: int, root: str) -> Graph:
    """P3, ФТ-011. Эго-сеть атома для окна: связи до глубины depth, затем рёбра метадерева; неизвестный — пусто.

    D: 2

    >>> graph('core_graph', 'нет', 2, 60, '.')
    Graph(nodes=(), edges=())
    """
    if not ledger.entries(home, (atom,)):
        return Graph((), ())
    edges = relations.claims(home)
    near = ledger.entries(home, relations.reach(atom, edges, depth, limit))
    own = relations.ego(atom, edges, near, root)
    return relations.merge(own, codegraph.around(home, atom, limit), limit)


@deal.pre(lambda home: home != "")
@deal.ensure(lambda home, result: result.atoms == result.live + result.retracted and result.superseded <= result.live)
@deal.has("io")
def stats(home: str) -> Stats:
    """Состояние базы.

    D: 1

    >>> stats('core_stats')
    Stats(atoms=0, live=0, retracted=0, superseded=0, relations=0, files=0, code=0, edges=0, spaces=())
    """
    atoms, live, retracted = ledger.counts(home)
    edges = relations.claims(home)
    code, tree = codegraph.counts(home)
    return Stats(atoms, live, retracted, relations.superseded(edges, ledger.live_ids(home)), len(edges),
                 len(ledger.processed(home)), code, tree, retrieval.spaces(home))


@deal.pre(lambda home, repos, binary: home != "" and len(set(repos)) == len(repos))
@deal.ensure(lambda home, repos, binary, result: result >= 0)
@deal.raises(FileNotFoundError, RuntimeError)
@deal.has("io")
def index_code(home: str, repos: tuple[str, ...], binary: str) -> int:
    """P4, ФТ-012. Граф кода выбранных репозиториев (codebase-memory-mcp); итог — символов. Швы — при save.

    D: 1

    >>> index_code('core_code', (), 'нет-такой-программы.exe')
    0
    """
    return codegraph.index(home, repos, binary)


@deal.pre(lambda home, store_dir, spec: home != "" and re.fullmatch(SPEC, spec) is not None)
@deal.ensure(lambda home, store_dir, spec, result: result.file == store_dir)
@deal.raises(FileNotFoundError, ValueError)
@deal.has("io")
def migrate_v6(home: str, store_dir: str, spec: str) -> Report:
    """P5, ФТ-014. Перенести живую базу формата 6 без пересчёта векторов.

    D: 1

    >>> migrate_v6('core_migrate', 'tests/fixtures/store_v6', 'v6=gguf:Qwen3-Embedding-4B-Q4_K_M@4')
    Report(file='tests/fixtures/store_v6', added=2, retracted=1, kept=0)
    """
    return migration.import_v6(home, store_dir, ptgfull.spaces.parse_space(spec), CHUNK)
