"""Черновики из прочитанного: как читать файл по имени, пары чата и текст файла → черновики атомов.

node: leaf
"""
import deal
import memshell

from ptgcore.types import Draft


@deal.ensure(lambda path, result: result in ('py', 'chat', 'text'))
@deal.ensure(lambda path, result: (result == 'py') == path.lower().endswith('.py'))
@deal.ensure(lambda path, result: (result == 'chat') == path.lower().endswith(('.json', '.jsonl', '.ndjson')))
@deal.has()
def file_kind(path: str) -> str:
    """ФТ-002. Как читать файл по расширению (регистр не важен): '.py' — 'py'; '.json', '.jsonl', '.ndjson' — 'chat';
    прочие и без расширения — 'text'.

    D: 3

    >>> file_kind('ptgcore/types.py')
    'py'
    >>> file_kind('exports/Chat.JSONL')
    'chat'
    >>> file_kind('docs/note.md')
    'text'
    >>> file_kind('Makefile')
    'text'
    """
    lower = path.lower()
    if lower.endswith('.py'):
        return 'py'
    if lower.endswith(('.json', '.jsonl', '.ndjson')):
        return 'chat'
    return 'text'


@deal.pre(lambda pairs, at: at > 0.0 and all(t >= 0.0 for _, _, t in pairs))
@deal.ensure(lambda pairs, at, result: [d.position for d in result] == list(range(len(pairs))))
@deal.ensure(lambda pairs, at, result: all(d.text == q + chr(10) * 2 + a and d.question == q and d.answer == a
                                          and d.session == '' and d.source == 'file'
                                          for d, (q, a, _) in zip(result, pairs, strict=True)))
@deal.ensure(lambda pairs, at, result: all(d.at == (t if t > 0.0 else at) for d, (_, _, t) in zip(result, pairs, strict=True)))
@deal.has()
def chat_drafts(pairs: tuple[tuple[str, str, float], ...], at: float) -> tuple[Draft, ...]:
    """ФТ-002. Черновик на каждую пару чата «вопрос, ответ, время»: Draft(position = номер с нуля, text = question +
    \"\\n\\n\" + answer, question, answer, at = время пары, а если оно 0.0 — at файла, session = '', source = 'file').

    D: 2

    >>> chat_drafts((('Какой базой?', 'LanceDB.', 1760000000.0), ('А граф?', 'codebase-memory-mcp.', 0.0)), 1760003600.0)
    (Draft(position=0, text='Какой базой?\\n\\nLanceDB.', question='Какой базой?', answer='LanceDB.', at=1760000000.0, session='', source='file'), Draft(position=1, text='А граф?\\n\\ncodebase-memory-mcp.', question='А граф?', answer='codebase-memory-mcp.', at=1760003600.0, session='', source='file'))
    >>> chat_drafts((), 1760003600.0)
    ()
    """
    return tuple(
        Draft(
            position=position,
            text=question + '\n\n' + answer,
            question=question,
            answer=answer,
            at=stamp if stamp > 0.0 else at,
            session='',
            source='file',
        )
        for position, (question, answer, stamp) in enumerate(pairs)
    )


@deal.pre(lambda path, text, limit, at: limit >= 200 and at > 0.0)
@deal.ensure(lambda path, text, limit, at, result: [d.position for d in result] == list(range(len(result))))
@deal.ensure(lambda path, text, limit, at, result: all(d.question == '' and d.answer == d.text and d.at == at
                                                      and d.session == '' and d.source == 'file' for d in result))
@deal.has()
def text_drafts(path: str, text: str, limit: int, at: float) -> tuple[Draft, ...]:
    """ФТ-002. Атомы текста файла: memshell.atomize(path, text, limit) — абзацы; по каждому Draft(position, text,
    question = '', answer = text, at, session = '', source = 'file').

    D: 1

    >>> text_drafts('n.md', 'Первый абзац.\\n\\nВторой абзац.\\n', 1200, 1760000000.0)
    (Draft(position=0, text='Первый абзац.', question='', answer='Первый абзац.', at=1760000000.0, session='', source='file'), Draft(position=1, text='Второй абзац.', question='', answer='Второй абзац.', at=1760000000.0, session='', source='file'))
    >>> text_drafts('n.md', '', 1200, 1760000000.0)
    ()
    >>> text_drafts('n.md', '   ', 1200, 1760000000.0)
    ()
    """
    return tuple(
        Draft(
            position=position,
            text=atom.text,
            question='',
            answer=atom.text,
            at=at,
            session='',
            source='file',
        )
        for position, atom in enumerate(memshell.atomize(path, text, limit))
    )
