"""Вычистка секретов до записи: черновики источника и текст заметки.

node: leaf
"""
import deal
import memcore

from ptgcore.types import Draft


@deal.ensure(lambda drafts, result: result[1] >= 0 and len(result[0]) == len(drafts))
@deal.ensure(lambda drafts, result: [d.position for d in result[0]] == [d.position for d in drafts]
             and [d.source for d in result[0]] == [d.source for d in drafts])
@deal.ensure(lambda drafts, result: (result[1] == 0) == all(a.text == b.text and a.question == b.question and a.answer == b.answer
                                                           for a, b in zip(drafts, result[0], strict=True)))
@deal.has()
def scrub_drafts(drafts: tuple[Draft, ...]) -> tuple[tuple[Draft, ...], int]:
    """ФТ-003. Черновики после memcore.scrub по text, question и answer (остальные поля — как были) и сумма found по
    всем трём полям всех черновиков.

    D: 2

    >>> scrub_drafts(())
    ((), 0)
    >>> scrub_drafts((Draft(position=0, text='Какой базой храним векторы памяти?\\n\\nХраним в LanceDB: вектор, полнотекст и SQL в одной таблице.\\n\\nTools: [Edit] H:/proj/memcore/filtering.py', question='Какой базой храним векторы памяти?', answer='Храним в LanceDB: вектор, полнотекст и SQL в одной таблице.\\n\\nTools: [Edit] H:/proj/memcore/filtering.py', at=1759996800.0, session='session', source='claude'), Draft(position=1, text='А граф кода откуда берём?\\n\\nГраф кода строит codebase-memory-mcp целиком.', question='А граф кода откуда берём?', answer='Граф кода строит codebase-memory-mcp целиком.', at=1759996920.0, session='session', source='claude')))
    ((Draft(position=0, text='Какой базой храним векторы памяти?\\n\\nХраним в LanceDB: вектор, полнотекст и SQL в одной таблице.\\n\\nTools: [Edit] H:/proj/memcore/filtering.py', question='Какой базой храним векторы памяти?', answer='Храним в LanceDB: вектор, полнотекст и SQL в одной таблице.\\n\\nTools: [Edit] H:/proj/memcore/filtering.py', at=1759996800.0, session='session', source='claude'), Draft(position=1, text='А граф кода откуда берём?\\n\\nГраф кода строит codebase-memory-mcp целиком.', question='А граф кода откуда берём?', answer='Граф кода строит codebase-memory-mcp целиком.', at=1759996920.0, session='session', source='claude')), 0)
    >>> scrub_drafts((Draft(0, 'key = sk-ant-api03-Xq7Lm2Pz9Rt4Vw8Ks3Nd6Fh1Jb5Gc0YaUe', '', 'key = sk-ant-api03-Xq7Lm2Pz9Rt4Vw8Ks3Nd6Fh1Jb5Gc0YaUe', 1760000000.0, '', 'file'),))
    ((Draft(position=0, text='key = <secret>', question='', answer='key = <secret>', at=1760000000.0, session='', source='file'),), 2)
    """
    def scrub_one(draft: Draft) -> tuple[Draft, int]:
        s_text = memcore.scrub(draft.text)
        s_question = memcore.scrub(draft.question)
        s_answer = memcore.scrub(draft.answer)
        clean = Draft(
            position=draft.position,
            text=s_text.text,
            question=s_question.text,
            answer=s_answer.text,
            at=draft.at,
            session=draft.session,
            source=draft.source,
        )
        return clean, s_text.found + s_question.found + s_answer.found

    done = tuple(scrub_one(draft) for draft in drafts)
    return tuple(item for item, _ in done), sum(found for _, found in done)


@deal.pre(lambda text, minimum: minimum >= 1)
@deal.ensure(lambda text, minimum, result: result == '' or len(result) >= minimum)
@deal.has()
def note_text(text: str, minimum: int) -> str:
    """ФТ-003. Текст заметки агента: memcore.scrub(text.strip()).text; короче minimum знаков — ''.

    D: 2

    >>> note_text('  Решение: векторы храним в LanceDB.  ', 8)
    'Решение: векторы храним в LanceDB.'
    >>> note_text('ок', 8)
    ''
    >>> note_text('token ghp_Xq7Lm2Pz9Rt4Vw8Ks3Nd6Fh1Jb5Gc0YaUe12', 8)
    'token <secret>'
    >>> note_text('', 8)
    ''
    """
    cleaned = memcore.scrub(text.strip()).text
    return cleaned if len(cleaned) >= minimum else ''
