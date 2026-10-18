"""Атомы приёма: атом из черновика (id и отпечаток) и план приёма одного источника — сверка memcore.plan_sync.

node: leaf
"""
import deal
import memcore
import memshell

from memcore.types import Atom, Stored
from ptgcore.types import Batch, Draft, Entry


@deal.ensure(lambda file, draft, result: result.file == file and result.position == draft.position
             and result.status == 'active' and result.retracted_at == 0.0 and len(result.id) == 16)
@deal.has()
def entry(file: str, draft: Draft) -> Entry:
    """ФТ-004. Атом из черновика: id = memshell.digest(chr(10).join([file, str(draft.position), repr(draft.at),
    draft.text])) — части через настоящий перевод строки; отпечаток digest = memshell.digest(draft.text); состояние
    'active', retracted_at = 0.0; position, text, question, answer, at, session, source — из черновика, file — аргумент.

    D: 1

    >>> entry('n.md', Draft(0, 'LanceDB выбран.', '', 'LanceDB выбран.', 1760000000.0, '', 'file'))
    Entry(id='2295c768ade7f62e', file='n.md', position=0, digest='0b00d70ea64ce933', text='LanceDB выбран.', question='', answer='LanceDB выбран.', at=1760000000.0, status='active', retracted_at=0.0, session='', source='file')
    """
    atom_id = memshell.digest('\n'.join([file, str(draft.position), repr(draft.at), draft.text]))
    return Entry(
        id=atom_id,
        file=file,
        position=draft.position,
        digest=memshell.digest(draft.text),
        text=draft.text,
        question=draft.question,
        answer=draft.answer,
        at=draft.at,
        status='active',
        retracted_at=0.0,
        session=draft.session,
        source=draft.source,
    )


@deal.pre(lambda file, stored, drafts, secrets: secrets >= 0 and all(e.file == file for e in stored))
@deal.ensure(lambda file, stored, drafts, secrets, result: result.file == file and result.secrets == secrets
             and set(result.retract) <= {e.id for e in stored if e.status == 'active'}
             and all(n in {f.id for f in result.fresh} and o in result.retract for n, o in result.revises)
             and result.kept + len(result.retract) == sum(e.status == 'active' for e in stored))
@deal.has()
def plan_batch(file: str, stored: tuple[Entry, ...], drafts: tuple[Draft, ...], secrets: int) -> Batch:
    """ФТ-005. План приёма. Атомы черновиков — entry(file, draft) по порядку. memcore.plan_sync требует уникальных id, а
    у хранимых и у черновиков они могут повториться, поэтому в Stored и Atom вместо id идёт номер строки:
    memcore.plan_sync(список Stored(str(i), e.file, e.position, e.digest, e.status == 'active') по stored, список
    Atom(str(j), a.file, a.position, a.digest) по атомам черновиков). Номера из плана возвращаются в настоящие id:
    fresh — атомы по номерам из add (в порядке add); retract — id хранимых по номерам из retract; revises — пары (id
    нового, id снятого) по номерам из revises; kept = len(keep); secrets — как дано. Итог Batch(file, fresh, retract,
    revises, kept, secrets).

    D: 6

    >>> plan_batch('n.md', (), (Draft(0, 'LanceDB выбран.', '', 'LanceDB выбран.', 1760000000.0, '', 'file'),), 0)
    Batch(file='n.md', fresh=(Entry(id='2295c768ade7f62e', file='n.md', position=0, digest='0b00d70ea64ce933', text='LanceDB выбран.', question='', answer='LanceDB выбран.', at=1760000000.0, status='active', retracted_at=0.0, session='', source='file'),), retract=(), revises=(), kept=0, secrets=0)
    >>> plan_batch('n.md', (Entry('o1', 'n.md', 0, '0b00d70ea64ce933', 'LanceDB выбран.', '', 'LanceDB выбран.', 1.0, 'active', 0.0, '', 'file'),), (Draft(0, 'LanceDB выбран.', '', 'LanceDB выбран.', 1760000000.0, '', 'file'),), 0)
    Batch(file='n.md', fresh=(), retract=(), revises=(), kept=1, secrets=0)
    >>> plan_batch('n.md', (Entry('o1', 'n.md', 0, '0b00d70ea64ce933', 'LanceDB выбран.', '', 'LanceDB выбран.', 1.0, 'active', 0.0, '', 'file'),), (Draft(0, 'LanceDB заменил SQLite.', '', 'LanceDB заменил SQLite.', 1760003600.0, '', 'file'),), 0)
    Batch(file='n.md', fresh=(Entry(id='b9b7a96df021571e', file='n.md', position=0, digest='3e24c9c9b6794eb2', text='LanceDB заменил SQLite.', question='', answer='LanceDB заменил SQLite.', at=1760003600.0, status='active', retracted_at=0.0, session='', source='file'),), retract=('o1',), revises=(('b9b7a96df021571e', 'o1'),), kept=0, secrets=0)
    >>> plan_batch('n.md', (Entry('o1', 'n.md', 0, '0b00d70ea64ce933', 'LanceDB выбран.', '', 'LanceDB выбран.', 1.0, 'active', 0.0, '', 'file'),), (), 0)
    Batch(file='n.md', fresh=(), retract=('o1',), revises=(), kept=0, secrets=0)
    """
    atoms = [entry(file, d) for d in drafts]
    plan = memcore.plan_sync(
        [Stored(str(i), e.file, e.position, e.digest, e.status == 'active') for i, e in enumerate(stored)],
        [Atom(str(j), a.file, a.position, a.digest) for j, a in enumerate(atoms)],
    )
    return Batch(
        file=file,
        fresh=tuple(atoms[int(j)] for j in plan.add),
        retract=tuple(stored[int(i)].id for i in plan.retract),
        revises=tuple((atoms[int(n)].id, stored[int(o)].id) for n, o in plan.revises),
        kept=len(plan.keep),
        secrets=secrets,
    )
