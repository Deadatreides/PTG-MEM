"""Синхронизация атомов файла после перечитки: что оставить, что снять, что добавить и что чем переписано.

node: leaf
"""
import deal

from memcore.types import Atom, Stored, SyncPlan


@deal.pre(lambda stored, fresh: len({s.id for s in stored}) == len(stored) and len({a.id for a in fresh}) == len(fresh))
@deal.ensure(lambda stored, fresh, result: len(result.add) + len(result.keep) == len(fresh))
@deal.ensure(lambda stored, fresh, result: set(result.keep) | set(result.retract) == {s.id for s in stored if s.live})
@deal.ensure(lambda stored, fresh, result: not set(result.keep) & set(result.retract))
@deal.ensure(lambda stored, fresh, result: all(new in result.add and old in result.retract for new, old in result.revises))
@deal.has()
def plan_sync(stored: list[Stored], fresh: list[Atom]) -> SyncPlan:
    """План синхронизации хранимых атомов со свежими атомами перечитанного файла.

    Снятые раньше (live=False) не участвуют. Свежий атом с тем же файлом и отпечатком, что у живого хранимого, —
    тот же атом: хранимый остаётся (keep), свежий не добавляется. Пара подбирается первому свежему с первым по
    порядку stored ещё не занятым хранимым; два одинаковых абзаца — два разных атома. Остальные свежие — add, остальные
    живые хранимые — retract. Добавленный атом на том же файле и месте (position), что и снятый, переписывает его:
    пара (новый, прежний) в revises, каждый снятый — не больше одного раза, первый по порядку.
    Порядок: add и revises — как в fresh; keep и retract — как в stored.

    D: 7

    >>> plan_sync([Stored("s1", "a.md", 0, "d1", True), Stored("s2", "a.md", 1, "d2", True)], [Atom("n1", "a.md", 0, "d1"), Atom("n2", "a.md", 1, "d9")])
    SyncPlan(add=('n2',), keep=('s1',), retract=('s2',), revises=(('n2', 's2'),))
    >>> plan_sync([], [])
    SyncPlan(add=(), keep=(), retract=(), revises=())
    >>> plan_sync([Stored("s1", "a.md", 0, "d1", True)], [])
    SyncPlan(add=(), keep=(), retract=('s1',), revises=())
    >>> plan_sync([Stored("s1", "a.md", 0, "d1", True), Stored("s0", "a.md", 5, "d1", False)], [Atom("n1", "a.md", 0, "d1"), Atom("n2", "a.md", 1, "d1")])
    SyncPlan(add=('n2',), keep=('s1',), retract=(), revises=())
    """
    pool: dict[tuple[str, str], list[str]] = {}
    for s in stored:
        if s.live:
            pool.setdefault((s.file, s.digest), []).append(s.id)
    taken: set[str] = set()
    added: list[Atom] = []
    for a in fresh:
        ids = pool.get((a.file, a.digest), [])
        if ids:
            taken.add(ids.pop(0))
        else:
            added.append(a)
    retracted = [s for s in stored if s.live and s.id not in taken]
    slots: dict[tuple[str, int], list[str]] = {}
    for s in retracted:
        slots.setdefault((s.file, s.position), []).append(s.id)
    revises: list[tuple[str, str]] = []
    for a in added:
        ids = slots.get((a.file, a.position), [])
        if ids:
            revises.append((a.id, ids.pop(0)))
    return SyncPlan(
        add=tuple(a.id for a in added),
        keep=tuple(filter(taken.__contains__, [s.id for s in stored])),
        retract=tuple(s.id for s in retracted),
        revises=tuple(revises),
    )
