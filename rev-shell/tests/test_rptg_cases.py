"""Тесты на свойства из условий карточек: deal.cases (Hypothesis). Пишет rptg kit."""
import deal
import hypothesis
import pytest

import memshell
import memshell.atoms
import memshell.decisions
import memshell.recall
import memshell.render
import memshell.render.budget
import memshell.render.lines
import memshell.render.text

# Настройки deal по умолчанию, и сверх них сняты порог «медленная генерация» и срок на вызов. Условие pre,
# связывающее аргументы (0 <= start < end <= len(text)), отсекает почти все случайные входы; проверка идёт на тех,
# что прошли (11.10, memcore.scrubbing: 7 годных из 350 — Hypothesis бросал проверку по порогу).
SETTINGS = hypothesis.settings(database=None, max_examples=50, deadline=None, verbosity=hypothesis.Verbosity.quiet,
                               report_multiple_bugs=False, print_blob=True,
                               suppress_health_check=[hypothesis.HealthCheck.filter_too_much, hypothesis.HealthCheck.too_slow])


def cases(func):
    """deal.cases для функции. Ни одного входа, который допускают условия pre (Unsatisfiable), — пропуск с причиной:
    свойства такой функции проверяет CrossHair, он решает условия, а не перебирает входы."""
    test = deal.cases(func, settings=SETTINGS)

    def run():
        try:
            test()
        except hypothesis.errors.Unsatisfiable:
            pytest.skip("условия pre отсекли все случайные входы — свойства этой функции проверяет CrossHair")
    return run

test_memshell_atoms__atomize = cases(memshell.atoms.atomize)
test_memshell_atoms__digest = cases(memshell.atoms.digest)
test_memshell_atoms__split_text = cases(memshell.atoms.split_text)
test_memshell_decisions__decision_pairs = cases(memshell.decisions.decision_pairs)
test_memshell_recall__in_window = cases(memshell.recall.in_window)
test_memshell_recall__is_noise = cases(memshell.recall.is_noise)
test_memshell_recall__session_ok = cases(memshell.recall.session_ok)
test_memshell_render_budget__item_limit = cases(memshell.render.budget.item_limit)
test_memshell_render_budget__pack_block = cases(memshell.render.budget.pack_block)
test_memshell_render_lines__note_line = cases(memshell.render.lines.note_line)
test_memshell_render_lines__recall_item = cases(memshell.render.lines.recall_item)
test_memshell_render_lines__recent_line = cases(memshell.render.lines.recent_line)
test_memshell_render_text__clip = cases(memshell.render.text.clip)
test_memshell_render_text__cut = cases(memshell.render.text.cut)
test_memshell_render_text__squash = cases(memshell.render.text.squash)
test_memshell_render__render_brief = cases(memshell.render.render_brief)
test_memshell_render__render_recall = cases(memshell.render.render_recall)
test_memshell__atomize = cases(memshell.atomize)
test_memshell__decision_pairs = cases(memshell.decision_pairs)
test_memshell__digest = cases(memshell.digest)
test_memshell__in_window = cases(memshell.in_window)
test_memshell__is_noise = cases(memshell.is_noise)
test_memshell__render_brief = cases(memshell.render_brief)
test_memshell__render_recall = cases(memshell.render_recall)
test_memshell__session_ok = cases(memshell.session_ok)
test_memshell__split_text = cases(memshell.split_text)
