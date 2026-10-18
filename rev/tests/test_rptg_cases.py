"""Тесты на свойства из условий карточек: deal.cases (Hypothesis). Пишет rptg kit."""
import deal
import hypothesis
import pytest

import memcore
import memcore.context
import memcore.context.dedup
import memcore.context.labeling
import memcore.context.packing
import memcore.context.ranking
import memcore.filtering
import memcore.relations
import memcore.scrubbing
import memcore.scrubbing.extension
import memcore.scrubbing.merging
import memcore.scrubbing.provider
import memcore.scrubbing.redaction
import memcore.scrubbing.starts
import memcore.scrubbing.vendor
import memcore.search
import memcore.sync

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

test_memcore_context_dedup__first_per_id = cases(memcore.context.dedup.first_per_id)
test_memcore_context_labeling__label_hits = cases(memcore.context.labeling.label_hits)
test_memcore_context_packing__take_within = cases(memcore.context.packing.take_within)
test_memcore_context_ranking__rank_hits = cases(memcore.context.ranking.rank_hits)
test_memcore_context__select_context = cases(memcore.context.select_context)
test_memcore_filtering__admit = cases(memcore.filtering.admit)
test_memcore_filtering__matches = cases(memcore.filtering.matches)
test_memcore_relations__current_version = cases(memcore.relations.current_version)
test_memcore_relations__record_relation = cases(memcore.relations.record_relation)
test_memcore_scrubbing_extension__extend_span = cases(memcore.scrubbing.extension.extend_span)
test_memcore_scrubbing_merging__merge_spans = cases(memcore.scrubbing.merging.merge_spans)
test_memcore_scrubbing_provider__provider_values = cases(memcore.scrubbing.provider.provider_values)
test_memcore_scrubbing_redaction__redact = cases(memcore.scrubbing.redaction.redact)
test_memcore_scrubbing_starts__secret_starts = cases(memcore.scrubbing.starts.secret_starts)
test_memcore_scrubbing_vendor__vendor_values = cases(memcore.scrubbing.vendor.vendor_values)
test_memcore_scrubbing__scrub = cases(memcore.scrubbing.scrub)
test_memcore_search__choose_search = cases(memcore.search.choose_search)
test_memcore_sync__plan_sync = cases(memcore.sync.plan_sync)
test_memcore__admit = cases(memcore.admit)
test_memcore__choose_search = cases(memcore.choose_search)
test_memcore__current_version = cases(memcore.current_version)
test_memcore__plan_sync = cases(memcore.plan_sync)
test_memcore__record_relation = cases(memcore.record_relation)
test_memcore__scrub = cases(memcore.scrub)
test_memcore__select_context = cases(memcore.select_context)
