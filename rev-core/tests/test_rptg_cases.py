"""Тесты на свойства из условий карточек: deal.cases (Hypothesis). Пишет rptg kit."""
import deal
import hypothesis
import pytest

import ptgcore
import ptgcore.codegraph
import ptgcore.codegraph.marks
import ptgcore.codegraph.rebuild
import ptgcore.codegraph.scan
import ptgcore.codegraph.seams
import ptgcore.context
import ptgcore.context.admission
import ptgcore.context.block
import ptgcore.context.digest
import ptgcore.context.marks
import ptgcore.context.selection
import ptgcore.context.sessions
import ptgcore.intake
import ptgcore.intake.clean
import ptgcore.intake.listing
import ptgcore.intake.shaping
import ptgcore.ledger
import ptgcore.ledger.atoms
import ptgcore.ledger.clause
import ptgcore.ledger.files
import ptgcore.ledger.order
import ptgcore.migration
import ptgcore.migration.atoms
import ptgcore.migration.layout
import ptgcore.migration.links
import ptgcore.migration.scalars
import ptgcore.relations
import ptgcore.relations.chain
import ptgcore.relations.claim
import ptgcore.relations.feed
import ptgcore.relations.hood
import ptgcore.relations.show
import ptgcore.relations.state
import ptgcore.retrieval
import ptgcore.retrieval.cards
import ptgcore.retrieval.prewrite
import ptgcore.retrieval.queries

# Настройки deal по умолчанию, и сверх них сняты порог «медленная генерация» и срок на вызов. Условие pre,
# связывающее аргументы (0 <= start < end <= len(text)), отсекает почти все случайные входы; проверка идёт на тех,
# что прошли (11.10, memcore.scrubbing: 7 годных из 350 — Hypothesis бросал проверку по порогу). Снят и порог
# «большие входы»: вложенные записи (карточка со связями и пометками) его превышают через раз, и тест падал без
# дефекта (13.10, ptgcore.context.block.render_block: 9 годных из 29).
SETTINGS = hypothesis.settings(database=None, max_examples=50, deadline=None, verbosity=hypothesis.Verbosity.quiet,
                               report_multiple_bugs=False, print_blob=True,
                               suppress_health_check=[hypothesis.HealthCheck.filter_too_much, hypothesis.HealthCheck.too_slow,
                                                      hypothesis.HealthCheck.data_too_large])


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

test_ptgcore_codegraph_marks__mark_line = cases(ptgcore.codegraph.marks.mark_line)
test_ptgcore_codegraph_marks__mark_text = cases(ptgcore.codegraph.marks.mark_text)
test_ptgcore_codegraph_rebuild__assemble = cases(ptgcore.codegraph.rebuild.assemble)
test_ptgcore_codegraph_rebuild__to_atoms = cases(ptgcore.codegraph.rebuild.to_atoms)
test_ptgcore_codegraph_scan__prefixed = cases(ptgcore.codegraph.scan.prefixed)
test_ptgcore_codegraph_seams__seam_lines = cases(ptgcore.codegraph.seams.seam_lines)
test_ptgcore_codegraph_seams__target = cases(ptgcore.codegraph.seams.target)
test_ptgcore_codegraph_seams__touching = cases(ptgcore.codegraph.seams.touching)
test_ptgcore_codegraph__tool_marks = cases(ptgcore.codegraph.tool_marks)
test_ptgcore_context_admission__admit = cases(ptgcore.context.admission.admit)
test_ptgcore_context_block__render_block = cases(ptgcore.context.block.render_block)
test_ptgcore_context_block__shown = cases(ptgcore.context.block.shown)
test_ptgcore_context_digest__brief = cases(ptgcore.context.digest.brief)
test_ptgcore_context_digest__recent_notes = cases(ptgcore.context.digest.recent_notes)
test_ptgcore_context_marks__code_notes = cases(ptgcore.context.marks.code_notes)
test_ptgcore_context_marks__conflict_note = cases(ptgcore.context.marks.conflict_note)
test_ptgcore_context_marks__notes_of = cases(ptgcore.context.marks.notes_of)
test_ptgcore_context_marks__version_note = cases(ptgcore.context.marks.version_note)
test_ptgcore_context_selection__choose = cases(ptgcore.context.selection.choose)
test_ptgcore_context_sessions__eligible = cases(ptgcore.context.sessions.eligible)
test_ptgcore_context_sessions__session_files = cases(ptgcore.context.sessions.session_files)
test_ptgcore_context__admit = cases(ptgcore.context.admit)
test_ptgcore_context__brief = cases(ptgcore.context.brief)
test_ptgcore_context__eligible = cases(ptgcore.context.eligible)
test_ptgcore_context__recent_notes = cases(ptgcore.context.recent_notes)
test_ptgcore_context__render = cases(ptgcore.context.render)
test_ptgcore_context__session_files = cases(ptgcore.context.session_files)
test_ptgcore_intake_clean__note_text = cases(ptgcore.intake.clean.note_text)
test_ptgcore_intake_clean__scrub_drafts = cases(ptgcore.intake.clean.scrub_drafts)
test_ptgcore_intake_listing__stamped = cases(ptgcore.intake.listing.stamped)
test_ptgcore_intake_shaping__chat_drafts = cases(ptgcore.intake.shaping.chat_drafts)
test_ptgcore_intake_shaping__file_kind = cases(ptgcore.intake.shaping.file_kind)
test_ptgcore_intake_shaping__text_drafts = cases(ptgcore.intake.shaping.text_drafts)
test_ptgcore_intake__note = cases(ptgcore.intake.note)
test_ptgcore_ledger_atoms__entry = cases(ptgcore.ledger.atoms.entry)
test_ptgcore_ledger_atoms__plan_batch = cases(ptgcore.ledger.atoms.plan_batch)
test_ptgcore_ledger_clause__equals = cases(ptgcore.ledger.clause.equals)
test_ptgcore_ledger_clause__one_of = cases(ptgcore.ledger.clause.one_of)
test_ptgcore_ledger_files__outdated = cases(ptgcore.ledger.files.outdated)
test_ptgcore_ledger_order__arrange = cases(ptgcore.ledger.order.arrange)
test_ptgcore_ledger_order__distinct = cases(ptgcore.ledger.order.distinct)
test_ptgcore_ledger_order__in_place = cases(ptgcore.ledger.order.in_place)
test_ptgcore_ledger_order__retire = cases(ptgcore.ledger.order.retire)
test_ptgcore_ledger__entry = cases(ptgcore.ledger.entry)
test_ptgcore_ledger__outdated = cases(ptgcore.ledger.outdated)
test_ptgcore_ledger__plan_batch = cases(ptgcore.ledger.plan_batch)
test_ptgcore_migration_atoms__args = cases(ptgcore.migration.atoms.args)
test_ptgcore_migration_atoms__build = cases(ptgcore.migration.atoms.build)
test_ptgcore_migration_atoms__rows = cases(ptgcore.migration.atoms.rows)
test_ptgcore_migration_atoms__tally = cases(ptgcore.migration.atoms.tally)
test_ptgcore_migration_layout__fit = cases(ptgcore.migration.layout.fit)
test_ptgcore_migration_layout__parse_meta = cases(ptgcore.migration.layout.parse_meta)
test_ptgcore_migration_links__batches = cases(ptgcore.migration.links.batches)
test_ptgcore_migration_links__build = cases(ptgcore.migration.links.build)
test_ptgcore_migration_links__known = cases(ptgcore.migration.links.known)
test_ptgcore_migration_scalars__number = cases(ptgcore.migration.scalars.number)
test_ptgcore_migration_scalars__text = cases(ptgcore.migration.scalars.text)
test_ptgcore_migration__v6_edge = cases(ptgcore.migration.v6_edge)
test_ptgcore_migration__v6_entry = cases(ptgcore.migration.v6_entry)
test_ptgcore_relations_chain__heads = cases(ptgcore.relations.chain.heads)
test_ptgcore_relations_chain__pick = cases(ptgcore.relations.chain.pick)
test_ptgcore_relations_chain__replaced = cases(ptgcore.relations.chain.replaced)
test_ptgcore_relations_chain__touching = cases(ptgcore.relations.chain.touching)
test_ptgcore_relations_chain__walk = cases(ptgcore.relations.chain.walk)
test_ptgcore_relations_claim__apply = cases(ptgcore.relations.claim.apply)
test_ptgcore_relations_claim__apply_all = cases(ptgcore.relations.claim.apply_all)
test_ptgcore_relations_claim__of_pair = cases(ptgcore.relations.claim.of_pair)
test_ptgcore_relations_feed__decide = cases(ptgcore.relations.feed.decide)
test_ptgcore_relations_feed__lineage = cases(ptgcore.relations.feed.lineage)
test_ptgcore_relations_feed__links = cases(ptgcore.relations.feed.links)
test_ptgcore_relations_feed__picked = cases(ptgcore.relations.feed.picked)
test_ptgcore_relations_hood__ego = cases(ptgcore.relations.hood.ego)
test_ptgcore_relations_hood__ends = cases(ptgcore.relations.hood.ends)
test_ptgcore_relations_hood__layer = cases(ptgcore.relations.hood.layer)
test_ptgcore_relations_hood__merge = cases(ptgcore.relations.hood.merge)
test_ptgcore_relations_show__day = cases(ptgcore.relations.show.day)
test_ptgcore_relations_show__rel = cases(ptgcore.relations.show.rel)
test_ptgcore_relations_state__status = cases(ptgcore.relations.state.status)
test_ptgcore_relations_state__successor = cases(ptgcore.relations.state.successor)
test_ptgcore_relations_state__superseded = cases(ptgcore.relations.state.superseded)
test_ptgcore_relations__day = cases(ptgcore.relations.day)
test_ptgcore_relations__decisions = cases(ptgcore.relations.decisions)
test_ptgcore_relations__ego = cases(ptgcore.relations.ego)
test_ptgcore_relations__ends = cases(ptgcore.relations.ends)
test_ptgcore_relations__lineage = cases(ptgcore.relations.lineage)
test_ptgcore_relations__links = cases(ptgcore.relations.links)
test_ptgcore_relations__merge = cases(ptgcore.relations.merge)
test_ptgcore_relations__reach = cases(ptgcore.relations.reach)
test_ptgcore_relations__rel = cases(ptgcore.relations.rel)
test_ptgcore_relations__scope = cases(ptgcore.relations.scope)
test_ptgcore_relations__status = cases(ptgcore.relations.status)
test_ptgcore_relations__successor = cases(ptgcore.relations.successor)
test_ptgcore_relations__superseded = cases(ptgcore.relations.superseded)
test_ptgcore_retrieval_cards__assemble = cases(ptgcore.retrieval.cards.assemble)
test_ptgcore_retrieval_cards__shortlist = cases(ptgcore.retrieval.cards.shortlist)
test_ptgcore_retrieval_cards__visible = cases(ptgcore.retrieval.cards.visible)
test_ptgcore_retrieval_prewrite__atoms = cases(ptgcore.retrieval.prewrite.atoms)
test_ptgcore_retrieval_prewrite__clash = cases(ptgcore.retrieval.prewrite.clash)
test_ptgcore_retrieval_queries__carry = cases(ptgcore.retrieval.queries.carry)
test_ptgcore_retrieval_queries__source_vector = cases(ptgcore.retrieval.queries.source_vector)
test_ptgcore_retrieval__card = cases(ptgcore.retrieval.card)
test_ptgcore_retrieval__pick = cases(ptgcore.retrieval.pick)
test_ptgcore__note_text = cases(ptgcore.note_text)
