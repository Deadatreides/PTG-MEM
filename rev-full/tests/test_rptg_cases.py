"""Тесты на свойства из условий карточек: deal.cases (Hypothesis). Пишет rptg kit."""
import deal
import hypothesis
import pytest

import ptgfull.adapter
import ptgfull.code
import ptgfull.code.process
import ptgfull.code.queries
import ptgfull.code.refs
import ptgfull.code.symbols
import ptgfull.code.table
import ptgfull.metatree
import ptgfull.metatree.folders
import ptgfull.metatree.marks
import ptgfull.metatree.pairs
import ptgfull.metatree.paths
import ptgfull.metatree.sources
import ptgfull.metatree.symbols
import ptgfull.spaces
import ptgfull.store.answer
import ptgfull.store.clause
import ptgfull.store.unique
import ptgfull.walk

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

test_ptgfull_adapter__adapt = cases(ptgfull.adapter.adapt)
test_ptgfull_adapter__fit = cases(ptgfull.adapter.fit)
test_ptgfull_code_process__command = cases(ptgfull.code.process.command)
test_ptgfull_code_queries__edge_query = cases(ptgfull.code.queries.edge_query)
test_ptgfull_code_queries__index_args = cases(ptgfull.code.queries.index_args)
test_ptgfull_code_queries__node_query = cases(ptgfull.code.queries.node_query)
test_ptgfull_code_queries__project_name = cases(ptgfull.code.queries.project_name)
test_ptgfull_code_queries__query_args = cases(ptgfull.code.queries.query_args)
test_ptgfull_code_refs__expand = cases(ptgfull.code.refs.expand)
test_ptgfull_code_refs__ref_map = cases(ptgfull.code.refs.ref_map)
test_ptgfull_code_symbols__edge_of = cases(ptgfull.code.symbols.edge_of)
test_ptgfull_code_symbols__inner_edges = cases(ptgfull.code.symbols.inner_edges)
test_ptgfull_code_symbols__node_of = cases(ptgfull.code.symbols.node_of)
test_ptgfull_code_table__block = cases(ptgfull.code.table.block)
test_ptgfull_code_table__cells = cases(ptgfull.code.table.cells)
test_ptgfull_code__code_edges = cases(ptgfull.code.code_edges)
test_ptgfull_code__code_nodes = cases(ptgfull.code.code_nodes)
test_ptgfull_code__parse_rows = cases(ptgfull.code.parse_rows)
test_ptgfull_metatree_folders__file_chain = cases(ptgfull.metatree.folders.file_chain)
test_ptgfull_metatree_marks__tool_paths = cases(ptgfull.metatree.marks.tool_paths)
test_ptgfull_metatree_marks__word_paths = cases(ptgfull.metatree.marks.word_paths)
test_ptgfull_metatree_pairs__first_per_pair = cases(ptgfull.metatree.pairs.first_per_pair)
test_ptgfull_metatree_paths__file_edges = cases(ptgfull.metatree.paths.file_edges)
test_ptgfull_metatree_paths__resolve = cases(ptgfull.metatree.paths.resolve)
test_ptgfull_metatree_sources__arrange = cases(ptgfull.metatree.sources.arrange)
test_ptgfull_metatree_sources__chain_edges = cases(ptgfull.metatree.sources.chain_edges)
test_ptgfull_metatree_symbols__lookup = cases(ptgfull.metatree.symbols.lookup)
test_ptgfull_metatree_symbols__names = cases(ptgfull.metatree.symbols.names)
test_ptgfull_metatree_symbols__symbol_edges = cases(ptgfull.metatree.symbols.symbol_edges)
test_ptgfull_metatree__source_edges = cases(ptgfull.metatree.source_edges)
test_ptgfull_metatree__stitch = cases(ptgfull.metatree.stitch)
test_ptgfull_metatree__tree_edges = cases(ptgfull.metatree.tree_edges)
test_ptgfull_spaces__fuse = cases(ptgfull.spaces.fuse)
test_ptgfull_spaces__hash_embed = cases(ptgfull.spaces.hash_embed)
test_ptgfull_spaces__parse_space = cases(ptgfull.spaces.parse_space)
test_ptgfull_spaces__plan_routes = cases(ptgfull.spaces.plan_routes)
test_ptgfull_store_answer__arrange_atoms = cases(ptgfull.store.answer.arrange_atoms)
test_ptgfull_store_answer__nearest_hits = cases(ptgfull.store.answer.nearest_hits)
test_ptgfull_store_answer__word_hits = cases(ptgfull.store.answer.word_hits)
test_ptgfull_store_clause__has_words = cases(ptgfull.store.clause.has_words)
test_ptgfull_store_clause__where_in = cases(ptgfull.store.clause.where_in)
test_ptgfull_store_unique__distinct_atoms = cases(ptgfull.store.unique.distinct_atoms)
test_ptgfull_store_unique__distinct_nodes = cases(ptgfull.store.unique.distinct_nodes)
test_ptgfull_store_unique__distinct_vectors = cases(ptgfull.store.unique.distinct_vectors)
test_ptgfull_walk__assemble = cases(ptgfull.walk.assemble)
test_ptgfull_walk__pieces = cases(ptgfull.walk.pieces)
test_ptgfull_walk__walk = cases(ptgfull.walk.walk)
