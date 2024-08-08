from typing import Dict
from collections import defaultdict
from itertools import product
from src.wrappers import _get_logical_tree, _get_full_plan, _get_explain_plan
from src.datasets.oracle import Oracle
from src.datasets.vectorization import extract_vertices_and_edges
from src.datasets.data_config import HINTSETS, DOPS, DEFAULT_HINTSET, DEFAULT_DOP
from src.utils import preprocess


def test_logical_tree():
    logical_trees = {_get_logical_tree(q_n, DEFAULT_HINTSET, DEFAULT_DOP, True) for q_n in ["6a", "6b", "6c", "6d"]}
    assert len(logical_trees) == 1


def test_get_full_plan():
    assert _get_full_plan("6b", 0, 64, True) == _get_full_plan("6d", 0, 64, True)


def test_consistency_with_ve_structure(oracles_dict: "Dict[str, Oracle]"):
    logical_tree_to_ve_structure = defaultdict(set)
    for _, oracle in oracles_dict.items():
        for q_n, hs, dop in product(oracle.get_query_names()[:42], HINTSETS[:4], DOPS[:2]):
            v, e = preprocess(*extract_vertices_and_edges(_get_explain_plan(q_n, hs, dop)))
            v[-2:, :] = 0
            ve_structure = str(v.flatten().tolist()), str(e.flatten().tolist())
            logical_tree = _get_logical_tree(q_n, hs, dop, with_rels=False)
            logical_tree_to_ve_structure[logical_tree].add(ve_structure)
    for ve_struct_set in logical_tree_to_ve_structure.values():
        assert len(ve_struct_set) == 1


def test_consistency_with_ve(oracles_dict: "Dict[str, Oracle]"):
    full_plan_to_ve = defaultdict(set)
    for _, oracle in oracles_dict.items():
        for q_n, hs, dop in product(oracle.get_query_names()[:42], HINTSETS[:4], DOPS[:2]):
            full_plan = _get_full_plan(q_n, hs, dop, with_rels=False)
            v, e = extract_vertices_and_edges(_get_explain_plan(q_n, hs, dop))
            ve = str(v.flatten().tolist()), str(e.flatten().tolist())
            full_plan_to_ve[full_plan].add(ve)
    for ve_set in full_plan_to_ve.values():
        assert len(ve_set) == 1
