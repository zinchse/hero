"""
This module provides functionalities for initializing oracles and 
retrieving the appropriate oracle for a given query. It includes 
a global list of oracles that can be initialized based on a root path
and a list of benchmark names. The module also uses LRU cache to 
optimize performance.

Functions:
- `initialize_oracles`: Initializes the global `ORACLES` list based 
  on the specified root path and benchmarks.
- `_get_oracle`: Retrieves the oracle for a given query name using an LRU cache, and
- many wrappers for getting `execution_time`, `e2e_time`, `logical_tree` and so on
"""

import os
from typing import List, Dict, Tuple
import numpy as np
import torch
from torch import Tensor
from src.datasets.vectorization import extract_vertices_and_edges
from src.utils import (
    get_cardinalities,
    get_selectivities,
    get_logical_tree,
    get_full_plan,
    preprocess,
    extract_list_info,
)
from functools import lru_cache
from src.datasets.data_types import QueryName, QueryDop, HintsetCode, Cardinality, Selectivity, Time, ExplainPlan
from src.datasets.data_config import DEFAULT_DOP, DEFAULT_HINTSET
from src.datasets.oracle import Oracle, OracleRequest, TIMEOUT
from src.models.neural_network.regressor import BinaryTreeRegressor


ORACLES_DICT: "Dict[str, Oracle]" = {}
PROCESSED_EX_TIMES: "Dict[Tuple[QueryName, HintsetCode, QueryDop], Time]" = {}


def initialize_oracles(bench_names: "List[str]") -> "None":
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for bench_name in bench_names:
        oracle = Oracle(f"{root_path}/data/processed/{bench_name}")
        for info in extract_list_info(oracle, oracle.get_query_names()):
            PROCESSED_EX_TIMES[(info["query_name"], info["hintset"], info["dop"])] = info["time"].item()
        ORACLES_DICT[bench_name] = oracle


@lru_cache(None)
def _get_oracle(q_n: "QueryName") -> "Oracle":
    for _, oracle in ORACLES_DICT.items():
        if q_n in oracle.get_query_names():
            return oracle
    assert False, f"Appropriate for query '{q_n}' oracle hasn't yet been initialized"  # pragma: no cover


@lru_cache(None)
def _get_selectivities(q_n: "QueryName", hs: "HintsetCode", dop: "QueryDop") -> "List[Selectivity]":
    return get_selectivities(q_n, _get_oracle(q_n), hs, dop)


@lru_cache(None)
def _get_cardinalities(q_n: "QueryName", hs: "HintsetCode", dop: "QueryDop") -> "List[Cardinality]":
    return get_cardinalities(q_n, _get_oracle(q_n), hs, dop)


@lru_cache(None)
def _get_template_id(q_n: "QueryName") -> "int":
    return _get_explain_plan(q_n, DEFAULT_HINTSET, DEFAULT_DOP).template_id


@lru_cache(None)
def _get_logical_tree(q_n: "QueryName", hs: "HintsetCode", dop: "QueryDop", with_rels: "bool" = True) -> "str":
    return get_logical_tree(q_n, _get_oracle(q_n), hs, dop, with_rels=with_rels)


@lru_cache(None)
def _get_full_plan(q_n: "QueryName", hs: "HintsetCode", dop: "QueryDop", with_rels: "bool" = True) -> "str":
    return get_full_plan(q_n, _get_oracle(q_n), hs, dop, with_rels=with_rels)


@lru_cache(None)
def _get_execution_time(q_n: "QueryName", hs: "HintsetCode", dop: "QueryDop", handle_timeout: "bool" = True) -> "Time":
    time_from_oracle = _get_oracle(q_n).get_execution_time(OracleRequest(query_name=q_n, hintset=hs, dop=dop)) / 1000
    if not handle_timeout:
        return time_from_oracle
    if time_from_oracle != TIMEOUT / 1000:
        assert np.isclose(time_from_oracle, PROCESSED_EX_TIMES[(q_n, hs, dop)]), "Inconsistent times"
        return time_from_oracle
    return PROCESSED_EX_TIMES[(q_n, hs, dop)]


@lru_cache(None)
def _get_planning_time(q_n: "QueryName", hs: "HintsetCode", dop: "QueryDop") -> "Time":
    return _get_oracle(q_n).get_planning_time(OracleRequest(query_name=q_n, hintset=hs, dop=dop)) / 1000


@lru_cache(None)
def _get_e2e_time(q_n: "QueryName", hs: "HintsetCode", dop: "QueryDop") -> "Time":
    return _get_execution_time(q_n, hs, dop) + _get_planning_time(q_n, hs, dop)


@lru_cache(None)
def _get_explain_plan(q_n: "QueryName", hs: "HintsetCode", dop: "QueryDop") -> "ExplainPlan":
    return _get_oracle(q_n).get_explain_plan(OracleRequest(query_name=q_n, hintset=hs, dop=dop))


@lru_cache(None)
def _get_processed_v_e(q_n: "QueryName", hs: "HintsetCode", dop: "QueryDop") -> "Tuple[Tensor, Tensor]":
    return preprocess(*extract_vertices_and_edges(_get_explain_plan(q_n, hs, dop)))


def _get_prediction(q_n: "QueryName", hs: "HintsetCode", dop: "QueryDop", model: "BinaryTreeRegressor") -> "Time":
    with torch.no_grad():
        v, e = _get_processed_v_e(q_n, hs, dop)
        return model(v.unsqueeze(0).to(model.device), e.unsqueeze(0).to(model.device)).squeeze(0).item()
