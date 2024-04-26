from typing import List, Tuple
import pytest
from src.datasets.oracle import Oracle, OracleRequest
from src.datasets.data_config import BENCH_NAMES, BENCH_NAME_TO_SIZE
from src.datasets.data_types import ExplainAnalyzePlan, ExplainPlan, Time, QueryDop, HintsetCode, QueryName


PATH_TO_DATASET: "str" = "data/processed"
EXPECTED_BENCHS: "List[str]" = BENCH_NAMES
TEST_DOP: "QueryDop" = 1
TEST_HINTSET: "HintsetCode" = 42
BENCH_QUERY_PAIRS: "List[Tuple[str, QueryName]]" = [
    ("JOB", "1b"),
    ("sample_queries", "q10_2a265"),
    ("tpch_10gb", "q01"),
]
BENCH_NAME_TO_EXPECTED_SIZE = BENCH_NAME_TO_SIZE


@pytest.mark.parametrize("bench_name, query_name", BENCH_QUERY_PAIRS)
def test_emulator_functionality(bench_name: "str", query_name: "str"):
    path_to_bench = f"{PATH_TO_DATASET}/{bench_name}"

    oracle = Oracle(path_to_bench=path_to_bench)
    oracle_request = OracleRequest(query_name=query_name, dop=TEST_DOP, hintset=TEST_HINTSET)

    assert isinstance(oracle.get_query_names(), list)
    assert len(oracle.get_query_names()) == BENCH_NAME_TO_EXPECTED_SIZE[bench_name]
    assert isinstance(oracle.get_planning_time(request=oracle_request), Time)
    assert isinstance(oracle.get_execution_time(request=oracle_request), Time)
    assert isinstance(oracle.get_explain_analyze_plan(request=oracle_request), (type(None), ExplainAnalyzePlan))
    assert isinstance(oracle.get_explain_plan(request=oracle_request), ExplainPlan)
