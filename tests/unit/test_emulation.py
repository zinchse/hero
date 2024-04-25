from typing import Any, Dict, List, Tuple
import pytest
from src.datasets import emulator as em
from src.datasets.data_config import BENCH_NAMES, BENCH_NAME_TO_SIZE
from src.datasets.data_types import ExplainAnalyzePlan, ExplainPlan, Time


PATH_TO_DATASET: "str" = "data/processed"
EXPECTED_BENCHS: "List[str]" = BENCH_NAMES
TEST_DOP: "int" = 1
TEST_HINTSET: "int" = 42
BENCH_QUERY_PAIRS: "List[Tuple[str, str]]" = [("JOB", "1b"), ("sample_queries", "q10_2a265"), ("tpch_10gb", "q01")]
BENCH_NAME_TO_EXPECTED_SIZE = BENCH_NAME_TO_SIZE


@pytest.mark.parametrize("bench_name, query_name", BENCH_QUERY_PAIRS)
def test_emulator_functionality(bench_name: "str", query_name: "str"):
    path_to_bench = f"{PATH_TO_DATASET}/{bench_name}"
    emulator = em.Emulator(path_to_bench=path_to_bench)

    kwargs: "Dict[str, Any]" = {"query_name": query_name, "dop": TEST_DOP, "hintset": TEST_HINTSET}

    assert isinstance(emulator.get_query_names(), list)
    assert len(emulator.get_query_names()) == BENCH_NAME_TO_EXPECTED_SIZE[bench_name]
    assert isinstance(emulator.get_planning_time(**kwargs), Time)
    assert isinstance(emulator.get_execution_time(**kwargs), Time)
    assert isinstance(emulator.get_explain_analyze_plan(**kwargs), (type(None), ExplainAnalyzePlan))
    assert isinstance(emulator.get_explain_plan(**kwargs), ExplainPlan)
