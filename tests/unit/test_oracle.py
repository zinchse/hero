from typing import Dict
from src.datasets.oracle import Oracle, OracleRequest
from src.datasets.data_config import BENCH_NAME_TO_SIZE
from src.datasets.data_types import ExplainAnalyzePlan, ExplainPlan, Time, QueryName, Cost


PATH_TO_DATA: "str" = "data/processed"
BENCH_NAME_TO_EXAMPLE_QUERY: "Dict[str, QueryName]" = {
    "JOB": "1b",
    "sample_queries": "q10_2a265",
    "tpch_10gb": "q01",
}


def test_types_and_sizes(oracles_dict: "Dict[str, Oracle]"):
    for bench_name, oracle in [("tpch_10gb", Oracle(f"{PATH_TO_DATA}/tpch_10gb"))] + list(oracles_dict.items()):
        oracle_request = OracleRequest(query_name=BENCH_NAME_TO_EXAMPLE_QUERY[bench_name], dop=1, hintset=42)
        assert isinstance(oracle.get_query_names(), list)
        assert len(oracle.get_query_names()) == BENCH_NAME_TO_SIZE[bench_name]
        assert isinstance(oracle.get_planning_time(request=oracle_request), Time)
        assert isinstance(oracle.get_execution_time(request=oracle_request), Time)
        assert isinstance(oracle.get_explain_analyze_plan(request=oracle_request), (type(None), ExplainAnalyzePlan))
        assert isinstance(oracle.get_explain_plan(request=oracle_request), ExplainPlan)
        assert isinstance(oracle.get_cost(request=oracle_request), Cost)
