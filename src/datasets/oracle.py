"""
A basic wrapper over the data, allowing, by specifying the query and the parameters used, 
to get almost any necessary information about its behaviour:
- explain plan,
- explain analyze plan,
- execution time,
- planning time,
- template_id, and so on
"""

import os
from json import load
from typing import Optional, Dict
from pydantic import BaseModel
from src.datasets.data_types import (
    QueryName,
    QueryDop,
    HintsetCode,
    QueryData,
    Time,
    Cost,
    ExplainAnalyzePlan,
    ExplainPlan,
    Plans,
)

TIMEOUT = float(2**42)


def _load_benchmark_data(path_to_bench: "str") -> "Dict[QueryName, QueryData]":
    benchmark_data = {}
    for file_name in os.listdir(path_to_bench):
        query_name = file_name.split(".")[0]
        with open(f"{path_to_bench}/{file_name}", "r") as query_file:
            query_data = load(query_file)
            for settings in query_data:
                query_data[settings] = Plans(**query_data[settings])
            benchmark_data[query_name] = query_data
    return benchmark_data


class OracleRequest(BaseModel):
    query_name: "QueryName"
    dop: "QueryDop"
    hintset: "HintsetCode"


class Oracle:
    def __init__(self, path_to_bench: "str"):
        self.benchmark_data: "Dict[QueryName, QueryData]" = _load_benchmark_data(path_to_bench=path_to_bench)

    def get_query_names(self):
        return list(self.benchmark_data.keys())

    def _get_plans(self, request: "OracleRequest") -> "Plans":
        assert request.query_name in self.benchmark_data, f"Unknown query {request.query_name}"
        settings = str((request.dop, request.hintset))
        return self.benchmark_data[request.query_name][settings]

    def get_planning_time(self, request: "OracleRequest") -> "Time":
        plans = self._get_plans(request=request)
        return plans.explain_plan.planning_time

    def get_cost(self, request: "OracleRequest") -> "Cost":
        plans = self._get_plans(request=request)
        return plans.explain_plan.plan.cost

    def get_execution_time(self, request: "OracleRequest") -> "Time":
        plans = self._get_plans(request=request)
        if plans.explain_analyze_plan:
            execution_time = plans.explain_analyze_plan.execution_time
        else:
            execution_time = TIMEOUT
        return execution_time

    def get_explain_plan(self, request: "OracleRequest") -> "ExplainPlan":
        plans = self._get_plans(request=request)
        return plans.explain_plan

    def get_explain_analyze_plan(self, request: "OracleRequest") -> "Optional[ExplainAnalyzePlan]":
        plans = self._get_plans(request=request)
        return plans.explain_analyze_plan
