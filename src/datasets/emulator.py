import os
from json import load
from typing import Optional
from src.datasets.data_types import BenchmarkData, Time, ExplainAnalyzePlan, ExplainPlan, Plans


TIMEOUT = float(2**42)


def _load_benchmark_data(path_to_bench: "str") -> "BenchmarkData":
    benchmark_data = {}
    for file_name in os.listdir(path_to_bench):
        if file_name.startswith("."):
            continue
        query_name = file_name.split(".")[0]
        with open(f"{path_to_bench}/{file_name}", "r") as query_file:
            query_data = load(query_file)
            for settings in query_data:
                query_data[settings] = Plans(**query_data[settings])
            benchmark_data[query_name] = query_data
    return benchmark_data


class Emulator:
    def __init__(self, path_to_bench: "str"):
        self.benchmark_data: "BenchmarkData" = _load_benchmark_data(path_to_bench=path_to_bench)

    def get_query_names(self):
        return list(self.benchmark_data.keys())

    def _get_plans(self, query_name: "str", dop: "int", hintset: "int") -> "Plans":
        assert query_name in self.benchmark_data, "Unknown query {query_name}"
        settings = str((dop, str(hintset)))
        return self.benchmark_data[query_name][settings]

    def get_planning_time(self, query_name: "str", dop: "int", hintset: "int") -> "Time":
        plans = self._get_plans(query_name=query_name, dop=dop, hintset=hintset)
        return plans.explain_plan.planning_time

    def get_execution_time(self, query_name: "str", dop: "int", hintset: "int") -> "Time":
        plans = self._get_plans(query_name=query_name, dop=dop, hintset=hintset)
        if plans.explain_analyze_plan:
            execution_time = plans.explain_analyze_plan.execution_time
        else:
            execution_time = TIMEOUT
        return execution_time

    def get_explain_plan(self, query_name: "str", dop: "int", hintset: "int") -> "ExplainPlan":
        plans = self._get_plans(query_name=query_name, dop=dop, hintset=hintset)
        return plans.explain_plan

    def get_explain_analyze_plan(self, query_name: "str", dop: "int", hintset: "int") -> "Optional[ExplainAnalyzePlan]":
        plans = self._get_plans(query_name=query_name, dop=dop, hintset=hintset)
        return plans.explain_analyze_plan
