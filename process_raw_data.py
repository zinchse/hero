from json import load, dumps, dump
import os
from typing import Dict
from src.datasets.data_types import Plans

BENCH_NAMES = ["JOB", "sample_queries", "tpch_10gb"]
DOPS = [1, 16, 64]


def _unify_actual_rows(root: "dict") -> "None":
    def recurse(node: "dict"):
        if "Actual Total Rows" in node:
            node["Actual Rows"] = node["Actual Total Rows"]
            del node["Actual Total Rows"]
        for child in node.get("Plans", []):
            recurse(node=child)

    recurse(node=root)


def process_raw_data(path_to_raw: "str", path_to_processed: "str", bench_name: "str") -> "None":
    bench_data: "Dict[str, Dict[str, Dict]]" = {}
    for dop in DOPS:
        with open(f"{path_to_raw}/dop{dop}/{bench_name}.json", "r") as raw_data_file:
            raw_data = load(raw_data_file)

        for sql_name, sql_data in raw_data.items():
            bench_data[sql_name] = bench_data.get(sql_name, {})

            for hs, explain_plan in sql_data["hs_to_explain_plan"].items():
                explain_analyze_plan = sql_data["explain_plan_to_explain_analyze_plan"][dumps(explain_plan)]
                if "Timeout" in explain_analyze_plan:
                    explain_analyze_plan = None
                else:
                    _unify_actual_rows(root=explain_analyze_plan["Plan"])

                explain_plan["Planner Runtime"] = sql_data["hs_to_planning_time"][str(hs)]
                if explain_analyze_plan:
                    explain_analyze_plan["Planner Runtime"] = sql_data["hs_to_planning_time"][str(hs)]

                bench_data[sql_name][str((dop, int(hs)))] = {
                    "explain_plan": explain_plan,
                    "explain_analyze_plan": explain_analyze_plan,
                }
                assert Plans(**bench_data[sql_name][str((dop, int(hs)))])

    os.mkdir(f"{path_to_processed}/{bench_name}")

    for sql_name, sql_data in bench_data.items():
        with open(f"{path_to_processed}/{bench_name}/{sql_name.split('.')[0]}.json", "w") as f:
            dump(sql_data, f)


if __name__ == "__main__":
    for name in BENCH_NAMES:
        if os.path.exists(f"data/processed/{name}"):
            print(f"Please, clear the folder 'data/processed/{name}' before processing")
            break
    else:
        for name in BENCH_NAMES:
            print(f"Processing raw data from '{name}' has been started")
            process_raw_data(
                path_to_processed="data/processed",
                path_to_raw="data/raw",
                bench_name=name,
            )
            print(f"Raw data for '{name}' has been processed")
