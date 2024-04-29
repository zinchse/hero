import os
import json

import pytest

from src.datasets.data_config import DOPS, HINTSETS, BENCH_NAME_TO_SIZE, BENCH_NAMES
from src.datasets.data_types import Plans


PATH_TO_DATASET = "data/processed"
EXPECTED_BENCHS = BENCH_NAMES
EXPECTED_QUERY_DATA_SIZE = len(DOPS) * len(HINTSETS)
BENCH_NAME_TO_EXPECTED_SIZE = BENCH_NAME_TO_SIZE


@pytest.mark.parametrize("bench_name", EXPECTED_BENCHS)
def test_existence(bench_name: str):
    assert bench_name in os.listdir(PATH_TO_DATASET), f"Can't find {bench_name} dataset at {PATH_TO_DATASET}"


@pytest.mark.parametrize("bench_name", EXPECTED_BENCHS)
def test_stucture(bench_name: str):
    path_to_bench = f"{PATH_TO_DATASET}/{bench_name}"

    expected_bench_size = BENCH_NAME_TO_EXPECTED_SIZE[bench_name]
    found_sqls = [f for f in os.listdir(path_to_bench) if f.endswith(".json")]
    assert (
        len(found_sqls) == expected_bench_size
    ), f"Expected # queries is {expected_bench_size}, real {len(found_sqls)}"

    for query_file in os.listdir(path_to_bench):
        if not query_file.endswith(".json"):
            continue

        with open(f"{path_to_bench}/{query_file}", "r") as query_data_file:
            expected_query_data_size = EXPECTED_QUERY_DATA_SIZE
            query_data = json.load(query_data_file)
            real_query_data_size = len(query_data)

            assert (
                real_query_data_size == expected_query_data_size
            ), f"Data size for {query_file} is {real_query_data_size} but expected {expected_query_data_size}"

            for settings_str, plans in query_data.items():
                assert Plans(**plans), f"Problem plans with {settings_str} for {query_file}"
