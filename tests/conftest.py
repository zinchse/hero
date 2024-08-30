import os
import sys

ROOT_PATH = os.getcwd()
HBO_BENCH_PATH = os.path.join(ROOT_PATH, "hbo_bench/src/hbo_bench")
sys.path.insert(0, ROOT_PATH)

from typing import Dict
import pytest
from hbo_bench.oracle import Oracle
from wrappers import ORACLES_DICT, initialize_oracles


@pytest.fixture(scope="session", autouse=True)
def setup_oracles():
    initialize_oracles(root_path=HBO_BENCH_PATH, bench_names=["JOB", "sample_queries"])


@pytest.fixture
def oracles_dict() -> "Dict[str, Oracle]":
    return ORACLES_DICT
