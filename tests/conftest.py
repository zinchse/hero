import sys
import os
from code.datasets.config import DOPS, HINTSETS, BENCH_NAME_TO_SIZE, BENCH_NAMES


sys.path.insert(0, os.getcwd())


PATH_TO_DATASET = "data/raw"
EXPECTED_BENCHS = BENCH_NAMES
EXPECTED_QUERY_DATA_SIZE = len(DOPS) * len(HINTSETS)
BENCH_NAME_TO_EXPECTED_SIZE = BENCH_NAME_TO_SIZE
