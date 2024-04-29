from typing import Dict, List
from src.datasets.data_types import Hint, HintsetCode, QueryDop, GUC


HINTS: "List[Hint]" = [
    "Nested Loop",
    "Merge",
    "Hash",
    "Bitmap",
    "Index Only Scan",
    "Index Scan",
    "Seq Scan",
]

GUCS: "List[GUC]" = [
    "nestloop",
    "mergejoin",
    "hashjoin",
    "bitmapscan",
    "indexonlyscan",
    "indexscan",
    "seqscan",
]

DEFAULT_HINTSET: "HintsetCode" = 0

DOPS: "List[QueryDop]" = [1, 16, 64]
HINTSETS: "List[HintsetCode]" = list(range(2 ** len(HINTS)))
BENCH_NAMES: "List[str]" = ["JOB", "sample_queries", "tpch_10gb"]
BENCH_NAME_TO_SIZE: "Dict[str, int]" = {
    "JOB": 113,
    "sample_queries": 40,
    "tpch_10gb": 22,
}
