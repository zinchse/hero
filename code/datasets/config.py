HINTS = [
    "Nested Loop",
    "Merge",
    "Hash",
    "Bitmap",
    "Index Only Scan",
    "Index Scan",
    "Seq Scan",
]

GUCS = [
    "nestloop",
    "mergejoin",
    "hashjoin",
    "bitmapscan",
    "indexonlyscan",
    "indexscan",
    "seqscan",
]

DEFAULT_HINTSET = 0

DOPS = [1, 16, 64]
HINTSETS = [hs for hs in range(2 ** len(HINTS))]
BENCH_NAMES = ["JOB", "sample_queries", "tpch_10gb"]
BENCH_NAME_TO_SIZE = {"JOB": 113, "sample_queries": 40, "tpch_10gb": 22}
