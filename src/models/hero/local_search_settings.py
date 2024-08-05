"""
Here're the basic configurations of the local search procedure: 
- the greedy algorithm
- the greedy algorithm with shortcuts (we call it just 'local'), as well as 
- their pruned versions and 
- the exhaustive search algorithm
"""

from src.datasets.data_config import DEFAULT_DOP, HINTSETS, DOPS
from src.models.hero.query_explorer import SearchingSettings

EMPTY_SS = SearchingSettings(
    hardcoded_hintsets=[],
    hardcoded_dops=[],
)

GREEDY_DEF_DOP_SS = SearchingSettings(
    disable_ops=True,
    max_iter=float("inf"),
)

PRUNED_GREEDY_DEF_DOP_SS = SearchingSettings(
    disable_ops=True,
    max_iter=1,
)

GREEDY_SS = SearchingSettings(
    disable_ops=True,
    decrease_dop=True,
    max_iter=float("inf"),
)

PRUNED_GREEDY_SS = SearchingSettings(
    disable_ops=True,
    decrease_dop=True,
    max_iter=1,
)

LOCAL_DEF_DOP_SS = SearchingSettings(
    disable_ops=True,
    disable_inl=True,
    force_join=True,
    force_only_nl=True,
    max_iter=float("inf"),
)

PRUNED_LOCAL_SS = SearchingSettings(
    disable_ops=True,
    disable_inl=True,
    force_join=True,
    force_only_nl=True,
    max_iter=1,
)

PRUNED_LOCAL_DEF_DOP_SS = SearchingSettings(
    disable_ops=True,
    disable_inl=True,
    force_join=True,
    force_only_nl=True,
    max_iter=1,
)

PRUNED_LOCAL_SS = SearchingSettings(
    disable_ops=True,
    decrease_dop=True,
    disable_inl=True,
    use_joined_search=True,
    force_join=True,
    force_only_nl=True,
    max_iter=1,
)

LOCAL_SS = SearchingSettings(
    disable_ops=True,
    decrease_dop=True,
    disable_inl=True,
    use_joined_search=True,
    force_join=True,
    force_only_nl=True,
    max_iter=float("inf"),
)

ALL_SS = SearchingSettings(
    hardcoded_hintsets=HINTSETS,
    hardcoded_dops=DOPS,
)

ALL_DEF_DOP_SS = SearchingSettings(
    hardcoded_hintsets=HINTSETS,
    hardcoded_dops=[DEFAULT_DOP],
)
