"""
The module represents an emulation of the logic of a local search 
algorithm for exploring suitable parameters for a query. 

It is assumed that the exploration takes place in parallel mode 
(i.e. within one iteration all candidates are explored simultaneously), 
and any interaction with the database is modeled by the corresponding 
wrapper (`_get_...`).
"""

from typing import Set, Optional, List, Tuple
from collections import namedtuple
from src.datasets.data_types import QueryName, Time
from src.datasets.data_config import DEFAULT_DOP, DEFAULT_HINTSET, DOPS, HINTS
from src.models.neural_network.regressor import BinaryTreeRegressor
from src.wrappers import (
    _get_planning_time,
    _get_execution_time,
    _get_prediction,
)


OFF_INL_HINT = 64 | 8 | 2
N_SCANS = 4
N_JOINS = 3
NL_POS = 2
assert N_SCANS + N_JOINS == len(HINTS)


SearchingState = namedtuple("SearchingState", ["hintset", "dop"], defaults=[DEFAULT_HINTSET, DEFAULT_DOP])

SearchingSettings = namedtuple(
    "SearchingSettings",
    [
        "disable_ops",
        "decrease_dop",
        "disable_inl",
        "relative_boost_threshold",
        "max_iter",
        "use_joined_search",
        "force_join",
        "force_only_nl",
        "default_hintset",
        "default_dop",
        "hardcoded_hintsets",
        "hardcoded_dops",
    ],
    defaults=[
        False,
        False,
        False,
        1.0,
        1,
        False,
        False,
        False,
        DEFAULT_HINTSET,
        DEFAULT_DOP,
        None,
        None,
    ],
)


class QueryExplorer:
    def __init__(
        self,
        query_name: "QueryName",
        settings: "SearchingSettings",
        execution_model: "Optional[BinaryTreeRegressor]" = None,
    ):
        self.query_name = query_name
        self.settings = settings

        self.execution_model = execution_model
        self.tried_states: "Set[SearchingState]" = set()
        self.explored_states: "Set[SearchingState]" = set()

        self.parallel_planning_time = 0.0
        self.parallel_e2e_time = 0.0

    def execute_state(self, state: "SearchingState") -> "Time":
        if self.execution_model:
            return _get_prediction(self.query_name, *state, self.execution_model)
        return _get_execution_time(self.query_name, *state, handle_timeout=False)

    def explore_in_parallel(self, neighbors: "List[SearchingState]", timeout: "Time") -> "Tuple[Time, SearchingState]":
        self.tried_states |= set(neighbors)
        min_e2e_time, best_st = min(
            (_get_planning_time(self.query_name, *st) + self.execute_state(st), st) for st in neighbors
        )
        timeout = min(timeout, min_e2e_time)
        plan_times = [_get_planning_time(self.query_name, *st) for st in neighbors]
        e2e_times = [_get_planning_time(self.query_name, *st) + self.execute_state(st) for st in neighbors]
        self.parallel_planning_time += max(min(plan_time, timeout) for plan_time in plan_times)
        self.parallel_e2e_time += min(min(e2e_time, timeout) for e2e_time in e2e_times)
        if min_e2e_time <= timeout:
            self.explored_states.add(best_st)
        return min_e2e_time, best_st

    def run(self) -> "SearchingState":
        def_state = SearchingState(self.settings.default_hintset, self.settings.default_dop)
        prev_state, record_state, record_time = None, def_state, float("inf")

        it = 0
        while it < self.settings.max_iter and prev_state != record_state:
            timeout, prev_state = record_time / self.settings.relative_boost_threshold, record_state
            neighbors = list(filter(lambda st: st not in self.tried_states, self.get_neighbors(state=record_state)))
            if not neighbors:
                break  # pragma: no cover
            best_ngb_time, best_ngb = self.explore_in_parallel(neighbors, timeout)
            if record_time / best_ngb_time > self.settings.relative_boost_threshold:
                record_state, record_time = best_ngb, best_ngb_time
            it += 1

        return record_state

    def get_neighbors(self, state: "SearchingState") -> "List[SearchingState]":
        current_dop, current_hintset = state.dop, state.hintset
        neighbors = set()

        if self.settings.use_joined_search:
            to_try_dops = DOPS
        else:
            to_try_dops = [current_dop]

        for dop in to_try_dops:
            if self.settings.disable_ops:
                for op_num in range(len(HINTS)):
                    neighbors.add(SearchingState(dop=dop, hintset=current_hintset | (1 << op_num)))

            if self.settings.disable_inl:
                neighbors.add(SearchingState(dop=dop, hintset=current_hintset | (OFF_INL_HINT)))

            if self.settings.decrease_dop:
                for new_dop in [new_dop for new_dop in DOPS if new_dop < dop]:
                    neighbors.add(SearchingState(dop=new_dop, hintset=current_hintset))

            if self.settings.force_join:
                for join_num in range(N_JOINS):
                    saved_scans = ((1 << N_SCANS) - 1) & current_hintset
                    only_one_join = (((1 << N_JOINS) - 1) - (1 << join_num)) << N_SCANS
                    neighbors.add(SearchingState(dop=dop, hintset=only_one_join | saved_scans))

            if self.settings.force_only_nl:
                join_num = NL_POS
                saved_scans = ((1 << N_SCANS) - 1) & current_hintset
                only_one_join = (((1 << N_JOINS) - 1) - (1 << join_num)) << N_SCANS
                neighbors.add(SearchingState(dop=dop, hintset=only_one_join | saved_scans))

        return [state] + list(neighbors)
