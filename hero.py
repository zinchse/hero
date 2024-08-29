"""
Here's a model based on the logic of the `Hero` algorithm that uses a local search procedure to explore queries
and report information about it to the graph storage. 

Inference looks as follows:
- get the template_id 
- fetch all potentially useful parameters from the graph storage
- plan them in parallel mode
- select those parameters which, judging by experience in graph storage, have the highest boost

Note, that in this situation the inference time is equal to the maximum planning time of one potentially good
parameter, which is much less than the time of a complete enumeration (albeit in parallel) and the running 
time of the greedy (which suffers because of the need for sequential planning calls)
"""

from typing import Tuple, Optional, List
from functools import lru_cache
from predictor import Predictor
from hbo_bench.data_types import Parameter, Time, QueryName, HintsetCode, QueryDop
from hbo_bench.data_config import DEFAULT_DOP, DEFAULT_HINTSET
from hbo_bench.local_search_settings import SearchingSettings
from hbo_bench.query_explorer import QueryExplorer
from storage import Storage, Node, Transition
from btcnn.regressor import BinaryTreeRegressor
from wrappers import (
    _get_oracle,
    _get_planning_time,
    _get_e2e_time,
    _get_cardinalities,
    _get_logical_tree,
    _get_selectivities,
    _get_template_id,
)


@lru_cache(None)
def _make_node(q_n: "QueryName", hs: "HintsetCode", dop: "QueryDop") -> "Node":
    return Node(
        _get_logical_tree(q_n, hs, dop),
        _get_template_id(q_n),
        _get_cardinalities(q_n, hs, dop),
        _get_selectivities(q_n, hs, dop),
    )


class Hero(Predictor):
    def __init__(
        self,
        fit_settings: "SearchingSettings",
        inference_settings: "SearchingSettings" = SearchingSettings(),
        execution_model: "Optional[BinaryTreeRegressor]" = None,
    ):
        super().__init__(fit_settings, inference_settings, execution_model)
        self.storage = Storage()

    def predict(self, query_name: "QueryName") -> "Tuple[Time, Parameter]":
        params = self.storage.get_promised_parameters(_get_template_id(query_name))
        assert len(params) + 1 <= self.fit_settings.default_dop, "Can't use single-batch planning"
        if not params:
            def_hs, def_dop = self.fit_settings.default_hintset, self.fit_settings.default_dop
            return _get_planning_time(query_name, def_hs, def_dop), Parameter(def_hs, def_dop)
        def_node = _make_node(query_name, self.fit_settings.default_hintset, self.fit_settings.default_dop)
        best_param = max(
            params, key=lambda p: self.storage.estimate_transition(Transition(def_node, _make_node(query_name, *p)))
        )
        inference_time = max((_get_planning_time(query_name, *p) for p in params), default=0)
        return inference_time, best_param

    def fit(self, list_q_n: "List[QueryName]") -> "Time":
        return sum(self.explore(q_n) for q_n in list_q_n)

    def explore(self, query_name: "QueryName") -> "Time":
        explorer = QueryExplorer(_get_oracle(query_name), query_name, self.fit_settings)
        hs, dop = explorer.run()
        transition = Transition(
            _make_node(query_name, self.default_hintset, self.default_dop), _make_node(query_name, hs, dop)
        )
        if (hs, dop) != (DEFAULT_HINTSET, DEFAULT_DOP):
            boost = _get_e2e_time(query_name, self.default_hintset, self.default_dop) - _get_e2e_time(
                query_name, hs, dop
            )
            self.storage.add_info(transition, boost, Parameter(hs, dop))
        return explorer.parallel_e2e_time
