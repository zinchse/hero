from typing import Dict
from src.datasets.oracle import Oracle
from src.models.hero.hero import Hero
from src.models.hero.local_search_settings import PRUNED_GREEDY_SS, PRUNED_LOCAL_SS, EMPTY_SS
from src.wrappers import _get_execution_time, _get_e2e_time


def test_superiority_of_local_strategy(oracles_dict: "Dict[str, Oracle]"):
    for _, oracle in oracles_dict.items():
        for q_n in oracle.get_query_names():
            local_heromodel = Hero(fit_settings=PRUNED_LOCAL_SS, inference_settings=EMPTY_SS)
            local_heromodel.fit([q_n])
            local_inference, local_pred = local_heromodel.predict(q_n)
            local_e2e_time = local_inference + _get_execution_time(q_n, *local_pred)

            greedy_heromodel = Hero(fit_settings=PRUNED_GREEDY_SS, inference_settings=EMPTY_SS)
            greedy_heromodel.fit([q_n])
            greedy_inference, greedy_pred = greedy_heromodel.predict(q_n)
            greedy_e2e_time = greedy_inference + _get_execution_time(q_n, *greedy_pred)

            assert greedy_e2e_time >= local_e2e_time


def test_empty_search(oracles_dict: "Dict[str, Oracle]"):
    for _, oracle in oracles_dict.items():
        for q_n in oracle.get_query_names():
            empty_heromodel = Hero(fit_settings=EMPTY_SS, inference_settings=PRUNED_LOCAL_SS)
            empty_heromodel.fit([q_n])
            empty_inference, empty_pred = empty_heromodel.predict(q_n)
            empty_e2e_time = empty_inference + _get_execution_time(q_n, *empty_pred)

            def_e2e_time = _get_e2e_time(q_n, EMPTY_SS.default_hintset, EMPTY_SS.default_dop)

            assert def_e2e_time == empty_e2e_time
