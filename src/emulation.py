"""
Small procedures are implemented here to emulate the online scenario and gather information 
about the behaviour of the predictor. 
"""

from itertools import product
from typing import List, Tuple, TypedDict, Optional
import torch
from src.datasets.data_types import QueryName, QueryDop, HintsetCode
from src.models.hero.query_explorer import SearchingSettings
from src.models.hero.local_search_settings import EMPTY_SS
from src.models.neural_network.regressor import get_bt_regressor
from src.models.predictor import Predictor
from src.models.neural_network.nn import NN
from src.datasets.oracle import TIMEOUT
from src.datasets.data_config import HINTSETS, DOPS, DEFAULT_HINTSET, DEFAULT_DOP
from src.wrappers import _get_e2e_time, _get_execution_time, _get_planning_time


class Report(TypedDict):
    workload: "str"
    searching_settings: "str"
    model: "str"
    only_def_dop: "bool"
    def_ex: "float"
    def_inference: "float"
    def_e2e: "float"
    custom_ex: "float"
    custom_inference: "float"
    custom_e2e: "float"
    opt_ex: "float"
    opt_inference: "float"
    opt_e2e: "float"
    n_timeouts: "int"
    predictions: "List[Tuple[QueryName, HintsetCode, QueryDop]]"
    boosts: "List[float]"


def get_report(
    model: "Predictor",
    model_name: "str",
    workload: "List[QueryName]",
    workload_name: "str",
    ss: "SearchingSettings",
    ss_descr: "str",
    only_def_dop: "bool" = False,
) -> "Report":
    """one useful property: all our models build resulting plans during inference"""
    report: "Report" = {
        "workload": workload_name,
        "searching_settings": ss_descr,
        "model": model_name,
        "only_def_dop": only_def_dop,
        "def_ex": 0.0,
        "def_inference": 0.0,
        "def_e2e": 0.0,
        "custom_ex": 0.0,
        "custom_inference": 0.0,
        "custom_e2e": 0.0,
        "opt_ex": 0.0,
        "opt_inference": 0.0,
        "opt_e2e": 0.0,
        "n_timeouts": 0,
        "predictions": [],
        "boosts": [],
    }
    report["def_ex"] = sum(_get_execution_time(q_n, model.default_hintset, model.default_dop) for q_n in workload)
    report["def_inference"] = sum(_get_planning_time(q_n, DEFAULT_HINTSET, DEFAULT_DOP) for q_n in workload)
    report["def_e2e"] = report["def_ex"] + report["def_inference"]

    for q_n in workload:
        opt_hs, opt_dop, record = DEFAULT_HINTSET, DEFAULT_DOP, float("inf")
        for hs, dop in product(HINTSETS, [DEFAULT_DOP] if only_def_dop else DOPS):
            record, opt_hs, opt_dop = min((_get_e2e_time(q_n, hs, dop), hs, dop), (record, opt_hs, opt_dop))
        report["opt_ex"] += _get_execution_time(q_n, opt_hs, opt_dop)
        report["opt_inference"] += _get_planning_time(q_n, opt_hs, opt_dop)
    report["opt_e2e"] = report["opt_ex"] + report["opt_inference"]

    model.inference_settings = ss
    for q_n in workload:
        inference_time, (hs, dop) = model.predict(q_n)
        report["custom_ex"] += _get_execution_time(q_n, hs, dop)
        report["custom_inference"] += inference_time
        report["predictions"].append((q_n, hs, dop))
        report["boosts"].append(
            _get_e2e_time(q_n, model.default_hintset, model.default_dop) - _get_e2e_time(q_n, hs, dop)
        )
        report["n_timeouts"] += _get_execution_time(q_n, hs, dop, handle_timeout=False) == TIMEOUT / 1000
    report["custom_e2e"] = report["custom_ex"] + report["custom_inference"]

    return report


def emulate_online_learning(
    model_name: "str",
    workload: "List[QueryName]",
    workload_name: "str",
    ss: "SearchingSettings",
    ss_descr: "str",
    only_def_dop: "bool" = False,
    with_default_data: "bool" = True,
    epochs: "int" = 300,
    n_runs: "int" = 10,
    path_to_save: "Optional[str]" = None,
    device: "torch.device" = torch.device("cpu"),
) -> "List[Report]":
    reports = []
    model = NN(
        fit_settings=EMPTY_SS,
        inference_settings=ss,
        model=get_bt_regressor("dummy_nn", device),
        path_to_save=path_to_save,
    )

    if with_default_data:
        default_history = [(q_n, DEFAULT_HINTSET, DEFAULT_DOP) for q_n in workload]
        model.add_history(default_history)
        model.update_model(epochs=epochs)
    reports.append(get_report(model, model_name, workload, workload_name, ss, ss_descr, only_def_dop))

    for _ in range(1, n_runs + 1):
        collected_history = reports[-1]["predictions"]
        model.add_history(collected_history)
        model.update_model(epochs=epochs)
        reports.append(get_report(model, model_name, workload, workload_name, ss, ss_descr, only_def_dop))

    return reports
