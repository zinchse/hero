from typing import Dict
import torch
from src.emulation import emulate_online_learning, get_report
from src.datasets.oracle import Oracle
from src.models.hero.local_search_settings import PRUNED_LOCAL_DEF_DOP_SS, ALL_DEF_DOP_SS, EMPTY_SS
from src.models.neural_network.nn import NN
from src.models.neural_network.train_utils import load_model


def test_smoke(oracles_dict: "Dict[str, Oracle]") -> "None":
    for workload_name, oracle in oracles_dict.items():
        workload = oracle.get_query_names()[:1]
        list_reports = []
        epochs, n_runs = 1, 2
        for ss, ss_descr in [
            (PRUNED_LOCAL_DEF_DOP_SS, "PRUNED LOCAL"),
            (ALL_DEF_DOP_SS, "EXHAUSTIVE"),
        ]:
            list_reports.append(
                emulate_online_learning(
                    model_name="NN",
                    workload=workload,
                    workload_name=workload_name,
                    ss=ss,
                    ss_descr=ss_descr,
                    only_def_dop=True,
                    with_default_data=True,
                    epochs=epochs,
                    n_runs=n_runs,
                    path_to_save="/tmp/model.pth",
                )
            )
        model = NN(
            fit_settings=EMPTY_SS,
            inference_settings=ss,
            model=load_model(torch.device("cpu"), "/tmp/model.pth"),
        )

        assert list_reports[-1][-1] == get_report(model, "NN", workload, workload_name, ss, ss_descr, True)
