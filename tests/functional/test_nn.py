import pytest
import torch
from src.datasets.data_types import QueryName
from src.models.neural_network.nn import NN
from src.models.neural_network.regressor import get_bt_regressor
from src.models.hero.local_search_settings import ALL_SS, PRUNED_LOCAL_SS
from src.wrappers import _get_e2e_time, _get_execution_time


@pytest.fixture
def q_n() -> "QueryName":
    return "29c"


def test_boost(q_n: "QueryName"):
    regressor = get_bt_regressor("dummy_nn", torch.device("cpu"))
    nnmodel = NN(fit_settings=ALL_SS, inference_settings=ALL_SS, model=regressor, batch_size=8)
    nnmodel.fit([q_n], epochs=30)
    _, pred = nnmodel.predict(q_n)
    def_params = nnmodel.fit_settings.default_hintset, nnmodel.fit_settings.default_dop
    def_time = _get_e2e_time(q_n, *def_params)
    custom_time = _get_e2e_time(q_n, *pred)
    assert custom_time < def_time


def test_local_search_applying(q_n: "QueryName"):
    regressor = get_bt_regressor("dummy_nn", torch.device("cpu"))
    exhaustive_nnmodel = NN(fit_settings=ALL_SS, inference_settings=ALL_SS, model=regressor, batch_size=8)
    exhaustive_nnmodel.fit([q_n], epochs=30)
    exhaustive_inference_time, exhaustive_pred = exhaustive_nnmodel.predict(q_n)
    exhaustive_e2e = exhaustive_inference_time + _get_execution_time(q_n, *exhaustive_pred)

    regressor = get_bt_regressor("dummy_nn", torch.device("cpu"))
    pruned_local_nnmodel = NN(
        fit_settings=PRUNED_LOCAL_SS, inference_settings=PRUNED_LOCAL_SS, model=regressor, batch_size=8
    )
    pruned_local_nnmodel.fit([q_n], epochs=30)
    pruned_local_inference_time, pruned_local_pred = pruned_local_nnmodel.predict(q_n)
    pruned_local_e2e = pruned_local_inference_time + _get_execution_time(q_n, *pruned_local_pred)

    assert exhaustive_inference_time > pruned_local_inference_time
    assert exhaustive_e2e > pruned_local_e2e
