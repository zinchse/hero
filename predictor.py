"""
Here's the main interface of the model for predicting hintset and dop. 
For simplicity of implementation we use as identifiers of queries their names, 
in the further they are captured by corresponding wrappers for modelling of 
interaction with the database.

A key feature is that the model can procedure local search (including both 
full search, the greedy algorithm, and the procedure we introduced) for both 
query exploration and inference. This allows us to cross `Bao`, `AutoSteer` and
`Hero` approaches together.
"""

from typing import Tuple, List, Optional
from hbo_bench.data_types import Time, Parameter, QueryName
from hbo_bench.local_search_settings import SearchingSettings
from btcnn.regressor import BinaryTreeRegressor


class Predictor:
    def __init__(
        self,
        fit_settings: "SearchingSettings",
        inference_settings: "SearchingSettings" = SearchingSettings(),
        execution_model: "Optional[BinaryTreeRegressor]" = None,
    ):
        self.fit_settings = fit_settings
        self.inference_settings = inference_settings
        self.execution_model = execution_model
        self.default_hintset = self.fit_settings.default_hintset
        self.default_dop = self.fit_settings.default_dop

    def predict(self, query_name: "QueryName") -> "Tuple[Time, Parameter]":
        raise NotImplementedError  # pragma: no cover

    def fit(self, list_q_n: "List[QueryName]") -> "Time":
        raise NotImplementedError  # pragma: no cover
