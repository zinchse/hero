"""
Here's an implementation of a predictor based on neural network estimates.

For the efficiency of experiments, the part of the logic related to exhaustive search is moved 
to a separate functions for the possibility of using batch inference on the GPU. 
"""

from typing import Optional, List, Tuple, Set
from heapq import heappop, heappush
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from src.utils import extract_vertices_and_edges
from src.datasets.binary_tree_dataset import WeightedBinaryTreeDataset, weighted_binary_tree_collate
from src.datasets.data_types import Time, QueryName, Parameter, QueryDop, HintsetCode
from src.config import MAX_TREE_LENGTH
from src.models.predictor import Predictor
from src.models.neural_network.regressor import BinaryTreeRegressor
from src.models.hero.query_explorer import QueryExplorer, SearchingSettings
from src.wrappers import (
    _get_processed_v_e,
    _get_planning_time,
    _get_logical_tree,
    _get_execution_time,
    _get_explain_plan,
)
from src.models.neural_network.train_utils import weighted_train_loop, set_seed, DEFAULT_LR, DEFAULT_BATCH_SIZE


class NN(Predictor):
    def __init__(
        self,
        fit_settings: "SearchingSettings",
        model: "BinaryTreeRegressor",
        inference_settings: "SearchingSettings" = SearchingSettings(),
        execution_model: "Optional[BinaryTreeRegressor]" = None,
        path_to_save: "Optional[str]" = None,
        lr: "float" = DEFAULT_LR,
        batch_size: "int" = DEFAULT_BATCH_SIZE,
    ):
        super().__init__(fit_settings, inference_settings, execution_model)

        self.model = model
        self.path_to_save = path_to_save
        self._lr = lr
        self._batch_size = batch_size

        self.list_vertices: "List[Tensor]" = []
        self.list_edges: "List[Tensor]" = []
        self.list_times: "List[Tensor]" = []
        self.seen: "Set[Tuple[QueryName, HintsetCode, QueryDop]]" = set()

        self.dataloader = DataLoader(WeightedBinaryTreeDataset([], [], [], self.model.device))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=20)

    def fit(self, list_q_n: "List[QueryName]", epochs: "int" = 100) -> "Time":
        learning_time, history = self.explore_and_get_history(list_q_n)
        self.add_history(history)
        self.update_model(epochs)
        return learning_time

    def predict(self, query_name: "QueryName") -> "Tuple[Time, Parameter]":
        if self.inference_settings.hardcoded_hintsets and self.inference_settings.hardcoded_dops:
            return self._get_exhaustive_prediction(
                query_name, self.inference_settings.hardcoded_hintsets, self.inference_settings.hardcoded_dops
            )
        return self.get_local_search_prediction(query_name)

    def get_local_search_prediction(self, query_name: "QueryName") -> "Tuple[Time, Parameter]":
        explorer = QueryExplorer(query_name, self.inference_settings, execution_model=self.model)
        hs, dop = explorer.run()
        return explorer.parallel_planning_time, Parameter(hs, dop)

    def _get_exhaustive_prediction(
        self, query_name: "QueryName", hintsets: "List[HintsetCode]", dops: "List[QueryDop]"
    ) -> "Tuple[Time, Parameter]":
        """special version of prediction for avoiding local search exploration and abusing the batch processing"""
        v_list, e_list = [], []
        parameters = []
        planning_times = []

        for hs in hintsets:
            for dop in dops:
                v, e = _get_processed_v_e(query_name, hs, dop)
                v_list.append(v)
                e_list.append(e)
                planning_times.append(_get_planning_time(query_name, hs, dop))
                parameters.append(Parameter(hs, dop))

        with torch.no_grad():
            v_batch, e_batch = torch.stack(v_list).to(self.model.device), torch.stack(e_list).to(self.model.device)
            predicted_times = self.model(v_batch, e_batch).squeeze(1).cpu().numpy()
            total_times = predicted_times + planning_times

        return self._calculate_parallel_planning_time(planning_times), parameters[total_times.argmin()]

    def _calculate_parallel_planning_time(self, planning_times: "List[Time]") -> "Time":
        threads: "List[float]" = [0.0 for _ in range(self.inference_settings.default_dop)]
        last_time = 0.0
        for plan_time in sorted(planning_times):
            finish_time = heappop(threads) + plan_time
            heappush(threads, finish_time)
            last_time = max(last_time, finish_time)
        return last_time

    def explore_and_get_history(
        self, list_q_n: "List[QueryName]"
    ) -> "Tuple[Time, List[Tuple[QueryName, HintsetCode, QueryDop]]]":
        if self.fit_settings.hardcoded_hintsets and self.fit_settings.hardcoded_dops:
            return self._explore_and_get_exhaustive_history(
                list_q_n, self.fit_settings.hardcoded_hintsets, self.fit_settings.hardcoded_dops
            )
        learning_time = 0.0
        history = []
        for q_n in list_q_n:
            explorer = QueryExplorer(q_n, self.fit_settings)
            explorer.run()
            learning_time += explorer.parallel_e2e_time
            for hs, dop in explorer.explored_states:
                history.append((q_n, hs, dop))
        return learning_time, history

    def _explore_and_get_exhaustive_history(
        self, list_q_n: "List[QueryName]", hintsets: "List[HintsetCode]", dops: "List[QueryDop]"
    ) -> "Tuple[Time, List[Tuple[QueryName, HintsetCode, QueryDop]]]":
        """special version of prediction for avoiding the local search exploration"""
        planning_time = 0.0
        ex_time = 0.0
        exhaustive_history = []
        for q_n in list_q_n:
            seen_trees = set()
            for hs in hintsets:
                for dop in dops:
                    exhaustive_history.append((q_n, hs, dop))
                    planning_time += _get_planning_time(q_n, hs, dop)
                    tree = _get_logical_tree(q_n, hs, dop)
                    ex_time += _get_execution_time(q_n, hs, dop) * (tree not in seen_trees)
                    seen_trees.add(tree)
        return planning_time + ex_time, exhaustive_history

    def add_history(self, history: "List[Tuple[QueryName, HintsetCode, QueryDop]]") -> "None":
        for q_n, hs, dop in history:
            if (q_n, hs, dop) in self.seen:
                continue
            self.seen.add((q_n, hs, dop))
            v, e = extract_vertices_and_edges(_get_explain_plan(q_n, hs, dop))
            time = _get_execution_time(q_n, hs, dop)
            self.list_vertices.append(v)
            self.list_edges.append(e)
            self.list_times.append(torch.Tensor([time]))
        self._prepare_dataloader()

    def _prepare_dataloader(self) -> "None":
        self.dataloader = DataLoader(
            dataset=WeightedBinaryTreeDataset(self.list_vertices, self.list_edges, self.list_times, self.model.device),
            batch_size=self._batch_size,
            shuffle=True,
            collate_fn=lambda el: weighted_binary_tree_collate(el, MAX_TREE_LENGTH),
            drop_last=False,
        )

    def update_model(self, epochs: "int") -> "None":
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=20)
        set_seed(2024)
        weighted_train_loop(
            model=self.model,
            optimizer=self.optimizer,
            criterion=nn.MSELoss(reduction="none"),
            scheduler=self.scheduler,
            train_dataloader=self.dataloader,
            num_epochs=epochs,
            ckpt_period=epochs,
            path_to_save=self.path_to_save,
        )