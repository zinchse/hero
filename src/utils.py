from typing import List, Tuple, Set, Dict
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from src.datasets.data_types import (
    QueryName,
    QueryDop,
    HintsetCode,
    ExplainNode,
)
from src.datasets.binary_tree_dataset import paddify_sequences, WeightedBinaryTreeDataset
from src.models.regressor import BinaryTreeRegressor
from src.datasets.oracle import Oracle, OracleRequest

MAX_TREE_LENGTH = 66  # hardcoded value


def get_logical_plan(query_name: "QueryName", oracle: "Oracle", hintset: "HintsetCode", dop: "QueryDop") -> "str":
    request = OracleRequest(query_name=query_name, hintset=hintset, dop=dop)
    plan = oracle.get_explain_plan(request=request)

    res = []

    def recurse(node: "ExplainNode") -> "None":
        res.append(f"{node.node_type} (Rel={node.relation_name}|Index={node.index_name})")
        res.append("[")
        for child in node.plans:
            recurse(child)
        res.append("]")

    recurse(node=plan.plan)
    return " ".join(res)


def get_full_plan(query_name: "QueryName", oracle: "Oracle", hintset: "HintsetCode", dop: "QueryDop") -> "str":
    request = OracleRequest(query_name=query_name, hintset=hintset, dop=dop)
    plan = oracle.get_explain_plan(request=request)

    res = []

    def recurse(node: "ExplainNode") -> "None":
        res.append(
            f"{node.node_type} (Rel={node.relation_name}|Index={node.index_name}|Cards={node.estimated_cardinality})"
        )
        res.append("[")
        for child in node.plans:
            recurse(child)
        res.append("]")

    recurse(node=plan.plan)
    return " ".join(res)


def get_template_id(query_name: "QueryName", oracle: "Oracle", hintset: "HintsetCode", dop: "QueryDop") -> "int":
    def_request = OracleRequest(query_name=query_name, hintset=hintset, dop=dop)
    default_plan = oracle.get_explain_plan(request=def_request)
    return default_plan.template_id


def get_selectivities(
    query_name: "QueryName", oracle: "Oracle", hintset: "HintsetCode", dop: "QueryDop"
) -> "List[float]":

    request = OracleRequest(query_name=query_name, hintset=hintset, dop=dop)
    plan = oracle.get_explain_plan(request=request)

    res = []

    def recurse(node: "ExplainNode") -> "None":
        max_possible_size = 1
        current_size = node.estimated_cardinality
        for child in node.plans:
            max_possible_size *= child.estimated_cardinality
        res.append(current_size / max_possible_size)

        for child in node.plans:
            recurse(child)

    recurse(node=plan.plan)
    return res


def preprocess(v: "Tensor", e: "Tensor") -> "Tuple[Tensor, Tensor]":
    """unifies tensors from dataset with tensors from dataloader; see `weighted_binary_tree_collate`"""
    v, e = v.clone(), e.clone()
    v = torch.stack(paddify_sequences([v], MAX_TREE_LENGTH)).transpose(1, 2)[0]
    e = torch.stack(paddify_sequences([e], MAX_TREE_LENGTH)).unsqueeze(1)[0]
    return v, e


def get_structure(v: "Tensor", e: "Tensor") -> "Tuple[str, str]":
    """cleans cards (and selectivities) and returns hashable repr for v, e"""
    v = v.clone()
    v[-2:, :] = 0
    return str(v.flatten().tolist()), str(e.flatten().tolist())


def get_tree(v: "Tensor", e: "Tensor") -> "Tuple[str, str]":
    """returns hashable repr for v, e"""
    v = v.clone()
    return str(v.flatten().tolist()), str(e.flatten().tolist())


def load_run(
    run: "int",
    trainval_dataset: "Dataset",
    model: "BinaryTreeRegressor",
    ckpt_path: "str",
    device: "torch.device",
) -> "Tuple[BinaryTreeRegressor, Dataset, Dataset]":
    generator = torch.Generator().manual_seed(42 + run - 1)
    train_dataset, val_dataset = torch.utils.data.dataset.random_split(
        trainval_dataset, [0.8, 0.2], generator=generator
    )
    ckpt_state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt_state["model_state_dict"])
    model = model.to(device)
    return model, train_dataset, val_dataset


def featurize_dataset(
    dataset: "WeightedBinaryTreeDataset",
    model: "BinaryTreeRegressor",
    train_structures: "Set[str]",
    train_trees: "Set[str]",
    data_type: "str",
) -> "pd.DataFrame":
    df = pd.DataFrame(iter(dataset), columns=["vertices", "edges", "frequency", "time"])
    df["data_type"] = data_type

    df["time_category"] = "small"
    df.loc[df["time"] > 0.2, "time_category"] = "medium"
    df.loc[df["time"] > 4, "time_category"] = "big"

    df["structure"] = df.apply(lambda row: get_structure(*preprocess(row["vertices"], row["edges"])), axis=1)
    df["structure_category"] = "unseen"
    df.loc[df["structure"].isin(train_structures), "structure_category"] = "seen"

    df["tree"] = df.apply(lambda row: get_tree(*preprocess(row["vertices"], row["edges"])), axis=1)
    df["tree_category"] = "unseen"
    df.loc[df["tree"].isin(train_trees), "tree_category"] = "seen"

    with torch.no_grad():
        df["embedding"] = df.apply(
            lambda row: model.btcnn(*[t.unsqueeze(0) for t in preprocess(row["vertices"], row["edges"])])
            .squeeze(0)
            .to("cpu"),
            axis=1,
        )
        df["prediction"] = df.apply(
            lambda row: model(*[t.unsqueeze(0) for t in preprocess(row["vertices"], row["edges"])]).to("cpu"), axis=1
        )

    for col in ["prediction", "frequency", "time"]:
        df[col] = df.apply(lambda row, col=col: row[col].item(), axis=1)

    df["error"] = df["prediction"] - df["time"]
    df["prediction_category"] = "underestimated"
    df.loc[df["error"] > 0, "prediction_category"] = "overestimated"

    return df


def filter_df(
    df: "pd.DataFrame",
    params: "Dict[str, str]",
) -> "pd.DataFrame":
    idx = df["data_type"] == params["data_type"]
    if params.get("structure", "all") != "all":
        idx = idx & (df["structure_category"] == params["structure"])
    if params.get("tree", "all") != "all":
        idx = idx & (df["tree_category"] == params["tree"])
    if params.get("time", "all") != "all":
        idx = idx & (df["time_category"] == params["time"])
    if params.get("prediction", "all") != "all":
        idx = idx & (df["prediction_category"] == params["prediction"])
    return df.loc[idx]
