from typing import Dict, Tuple, List
import math
import torch
from torch import Tensor
from src.datasets.data_types import ExplainNode, ExplainPlan


ALL_OPERATIONS = [
    "Bitmap Index Scan",
    "Sort",
    "Merge Join",
    "Seq Scan",
    "Streaming(type: LOCAL REDISTRIBUTE dop: 64/64)",
    "Streaming(type: LOCAL ROUNDROBIN dop: 64/1)",
    "Streaming(type: LOCAL REDISTRIBUTE dop: 16/16)",
    "Streaming(type: BROADCAST dop: 16/1)",
    "Materialize",
    "Streaming(type: LOCAL GATHER dop: 1/64)",
    "Result",
    "Index Only Scan",
    "Streaming(type: LOCAL ROUNDROBIN dop: 16/1)",
    "Streaming(type: BROADCAST dop: 16/16)",
    "Streaming(type: LOCAL REDISTRIBUTE dop: 64/1)",
    "Bitmap Heap Scan",
    "Streaming(type: BROADCAST dop: 64/64)",
    "Aggregate",
    "Streaming(type: LOCAL REDISTRIBUTE dop: 16/1)",
    "Hash Join",
    "Nested Loop",
    "Index Scan",
    "Streaming(type: BROADCAST dop: 64/1)",
    "Streaming(type: LOCAL GATHER dop: 1/16)",
    "Hash",
]

ALL_FEATURES = ALL_OPERATIONS + ["Cardinality", "Selectivity"]


def node_to_features(node: "ExplainNode") -> "Dict[str, float]":
    features = {}

    assert node.node_type in ALL_OPERATIONS
    for op in ALL_OPERATIONS:
        features[op] = float(op == node.node_type)

    features["Cardinality"] = math.log(node.estimated_cardinality)

    if not node.plans:
        features["Selectivity"] = 1.0
    else:
        max_possible_size = 1.0
        current_size = node.estimated_cardinality
        for child in node.plans:
            max_possible_size *= child.estimated_cardinality
        features["Selectivity"] = current_size / max_possible_size

    return features


def features_to_tensor(features: "Dict[str, float]") -> "Tensor":
    return torch.tensor([features[f] for f in ALL_FEATURES], dtype=torch.float32)


def node_to_feature_tensor(node: "ExplainNode") -> "Tensor":
    return features_to_tensor(features=node_to_features(node=node))


def extract_vertices_and_edges(plan: "ExplainPlan") -> "Tuple[Tensor, Tensor]":
    """
    Traverses plan and extracts a) embeddings of its nodes and b) its edges.
    Returns 2-d tensor of flattened vertex embeddings and 2-d tensor of edges,
    where `edges[i][j]` contains index of neighbor node `i`.

    P.S. all vertices have exactly 3 edges (if there are less in the plan, we
    use a dummy edge at index 0, where there will be a padding node in the future)
    """
    padding_num = 0
    padding_shift = 1
    vertices: "List[Tensor]" = []
    edges: "List[List[int]]" = []

    def recurse(node: "ExplainNode") -> "None":
        cur_num = len(vertices)
        vertex = node_to_feature_tensor(node)
        vertices.append(vertex)
        edges.append([cur_num + padding_shift])

        for child in node.plans:
            child_num = len(vertices)
            edges[cur_num].append(child_num + padding_shift)
            recurse(node=child)

        n_missing_children = 2 - len(node.plans)
        for _ in range(n_missing_children):
            edges[cur_num].append(padding_num)

    recurse(node=plan.plan)
    return torch.stack(vertices), torch.tensor(edges, dtype=torch.long)
