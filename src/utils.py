"""
This module provides commonly used functions for processing of oracle's data 
(for logical tree, cardinalities, and so on). Unlike the `wrapper.py`'s 
functions, they require passing an oracle directly.

It also contains a simple script to emulate the use of a NN in an online scenario.
Note that the time calculation assumes that all predictors during the inference, 
in addition to searching for parameters, **will also construct a corresponding plan**.
"""

from collections import defaultdict
from typing import List, Tuple, TypedDict
import torch
from torch import Tensor
from src.datasets.data_types import QueryName, QueryDop, HintsetCode, ExplainNode, Cardinality, Selectivity
from src.datasets.binary_tree_dataset import paddify_sequences
from src.datasets.oracle import Oracle, OracleRequest, TIMEOUT
from src.datasets.vectorization import extract_vertices_and_edges
from src.datasets.data_config import HINTSETS, DOPS, DEFAULT_HINTSET
from src.config import MAX_TREE_LENGTH


def get_logical_tree(
    query_name: "QueryName", oracle: "Oracle", hintset: "HintsetCode", dop: "QueryDop", with_rels: "bool" = True
) -> "str":
    request = OracleRequest(query_name=query_name, hintset=hintset, dop=dop)
    plan = oracle.get_explain_plan(request=request)
    res = []

    def recurse(node: "ExplainNode") -> "None":
        if with_rels:
            res.append(f"{node.node_type} (Rel={node.relation_name}|Index={node.index_name})")
        else:
            res.append(f"{node.node_type}")
        res.append("[")
        for child in node.plans:
            recurse(child)
        res.append("]")

    recurse(node=plan.plan)
    return " ".join(res)


def get_full_plan(
    query_name: "QueryName", oracle: "Oracle", hintset: "HintsetCode", dop: "QueryDop", with_rels: "bool" = True
) -> "str":
    request = OracleRequest(query_name=query_name, hintset=hintset, dop=dop)
    plan = oracle.get_explain_plan(request=request)
    res = []

    def recurse(node: "ExplainNode") -> "None":
        node_type, cardinalities = node.node_type, node.estimated_cardinality
        if with_rels:
            rel_name, index_name = node.relation_name, node.index_name
            res.append(f"{node_type} (Rel={rel_name}|Index={index_name}|Cards={cardinalities})")
        else:
            res.append(f"{node_type} (Cards={cardinalities})")
        res.append("[")
        for child in node.plans:
            recurse(child)
        res.append("]")

    recurse(node=plan.plan)
    return " ".join(res)


def get_selectivities(
    query_name: "QueryName", oracle: "Oracle", hintset: "HintsetCode", dop: "QueryDop"
) -> "List[Selectivity]":
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


def get_cardinalities(
    query_name: "QueryName", oracle: "Oracle", hintset: "HintsetCode", dop: "QueryDop"
) -> "List[Cardinality]":
    request = OracleRequest(query_name=query_name, hintset=hintset, dop=dop)
    plan = oracle.get_explain_plan(request=request)
    res = []

    def recurse(node: "ExplainNode") -> "None":
        res.append(node.estimated_cardinality)
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


class QueryInfo(TypedDict):
    query_name: "QueryName"
    hintset: "HintsetCode"
    dop: "QueryDop"
    vertices: "Tensor"
    edges: "Tensor"
    time: "Tensor"


def extract_list_info(oracle: "Oracle", query_names: "List[QueryName]") -> "List[QueryInfo]":
    """initial plan processing and T/O handling with search for maximum lower bound of execution time"""
    list_info = []

    for query_name in query_names:
        seen_logical_plans = set()
        timeouted_logical_plans_to_dops = defaultdict(set)
        timeouted_logical_plans_to_settings = defaultdict(list)
        logical_plan_to_times = defaultdict(list)

        for dop in DOPS:
            for hintset in HINTSETS:
                custom_request = OracleRequest(query_name=query_name, hintset=hintset, dop=dop)
                custom_logical_plan = get_logical_tree(query_name=query_name, oracle=oracle, hintset=hintset, dop=dop)
                custom_time = oracle.get_execution_time(custom_request)
                if custom_time != TIMEOUT:
                    time = torch.tensor(custom_time / 1000, dtype=torch.float32)
                    vertices, edges = extract_vertices_and_edges(oracle.get_explain_plan(request=custom_request))
                    seen_logical_plans.add(custom_logical_plan)
                    info: "QueryInfo" = {
                        "query_name": query_name,
                        "hintset": hintset,
                        "dop": dop,
                        "time": time,
                        "vertices": vertices,
                        "edges": edges,
                    }
                    list_info.append(info)
                    logical_plan_to_times[custom_logical_plan].append(time)
                else:
                    timeouted_logical_plans_to_dops[custom_logical_plan].add(dop)
                    timeouted_logical_plans_to_settings[custom_logical_plan].append((dop, hintset))

        for custom_logical_plan in timeouted_logical_plans_to_settings:
            if custom_logical_plan in logical_plan_to_times:
                time = torch.mean(torch.stack(logical_plan_to_times[custom_logical_plan]))
            else:
                max_def_time = 0.0
                for dop in timeouted_logical_plans_to_dops[custom_logical_plan]:
                    def_request = OracleRequest(query_name=query_name, hintset=DEFAULT_HINTSET, dop=dop)
                    def_time = oracle.get_execution_time(request=def_request)
                    max_def_time = max(max_def_time, def_time)
                time = torch.tensor(2 * max_def_time / 1000, dtype=torch.float32)

            for dop, hintset in timeouted_logical_plans_to_settings[custom_logical_plan]:
                custom_request = OracleRequest(query_name=query_name, hintset=hintset, dop=dop)
                vertices, edges = extract_vertices_and_edges(oracle.get_explain_plan(request=custom_request))
                timeouted_info: "QueryInfo" = {
                    "query_name": query_name,
                    "hintset": hintset,
                    "dop": dop,
                    "time": time,
                    "vertices": vertices,
                    "edges": edges,
                }

                list_info.append(timeouted_info)

    return list_info
