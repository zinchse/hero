from typing import List
from src.datasets.data_types import (
    QueryName,
    QueryDop,
    HintsetCode,
    ExplainNode,
)

from src.datasets.oracle import Oracle, OracleRequest


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
