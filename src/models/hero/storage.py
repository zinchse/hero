"""
Here's the simplest scheme of storage module, which uses the 
graph structure of the problem. The node in it is the actually 
observed plan, and parameters (hintset and dop) are used exclusively 
as a way of transition from one plan to another plan, i.e. they are
 *an attribute of transition between plans*. Another important attribute 
 of the transition between plans is obviously the observed acceleration (boost).
"""

from typing import List, Set, Dict
from collections import defaultdict, namedtuple
from src.datasets.data_types import HintsetCode, QueryDop, TemplateID, Parameter
from src.datasets.data_config import DEFAULT_HINTSET, DEFAULT_DOP
from src.config import DISTANCE_THRESHOLD

Node = namedtuple("Node", ["logical_plan", "template_id", "selectivities", "cardinalities"])
Transition = namedtuple("Transition", ["node_from", "node_to"])
Edge = namedtuple("Edge", ["node_from", "node_to", "boosts", "parameters"])


class Storage:
    def __init__(self, default_hintset: "HintsetCode" = DEFAULT_HINTSET, default_dop: "QueryDop" = DEFAULT_DOP):
        self.logical_plan_to_edges: "Dict" = defaultdict(list)
        self._template_to_promised_params: "Dict" = defaultdict(set)
        self.threshold = DISTANCE_THRESHOLD
        self.default_hintset = default_hintset
        self.default_dop = default_dop

    @staticmethod
    def get_distance(x: "Node", y: "Node") -> "float":
        if x.logical_plan != y.logical_plan:
            return float("inf")
        return max(
            max((max(xi / yi, yi / xi) for xi, yi in zip(x.cardinalities, y.cardinalities))),
            max((max(xi / yi, yi / xi) for xi, yi in zip(x.selectivities, y.selectivities))),
        )

    def add_info(self, transition: "Transition", boost: "float", parameter: "Parameter") -> "None":
        new_edge = Edge(transition.node_from, transition.node_to, [boost], [parameter])
        if boost > 0:
            self._template_to_promised_params[transition.node_from.template_id].add(parameter)
        for edge in self.logical_plan_to_edges[transition.node_from.logical_plan]:
            if edge.node_from == transition.node_from and edge.node_to == transition.node_to:
                edge.boosts.append(boost)
                edge.parameters.append(parameter)
                break
        else:
            self.logical_plan_to_edges[transition.node_from.logical_plan].append(new_edge)

    def get_promised_parameters(self, template_id: "TemplateID") -> "Set[Parameter]":
        return self._template_to_promised_params[template_id]

    def _get_relevant_edges(self, node: "Node") -> "List[Edge]":
        min_dist, argmin = float("inf"), None
        for cur_edge in self.logical_plan_to_edges[node.logical_plan]:
            cur_dist = self.get_distance(cur_edge.node_from, node)
            if cur_dist < min_dist and cur_dist < self.threshold:
                min_dist, argmin = cur_dist, cur_edge.node_from
        return [] if argmin is None else self.logical_plan_to_edges[argmin.logical_plan]

    def estimate_transition(self, transition: "Transition") -> "float":
        if transition.node_from == transition.node_to:
            return 0.0
        max_boost = float("-inf")
        for cur_edge in self._get_relevant_edges(transition.node_from):
            if self.get_distance(transition.node_to, cur_edge.node_to) > self.threshold:
                continue
            max_boost = max(max_boost, sum(cur_edge.boosts) / len(cur_edge.boosts))
        return max_boost
