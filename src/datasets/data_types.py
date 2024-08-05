"""
Definition of standard types of the most popular objects, advantage has been introduced 
to give semantics and possibility of data validation through the pydantic.
"""

from typing import List, Optional, Dict
from collections import namedtuple
from pydantic import BaseModel, Field

Time = float
Cardinality = int
Selectivity = float
Cost = float
RelationName = str
NodeType = str
TemplateID = int
QueryName = str

QueryDop = int
Hint = str
HintsetCode = int
GUC = str
Settings = str  # str((query_dop, hintset_code)), we need it to simplify serialization
Parameter = namedtuple("Parameter", ["hintset", "dop"])


class ExplainNode(BaseModel):
    node_type: "NodeType" = Field(alias="Node Type")
    plans: "List[ExplainNode]" = Field(default=[], alias="Plans")
    estimated_cardinality: "Cardinality" = Field(alias="Plan Rows")
    index_name: "Optional[RelationName]" = Field(default=None, alias="Index Name")
    relation_name: "Optional[RelationName]" = Field(default=None, alias="Relation Name")
    cost: "Cost" = Field(alias="Total Cost")


class ExplainAnalyzeNode(ExplainNode):
    real_cardinality: "Cardinality" = Field(alias="Actual Rows")
    cost: "Cost" = Field(alias="Total Cost")


class ExplainPlan(BaseModel):
    plan: "ExplainNode" = Field(alias="Plan")
    template_id: "TemplateID" = Field(alias="Unique SQL Id")
    planning_time: "Time" = Field(alias="Planner Runtime")


class ExplainAnalyzePlan(ExplainPlan):
    plan: "ExplainAnalyzeNode" = Field(alias="Plan")
    execution_time: "Time" = Field(alias="Total Runtime")


class Plans(BaseModel):
    explain_plan: "ExplainPlan"
    explain_analyze_plan: "Optional[ExplainAnalyzePlan]" = None


QueryData = Dict[Settings, Plans]
