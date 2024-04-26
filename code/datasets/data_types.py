from typing import List, Optional, Dict
from pydantic import BaseModel, Field

Time = float
Cardinality = int
Selectivity = float
RelationName = str
NodeType = str
TemplateID = int
QueryName = str
Settings = str


class ExplainNode(BaseModel):
    node_type: "NodeType" = Field(alias="Node Type")
    plans: "Optional[List[ExplainNode]]" = Field(default=[], alias="Plans")
    estimated_cardinality: "Cardinality" = Field(alias="Plan Rows")
    index_name: "Optional[RelationName]" = Field(default=None, alias="Index Name")
    relation_name: "Optional[RelationName]" = Field(default=None, alias="Relation Name")


class ExplainAnalyzeNode(ExplainNode):
    real_cardinality: "Cardinality" = Field(alias="Actual Rows")


class ExplainPlan(BaseModel):
    plan: "ExplainNode" = Field(alias="Plan")
    template_id: "TemplateID" = Field(alias="Unique SQL Id")


class ExplainAnalyzePlan(ExplainPlan):
    execution_time: "Time" = Field(alias="Total Runtime")


class Plans(BaseModel):
    explain_plan: "ExplainPlan"
    explain_analyze_plan: "Optional[ExplainAnalyzePlan]" = None


QueryData = Dict[Settings, Plans]


class BenchmarkData(BaseModel):
    data: "Dict[QueryName, QueryData]"
