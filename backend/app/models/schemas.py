from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TransactionIn(BaseModel):
    transaction_id: str = Field(..., min_length=3)
    user_id: str
    merchant_id: str
    device_id: str
    amount: float = Field(..., gt=0)
    timestamp: datetime
    ip_address: str
    channel: Literal["web", "mobile", "pos", "atm"]
    country: str = Field(default="IN")


class ScoringResult(BaseModel):
    transaction_id: str
    risk_score: int
    decision: Literal["allow", "review", "block"]
    reasons: list[str]
    created_at: datetime


class GraphSummary(BaseModel):
    node_count: int
    edge_count: int
    suspicious_components: int
    nodes: list[dict]
    edges: list[dict]


class DashboardMetrics(BaseModel):
    total_transactions: int
    avg_risk_score: float
    blocked_count: int
    review_count: int
    allow_count: int
    high_risk_percentage: float
