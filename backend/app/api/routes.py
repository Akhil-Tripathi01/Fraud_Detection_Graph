from fastapi import APIRouter, Query

from backend.app.models.schemas import DashboardMetrics, GraphSummary, ScoringResult, TransactionIn
from backend.app.services.fraud_service import fraud_service

router = APIRouter(prefix="/api", tags=["fraud"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/transactions")
def transactions() -> list[dict]:
    return fraud_service.list_transactions()


@router.post("/transactions/score", response_model=ScoringResult)
def score_transaction(payload: TransactionIn) -> ScoringResult:
    return fraud_service.score_transaction(payload)


@router.get("/alerts")
def alerts(min_score: int = Query(default=70, ge=1, le=100)) -> list[dict]:
    return fraud_service.list_alerts(min_score=min_score)


@router.get("/graph/summary", response_model=GraphSummary)
def graph_summary() -> GraphSummary:
    return fraud_service.graph_summary()


@router.get("/dashboard/metrics", response_model=DashboardMetrics)
def dashboard_metrics() -> DashboardMetrics:
    return fraud_service.metrics()
