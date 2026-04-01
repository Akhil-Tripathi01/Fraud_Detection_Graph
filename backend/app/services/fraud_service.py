import json
from datetime import datetime, timezone
from pathlib import Path

from backend.app.models.schemas import DashboardMetrics, GraphSummary, ScoringResult, TransactionIn
from backend.app.services.decision_engine import DecisionEngine
from backend.app.services.graph_service import GraphService
from backend.app.services.risk_engine import RiskEngine


class FraudService:
    def __init__(self) -> None:
        self.risk_engine = RiskEngine()
        self.decision_engine = DecisionEngine()
        self.graph_service = GraphService()
        self.transactions = self._load_seed_data()
        self.results: list[dict] = []

    def _load_seed_data(self) -> list[dict]:
        data_path = Path(__file__).resolve().parent.parent / "data" / "sample_transactions.json"
        if not data_path.exists():
            return []
        with data_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def list_transactions(self) -> list[dict]:
        return sorted(self.transactions, key=lambda x: x["timestamp"], reverse=True)

    def score_transaction(self, payload: TransactionIn) -> ScoringResult:
        tx = payload.model_dump()
        tx["timestamp"] = payload.timestamp.astimezone(timezone.utc).isoformat()

        score, reasons = self.risk_engine.score(self.transactions, tx)
        decision = self.decision_engine.decide(score)
        created_at = datetime.now(timezone.utc)

        self.transactions.append(tx)
        result = {
            "transaction_id": tx["transaction_id"],
            "risk_score": score,
            "decision": decision,
            "reasons": reasons,
            "created_at": created_at.isoformat(),
        }
        self.results.append(result)

        return ScoringResult(
            transaction_id=result["transaction_id"],
            risk_score=result["risk_score"],
            decision=result["decision"],
            reasons=result["reasons"],
            created_at=created_at,
        )

    def list_alerts(self, min_score: int = 70) -> list[dict]:
        alerts = [r for r in self.results if r["risk_score"] >= min_score]
        return sorted(alerts, key=lambda x: x["created_at"], reverse=True)

    def graph_summary(self) -> GraphSummary:
        nodes, edges = self.graph_service.build_graph(self.transactions)
        suspicious_components = self.graph_service.suspicious_component_count(nodes, edges)
        return GraphSummary(
            node_count=len(nodes),
            edge_count=len(edges),
            suspicious_components=suspicious_components,
            nodes=nodes,
            edges=edges,
        )

    def metrics(self) -> DashboardMetrics:
        scored = self.results
        total = len(scored)
        if total == 0:
            return DashboardMetrics(
                total_transactions=len(self.transactions),
                avg_risk_score=0,
                blocked_count=0,
                review_count=0,
                allow_count=0,
                high_risk_percentage=0,
            )

        blocked = sum(1 for r in scored if r["decision"] == "block")
        review = sum(1 for r in scored if r["decision"] == "review")
        allow = sum(1 for r in scored if r["decision"] == "allow")
        avg = sum(r["risk_score"] for r in scored) / total
        high = sum(1 for r in scored if r["risk_score"] >= 70)

        return DashboardMetrics(
            total_transactions=len(self.transactions),
            avg_risk_score=round(avg, 2),
            blocked_count=blocked,
            review_count=review,
            allow_count=allow,
            high_risk_percentage=round((high / total) * 100, 2),
        )


fraud_service = FraudService()
