import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from backend.app.models.schemas import DashboardMetrics, ExampleCaseSummary, GraphSummary, ScoringResult, TransactionIn
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

    def example_cases(self, limit: int = 100) -> list[dict]:
        history = deepcopy(self.transactions)
        cases: list[dict] = []

        for idx in range(limit):
            case = self._build_example_case(idx)
            tx = case["transaction"]
            score, reasons = self.risk_engine.score(history, tx)
            decision = self.decision_engine.decide(score)

            cases.append(
                {
                    "case_id": case["case_id"],
                    "scenario": case["scenario"],
                    "expected_pattern": case["expected_pattern"],
                    "transaction": tx,
                    "risk_score": score,
                    "decision": decision,
                    "reasons": reasons,
                }
            )
            history.append(tx)

        return cases

    def example_case_summary(self, limit: int = 100) -> ExampleCaseSummary:
        cases = self.example_cases(limit=limit)
        blocked = sum(1 for case in cases if case["decision"] == "block")
        review = sum(1 for case in cases if case["decision"] == "review")
        allow = sum(1 for case in cases if case["decision"] == "allow")
        avg_score = round(sum(case["risk_score"] for case in cases) / len(cases), 2) if cases else 0.0
        highest_case = max(cases, key=lambda case: case["risk_score"], default=None)

        return ExampleCaseSummary(
            total_cases=len(cases),
            blocked_count=blocked,
            review_count=review,
            allow_count=allow,
            average_risk_score=avg_score,
            highest_risk_case=None if highest_case is None else highest_case["case_id"],
        )

    def _build_example_case(self, idx: int) -> dict:
        series = idx // 20
        sequence = idx % 20
        timestamp = f"2026-04-{(idx % 5) + 1:02d}T{(9 + (idx % 10)):02d}:{(idx * 7) % 60:02d}:00+00:00"

        if series == 0:
            return {
                "case_id": f"CASE_{idx + 1:03d}",
                "scenario": "fraud_ring_velocity",
                "expected_pattern": "Shared device and IP reused across many users in a short window.",
                "transaction": {
                    "transaction_id": f"EX-{idx + 1:03d}",
                    "user_id": f"UR{sequence + 1:03d}",
                    "merchant_id": "MRING",
                    "device_id": f"DRING_{sequence % 3}",
                    "amount": 18500.0 + sequence * 520,
                    "timestamp": timestamp,
                    "ip_address": f"203.0.113.{(sequence % 4) + 10}",
                    "channel": "web",
                    "country": "SG" if sequence % 2 else "AE",
                },
            }

        if series == 1:
            return {
                "case_id": f"CASE_{idx + 1:03d}",
                "scenario": "amount_spike",
                "expected_pattern": "Known user suddenly attempts a much larger transaction than normal.",
                "transaction": {
                    "transaction_id": f"EX-{idx + 1:03d}",
                    "user_id": "U101",
                    "merchant_id": f"MSPIKE_{sequence % 5}",
                    "device_id": "D45",
                    "amount": 15000.0 + sequence * 3500,
                    "timestamp": timestamp,
                    "ip_address": "49.32.100.10",
                    "channel": "mobile",
                    "country": "IN",
                },
            }

        if series == 2:
            return {
                "case_id": f"CASE_{idx + 1:03d}",
                "scenario": "geo_channel_shift",
                "expected_pattern": "User shifts country and channel away from historical behavior.",
                "transaction": {
                    "transaction_id": f"EX-{idx + 1:03d}",
                    "user_id": "U305",
                    "merchant_id": f"MTRAVEL_{sequence % 6}",
                    "device_id": f"DTRAVEL_{sequence % 4}",
                    "amount": 4200.0 + sequence * 260,
                    "timestamp": timestamp,
                    "ip_address": f"198.51.100.{sequence + 30}",
                    "channel": "web" if sequence % 2 == 0 else "atm",
                    "country": "US" if sequence % 2 == 0 else "AE",
                },
            }

        if series == 3:
            return {
                "case_id": f"CASE_{idx + 1:03d}",
                "scenario": "normal_repeat_customer",
                "expected_pattern": "Stable domestic repeat customer using familiar device and IP.",
                "transaction": {
                    "transaction_id": f"EX-{idx + 1:03d}",
                    "user_id": "U101",
                    "merchant_id": f"MREG_{sequence % 8}",
                    "device_id": "D45",
                    "amount": 900.0 + sequence * 80,
                    "timestamp": timestamp,
                    "ip_address": "49.32.100.10",
                    "channel": "mobile",
                    "country": "IN",
                },
            }

        return {
            "case_id": f"CASE_{idx + 1:03d}",
            "scenario": "new_user_high_value",
            "expected_pattern": "Low-history user attempts a high-value online purchase.",
            "transaction": {
                "transaction_id": f"EX-{idx + 1:03d}",
                "user_id": f"UNEW_{sequence + 1:03d}",
                "merchant_id": f"MLUX_{sequence % 5}",
                "device_id": f"DNEW_{sequence % 7}",
                "amount": 52000.0 + sequence * 1800,
                "timestamp": timestamp,
                "ip_address": f"145.14.2.{sequence + 40}",
                "channel": "web",
                "country": "IN" if sequence % 3 else "SG",
            },
        }


fraud_service = FraudService()
