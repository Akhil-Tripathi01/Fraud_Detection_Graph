from backend.app.core.config import settings


class DecisionEngine:
    def decide(self, score: int) -> str:
        if score >= settings.high_risk_threshold:
            return "block"
        if score >= settings.review_risk_threshold:
            return "review"
        return "allow"
