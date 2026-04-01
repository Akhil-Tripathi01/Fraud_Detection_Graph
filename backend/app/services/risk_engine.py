from collections import defaultdict
from datetime import timedelta
from statistics import mean

from dateutil.parser import isoparse

from backend.app.services.graph_service import GraphService


class RiskEngine:
    def __init__(self) -> None:
        self.graph_service = GraphService()

    def score(self, transactions: list[dict], tx: dict) -> tuple[int, list[str]]:
        reasons = []
        score = 0

        user_history = [t for t in transactions if t["user_id"] == tx["user_id"] and t["transaction_id"] != tx["transaction_id"]]
        score_amount, amount_reason = self._amount_risk(user_history, tx)
        if amount_reason:
            reasons.append(amount_reason)
        score += score_amount

        velocity_score, velocity_reason = self._velocity_risk(transactions, tx)
        if velocity_reason:
            reasons.append(velocity_reason)
        score += velocity_score

        graph_pressure = self.graph_service.shared_resource_pressure(transactions, tx)
        if graph_pressure >= 15:
            reasons.append("Shared infrastructure pattern detected (device/IP reused across users).")
        score += graph_pressure

        geo_score, geo_reason = self._geo_risk(user_history, tx)
        if geo_reason:
            reasons.append(geo_reason)
        score += geo_score

        channel_score, channel_reason = self._channel_risk(user_history, tx)
        if channel_reason:
            reasons.append(channel_reason)
        score += channel_score

        score = min(100, int(round(score)))
        if not reasons:
            reasons.append("No strong anomaly signal found.")
        return score, reasons

    def _amount_risk(self, user_history: list[dict], tx: dict) -> tuple[int, str]:
        if len(user_history) < 3:
            if tx["amount"] > 50000:
                return 20, "High-value transaction from low-history user."
            return 8, ""

        avg_amt = mean(t["amount"] for t in user_history)
        if avg_amt <= 0:
            return 0, ""
        ratio = tx["amount"] / avg_amt
        if ratio >= 6:
            return 35, f"Amount spike detected ({ratio:.1f}x user's normal average)."
        if ratio >= 3:
            return 20, f"Amount elevated ({ratio:.1f}x above user average)."
        return 5, ""

    def _velocity_risk(self, transactions: list[dict], tx: dict) -> tuple[int, str]:
        now = isoparse(tx["timestamp"])
        window_start = now - timedelta(minutes=15)

        recent_user = 0
        recent_device = 0
        for t in transactions:
            ts = isoparse(t["timestamp"])
            if ts < window_start or ts > now:
                continue
            if t["user_id"] == tx["user_id"]:
                recent_user += 1
            if t["device_id"] == tx["device_id"]:
                recent_device += 1

        score = 0
        reason_parts = []
        if recent_user >= 4:
            score += min(20, recent_user * 3)
            reason_parts.append(f"{recent_user} user transactions in 15 minutes")
        if recent_device >= 6:
            score += min(20, recent_device * 2)
            reason_parts.append(f"{recent_device} device transactions in 15 minutes")

        if reason_parts:
            return score, "Unusual transaction velocity: " + ", ".join(reason_parts) + "."
        return score, ""

    def _geo_risk(self, user_history: list[dict], tx: dict) -> tuple[int, str]:
        if not user_history:
            return (5, "") if tx["country"] != "IN" else (0, "")

        countries = defaultdict(int)
        for t in user_history:
            countries[t["country"]] += 1

        dominant_country = max(countries, key=countries.get)
        if tx["country"] != dominant_country and countries[dominant_country] >= 3:
            return 10, f"Country mismatch vs historical behavior ({dominant_country} -> {tx['country']})."
        return 0, ""

    def _channel_risk(self, user_history: list[dict], tx: dict) -> tuple[int, str]:
        if len(user_history) < 5:
            return 0, ""

        channels = defaultdict(int)
        for t in user_history:
            channels[t["channel"]] += 1

        dominant_channel = max(channels, key=channels.get)
        if tx["channel"] != dominant_channel and channels[dominant_channel] >= 4:
            return 8, f"Channel switch detected ({dominant_channel} -> {tx['channel']})."
        return 0, ""
