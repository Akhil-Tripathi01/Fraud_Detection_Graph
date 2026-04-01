from backend.app.services.risk_engine import RiskEngine


def test_risk_engine_amount_spike():
    engine = RiskEngine()
    base = [
        {
            "transaction_id": "A1",
            "user_id": "U1",
            "merchant_id": "M1",
            "device_id": "D1",
            "amount": 1000.0,
            "timestamp": "2026-03-29T10:00:00+00:00",
            "ip_address": "1.1.1.1",
            "channel": "mobile",
            "country": "IN",
        },
        {
            "transaction_id": "A2",
            "user_id": "U1",
            "merchant_id": "M1",
            "device_id": "D1",
            "amount": 1200.0,
            "timestamp": "2026-03-29T10:05:00+00:00",
            "ip_address": "1.1.1.1",
            "channel": "mobile",
            "country": "IN",
        },
        {
            "transaction_id": "A3",
            "user_id": "U1",
            "merchant_id": "M1",
            "device_id": "D1",
            "amount": 1300.0,
            "timestamp": "2026-03-29T10:10:00+00:00",
            "ip_address": "1.1.1.1",
            "channel": "mobile",
            "country": "IN",
        },
    ]
    tx = {
        "transaction_id": "A4",
        "user_id": "U1",
        "merchant_id": "M2",
        "device_id": "D1",
        "amount": 10000.0,
        "timestamp": "2026-03-29T10:12:00+00:00",
        "ip_address": "1.1.1.1",
        "channel": "mobile",
        "country": "IN",
    }
    score, reasons = engine.score(base, tx)
    assert score >= 35
    assert any("Amount" in r for r in reasons)
