from fastapi.testclient import TestClient

from backend.app.main import app


def test_ml_api_train_report_predict():
    client = TestClient(app)

    train_resp = client.post(
        "/api/ml/train",
        json={"n_transactions": 900, "n_accounts": 180, "fraud_rate": 0.1, "random_seed": 9},
    )
    assert train_resp.status_code == 200
    train_data = train_resp.json()
    assert "trained_at" in train_data
    assert "top_features" in train_data and len(train_data["top_features"]) > 0
    assert "confusion_matrix" in train_data

    report_resp = client.get("/api/ml/report")
    assert report_resp.status_code == 200
    report = report_resp.json()
    assert report["metrics"]["accuracy"] >= 0
    assert "top_features" in report

    profile_resp = client.get("/api/ml/data-profile")
    assert profile_resp.status_code == 200
    profile = profile_resp.json()
    assert "channel_distribution" in profile
    assert "top_risky_devices" in profile

    research_resp = client.get("/api/ml/research")
    assert research_resp.status_code == 200
    research = research_resp.json()
    assert "findings" in research and len(research["findings"]) > 0
    assert "recommendations" in research and len(research["recommendations"]) > 0

    predict_resp = client.post("/api/ml/predict", json={"account_id": "ACC_UNKNOWN_001"})
    assert predict_resp.status_code == 200
    pred = predict_resp.json()
    assert pred["account_id"] == "ACC_UNKNOWN_001"
    assert pred["account_found"] is False
    assert "threshold_used" in pred

    status_resp = client.get("/api/ml/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert "trained" in status_data

    save_resp = client.post("/api/ml/save")
    assert save_resp.status_code == 200
    saved_tag = save_resp.json()["model_tag"]
    assert saved_tag

    save_resp2 = client.post("/api/ml/save?model_tag=test_model_candidate")
    assert save_resp2.status_code == 200
    saved_tag2 = save_resp2.json()["model_tag"]
    assert saved_tag2 == "test_model_candidate"

    models_resp = client.get("/api/ml/models")
    assert models_resp.status_code == 200
    assert saved_tag in models_resp.json()
    assert saved_tag2 in models_resp.json()

    load_resp = client.post(f"/api/ml/load?model_tag={saved_tag}")
    assert load_resp.status_code == 200
    assert load_resp.json()["model_tag"] == saved_tag

    manifest_resp = client.get(f"/api/ml/model-manifest?model_tag={saved_tag}")
    assert manifest_resp.status_code == 200
    manifest = manifest_resp.json()
    assert manifest["model_tag"] == saved_tag
    assert "metrics" in manifest

    compare_resp = client.get(f"/api/ml/compare?base_tag={saved_tag}&candidate_tag={saved_tag2}")
    assert compare_resp.status_code == 200
    compare = compare_resp.json()
    assert compare["base_tag"] == saved_tag
    assert compare["candidate_tag"] == saved_tag2
    assert "roc_auc_delta" in compare["deltas"]


def test_example_cases_endpoints():
    client = TestClient(app)

    summary_resp = client.get("/api/example-cases/summary")
    assert summary_resp.status_code == 200
    summary = summary_resp.json()
    assert summary["total_cases"] == 100
    assert summary["blocked_count"] + summary["review_count"] + summary["allow_count"] == 100

    cases_resp = client.get("/api/example-cases")
    assert cases_resp.status_code == 200
    cases = cases_resp.json()
    assert len(cases) == 100
    assert {"case_id", "scenario", "risk_score", "decision", "transaction"}.issubset(cases[0].keys())
