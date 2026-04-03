from fastapi.testclient import TestClient

from backend.app.main import app
from fraud_detection.config import load_config
from fraud_detection.pipeline import run_training


def test_ml_api_train_report_predict():
    client = TestClient(app)

    train_resp = client.post(
        "/api/ml/train",
        json={
            "n_transactions": 900,
            "n_accounts": 180,
            "fraud_rate": 0.1,
            "random_seed": 9,
            "test_size": 0.3,
            "n_estimators_start": 40,
            "n_estimators_end": 160,
            "n_estimators_step": 40,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "feature_set": "extended",
            "fraud_ring_device_count": 5,
            "fraud_ring_ip_count": 6,
            "burst_fraction": 0.4,
        },
    )
    assert train_resp.status_code == 200
    train_data = train_resp.json()
    assert "trained_at" in train_data
    assert "top_features" in train_data and len(train_data["top_features"]) > 0
    assert "confusion_matrix" in train_data
    assert "training_history" in train_data and len(train_data["training_history"]) > 0
    assert "training_config" in train_data
    assert train_data["training_config"]["feature_set"] == "extended"

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

    results_resp = client.get("/api/ml/results-summary")
    assert results_resp.status_code == 200
    results = results_resp.json()
    assert "graph_overview" in results
    assert "performance" in results
    assert "priority_note" in results

    visual_resp = client.get("/api/ml/visual-summary")
    assert visual_resp.status_code == 200
    visual = visual_resp.json()
    assert len(visual["metric_series"]) > 0
    assert len(visual["risk_distribution"]) == 3

    history_resp = client.get("/api/ml/training-history")
    assert history_resp.status_code == 200
    history = history_resp.json()
    assert len(history["history"]) > 0

    hetero_resp = client.get("/api/ml/hetero-graph-summary")
    assert hetero_resp.status_code == 200
    hetero = hetero_resp.json()
    assert "node_type_counts" in hetero
    assert "edge_type_counts" in hetero

    catalog_resp = client.get("/api/ml/model-catalog")
    assert catalog_resp.status_code == 200
    catalog = catalog_resp.json()
    assert any(entry["model_name"] == "random_forest" for entry in catalog)

    configs_resp = client.get("/api/ml/configs")
    assert configs_resp.status_code == 200
    configs = configs_resp.json()
    assert any(entry["name"] == "default_training" for entry in configs)
    assert any(entry["name"] == "hetero_temporal_experiment" for entry in configs)

    run_config_resp = client.post("/api/ml/run-config?config_name=default_training")
    assert run_config_resp.status_code == 200
    run_config = run_config_resp.json()
    assert run_config["model_name"] == "random_forest"
    assert run_config["model_info"]["status"] == "implemented"
    assert run_config["execution_mode"] == "native"

    planned_run_resp = client.post("/api/ml/run-config?config_name=graphsage_experiment")
    assert planned_run_resp.status_code == 200
    planned_run = planned_run_resp.json()
    assert planned_run["model_name"] == "graphsage"
    assert planned_run["effective_model_name"] == "random_forest"
    assert planned_run["execution_mode"] == "fallback_baseline"

    landscape_resp = client.get("/api/ml/research-landscape")
    assert landscape_resp.status_code == 200
    landscape = landscape_resp.json()
    assert len(landscape["research_trends"]) > 0
    assert len(landscape["next_steps"]) > 0

    source_resp = client.get("/api/ml/research-sources")
    assert source_resp.status_code == 200
    sources = source_resp.json()
    assert len(sources["sources"]) > 0
    assert any(item["category"] == "official_docs" for item in sources["sources"])

    bundle_resp = client.post("/api/ml/export-bundle?bundle_name=test_bundle_api")
    assert bundle_resp.status_code == 200
    bundle = bundle_resp.json()
    assert bundle["bundle_name"] == "test_bundle_api"
    assert len(bundle["files"]) >= 4

    cases_resp = client.get("/api/ml/training-cases")
    assert cases_resp.status_code == 200
    cases = cases_resp.json()
    assert len(cases) >= 3
    assert any(c["case_name"] == "balanced_baseline" for c in cases)

    run_case_resp = client.post("/api/ml/run-case?case_name=ring_heavy")
    assert run_case_resp.status_code == 200
    run_case = run_case_resp.json()
    assert run_case["case_name"] == "ring_heavy"
    assert "training_config" in run_case

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


def test_config_driven_training_pipeline():
    config = load_config("configs/default_training.json")
    result = run_training(config)
    assert result["model_name"] == "random_forest"
    assert result["model_info"]["status"] == "implemented"
    assert result["metrics"]["accuracy"] >= 0
    assert result["graph_nodes"] > 0
