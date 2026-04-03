from backend.app.services.graph_ml_service import GraphMLService


def test_graph_ml_train_and_predict():
    svc = GraphMLService()
    artifacts = svc.train_pipeline(
        n_transactions=1200,
        n_accounts=220,
        fraud_rate=0.1,
        random_seed=7,
        test_size=0.3,
        n_estimators_start=40,
        n_estimators_end=120,
        n_estimators_step=40,
        feature_set="extended",
        fraud_ring_device_count=4,
        fraud_ring_ip_count=5,
        burst_fraction=0.45,
    )

    assert artifacts.transaction_count == 1200
    assert artifacts.account_count > 0
    assert 0 <= artifacts.metrics["accuracy"] <= 1
    assert 0 <= artifacts.metrics["roc_auc"] <= 1
    assert isinstance(artifacts.top_features, list) and len(artifacts.top_features) > 0
    assert set(artifacts.confusion_matrix.keys()) == {"tn", "fp", "fn", "tp"}
    assert artifacts.training_config["feature_set"] == "extended"
    assert artifacts.training_config["fraud_ring_device_count"] == 4

    sample_account = svc.nodes_df.iloc[0]["account_id"]
    pred = svc.predict_account(sample_account)
    assert pred["account_id"] == sample_account
    assert pred["account_found"] is True
    assert pred["threshold_used"] == 0.5
    assert 0 <= pred["fraud_probability"] <= 1
    assert pred["risk_tier"] in {"LOW", "MEDIUM", "HIGH"}

    missing = svc.predict_account("ACC_UNKNOWN_9999")
    assert missing["account_id"] == "ACC_UNKNOWN_9999"
    assert missing["account_found"] is False

    low_threshold = svc.predict_account(sample_account, threshold=0.2)
    high_threshold = svc.predict_account(sample_account, threshold=0.8)
    assert low_threshold["threshold_used"] == 0.2
    assert high_threshold["threshold_used"] == 0.8

    cases = svc.list_training_cases()
    assert any(c["case_name"] == "balanced_baseline" for c in cases)
    case_result = svc.run_training_case("balanced_baseline")
    assert case_result["case_name"] == "balanced_baseline"
    assert "metrics" in case_result
