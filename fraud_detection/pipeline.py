from __future__ import annotations

from backend.app.services.graph_ml_service import graph_ml_service
from fraud_detection.model_registry import get_model_info


def run_training(config: dict) -> dict:
    model_name = config.get("model_name", "random_forest")
    model_info = get_model_info(model_name)
    training = config.get("training", {})

    artifacts = graph_ml_service.train_pipeline(
        n_transactions=int(training.get("n_transactions", 3000)),
        n_accounts=int(training.get("n_accounts", 500)),
        fraud_rate=float(training.get("fraud_rate", 0.08)),
        random_seed=int(training.get("random_seed", 42)),
    )
    model_tag = graph_ml_service.save_model(model_tag=config.get("model_tag"))

    return {
      "model_name": model_name,
      "model_info": model_info,
      "model_tag": model_tag,
      "trained_at": artifacts.trained_at,
      "metrics": artifacts.metrics,
      "graph_nodes": artifacts.graph_nodes,
      "graph_edges": artifacts.graph_edges,
      "transactions": artifacts.transaction_count,
      "accounts": artifacts.account_count,
      "top_features": artifacts.top_features,
    }
