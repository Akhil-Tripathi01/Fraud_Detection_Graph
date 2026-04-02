from backend.app.services.graph_ml_service import graph_ml_service


def main() -> None:
    artifacts = graph_ml_service.train_pipeline(
        n_transactions=3000,
        n_accounts=500,
        fraud_rate=0.08,
        random_seed=42,
    )
    model_tag = graph_ml_service.save_model()
    print("Training complete")
    print(f"trained_at={artifacts.trained_at}")
    print(f"accuracy={artifacts.metrics['accuracy']}")
    print(f"precision={artifacts.metrics['precision']}")
    print(f"recall={artifacts.metrics['recall']}")
    print(f"f1={artifacts.metrics['f1']}")
    print(f"roc_auc={artifacts.metrics['roc_auc']}")
    print(f"saved_model={model_tag}")


if __name__ == "__main__":
    main()
