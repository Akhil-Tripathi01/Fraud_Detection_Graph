from pathlib import Path

from fastapi import APIRouter, Query

from backend.app.models.schemas import (
    DashboardMetrics,
    ExampleCase,
    ExampleCaseSummary,
    GraphSummary,
    MLConfigEntry,
    MLConfigRunResponse,
    MLBundleExportResponse,
    MLDataProfileResponse,
    MLEvaluationMetrics,
    MLHeteroGraphSummaryResponse,
    MLModelActionResponse,
    MLModelCatalogEntry,
    MLModelCompareResponse,
    MLModelManifestResponse,
    MLNotebookVisualsResponse,
    MLTrainingCaseEntry,
    MLTrainingCaseRunResponse,
    MLPredictRequest,
    MLPredictResponse,
    MLResearchResponse,
    MLResearchLandscapeResponse,
    MLResearchSourceResponse,
    MLResultsSummaryResponse,
    MLStatusResponse,
    MLTrainingHistoryResponse,
    MLTrainRequest,
    MLTrainResponse,
    MLVisualSummaryResponse,
    ScoringResult,
    TransactionIn,
)
from backend.app.services.fraud_service import fraud_service
from backend.app.services.graph_ml_service import graph_ml_service
from fraud_detection.config import load_config
from fraud_detection.model_registry import SUPPORTED_MODELS
from fraud_detection.pipeline import run_training

router = APIRouter(prefix="/api", tags=["fraud"])
config_dir = Path(__file__).resolve().parents[3] / "configs"


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/transactions")
def transactions() -> list[dict]:
    return fraud_service.list_transactions()


@router.post("/transactions/score", response_model=ScoringResult)
def score_transaction(payload: TransactionIn) -> ScoringResult:
    return fraud_service.score_transaction(payload)


@router.get("/alerts")
def alerts(min_score: int = Query(default=70, ge=1, le=100)) -> list[dict]:
    return fraud_service.list_alerts(min_score=min_score)


@router.get("/graph/summary", response_model=GraphSummary)
def graph_summary() -> GraphSummary:
    return fraud_service.graph_summary()


@router.get("/dashboard/metrics", response_model=DashboardMetrics)
def dashboard_metrics() -> DashboardMetrics:
    return fraud_service.metrics()


@router.get("/example-cases", response_model=list[ExampleCase])
def example_cases(limit: int = Query(default=100, ge=1, le=100)) -> list[ExampleCase]:
    return [ExampleCase(**case) for case in fraud_service.example_cases(limit=limit)]


@router.get("/example-cases/summary", response_model=ExampleCaseSummary)
def example_case_summary(limit: int = Query(default=100, ge=1, le=100)) -> ExampleCaseSummary:
    return fraud_service.example_case_summary(limit=limit)


@router.post("/ml/train", response_model=MLTrainResponse)
def train_graph_ml(payload: MLTrainRequest) -> MLTrainResponse:
    graph_ml_service.train_pipeline(
        n_transactions=payload.n_transactions,
        n_accounts=payload.n_accounts,
        fraud_rate=payload.fraud_rate,
        random_seed=payload.random_seed,
        test_size=payload.test_size,
        n_estimators_start=payload.n_estimators_start,
        n_estimators_end=payload.n_estimators_end,
        n_estimators_step=payload.n_estimators_step,
        max_depth=payload.max_depth,
        min_samples_leaf=payload.min_samples_leaf,
        feature_set=payload.feature_set,
        fraud_ring_device_count=payload.fraud_ring_device_count,
        fraud_ring_ip_count=payload.fraud_ring_ip_count,
        burst_fraction=payload.burst_fraction,
    )
    data = graph_ml_service.metrics()
    return MLTrainResponse(
        message=data["message"],
        trained_at=data["trained_at"],
        transactions=data["transactions"],
        accounts=data["accounts"],
        graph_nodes=data["graph_nodes"],
        graph_edges=data["graph_edges"],
        metrics=MLEvaluationMetrics(**data["metrics"]),
        feature_columns=data["feature_columns"],
        top_features=data["top_features"],
        confusion_matrix=data["confusion_matrix"],
        training_history=data["training_history"],
        training_config=data["training_config"],
    )


@router.get("/ml/metrics", response_model=MLTrainResponse)
def get_graph_ml_metrics() -> MLTrainResponse:
    data = graph_ml_service.metrics()
    return MLTrainResponse(
        message=data["message"],
        trained_at=data["trained_at"],
        transactions=data["transactions"],
        accounts=data["accounts"],
        graph_nodes=data["graph_nodes"],
        graph_edges=data["graph_edges"],
        metrics=MLEvaluationMetrics(**data["metrics"]),
        feature_columns=data["feature_columns"],
        top_features=data["top_features"],
        confusion_matrix=data["confusion_matrix"],
        training_history=data["training_history"],
        training_config=data["training_config"],
    )


@router.get("/ml/report")
def get_ml_report() -> dict:
    return graph_ml_service.metrics()


@router.post("/ml/predict", response_model=MLPredictResponse)
def predict_account(payload: MLPredictRequest) -> MLPredictResponse:
    result = graph_ml_service.predict_account(payload.account_id, threshold=payload.threshold)
    return MLPredictResponse(**result)


@router.get("/ml/simulate")
def simulate_realtime(n: int = Query(default=5, ge=1, le=20)) -> list[dict]:
    return graph_ml_service.simulate_realtime_transactions(n=n)


@router.get("/ml/data-profile", response_model=MLDataProfileResponse)
def get_ml_data_profile() -> MLDataProfileResponse:
    return MLDataProfileResponse(**graph_ml_service.data_profile())


@router.get("/ml/research", response_model=MLResearchResponse)
def get_ml_research() -> MLResearchResponse:
    return MLResearchResponse(**graph_ml_service.research_report())


@router.get("/ml/research-landscape", response_model=MLResearchLandscapeResponse)
def get_ml_research_landscape() -> MLResearchLandscapeResponse:
    return MLResearchLandscapeResponse(**graph_ml_service.research_landscape())


@router.get("/ml/research-sources", response_model=MLResearchSourceResponse)
def get_ml_research_sources() -> MLResearchSourceResponse:
    return MLResearchSourceResponse(**graph_ml_service.research_sources())


@router.get("/ml/results-summary", response_model=MLResultsSummaryResponse)
def get_ml_results_summary() -> MLResultsSummaryResponse:
    return MLResultsSummaryResponse(**graph_ml_service.results_summary())


@router.get("/ml/visual-summary", response_model=MLVisualSummaryResponse)
def get_ml_visual_summary() -> MLVisualSummaryResponse:
    return MLVisualSummaryResponse(**graph_ml_service.visual_summary())


@router.get("/ml/notebook-visuals", response_model=MLNotebookVisualsResponse)
def get_ml_notebook_visuals() -> MLNotebookVisualsResponse:
    return MLNotebookVisualsResponse(**graph_ml_service.notebook_visuals())


@router.get("/ml/training-history", response_model=MLTrainingHistoryResponse)
def get_ml_training_history() -> MLTrainingHistoryResponse:
    return MLTrainingHistoryResponse(**graph_ml_service.training_history())


@router.get("/ml/hetero-graph-summary", response_model=MLHeteroGraphSummaryResponse)
def get_ml_hetero_graph_summary() -> MLHeteroGraphSummaryResponse:
    return MLHeteroGraphSummaryResponse(**graph_ml_service.heterogeneous_graph_summary())


@router.get("/ml/model-catalog", response_model=list[MLModelCatalogEntry])
def get_ml_model_catalog() -> list[MLModelCatalogEntry]:
    return [MLModelCatalogEntry(model_name=name, **info) for name, info in sorted(SUPPORTED_MODELS.items())]


@router.get("/ml/configs", response_model=list[MLConfigEntry])
def get_ml_configs() -> list[MLConfigEntry]:
    entries = []
    for path in sorted(config_dir.glob("*.json")):
        config = load_config(path)
        entries.append(
            MLConfigEntry(
                name=path.stem,
                path=str(path.relative_to(config_dir.parent)),
                model_name=config.get("model_name", "random_forest"),
                exists=path.exists(),
            )
        )
    return entries


@router.post("/ml/run-config", response_model=MLConfigRunResponse)
def run_ml_config(config_name: str = Query(..., min_length=1)) -> MLConfigRunResponse:
    path = config_dir / f"{config_name}.json"
    result = run_training(load_config(path))
    return MLConfigRunResponse(**result)


@router.post("/ml/export-bundle", response_model=MLBundleExportResponse)
def export_ml_bundle(bundle_name: str | None = Query(default=None)) -> MLBundleExportResponse:
    return MLBundleExportResponse(**graph_ml_service.export_bundle(bundle_name=bundle_name))


@router.get("/ml/training-cases", response_model=list[MLTrainingCaseEntry])
def get_ml_training_cases() -> list[MLTrainingCaseEntry]:
    return [MLTrainingCaseEntry(**item) for item in graph_ml_service.list_training_cases()]


@router.post("/ml/run-case", response_model=MLTrainingCaseRunResponse)
def run_ml_training_case(case_name: str = Query(..., min_length=3)) -> MLTrainingCaseRunResponse:
    return MLTrainingCaseRunResponse(**graph_ml_service.run_training_case(case_name))


@router.get("/ml/status", response_model=MLStatusResponse)
def get_ml_status() -> MLStatusResponse:
    return MLStatusResponse(**graph_ml_service.status())


@router.get("/ml/models")
def list_ml_models() -> list[str]:
    return graph_ml_service.list_saved_models()


@router.post("/ml/save", response_model=MLModelActionResponse)
def save_ml_model(model_tag: str | None = Query(default=None)) -> MLModelActionResponse:
    tag = graph_ml_service.save_model(model_tag=model_tag)
    return MLModelActionResponse(message="Model saved.", model_tag=tag)


@router.post("/ml/load", response_model=MLModelActionResponse)
def load_ml_model(model_tag: str | None = Query(default=None)) -> MLModelActionResponse:
    tag = graph_ml_service.load_model(model_tag=model_tag)
    return MLModelActionResponse(message="Model loaded.", model_tag=tag)


@router.get("/ml/model-manifest", response_model=MLModelManifestResponse)
def get_model_manifest(model_tag: str | None = Query(default=None)) -> MLModelManifestResponse:
    return MLModelManifestResponse(**graph_ml_service.model_manifest(model_tag=model_tag))


@router.get("/ml/compare", response_model=MLModelCompareResponse)
def compare_models(base_tag: str = Query(...), candidate_tag: str = Query(...)) -> MLModelCompareResponse:
    return MLModelCompareResponse(**graph_ml_service.compare_models(base_tag=base_tag, candidate_tag=candidate_tag))
