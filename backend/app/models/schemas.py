from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TransactionIn(BaseModel):
    transaction_id: str = Field(..., min_length=3)
    user_id: str
    merchant_id: str
    device_id: str
    amount: float = Field(..., gt=0)
    timestamp: datetime
    ip_address: str
    channel: Literal["web", "mobile", "pos", "atm"]
    country: str = Field(default="IN")


class ScoringResult(BaseModel):
    transaction_id: str
    risk_score: int
    decision: Literal["allow", "review", "block"]
    reasons: list[str]
    created_at: datetime


class GraphSummary(BaseModel):
    node_count: int
    edge_count: int
    suspicious_components: int
    nodes: list[dict]
    edges: list[dict]


class DashboardMetrics(BaseModel):
    total_transactions: int
    avg_risk_score: float
    blocked_count: int
    review_count: int
    allow_count: int
    high_risk_percentage: float


class ExampleCase(BaseModel):
    case_id: str
    scenario: str
    expected_pattern: str
    transaction: dict
    risk_score: int
    decision: Literal["allow", "review", "block"]
    reasons: list[str]


class ExampleCaseSummary(BaseModel):
    total_cases: int
    blocked_count: int
    review_count: int
    allow_count: int
    average_risk_score: float
    highest_risk_case: str | None


class MLTrainRequest(BaseModel):
    n_transactions: int = Field(default=3000, ge=500, le=20000)
    n_accounts: int = Field(default=500, ge=100, le=5000)
    fraud_rate: float = Field(default=0.08, gt=0.01, lt=0.5)
    random_seed: int = Field(default=42, ge=1, le=99999)
    test_size: float = Field(default=0.25, gt=0.1, lt=0.5)
    n_estimators_start: int = Field(default=40, ge=20, le=400)
    n_estimators_end: int = Field(default=280, ge=40, le=1200)
    n_estimators_step: int = Field(default=40, ge=10, le=200)
    max_depth: int = Field(default=10, ge=3, le=40)
    min_samples_leaf: int = Field(default=2, ge=1, le=20)
    feature_set: Literal["core", "extended"] = "extended"
    fraud_ring_device_count: int = Field(default=6, ge=2, le=30)
    fraud_ring_ip_count: int = Field(default=7, ge=2, le=50)
    burst_fraction: float = Field(default=0.33, gt=0.05, lt=0.9)


class MLEvaluationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    support: int


class MLTrainResponse(BaseModel):
    message: str
    trained_at: datetime
    transactions: int
    accounts: int
    graph_nodes: int
    graph_edges: int
    metrics: MLEvaluationMetrics
    feature_columns: list[str]
    top_features: list[dict]
    confusion_matrix: dict
    training_history: list[dict] = []
    training_config: dict = {}


class MLPredictRequest(BaseModel):
    account_id: str
    threshold: float = Field(default=0.5, ge=0.05, le=0.95)


class MLPredictResponse(BaseModel):
    account_id: str
    account_found: bool
    fraud_probability: float
    threshold_used: float
    prediction: int
    risk_tier: str
    explanation: list[str]


class MLDataProfileResponse(BaseModel):
    transactions: int
    accounts: int
    fraud_rate: float
    avg_amount: float
    p95_amount: float
    channel_distribution: dict
    country_distribution: dict
    top_risky_devices: list[dict]


class MLResearchResponse(BaseModel):
    title: str
    summary: str
    findings: list[str]
    recommendations: list[str]


class MLResultsSummaryResponse(BaseModel):
    title: str
    graph_overview: dict
    performance: dict
    confusion_matrix: dict
    priority_note: str
    top_features: list[dict]
    recommendations: list[str]


class MLModelCatalogEntry(BaseModel):
    model_name: str
    family: str
    description: str
    status: str


class MLConfigEntry(BaseModel):
    name: str
    path: str
    model_name: str
    exists: bool


class MLVisualSummaryResponse(BaseModel):
    metric_series: list[dict]
    feature_importance_series: list[dict]
    risk_distribution: list[dict]
    graph_snapshot: dict


class MLNotebookVisualsResponse(BaseModel):
    daily_volume: list[dict]
    feature_gap: list[dict]
    correlation_matrix: dict
    confusion_heatmap: list[list[int]]
    top_risk_accounts: list[dict]
    fraud_probability_distribution: list[dict]


class MLConfigRunResponse(BaseModel):
    model_name: str
    effective_model_name: str
    execution_mode: str
    model_info: dict
    model_tag: str
    trained_at: datetime
    metrics: dict
    graph_nodes: int
    graph_edges: int
    transactions: int
    accounts: int
    top_features: list[dict]


class MLResearchLandscapeResponse(BaseModel):
    title: str
    repo_learnings: list[str]
    research_trends: list[str]
    project_upgrades: list[str]
    next_steps: list[str]


class MLResearchSourceResponse(BaseModel):
    title: str
    sources: list[dict]


class MLTrainingHistoryResponse(BaseModel):
    history: list[dict]


class MLHeteroGraphSummaryResponse(BaseModel):
    title: str
    node_type_counts: dict
    edge_type_counts: dict
    graph_density_view: dict
    top_shared_devices: list[dict]
    top_shared_ips: list[dict]


class MLBundleExportResponse(BaseModel):
    bundle_name: str
    bundle_dir: str
    files: list[str]


class MLStatusResponse(BaseModel):
    trained: bool
    trained_at: datetime | None
    model_count: int
    latest_model_tag: str | None


class MLModelActionResponse(BaseModel):
    message: str
    model_tag: str


class MLModelManifestResponse(BaseModel):
    model_tag: str
    saved_at: datetime
    trained_at: datetime
    metrics: dict
    top_features: list[dict]
    transaction_count: int
    account_count: int


class MLModelCompareResponse(BaseModel):
    base_tag: str
    candidate_tag: str
    deltas: dict
    winner: str


class MLTrainingCaseEntry(BaseModel):
    case_name: str
    description: str
    config: dict


class MLTrainingCaseRunResponse(BaseModel):
    case_name: str
    message: str
    trained_at: datetime
    metrics: dict
    training_config: dict
