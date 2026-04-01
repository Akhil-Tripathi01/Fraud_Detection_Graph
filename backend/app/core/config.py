from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "Project 74 Fraud Detection Platform"
    app_version: str = "1.0.0"
    high_risk_threshold: int = 80
    review_risk_threshold: int = 55
    max_graph_nodes_preview: int = 40


settings = Settings()
