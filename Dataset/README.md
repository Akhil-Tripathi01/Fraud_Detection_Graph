# Dataset

This folder mirrors the notebook-first layout of the reference repository while keeping the data lightweight and runnable in this project.

## Files
- `example_cases_100.json`: `100` generated fraud and non-fraud style cases used by the dashboard and API demos

## Notes
- The original reference repository discusses a large Kaggle transaction dataset and GNN-based exploration.
- This project ships generated example data plus synthetic graph-training data so the repository remains easy to run locally.
- Live ML training data is generated inside [backend/app/services/graph_ml_service.py](../backend/app/services/graph_ml_service.py).
