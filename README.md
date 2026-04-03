# Fraud Detection Using Graph Learning

This repository was reshaped to feel closer to the reference project at `bimbomuri/Fraud_Detection_GNN`, but with a stronger end-to-end implementation. It keeps the notebook-first research feel while also delivering a working FastAPI backend, middle-layer fraud logic, and a live frontend dashboard.

It also now borrows a second useful pattern from `waittim/graph-fraud-detection`: staged notebooks, explicit `data/model/output` folders, and a simple training entrypoint.

A third reference, `arxyzan/fraud-detection-gnn`, pushed this repository toward a cleaner package-oriented structure with config-driven training and a multi-model roadmap.

After reviewing all the reference links together, this project now aims to combine their best ideas in one place:
- notebook-first research storytelling
- staged workflow organization
- config-driven experimentation
- multi-model GNN roadmap
- a working full-stack product instead of notebook-only exploration

## Introduction

Fraud detection benefits from graph-based learning because suspicious behavior is often relational rather than isolated. Shared devices, shared IPs, burst timing, repeated merchant patterns, and account-to-account connectivity can expose fraud rings that row-wise models miss.

This repository now combines:
- a notebook-style exploration flow
- a true Jupyter notebook file
- a dataset folder
- a working web application
- graph-aware ML training and inference APIs
- heterogeneous-graph style summaries and export bundles
- training-history traces for experiment review

## Repository Layout

- [Fraud_Detection.ipynb](Fraud_Detection.ipynb): notebook-style walkthrough inspired by the reference repo
- [10_data_loader.ipynb](10_data_loader.ipynb): staged data-preparation notebook
- [20_modeling.ipynb](20_modeling.ipynb): staged modeling notebook
- [30_visual.ipynb](30_visual.ipynb): staged results/visual notebook
- [Dataset/README.md](Dataset/README.md): dataset notes
- [Dataset/example_cases_100.json](Dataset/example_cases_100.json): `100` generated example cases
- [configs/default_training.json](configs/default_training.json): default config-driven training setup
- [configs/graphsage_experiment.json](configs/graphsage_experiment.json): future-facing GraphSAGE experiment config
- [configs/hetero_temporal_experiment.json](configs/hetero_temporal_experiment.json): heterogeneous and temporal roadmap config
- [configs/robust_fraudgt_experiment.json](configs/robust_fraudgt_experiment.json): robustness and graph-transformer roadmap config
- [configs/temporal_signal_experiment.json](configs/temporal_signal_experiment.json): time-aware graph roadmap config
- [data/README.md](data/README.md): prepared-data folder
- [model/README.md](model/README.md): model artifact folder
- [output/README.md](output/README.md): output artifact folder
- [ieee-data/README.md](ieee-data/README.md): placeholder for external raw datasets
- [gnn/README.md](gnn/README.md): placeholder for future graph deep-learning modules
- [fraud_detection/](fraud_detection): config-driven experiment package
- [train.py](train.py): standalone training script
- [visualize.py](visualize.py): result-summary export script
- [docs/REFERENCE_LEARNINGS.md](docs/REFERENCE_LEARNINGS.md): combined learnings from reference repositories and research themes
- [docs/INTERNET_RESEARCH_ROADMAP.md](docs/INTERNET_RESEARCH_ROADMAP.md): ongoing internet research mapped to project upgrades
- [frontend/index.html](frontend/index.html): full web frontend
- [frontend/notebook.html](frontend/notebook.html): notebook-like browser view
- [backend/app/main.py](backend/app/main.py): FastAPI entrypoint
- [backend/app/services/graph_ml_service.py](backend/app/services/graph_ml_service.py): graph ML pipeline
- [backend/app/services/fraud_service.py](backend/app/services/fraud_service.py): transaction scoring and example-case generation

## Dataset Overview

The reference repository is centered on a research notebook and a large external fraud dataset. This repo keeps that structure but uses runnable synthetic data so anyone can launch the project locally without a heavy data dependency.

Included data paths:
- seed transactions for live scoring
- synthetic graph-training data generated on demand
- `100` example cases for demos, filters, and GitHub presentation

## Challenges in Fraud Detection Datasets

### 1. Imbalance
Fraud is rare, so models can look accurate while missing the class that matters most.

### 2. Privacy
Real transaction datasets often require masking or anonymization.

### 3. Network Representation
Useful fraud signals live in relationships: shared devices, IP reuse, cluster pressure, and unusual interaction paths.

### 4. Operational Explainability
A useful fraud system must justify why a transaction or account was flagged.

## Aim and Tools

The project aims to show how graph-inspired fraud detection can move from research framing to a deployable application.

Main tools:
- `NetworkX` for graph construction and graph statistics
- `pandas` and `numpy` for feature engineering and synthetic data creation
- `scikit-learn` for the current production-ready baseline
- `FastAPI` for serving APIs and the frontend
- `HTML`, `CSS`, and `JavaScript` for the connected web interface
- staged notebooks for data, modeling, and visual interpretation
- config-driven experiment runs inspired by multi-model GNN repositories
- model catalog and config-runner APIs for experiment management

Future-ready path:
- `PyTorch Geometric`
- `DGL`

Those libraries are part of the roadmap, and the current API shape is designed so a deeper GNN backend can replace the baseline model later.

## What Is Working

- Live transaction scoring with `allow`, `review`, and `block`
- Investigator dashboard with graph visualization
- Graph ML training, reporting, prediction, simulation, save, load, and compare
- Training-history tracking across estimator growth steps
- Heterogeneous graph summaries across accounts, devices, IPs, merchants, and countries
- Research bundle export with transactions, features, edges, and summaries
- Research-style notebook browser page
- `100` generated example cases with backend endpoints and frontend filtering
- Automated tests for risk engine and ML API flows

## Run the Project

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --app-dir d:\Fraud_detection_Graph
```

Open:
- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/notebook`
- `http://127.0.0.1:8000/docs`

One-click startup (recommended on Windows):

```powershell
.\run_project.ps1
```

Or double-click `run_project.bat`.


## Main APIs

- `GET /api/health`
- `POST /api/transactions/score`
- `GET /api/alerts`
- `GET /api/graph/summary`
- `GET /api/dashboard/metrics`
- `GET /api/example-cases`
- `GET /api/example-cases/summary`
- `POST /api/ml/train`
- `GET /api/ml/metrics`
- `GET /api/ml/report`
- `POST /api/ml/predict`
- `GET /api/ml/simulate`
- `GET /api/ml/data-profile`
- `GET /api/ml/research`
- `GET /api/ml/results-summary`
- `GET /api/ml/visual-summary`
- `GET /api/ml/training-history`
- `GET /api/ml/hetero-graph-summary`
- `GET /api/ml/model-catalog`
- `GET /api/ml/configs`
- `POST /api/ml/run-config?config_name=<name>`
- `POST /api/ml/export-bundle`
- `GET /api/ml/research-landscape`
- `GET /api/ml/research-sources`
- `GET /api/ml/status`
- `POST /api/ml/save`
- `POST /api/ml/load`
- `GET /api/ml/model-manifest`
- `GET /api/ml/compare`

## Verification

```powershell
pytest -q
```

Standalone training:

```powershell
python train.py --config configs/default_training.json
```

Results export:

```powershell
python visualize.py
```

## Improvement Over the Reference Repo

The reference repository is useful as a concept and learning starting point, but this repository now goes further by adding:
- a fully working backend
- a real browser frontend
- middle-layer fraud services
- a richer README
- test coverage
- repeatable example data
- deployment-style model lifecycle endpoints
- experiment catalog and config execution APIs
- visual-summary exports for charts and reporting
- heterograph-aware summaries and portable research bundles
- documented roadmap for heterogeneous, temporal, and robustness-focused GNN work
