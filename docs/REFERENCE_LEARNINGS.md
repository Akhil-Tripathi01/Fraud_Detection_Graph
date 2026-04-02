# Reference Learnings

This project has been improved by combining patterns learned from multiple public fraud-detection repositories and current graph-fraud-detection research directions.

## Repository Patterns Adopted

- Notebook-first storytelling for research readability
- Staged notebooks for data, modeling, and visual interpretation
- Explicit `Dataset`, `data`, `model`, and `output` folders
- Config-driven training and a dedicated experiment package
- Model catalog and future-model placeholders

## Current Research Directions Reflected Here

- Heterophily-aware fraud detection
- Heterogeneous graph modeling across users, devices, IPs, and merchants
- Temporal graph modeling for burst behavior
- Robustness against adaptive fraud and graph manipulation
- Graph-transformer style roadmap beyond baseline message passing

## How This Repo Responds

- Keeps a runnable baseline today
- Exposes planned GNN families through config files and model catalog APIs
- Documents a clean path to heterogeneous and temporal graph upgrades
- Preserves the working FastAPI + frontend app while expanding the research surface area
