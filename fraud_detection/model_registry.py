from __future__ import annotations


SUPPORTED_MODELS = {
    "random_forest": {
        "family": "baseline",
        "description": "Production-ready tree ensemble currently used by the app.",
        "status": "implemented",
    },
    "gcn": {
        "family": "gnn",
        "description": "Graph Convolutional Network placeholder for future implementation.",
        "status": "planned",
    },
    "gat": {
        "family": "gnn",
        "description": "Graph Attention Network placeholder for future implementation.",
        "status": "planned",
    },
    "gin": {
        "family": "gnn",
        "description": "Graph Isomorphism Network placeholder for future implementation.",
        "status": "planned",
    },
    "graphsage": {
        "family": "gnn",
        "description": "GraphSAGE placeholder for future implementation.",
        "status": "planned",
    },
    "hetero_gnn": {
        "family": "gnn",
        "description": "Heterogeneous graph fraud model placeholder for multi-entity transaction networks.",
        "status": "planned",
    },
    "temporal_gnn": {
        "family": "gnn",
        "description": "Temporal fraud model placeholder for time-aware graph sequences and bursts.",
        "status": "planned",
    },
    "fraudgt": {
        "family": "graph_transformer",
        "description": "Graph transformer placeholder inspired by newer fraud-specific transformer research.",
        "status": "planned",
    },
    "mpnn": {
        "family": "gnn",
        "description": "Message Passing Neural Network placeholder for future implementation.",
        "status": "planned",
    },
    "gtn": {
        "family": "gnn",
        "description": "Graph Transformer Network placeholder for future implementation.",
        "status": "planned",
    },
}


def get_model_info(model_name: str) -> dict:
    key = model_name.strip().lower()
    if key not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Supported models: {', '.join(sorted(SUPPORTED_MODELS))}")
    return SUPPORTED_MODELS[key]
