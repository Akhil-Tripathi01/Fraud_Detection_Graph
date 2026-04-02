from __future__ import annotations

import json
from pathlib import Path

from backend.app.services.graph_ml_service import graph_ml_service
from fraud_detection.config import load_config
from fraud_detection.model_registry import get_model_info


def main() -> None:
    config = load_config("configs/default_training.json")
    model_name = config.get("model_name", "random_forest")
    model_info = get_model_info(model_name)
    summary = graph_ml_service.results_summary()

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    out_file = output_dir / "latest_results_summary.json"
    payload = {
        "requested_model": model_name,
        "model_info": model_info,
        "results_summary": summary,
    }
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
