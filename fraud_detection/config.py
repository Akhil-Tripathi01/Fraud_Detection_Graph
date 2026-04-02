from __future__ import annotations

import json
from pathlib import Path


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix.lower() not in {".json"}:
        raise ValueError("Only JSON config files are supported in this project.")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
