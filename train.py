from __future__ import annotations

import argparse
import json

from fraud_detection.config import load_config
from fraud_detection.pipeline import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fraud detection pipeline from a config file.")
    parser.add_argument("--config", default="configs/default_training.json", help="Path to JSON config file.")
    args = parser.parse_args()

    result = run_training(load_config(args.config))
    print("Training complete")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
