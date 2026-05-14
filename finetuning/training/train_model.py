from __future__ import annotations

from pathlib import Path

import yaml
from ultralytics import YOLO


FINETUNING_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = FINETUNING_DIR.parent
TRAIN_CONFIG_PATH = FINETUNING_DIR / "training" / "train_config.yaml"


def main() -> None:
    if not TRAIN_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Training config not found: {TRAIN_CONFIG_PATH}")

    config = load_training_config(TRAIN_CONFIG_PATH)
    model_variant = str(config.pop("model_variant")).strip()
    if not model_variant:
        raise ValueError(f"'model_variant' must not be empty: {TRAIN_CONFIG_PATH}")

    model_path = REPO_ROOT / f"yolo{model_variant}.pt"
    run_name = f"yolo{model_variant}_{config['imgsz']}_finetune"

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    config["data"] = str(resolve_repo_path(config["data"]))
    config["project"] = str(resolve_repo_path(config["project"]))
    model = YOLO(str(model_path))
    config["model"] = str(model_path)
    config["name"] = run_name
    model.train(**config)


def load_training_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a mapping: {config_path}")
    if "model_variant" not in config:
        raise ValueError(f"Training config is missing 'model_variant': {config_path}")

    return config


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


if __name__ == "__main__":
    main()
