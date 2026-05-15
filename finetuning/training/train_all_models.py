from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, Tuple

from ultralytics import YOLO

from finetuning.training.train_model import (
    REPO_ROOT,
    TRAIN_CONFIG_PATH,
    load_training_config,
    resolve_repo_path,
)


TRAINING_TARGETS: Tuple[Tuple[str, int], ...] = (
    ("26n", 640),
    ("26n", 960),
    ("26s", 640),
    ("26s", 960),
    ("26m", 960),
    ("26m", 1280),
)


def main() -> None:
    if not TRAIN_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Training config not found: {TRAIN_CONFIG_PATH}")

    base_config = load_training_config(TRAIN_CONFIG_PATH)

    for model_variant, image_size in iter_training_targets():
        train_single_configuration(base_config, model_variant=model_variant, image_size=image_size)


def iter_training_targets() -> Iterable[Tuple[str, int]]:
    yield from TRAINING_TARGETS


def train_single_configuration(base_config: Dict[str, object], *, model_variant: str, image_size: int) -> None:
    config = deepcopy(base_config)
    model_path = REPO_ROOT / f"yolo{model_variant}.pt"
    run_name = f"yolo{model_variant}_{image_size}_finetune"

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    config["model_variant"] = model_variant
    config["imgsz"] = image_size
    config["data"] = str(resolve_repo_path(str(config["data"])))
    config["project"] = str(resolve_repo_path(str(config["project"])))
    config["model"] = str(model_path)
    config["name"] = run_name
    config.pop("model_variant", None)

    print(f"Training {run_name} from {model_path.name}")
    model = YOLO(str(model_path))
    model.train(**config)


if __name__ == "__main__":
    main()
