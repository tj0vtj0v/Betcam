from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence

import yaml

from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
FINETUNING_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = FINETUNING_DIR.parent

# Edit these constants instead of passing CLI flags.
SOURCE_DIR = FINETUNING_DIR / "to_be_labled" / "images"
DATASET_DIR = FINETUNING_DIR / "dataset"
DATA_YAML_PATH = FINETUNING_DIR / "training" / "data.yaml"
MODEL_PATH = REPO_ROOT / "yolo26m_1280_finetune.pt"

TRAIN_RATIO = 0.9
SPLIT_SEED = 69
INDEX_WIDTH = 3

INFERENCE_IMGSZ = 1280
INFERENCE_CONF = 0.5
INFERENCE_IOU = 0.5
INFERENCE_DEVICE = "auto"
INFERENCE_BATCH = 8

OVERWRITE_EXISTING = False


def main() -> None:
    validate_configuration()

    class_names = load_class_names(DATA_YAML_PATH)
    class_id_by_name = {name: index for index, name in enumerate(class_names)}

    image_paths = list_images(SOURCE_DIR)

    train_images_dir = DATASET_DIR / "images" / "train"
    val_images_dir = DATASET_DIR / "images" / "val"
    train_labels_dir = DATASET_DIR / "labels" / "train"
    val_labels_dir = DATASET_DIR / "labels" / "val"
    target_dirs = [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]

    ensure_target_dirs(target_dirs, overwrite=OVERWRITE_EXISTING)

    train_images, val_images = split_images(image_paths, train_ratio=TRAIN_RATIO, seed=SPLIT_SEED)
    copied_train = copy_and_rename(train_images, train_images_dir, prefix="train", index_width=INDEX_WIDTH)
    copied_val = copy_and_rename(val_images, val_images_dir, prefix="val", index_width=INDEX_WIDTH)

    model = YOLO(str(MODEL_PATH))
    write_labels_for_split(
        model=model,
        image_paths=copied_train,
        output_dir=train_labels_dir,
        class_id_by_name=class_id_by_name,
        imgsz=INFERENCE_IMGSZ,
        conf=INFERENCE_CONF,
        iou=INFERENCE_IOU,
        device=INFERENCE_DEVICE,
        batch=INFERENCE_BATCH,
    )
    write_labels_for_split(
        model=model,
        image_paths=copied_val,
        output_dir=val_labels_dir,
        class_id_by_name=class_id_by_name,
        imgsz=INFERENCE_IMGSZ,
        conf=INFERENCE_CONF,
        iou=INFERENCE_IOU,
        device=INFERENCE_DEVICE,
        batch=INFERENCE_BATCH,
    )
    fill_missing_labels_for_split(
        model=model,
        images_dir=train_images_dir,
        labels_dir=train_labels_dir,
        class_id_by_name=class_id_by_name,
        imgsz=INFERENCE_IMGSZ,
        conf=INFERENCE_CONF,
        iou=INFERENCE_IOU,
        device=INFERENCE_DEVICE,
        batch=INFERENCE_BATCH,
    )
    fill_missing_labels_for_split(
        model=model,
        images_dir=val_images_dir,
        labels_dir=val_labels_dir,
        class_id_by_name=class_id_by_name,
        imgsz=INFERENCE_IMGSZ,
        conf=INFERENCE_CONF,
        iou=INFERENCE_IOU,
        device=INFERENCE_DEVICE,
        batch=INFERENCE_BATCH,
    )

    print(f"Prepared {len(copied_train)} train images and {len(copied_val)} val images.")
    print(f"Labels written under '{DATASET_DIR / 'labels'}'.")


def validate_configuration() -> None:
    if not SOURCE_DIR.exists():
        SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_YAML_PATH.exists():
        raise FileNotFoundError(f"data.yaml not found: {DATA_YAML_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}. "
            "Edit MODEL_PATH at the top of finetuning/data_collection/prepare_dataset.py."
        )
    if not 0 < TRAIN_RATIO < 1:
        raise ValueError("TRAIN_RATIO must be between 0 and 1.")
    if INDEX_WIDTH <= 0:
        raise ValueError("INDEX_WIDTH must be greater than 0.")
    if INFERENCE_BATCH <= 0:
        raise ValueError("INFERENCE_BATCH must be greater than 0.")


def load_class_names(data_yaml_path: Path) -> List[str]:
    with data_yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    names = data.get("names")
    if isinstance(names, dict):
        return [names[index] for index in sorted(names)]
    if isinstance(names, list):
        return names
    raise ValueError(f"Unsupported names format in {data_yaml_path}.")


def list_images(source_dir: Path) -> List[Path]:
    return sorted(
        path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def ensure_target_dirs(directories: Sequence[Path], *, overwrite: bool) -> None:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

        existing_files = [path for path in directory.iterdir() if path.is_file() and path.name != ".gitkeep"]
        if overwrite:
            for path in existing_files:
                path.unlink()


def split_images(image_paths: Sequence[Path], *, train_ratio: float, seed: int) -> tuple[List[Path], List[Path]]:
    shuffled = list(image_paths)
    random.Random(seed).shuffle(shuffled)

    train_count = int(len(shuffled) * train_ratio)
    train_count = max(1, min(train_count, len(shuffled) - 1)) if len(shuffled) > 1 else len(shuffled)

    return shuffled[:train_count], shuffled[train_count:]


def copy_and_rename(
    image_paths: Sequence[Path],
    target_dir: Path,
    *,
    prefix: str,
    index_width: int,
) -> List[Path]:
    starting_index = next_dataset_index(target_dir, prefix=prefix)
    highest_index = starting_index + max(0, len(image_paths) - 1)
    width = max(index_width, len(str(highest_index)))
    copied_paths: List[Path] = []

    for offset, source_path in enumerate(image_paths):
        destination_index = starting_index + offset
        destination_name = f"{prefix}_{destination_index:0{width}d}{source_path.suffix.lower()}"
        destination_path = target_dir / destination_name
        shutil.copy2(source_path, destination_path)
        copied_paths.append(destination_path)

    return copied_paths


def next_dataset_index(target_dir: Path, *, prefix: str) -> int:
    highest_index = -1

    for existing_path in target_dir.iterdir():
        if not existing_path.is_file():
            continue
        if existing_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not existing_path.stem.startswith(f"{prefix}_"):
            continue

        try:
            existing_index = int(existing_path.stem.removeprefix(f"{prefix}_"))
        except ValueError:
            continue

        highest_index = max(highest_index, existing_index)

    return highest_index + 1


def write_labels_for_split(
    *,
    model: YOLO,
    image_paths: Sequence[Path],
    output_dir: Path,
    class_id_by_name: dict[str, int],
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    batch: int,
) -> None:
    if not image_paths:
        return

    resolved_device = resolve_inference_device(device)

    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=resolved_device,
        batch=batch,
        verbose=False,
    )

    for image_path, result in zip(image_paths, results):
        label_path = output_dir / f"{image_path.stem}.txt"
        lines = list(format_result_lines(result, class_id_by_name))
        label_path.write_text("\n".join(lines), encoding="utf-8")


def fill_missing_labels_for_split(
    *,
    model: YOLO,
    images_dir: Path,
    labels_dir: Path,
    class_id_by_name: dict[str, int],
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    batch: int,
) -> None:
    missing_image_paths = list_images_missing_labels(images_dir, labels_dir)
    if not missing_image_paths:
        return

    write_labels_for_split(
        model=model,
        image_paths=missing_image_paths,
        output_dir=labels_dir,
        class_id_by_name=class_id_by_name,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        batch=batch,
    )


def list_images_missing_labels(images_dir: Path, labels_dir: Path) -> List[Path]:
    missing_image_paths: List[Path] = []

    for image_path in list_images(images_dir):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            continue
        missing_image_paths.append(image_path)

    return missing_image_paths


def format_result_lines(result: object, class_id_by_name: dict[str, int]) -> Iterable[str]:
    boxes = getattr(result, "boxes", None)
    names = getattr(result, "names", {})
    if boxes is None:
        return ()

    lines: List[str] = []
    for box in boxes:
        model_class_id = int(box.cls[0])
        class_name = names[model_class_id]
        output_class_id = class_id_by_name.get(class_name)
        if output_class_id is None:
            continue

        x_center, y_center, width, height = box.xywhn[0].tolist()
        lines.append(
            f"{output_class_id} "
            f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    return lines


def resolve_inference_device(device: str) -> str:
    requested_device = device.strip().lower()
    if requested_device != "auto":
        return device

    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"

    if torch.cuda.is_available():
        return "0"

    return "cpu"


if __name__ == "__main__":
    main()
