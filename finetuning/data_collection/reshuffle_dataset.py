from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Sequence


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
FINETUNING_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = FINETUNING_DIR / "dataset"

# Edit these constants instead of passing CLI flags.
TRAIN_RATIO = 0.9
INDEX_WIDTH = 3


@dataclass(frozen=True)
class DatasetSample:
    image_path: Path
    label_path: Path


def main() -> None:
    train_samples = load_split_samples(split_name="train")
    val_samples = load_split_samples(split_name="val")
    all_samples = train_samples + val_samples

    if not all_samples:
        raise RuntimeError(f"No dataset samples found under '{DATASET_DIR}'.")

    shuffled_samples = list(all_samples)
    random.Random().shuffle(shuffled_samples)

    train_count = split_train_count(len(shuffled_samples), train_ratio=TRAIN_RATIO)
    rewritten_train = shuffled_samples[:train_count]
    rewritten_val = shuffled_samples[train_count:]

    rewrite_dataset(
        split_samples={
            "train": rewritten_train,
            "val": rewritten_val,
        }
    )

    print(
        f"Reshuffled {len(all_samples)} samples. "
        f"Applied TRAIN_RATIO={TRAIN_RATIO}: train={len(rewritten_train)}, val={len(rewritten_val)}."
    )


def split_train_count(total_count: int, *, train_ratio: float) -> int:
    if not 0 < train_ratio < 1:
        raise ValueError("TRAIN_RATIO must be between 0 and 1.")

    if total_count <= 1:
        return total_count

    train_count = int(total_count * train_ratio)
    return max(1, min(train_count, total_count - 1))


def load_split_samples(*, split_name: str) -> List[DatasetSample]:
    images_dir = DATASET_DIR / "images" / split_name
    labels_dir = DATASET_DIR / "labels" / split_name

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory does not exist: {labels_dir}")

    image_paths = sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    samples: List[DatasetSample] = []
    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for image '{image_path.name}': {label_path}")
        samples.append(DatasetSample(image_path=image_path, label_path=label_path))

    return samples


def rewrite_dataset(*, split_samples: Dict[str, Sequence[DatasetSample]]) -> None:
    with TemporaryDirectory(dir=DATASET_DIR) as temp_dir_name:
        temp_dataset_dir = Path(temp_dir_name) / "dataset"
        stage_split_directories(temp_dataset_dir=temp_dataset_dir, split_samples=split_samples)
        replace_dataset_split_directories(temp_dataset_dir=temp_dataset_dir, split_names=tuple(split_samples))


def stage_split_directories(*, temp_dataset_dir: Path, split_samples: Dict[str, Sequence[DatasetSample]]) -> None:
    for split_name, samples in split_samples.items():
        temp_images_dir = temp_dataset_dir / "images" / split_name
        temp_labels_dir = temp_dataset_dir / "labels" / split_name
        temp_images_dir.mkdir(parents=True, exist_ok=True)
        temp_labels_dir.mkdir(parents=True, exist_ok=True)

        width = max(INDEX_WIDTH, len(str(max(0, len(samples) - 1))))
        for index, sample in enumerate(samples):
            image_name = f"{split_name}_{index:0{width}d}{sample.image_path.suffix.lower()}"
            label_name = f"{split_name}_{index:0{width}d}.txt"
            shutil.copy2(sample.image_path, temp_images_dir / image_name)
            shutil.copy2(sample.label_path, temp_labels_dir / label_name)


def replace_dataset_split_directories(*, temp_dataset_dir: Path, split_names: Sequence[str]) -> None:
    for split_name in split_names:
        live_images_dir = DATASET_DIR / "images" / split_name
        live_labels_dir = DATASET_DIR / "labels" / split_name
        staged_images_dir = temp_dataset_dir / "images" / split_name
        staged_labels_dir = temp_dataset_dir / "labels" / split_name

        for live_dir, staged_dir in (
            (live_images_dir, staged_images_dir),
            (live_labels_dir, staged_labels_dir),
        ):
            live_dir.parent.mkdir(parents=True, exist_ok=True)
            backup_dir = live_dir.with_name(f"{live_dir.name}__backup")
            if backup_dir.exists():
                shutil.rmtree(backup_dir)

            if live_dir.exists():
                live_dir.replace(backup_dir)

            try:
                staged_dir.replace(live_dir)
            except Exception:
                if backup_dir.exists() and not live_dir.exists():
                    backup_dir.replace(live_dir)
                raise

            if backup_dir.exists():
                shutil.rmtree(backup_dir)


if __name__ == "__main__":
    main()
