from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import List


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS_DIR = REPO_ROOT / "snapshots"
TARGET_DIR = REPO_ROOT / "finetuning" / "to_be_labled" / "images"

# Edit these constants if needed.
SAMPLES_PER_FOLDER = 1
RANDOM_SEED = 420
OVERWRITE_EXISTING = False


def main() -> None:
    validate_configuration()
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    if OVERWRITE_EXISTING:
        clear_target_directory(TARGET_DIR)

    random_generator = random.Random(RANDOM_SEED)
    copied_count = 0
    folder_count = 0

    for snapshot_folder in find_leaf_snapshot_folders(SNAPSHOTS_DIR):
        image_paths = list_images(snapshot_folder)
        if not image_paths:
            continue

        sample_count = min(SAMPLES_PER_FOLDER, len(image_paths))
        selected_images = random_generator.sample(image_paths, sample_count)

        for image_path in sorted(selected_images):
            destination_path = build_destination_path(
                image_path=image_path,
                base_dir=SNAPSHOTS_DIR,
                target_dir=TARGET_DIR,
            )
            shutil.copy2(image_path, destination_path)
            copied_count += 1

        folder_count += 1

    print(f"Copied {copied_count} images from {folder_count} snapshot folders into '{TARGET_DIR}'.")


def validate_configuration() -> None:
    if not SNAPSHOTS_DIR.exists():
        raise FileNotFoundError(f"Snapshots directory does not exist: {SNAPSHOTS_DIR}")
    if SAMPLES_PER_FOLDER <= 0:
        raise ValueError("SAMPLES_PER_FOLDER must be greater than 0.")


def clear_target_directory(target_dir: Path) -> None:
    for path in target_dir.iterdir():
        if path.is_file() and path.name != ".gitkeep":
            path.unlink()


def find_leaf_snapshot_folders(root_dir: Path) -> List[Path]:
    leaf_folders: List[Path] = []

    for directory in sorted(path for path in root_dir.rglob("*") if path.is_dir()):
        child_directories = [child for child in directory.iterdir() if child.is_dir()]
        if child_directories:
            continue
        leaf_folders.append(directory)

    return leaf_folders


def list_images(directory: Path) -> List[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_destination_path(*, image_path: Path, base_dir: Path, target_dir: Path) -> Path:
    relative_parts = image_path.relative_to(base_dir).parts
    flattened_prefix = "__".join(relative_parts[:-1])
    destination_name = f"{flattened_prefix}__{image_path.name}"
    return target_dir / destination_name


if __name__ == "__main__":
    main()
