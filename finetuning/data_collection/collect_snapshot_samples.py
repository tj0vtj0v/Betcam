from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import List


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
REPO_ROOT = Path(__file__).resolve().parents[2]
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
    if len(relative_parts) < 3:
        raise ValueError(
            f"Expected snapshot path to include at least town and location below '{base_dir}', got '{image_path}'."
        )

    town = relative_parts[0]
    location = relative_parts[1]
    prefix = f"{town}__{location}"
    next_index = next_destination_index(target_dir, prefix=prefix)
    destination_name = f"{prefix}__{next_index:04d}{image_path.suffix.lower()}"
    return target_dir / destination_name


def next_destination_index(target_dir: Path, *, prefix: str) -> int:
    highest_index = -1

    for existing_path in target_dir.iterdir():
        if not existing_path.is_file():
            continue
        if existing_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not existing_path.stem.startswith(f"{prefix}__"):
            continue

        suffix = existing_path.stem.removeprefix(f"{prefix}__")
        try:
            existing_index = int(suffix)
        except ValueError:
            continue

        highest_index = max(highest_index, existing_index)

    return highest_index + 1


if __name__ == "__main__":
    main()
