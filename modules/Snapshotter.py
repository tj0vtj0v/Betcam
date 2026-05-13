from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Dict, Optional

import cv2

from config.types import StreamFrame


class Snapshotter:
    def __init__(
        self,
        *,
        base_directory: str = "snapshots",
        save_every_n_frames: int = 1,
        scale_factor: float = 1.0,
    ) -> None:
        if save_every_n_frames <= 0:
            raise ValueError("save_every_n_frames must be greater than 0.")
        if scale_factor <= 0:
            raise ValueError("scale_factor must be greater than 0.")

        self.base_directory = Path(base_directory)
        self.save_every_n_frames = save_every_n_frames
        self.scale_factor = scale_factor
        self._save_counters: Dict[Path, int] = {}
        self._lock = Lock()

    def save(
        self,
        stream_frame: StreamFrame,
        *,
        town: Optional[str] = None,
        location: Optional[str] = None,
        annotated: bool = True,
    ) -> Optional[Path]:
        snapshot_directory = self._snapshot_directory(town=town, location=location)
        if not self._should_save(snapshot_directory):
            return None

        snapshot_directory.mkdir(parents=True, exist_ok=True)

        snapshot_index = self._next_snapshot_index(snapshot_directory)
        snapshot_path = snapshot_directory / f"{snapshot_index:04d}.jpg"
        snapshot_frame = stream_frame.annotated_frame if annotated else stream_frame.raw_frame
        snapshot_frame = self._scale_frame(snapshot_frame)

        if not cv2.imwrite(str(snapshot_path), snapshot_frame):
            raise RuntimeError(f"Could not write snapshot to '{snapshot_path}'.")

        return snapshot_path

    def _snapshot_directory(self, *, town: Optional[str], location: Optional[str]) -> Path:
        if town is None or location is None:
            return self.base_directory / "manual"

        return self.base_directory / self._sanitize_path_component(town) / self._sanitize_path_component(location)

    def _should_save(self, snapshot_directory: Path) -> bool:
        with self._lock:
            next_counter = self._save_counters.get(snapshot_directory, 0) + 1
            self._save_counters[snapshot_directory] = next_counter
            return next_counter % self.save_every_n_frames == 0

    def _scale_frame(self, frame):
        if self.scale_factor == 1.0:
            return frame

        height, width = frame.shape[:2]
        scaled_width = max(1, int(round(width * self.scale_factor)))
        scaled_height = max(1, int(round(height * self.scale_factor)))
        interpolation = cv2.INTER_AREA if self.scale_factor < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(frame, (scaled_width, scaled_height), interpolation=interpolation)

    @staticmethod
    def _sanitize_path_component(value: str) -> str:
        return value.replace(" ", "_")

    @staticmethod
    def _next_snapshot_index(snapshot_directory: Path) -> int:
        existing_indices = []

        for snapshot_file in snapshot_directory.glob("*.jpg"):
            try:
                existing_indices.append(int(snapshot_file.stem))
            except ValueError:
                continue

        return (max(existing_indices) + 1) if existing_indices else 1
