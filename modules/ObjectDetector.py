from __future__ import annotations

import sys
import threading
import time
import subprocess
from math import ceil
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2

from config.types import Detection, Point, RawFrame, TrackedTrail, TrailPoint

ANIMAL_CLASS_NAMES = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}
PEOPLE_CLASS_NAMES = {"person"}
TRAFFIC_CLASS_NAMES = {
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "stop sign",
    "parking meter",
}
ALLOWED_CLASS_NAMES = ANIMAL_CLASS_NAMES | PEOPLE_CLASS_NAMES | TRAFFIC_CLASS_NAMES
GENERIC_DISPLAY_ADAPTER_NAMES = {
    "microsoft basic display adapter",
    "microsoft basic render driver",
}


@dataclass
class TrackState:
    category: str
    points: Deque[TrailPoint]
    last_seen_at: float


@dataclass(frozen=True)
class DetectionCandidate:
    track_id: Optional[int]
    class_id: int
    class_name: str
    category: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]


class TrackedObjectDetector:
    def __init__(
        self,
        *,
        model_path: str = "yolo11s.pt",
        inference_device: str = "auto",
        model_input_size: int = 480,
        detection_confidence_threshold: float = 0.25,
        tracker_config_path: str = "bytetrack.yaml",
        detection_start_delay_seconds: float = 0.0,
        track_history_timeout_seconds: float = 3.0,
        max_track_history_points: int = 256,
        max_inference_frame_dimension: int = 480,
        fixed_inference_frame_budget_ms: int = 30,
        max_frames_to_skip_after_inference: int = 5,
        inference_time_ema_alpha: float = 0.3,
        same_class_iou_suppression_threshold: float = 0.5,
        allowed_class_names: Optional[Iterable[str]] = None,
    ) -> None:
        if model_input_size <= 0:
            raise ValueError("model_input_size must be greater than 0.")
        if not inference_device:
            raise ValueError("inference_device must not be empty.")
        if not 0 < detection_confidence_threshold <= 1:
            raise ValueError("detection_confidence_threshold must be in the range (0, 1].")
        if detection_start_delay_seconds < 0:
            raise ValueError("detection_start_delay_seconds must be non-negative.")
        if track_history_timeout_seconds < 0:
            raise ValueError("track_history_timeout_seconds must be non-negative.")
        if max_track_history_points <= 0:
            raise ValueError("max_track_history_points must be greater than 0.")
        if max_inference_frame_dimension <= 0:
            raise ValueError("max_inference_frame_dimension must be greater than 0.")
        if fixed_inference_frame_budget_ms <= 0:
            raise ValueError("fixed_inference_frame_budget_ms must be greater than 0.")
        if max_frames_to_skip_after_inference < 0:
            raise ValueError("max_frames_to_skip_after_inference must be non-negative.")
        if not 0 < inference_time_ema_alpha <= 1:
            raise ValueError("inference_time_ema_alpha must be in the range (0, 1].")
        if not 0 <= same_class_iou_suppression_threshold <= 1:
            raise ValueError("same_class_iou_suppression_threshold must be in the range [0, 1].")

        self.model_path = model_path
        self.inference_device = inference_device
        self.model_input_size = model_input_size
        self.detection_confidence_threshold = detection_confidence_threshold
        self.tracker_config_path = tracker_config_path
        self.detection_start_delay_seconds = detection_start_delay_seconds
        self.track_history_timeout_seconds = track_history_timeout_seconds
        self.max_track_history_points = max_track_history_points
        self.max_inference_frame_dimension = max_inference_frame_dimension
        self.fixed_inference_frame_budget_ms = fixed_inference_frame_budget_ms
        self.max_frames_to_skip_after_inference = max_frames_to_skip_after_inference
        self.inference_time_ema_alpha = inference_time_ema_alpha
        self.same_class_iou_suppression_threshold = same_class_iou_suppression_threshold
        self.allowed_class_names = set(allowed_class_names or ALLOWED_CLASS_NAMES)

        self._model = None
        self._allowed_class_ids: Optional[List[int]] = None
        self._resolved_inference_device: Optional[str] = None
        self._gpu_hardware_names: Optional[Tuple[str, ...]] = None
        self._track_states: Dict[int, TrackState] = {}
        self._cached_detections: Tuple[Detection, ...] = ()
        self._frames_until_next_inference = 0
        self._average_inference_seconds: Optional[float] = None
        self._detection_enabled_at_timestamp: Optional[float] = None
        self._model_preload_started = False
        self._model_preload_thread: Optional[threading.Thread] = None

    def clone(self) -> "TrackedObjectDetector":
        return TrackedObjectDetector(
            model_path=self.model_path,
            inference_device=self.inference_device,
            model_input_size=self.model_input_size,
            detection_confidence_threshold=self.detection_confidence_threshold,
            tracker_config_path=self.tracker_config_path,
            detection_start_delay_seconds=self.detection_start_delay_seconds,
            track_history_timeout_seconds=self.track_history_timeout_seconds,
            max_track_history_points=self.max_track_history_points,
            max_inference_frame_dimension=self.max_inference_frame_dimension,
            fixed_inference_frame_budget_ms=self.fixed_inference_frame_budget_ms,
            max_frames_to_skip_after_inference=self.max_frames_to_skip_after_inference,
            inference_time_ema_alpha=self.inference_time_ema_alpha,
            same_class_iou_suppression_threshold=self.same_class_iou_suppression_threshold,
            allowed_class_names=tuple(self.allowed_class_names),
        )

    def reset(self) -> None:
        self._track_states.clear()
        self._cached_detections = ()
        self._frames_until_next_inference = 0
        self._average_inference_seconds = None
        self._detection_enabled_at_timestamp = time.monotonic() + self.detection_start_delay_seconds
        self._ensure_model_preload()
        if self._model is not None:
            self._model.predictor = None

    def detect(self, frame: RawFrame, *, timestamp: Optional[float] = None) -> Tuple[Detection, ...]:
        active_timestamp = time.monotonic() if timestamp is None else timestamp
        if not self._detection_is_enabled(active_timestamp):
            return ()

        self._advance_trail_ages()

        if self._should_skip_inference():
            self._prune_stale_tracks(active_timestamp)
            self._frames_until_next_inference -= 1
            return self._filter_cached_detections()

        model = self._get_model()
        allowed_class_ids = self._get_allowed_class_ids()
        inference_frame, scale_x, scale_y = self._prepare_inference_frame(frame)
        inference_started_at = time.monotonic()

        results = model.track(
            inference_frame,
            imgsz=self.model_input_size,
            conf=self.detection_confidence_threshold,
            classes=allowed_class_ids,
            tracker=self.tracker_config_path,
            device=self._get_inference_device(),
            persist=True,
            verbose=False,
        )
        inference_duration_seconds = time.monotonic() - inference_started_at
        detections = self._parse_results(results, active_timestamp, scale_x=scale_x, scale_y=scale_y)
        self._prune_stale_tracks(active_timestamp)
        self._cached_detections = tuple(detections)
        self._update_adaptive_skip_state(inference_duration_seconds)
        return self._cached_detections

    def active_trails(self, *, timestamp: Optional[float] = None) -> Tuple[TrackedTrail, ...]:
        active_timestamp = time.monotonic() if timestamp is None else timestamp
        self._prune_stale_tracks(active_timestamp)
        trails = []

        for track_id, state in self._track_states.items():
            trails.append(
                TrackedTrail(
                    track_id=track_id,
                    category=state.category,
                    points=tuple(state.points),
                )
            )

        return tuple(trails)

    def _get_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "Ultralytics is required for object detection. Install dependencies from requirements.txt first."
                ) from error

            self._model = YOLO(self.model_path)

        return self._model

    def _ensure_model_preload(self) -> None:
        if self.detection_start_delay_seconds <= 0:
            return
        if self._model is not None or self._model_preload_started:
            return

        self._model_preload_started = True
        self._model_preload_thread = threading.Thread(
            target=self._preload_model,
            daemon=True,
        )
        self._model_preload_thread.start()

    def _preload_model(self) -> None:
        try:
            self._get_model()
            self._get_inference_device()
            self._get_allowed_class_ids()
        except Exception:
            self._model_preload_started = False

    def _get_allowed_class_ids(self) -> List[int]:
        if self._allowed_class_ids is not None:
            return self._allowed_class_ids

        model = self._get_model()
        allowed_class_ids = []
        if isinstance(model.names, dict):
            for class_id, class_name in model.names.items():
                if class_name in self.allowed_class_names:
                    allowed_class_ids.append(int(class_id))
        else:
            for class_id, class_name in enumerate(model.names):
                if class_name in self.allowed_class_names:
                    allowed_class_ids.append(class_id)

        self._allowed_class_ids = allowed_class_ids
        return allowed_class_ids

    def _get_inference_device(self) -> str:
        if self._resolved_inference_device is None:
            self._resolved_inference_device = self._resolve_inference_device()
            self._log_inference_runtime()

        return self._resolved_inference_device

    def _resolve_inference_device(self) -> str:
        requested_device = self.inference_device.strip().lower()
        if requested_device != "auto":
            return self.inference_device

        try:
            import torch
        except ModuleNotFoundError:
            return "cpu"

        if torch.cuda.is_available():
            return "0"

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"

        return "cpu"

    def _log_inference_runtime(self) -> None:
        resolved_device = self._resolved_inference_device or "cpu"
        print(f"Using inference device: {resolved_device}", file=sys.stderr)

        gpu_hardware_names = self._get_gpu_hardware_names()
        if gpu_hardware_names:
            gpu_list = ", ".join(gpu_hardware_names)
            print(f"Detected GPU hardware: {gpu_list}", file=sys.stderr)
            if resolved_device == "cpu" and not self._torch_cuda_is_available():
                print(
                    "GPU acceleration is unavailable because the installed PyTorch build does not include CUDA support.",
                    file=sys.stderr,
                )
        else:
            print("No dedicated GPU hardware detected.", file=sys.stderr)

    def _get_gpu_hardware_names(self) -> Tuple[str, ...]:
        if self._gpu_hardware_names is None:
            detected_names = []
            detected_names.extend(self._detect_torch_cuda_gpu_names())
            detected_names.extend(self._detect_nvidia_smi_gpu_names())
            detected_names.extend(self._detect_windows_video_controller_names())
            self._gpu_hardware_names = tuple(dict.fromkeys(detected_names))

        return self._gpu_hardware_names

    def _detect_torch_cuda_gpu_names(self) -> List[str]:
        try:
            import torch
        except ModuleNotFoundError:
            return []

        if not torch.cuda.is_available():
            return []

        detected_names = []
        for index in range(torch.cuda.device_count()):
            try:
                detected_names.append(torch.cuda.get_device_name(index).strip())
            except Exception:
                continue
        return [name for name in detected_names if name]

    def _detect_nvidia_smi_gpu_names(self) -> List[str]:
        process = self._run_command(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ]
        )
        if process is None or process.returncode != 0:
            return []

        return self._parse_gpu_names_from_lines(process.stdout.splitlines())

    def _detect_windows_video_controller_names(self) -> List[str]:
        if not sys.platform.startswith("win"):
            return []

        process = self._run_command(
            [
                "wmic",
                "path",
                "win32_VideoController",
                "get",
                "name",
            ]
        )
        if process is None or process.returncode != 0:
            return []

        parsed_names = self._parse_gpu_names_from_lines(process.stdout.splitlines())
        return [
            name
            for name in parsed_names
            if name.lower() not in GENERIC_DISPLAY_ADAPTER_NAMES
        ]

    def _parse_gpu_names_from_lines(self, lines: Sequence[str]) -> List[str]:
        detected_names = []
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.lower() == "name":
                continue
            detected_names.append(line)
        return detected_names

    def _run_command(self, command: Sequence[str]) -> Optional[subprocess.CompletedProcess[str]]:
        try:
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            return None

    def _torch_cuda_is_available(self) -> bool:
        try:
            import torch
        except ModuleNotFoundError:
            return False

        return bool(torch.cuda.is_available())

    def _parse_results(
        self,
        results: Sequence[object],
        timestamp: float,
        *,
        scale_x: float,
        scale_y: float,
    ) -> List[Detection]:
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        candidates: List[DetectionCandidate] = []
        names = result.names

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = names[class_id]
            category = self._category_for_class_name(class_name)
            if category is None:
                continue

            scaled_box = box.xyxy[0].tolist()
            x1, y1, x2, y2 = self._scale_bounding_box(scaled_box, scale_x=scale_x, scale_y=scale_y)
            track_id = int(box.id[0]) if box.id is not None else None

            candidates.append(
                DetectionCandidate(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    category=category,
                    confidence=float(box.conf[0]),
                    bounding_box=(x1, y1, x2, y2),
                )
            )

        filtered_candidates = self._suppress_same_class_overlaps(candidates)
        detections: List[Detection] = []
        active_track_ids: Set[int] = set()

        for candidate in filtered_candidates:
            x1, y1, x2, y2 = candidate.bounding_box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            trail: Tuple[Point, ...] = ()

            if candidate.track_id is not None:
                active_track_ids.add(candidate.track_id)
                state = self._track_states.get(candidate.track_id)
                if state is None:
                    state = TrackState(
                        category=candidate.category,
                        points=deque(maxlen=self.max_track_history_points),
                        last_seen_at=timestamp,
                    )
                    self._track_states[candidate.track_id] = state

                state.category = candidate.category
                state.points.append(TrailPoint(position=center, age_frames=0))
                state.last_seen_at = timestamp
                trail = tuple(point.position for point in state.points)

            detections.append(
                Detection(
                    track_id=candidate.track_id,
                    class_id=candidate.class_id,
                    class_name=candidate.class_name,
                    category=candidate.category,
                    confidence=candidate.confidence,
                    bounding_box=candidate.bounding_box,
                    center=center,
                    trail=trail,
                )
            )

        self._drop_inactive_tracks(active_track_ids, timestamp)
        return detections

    def _drop_inactive_tracks(self, active_track_ids: Set[int], timestamp: float) -> None:
        for track_id, state in list(self._track_states.items()):
            if track_id in active_track_ids:
                continue

            if timestamp - state.last_seen_at > self.track_history_timeout_seconds:
                del self._track_states[track_id]

    def _prune_stale_tracks(self, timestamp: float) -> None:
        for track_id, state in list(self._track_states.items()):
            if timestamp - state.last_seen_at > self.track_history_timeout_seconds:
                del self._track_states[track_id]

    def _advance_trail_ages(self) -> None:
        if not self._track_states:
            return

        for track_id, state in list(self._track_states.items()):
            aged_points = deque(maxlen=self.max_track_history_points)

            for trail_point in state.points:
                next_age_frames = trail_point.age_frames + 1
                aged_points.append(
                    TrailPoint(
                        position=trail_point.position,
                        age_frames=next_age_frames,
                    )
                )

            state.points = aged_points

    def _should_skip_inference(self) -> bool:
        return self._frames_until_next_inference > 0 and self._average_inference_seconds is not None

    def _update_adaptive_skip_state(self, inference_duration_seconds: float) -> None:
        if self._average_inference_seconds is None:
            self._average_inference_seconds = inference_duration_seconds
        else:
            alpha = self.inference_time_ema_alpha
            self._average_inference_seconds = (
                (alpha * inference_duration_seconds) + ((1 - alpha) * self._average_inference_seconds)
            )

        inference_time_ms = self._average_inference_seconds * 1000
        inference_to_budget_ratio = inference_time_ms / self.fixed_inference_frame_budget_ms

        if inference_to_budget_ratio <= 1:
            self._frames_until_next_inference = 0
            return

        self._frames_until_next_inference = min(
            self.max_frames_to_skip_after_inference,
            ceil(inference_to_budget_ratio),
        )

    def _filter_cached_detections(self) -> Tuple[Detection, ...]:
        active_track_ids = set(self._track_states)
        filtered_detections = []

        for detection in self._cached_detections:
            if detection.track_id is None or detection.track_id in active_track_ids:
                filtered_detections.append(detection)

        self._cached_detections = tuple(filtered_detections)
        return self._cached_detections

    def _suppress_same_class_overlaps(self, candidates: Sequence[DetectionCandidate]) -> List[DetectionCandidate]:
        if not candidates:
            return []

        kept_candidates: List[DetectionCandidate] = []

        for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
            if any(
                kept.class_id == candidate.class_id
                and self._intersection_over_union(kept.bounding_box, candidate.bounding_box)
                >= self.same_class_iou_suppression_threshold
                for kept in kept_candidates
            ):
                continue

            kept_candidates.append(candidate)

        return kept_candidates

    def _prepare_inference_frame(self, frame: RawFrame) -> Tuple[RawFrame, float, float]:
        frame_height, frame_width = frame.shape[:2]
        longest_side = max(frame_width, frame_height)

        if longest_side <= self.max_inference_frame_dimension:
            return frame, 1.0, 1.0

        resize_ratio = self.max_inference_frame_dimension / longest_side
        inference_width = max(1, int(round(frame_width * resize_ratio)))
        inference_height = max(1, int(round(frame_height * resize_ratio)))
        inference_frame = cv2.resize(frame, (inference_width, inference_height), interpolation=cv2.INTER_AREA)

        return inference_frame, frame_width / inference_width, frame_height / inference_height

    def _detection_is_enabled(self, timestamp: float) -> bool:
        if self._detection_enabled_at_timestamp is None:
            self._detection_enabled_at_timestamp = timestamp + self.detection_start_delay_seconds

        return timestamp >= self._detection_enabled_at_timestamp

    @staticmethod
    def _scale_bounding_box(
        bounding_box: Sequence[float],
        *,
        scale_x: float,
        scale_y: float,
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bounding_box
        return (
            int(round(x1 * scale_x)),
            int(round(y1 * scale_y)),
            int(round(x2 * scale_x)),
            int(round(y2 * scale_y)),
        )

    @staticmethod
    def _category_for_class_name(class_name: str) -> Optional[str]:
        if class_name in PEOPLE_CLASS_NAMES:
            return "people"
        if class_name in ANIMAL_CLASS_NAMES:
            return "animals"
        if class_name in TRAFFIC_CLASS_NAMES:
            return "traffic"
        return None

    @staticmethod
    def _intersection_over_union(
        left_box: Tuple[int, int, int, int],
        right_box: Tuple[int, int, int, int],
    ) -> float:
        left_x1, left_y1, left_x2, left_y2 = left_box
        right_x1, right_y1, right_x2, right_y2 = right_box

        intersection_x1 = max(left_x1, right_x1)
        intersection_y1 = max(left_y1, right_y1)
        intersection_x2 = min(left_x2, right_x2)
        intersection_y2 = min(left_y2, right_y2)

        intersection_width = max(0, intersection_x2 - intersection_x1)
        intersection_height = max(0, intersection_y2 - intersection_y1)
        intersection_area = intersection_width * intersection_height
        if intersection_area <= 0:
            return 0.0

        left_area = max(0, left_x2 - left_x1) * max(0, left_y2 - left_y1)
        right_area = max(0, right_x2 - right_x1) * max(0, right_y2 - right_y1)
        union_area = left_area + right_area - intersection_area
        if union_area <= 0:
            return 0.0

        return intersection_area / union_area
