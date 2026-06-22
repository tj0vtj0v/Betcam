from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

RawFrame = Any
Point = Tuple[int, int]
BoundingBox = Tuple[int, int, int, int]
QueueSample = Tuple[float, int]


@dataclass(frozen=True)
class Detection:
    track_id: Optional[int]
    class_id: int
    class_name: str
    category: str
    confidence: float
    bounding_box: BoundingBox
    center: Point
    trail: Tuple[Point, ...]


@dataclass(frozen=True)
class TrailPoint:
    position: Point
    age_frames: int


@dataclass(frozen=True)
class TrackedTrail:
    track_id: int
    category: str
    points: Tuple[TrailPoint, ...]


@dataclass
class DetectionResult:
    """Mutable per-frame output shared by a detector and its consumers."""

    detections: Tuple[Detection, ...] = field(default_factory=tuple)
    active_trails: Tuple[TrackedTrail, ...] = field(default_factory=tuple)

    def replace(
        self,
        detections: Tuple[Detection, ...],
        active_trails: Tuple[TrackedTrail, ...],
    ) -> None:
        self.detections = detections
        self.active_trails = active_trails
