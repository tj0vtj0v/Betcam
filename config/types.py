from dataclasses import dataclass
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


@dataclass(frozen=True)
class StreamFrame:
    raw_frame: RawFrame
    annotated_frame: RawFrame
    detections: Tuple[Detection, ...]
    active_trails: Tuple[TrackedTrail, ...]
    timestamp: float
