from __future__ import annotations

import colorsys
from typing import Dict, Tuple

import cv2

from config.types import Detection, RawFrame, TrackedTrail, TrailPoint

Color = Tuple[int, int, int]


class DetectionAnnotator:
    CATEGORY_HUES: Dict[str, float] = {
        "people": 0.08,
        "animals": 0.33,
        "traffic": 0.62,
    }

    def __init__(
        self,
        *,
        show_labels: bool = True,
        label_font_scale: float = 0.4,
        trail_thickness: int = 2,
        trail_point_fade_frames: int = 0,
    ) -> None:
        self.show_labels = show_labels
        self.label_font_scale = label_font_scale
        self.trail_thickness = trail_thickness
        self.trail_point_fade_frames = trail_point_fade_frames

    def annotate(
        self,
        frame: RawFrame,
        detections: Tuple[Detection, ...],
        active_trails: Tuple[TrackedTrail, ...],
    ) -> RawFrame:
        if not detections and not active_trails:
            return frame

        annotated_frame = frame.copy()

        for trail in active_trails:
            if len(trail.points) >= 2:
                trail_color = self._color_for_track(trail.category, trail.track_id)
                self._draw_trail(annotated_frame, trail.points, trail_color)

        for detection in detections:
            color = self._color_for_detection(detection)
            x1, y1, x2, y2 = detection.bounding_box

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)

            if self.show_labels:
                label = self._label_for_detection(detection)
                text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.label_font_scale, 1)
                _, text_height = text_size
                text_x = x1
                text_y = max(text_height + baseline + 1, y1 - 3)
                cv2.putText(
                    annotated_frame,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.label_font_scale,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        return annotated_frame

    @classmethod
    def _color_for_detection(cls, detection: Detection) -> Color:
        if detection.track_id is None:
            return cls._color_for_track(detection.category, detection.class_id)

        return cls._color_for_track(detection.category, detection.track_id)

    @classmethod
    def _color_for_track(cls, category: str, identifier: int) -> Color:
        base_hue = cls.CATEGORY_HUES.get(category, 0.0)
        hue_offset = ((identifier * 37) % 100) / 1000
        hue = (base_hue + hue_offset) % 1.0
        saturation = 0.65 + (((identifier * 17) % 20) / 100)
        value = 0.75 + (((identifier * 29) % 20) / 100)
        red, green, blue = colorsys.hsv_to_rgb(hue, min(saturation, 0.9), min(value, 0.95))
        return (
            int(round(blue * 255)),
            int(round(green * 255)),
            int(round(red * 255)),
        )

    @staticmethod
    def _label_for_detection(detection: Detection) -> str:
        if detection.track_id is None:
            return f"{detection.class_name} {detection.confidence:.2f}"

        return f"{detection.class_name} #{detection.track_id} {detection.confidence:.2f}"

    @staticmethod
    def _segment_alpha(start_point: TrailPoint, end_point: TrailPoint, fade_after_frames: int) -> float:
        if fade_after_frames <= 0:
            return 1.0

        segment_age_frames = max(start_point.age_frames, end_point.age_frames)
        fade_starts_after_frames = (2 * fade_after_frames) / 3
        if segment_age_frames <= fade_starts_after_frames:
            return 1.0

        fade_window_frames = max(1.0, fade_after_frames - fade_starts_after_frames)
        fade_progress = (segment_age_frames - fade_starts_after_frames) / fade_window_frames
        return max(0.0, 1.0 - min(fade_progress, 1.0))

    def _draw_trail(self, annotated_frame: RawFrame, trail_points: Tuple[TrailPoint, ...], color: Color) -> None:
        for start_point, end_point in zip(trail_points, trail_points[1:]):
            alpha = self._segment_alpha(start_point, end_point, self.trail_point_fade_frames)
            if alpha <= 0:
                continue
            self._draw_blended_line_segment(
                annotated_frame,
                start_point.position,
                end_point.position,
                color,
                alpha,
            )

    def _draw_blended_line_segment(
        self,
        frame: RawFrame,
        start_point: Tuple[int, int],
        end_point: Tuple[int, int],
        color: Color,
        alpha: float,
    ) -> None:
        margin = self.trail_thickness + 3
        min_x = max(0, min(start_point[0], end_point[0]) - margin)
        min_y = max(0, min(start_point[1], end_point[1]) - margin)
        max_x = min(frame.shape[1], max(start_point[0], end_point[0]) + margin + 1)
        max_y = min(frame.shape[0], max(start_point[1], end_point[1]) + margin + 1)

        if min_x >= max_x or min_y >= max_y:
            return

        frame_region = frame[min_y:max_y, min_x:max_x]
        overlay_region = frame_region.copy()
        local_start = (start_point[0] - min_x, start_point[1] - min_y)
        local_end = (end_point[0] - min_x, end_point[1] - min_y)

        cv2.line(
            overlay_region,
            local_start,
            local_end,
            color,
            self.trail_thickness,
            cv2.LINE_AA,
        )
        cv2.addWeighted(overlay_region, alpha, frame_region, 1.0 - alpha, 0, frame_region)
