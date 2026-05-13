from collections import deque
from typing import Deque, List, Tuple

import cv2
import numpy as np

DEBUG_WINDOW_NAME = "Debug Plot"
DEBUG_PLOT_TIME_WINDOW_SECONDS = 60
DEBUG_PLOT_WIDTH = 900
DEBUG_PLOT_HEIGHT = 520

DebugSample = Tuple[float, int, float, int]
PlotSeries = Tuple[str, List[float], Tuple[int, int, int], float]


class DebugPlot:
    def __init__(
        self,
        maximum_buffer_size: int,
        maximum_display_delay_ms: int,
        queue_sample_window_seconds: int,
        window_name: str = DEBUG_WINDOW_NAME,
        time_window_seconds: int = DEBUG_PLOT_TIME_WINDOW_SECONDS,
        plot_width: int = DEBUG_PLOT_WIDTH,
        plot_height: int = DEBUG_PLOT_HEIGHT,
    ) -> None:
        self.maximum_buffer_size = maximum_buffer_size
        self.maximum_display_delay_ms = maximum_display_delay_ms
        self.queue_sample_window_seconds = queue_sample_window_seconds
        self.window_name = window_name
        self.time_window_seconds = time_window_seconds
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.samples: Deque[DebugSample] = deque()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def add_sample(
        self,
        elapsed_time_seconds: float,
        queue_size: int,
        average_queue_size: float,
        delay_ms: int,
    ) -> None:
        self.samples.append((elapsed_time_seconds, queue_size, average_queue_size, delay_ms))
        self._remove_old_samples(elapsed_time_seconds)

    def show(self) -> None:
        plot_image = self._create_plot_image()
        cv2.imshow(self.window_name, plot_image)

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)

    def _remove_old_samples(self, current_elapsed_time_seconds: float) -> None:
        oldest_allowed_time_seconds = current_elapsed_time_seconds - self.time_window_seconds
        while self.samples and self.samples[0][0] < oldest_allowed_time_seconds:
            self.samples.popleft()

    def _create_plot_image(self) -> np.ndarray:
        plot_image = np.full((self.plot_height, self.plot_width, 3), 245, dtype=np.uint8)
        plot_left = 70
        plot_right = self.plot_width - 25
        first_plot_top = 45
        first_plot_bottom = (self.plot_height // 2) - 25
        second_plot_top = (self.plot_height // 2) + 35
        second_plot_bottom = self.plot_height - 55

        self._draw_panel(
            plot_image,
            title=f"Buffer size over time ({self.queue_sample_window_seconds}s average overlay)",
            top=first_plot_top,
            bottom=first_plot_bottom,
            left=plot_left,
            right=plot_right,
            series=[
                (
                    "actual",
                    [queue_size for _, queue_size, _, _ in self.samples],
                    (30, 130, 220),
                    self.maximum_buffer_size,
                ),
                (
                    "avg",
                    [average_queue_size for _, _, average_queue_size, _ in self.samples],
                    (205, 90, 45),
                    self.maximum_buffer_size,
                ),
            ],
            value_limit=self.maximum_buffer_size,
        )
        self._draw_panel(
            plot_image,
            title=f"Display delay ({self.queue_sample_window_seconds}s sample window)",
            top=second_plot_top,
            bottom=second_plot_bottom,
            left=plot_left,
            right=plot_right,
            series=[
                (
                    "delay",
                    [delay_ms for _, _, _, delay_ms in self.samples],
                    (40, 170, 70),
                    self.maximum_display_delay_ms,
                ),
            ],
            value_limit=self.maximum_display_delay_ms,
        )
        self._draw_time_axis(plot_image, plot_left, plot_right, second_plot_bottom)

        return plot_image

    def _draw_panel(
        self,
        plot_image: np.ndarray,
        title: str,
        top: int,
        bottom: int,
        left: int,
        right: int,
        series: List[PlotSeries],
        value_limit: int,
    ) -> None:
        axis_color = (70, 70, 70)
        grid_color = (215, 215, 215)

        cv2.rectangle(plot_image, (left, top), (right, bottom), axis_color, 1)
        cv2.putText(plot_image, title, (left, top - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_color, 1)

        for grid_index in range(1, 4):
            grid_y = top + round((bottom - top) * grid_index / 4)
            cv2.line(plot_image, (left, grid_y), (right, grid_y), grid_color, 1)

        cv2.putText(plot_image, str(value_limit), (8, top + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, axis_color, 1)
        cv2.putText(plot_image, "0", (42, bottom + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, axis_color, 1)

        if not series:
            return

        legend_x = right - 190
        for series_index, (label, values, line_color, series_value_limit) in enumerate(series):
            if not values:
                continue

            latest_value = values[-1]
            latest_value_text = self._format_legend_value(label, latest_value)
            legend_y = top - 14 + (series_index * 17)
            cv2.putText(
                plot_image,
                f"{label}: {latest_value_text}",
                (legend_x, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                line_color,
                1,
            )

            line_points = self._calculate_line_points(top, bottom, left, right, values, series_value_limit)
            if len(line_points) == 1:
                cv2.circle(plot_image, line_points[0], 2, line_color, -1)
            elif line_points:
                cv2.polylines(plot_image, [np.array(line_points, dtype=np.int32)], False, line_color, 2)

    def _format_legend_value(self, label: str, value: float) -> str:
        if label == "delay":
            return f"{round(value)} ms / {self.display_delay_ms_to_fps(round(value)):.1f} FPS"

        return f"{value:.1f}" if isinstance(value, float) else str(value)

    def _calculate_line_points(
        self,
        top: int,
        bottom: int,
        left: int,
        right: int,
        values: List[float],
        value_limit: float,
    ) -> List[Tuple[int, int]]:
        if not self.samples:
            return []

        newest_elapsed_time_seconds = self.samples[-1][0]
        oldest_visible_time_seconds = max(0, newest_elapsed_time_seconds - self.time_window_seconds)
        visible_time_span_seconds = max(1, newest_elapsed_time_seconds - oldest_visible_time_seconds)
        plot_width = right - left
        plot_height = bottom - top
        line_points = []

        for sample_index, sample in enumerate(self.samples):
            elapsed_time_seconds = sample[0]
            value = values[sample_index]
            time_ratio = (elapsed_time_seconds - oldest_visible_time_seconds) / visible_time_span_seconds
            value_ratio = min(max(value / value_limit, 0), 1)
            point_x = left + round(plot_width * time_ratio)
            point_y = bottom - round(plot_height * value_ratio)
            line_points.append((point_x, point_y))

        return line_points

    def _draw_time_axis(self, plot_image: np.ndarray, left: int, right: int, axis_y: int) -> None:
        if not self.samples:
            return

        newest_elapsed_time_seconds = self.samples[-1][0]
        oldest_visible_time_seconds = max(0, newest_elapsed_time_seconds - self.time_window_seconds)
        axis_color = (70, 70, 70)

        cv2.putText(
            plot_image,
            f"{oldest_visible_time_seconds:.1f}s",
            (left, axis_y + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            axis_color,
            1,
        )
        cv2.putText(
            plot_image,
            f"{newest_elapsed_time_seconds:.1f}s",
            (right - 55, axis_y + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            axis_color,
            1,
        )
        cv2.putText(
            plot_image,
            "time",
            ((left + right) // 2 - 18, axis_y + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            axis_color,
            1,
        )

    @staticmethod
    def display_delay_ms_to_fps(delay_ms: int) -> float:
        if delay_ms <= 0:
            return 0

        return 1000 / delay_ms
