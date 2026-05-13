from __future__ import annotations

from typing import Optional

import cv2

from config.config import MAX_BUFFER_SIZE, MAX_DISPLAY_DELAY_MS, QUEUE_SAMPLE_WINDOW_SECONDS
from modules.DebugPlot import DebugPlot


class DebugPlotter:
    def __init__(
        self,
        *,
        maximum_display_delay_ms: int = MAX_DISPLAY_DELAY_MS,
        queue_sample_window_seconds: int = QUEUE_SAMPLE_WINDOW_SECONDS,
    ) -> None:
        self.maximum_display_delay_ms = maximum_display_delay_ms
        self.queue_sample_window_seconds = queue_sample_window_seconds
        self._debug_plot: Optional[DebugPlot] = None

    def open(self, *, window_name: str, maximum_buffer_size: int = MAX_BUFFER_SIZE) -> None:
        if self._debug_plot is not None:
            return

        self._debug_plot = DebugPlot(
            maximum_buffer_size=maximum_buffer_size,
            maximum_display_delay_ms=self.maximum_display_delay_ms,
            queue_sample_window_seconds=self.queue_sample_window_seconds,
            window_name=window_name,
        )

    def update(
        self,
        *,
        elapsed_time_seconds: float,
        queue_size: int,
        average_queue_size: float,
        delay_ms: int,
    ) -> None:
        if self._debug_plot is None:
            return

        self._debug_plot.add_sample(
            elapsed_time_seconds=elapsed_time_seconds,
            queue_size=queue_size,
            average_queue_size=average_queue_size,
            delay_ms=delay_ms,
        )
        self._debug_plot.show()

    def close(self) -> None:
        if self._debug_plot is None:
            return

        try:
            self._debug_plot.close()
        except cv2.error:
            pass
        finally:
            self._debug_plot = None
