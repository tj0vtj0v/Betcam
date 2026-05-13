import threading
import time
from collections import deque
from typing import Deque, Optional, Tuple

from WebcamOperations.config.config import (
    MAX_BUFFER_SIZE,
    MAX_DISPLAY_DELAY_MS,
    MIN_DISPLAY_DELAY_MS,
    QUEUE_SAMPLE_WINDOW_SECONDS,
    TARGET_BUFFER_SIZE,
)
from WebcamOperations.config.types import Frame, QueueSample


class WebcamBuffer:
    def __init__(self, max_size: int = MAX_BUFFER_SIZE) -> None:
        self.frames: Deque[Frame] = deque(maxlen=max_size)
        self.queue_size_samples: Deque[QueueSample] = deque()
        self.lock = threading.Lock()

    def clear(self) -> None:
        self.queue_size_samples.clear()

        with self.lock:
            self.frames.clear()

    def append(self, frame: Frame) -> None:
        with self.lock:
            self.frames.append(frame)

    def pop(self) -> Tuple[Optional[Frame], int]:
        with self.lock:
            frame = self.frames.popleft() if self.frames else None
            queue_size = len(self.frames)

        return frame, queue_size

    def read(self) -> Optional[Frame]:
        frame, _ = self.pop()
        return frame

    def calculate_display_delay(self, queue_size: int) -> int:
        current_time = time.monotonic()
        self.queue_size_samples.append((current_time, queue_size))
        self.remove_old_queue_samples(self.queue_size_samples, current_time)

        average_queue_size = self.calculate_average_queue_size(self.queue_size_samples)
        return self.average_queue_size_to_interframe_delay(average_queue_size)

    @staticmethod
    def average_queue_size_to_interframe_delay(average_queue_size: float) -> int:
        capped_queue_size = min(max(average_queue_size, 0), TARGET_BUFFER_SIZE)
        delay_range = MAX_DISPLAY_DELAY_MS - MIN_DISPLAY_DELAY_MS
        buffer_fill_ratio = capped_queue_size / TARGET_BUFFER_SIZE
        display_delay = MAX_DISPLAY_DELAY_MS - (delay_range * buffer_fill_ratio)

        return max(MIN_DISPLAY_DELAY_MS, round(display_delay))

    @staticmethod
    def remove_old_queue_samples(queue_size_samples: Deque[QueueSample], current_time: float) -> None:
        oldest_allowed_sample_time = current_time - QUEUE_SAMPLE_WINDOW_SECONDS
        while queue_size_samples and queue_size_samples[0][0] < oldest_allowed_sample_time:
            queue_size_samples.popleft()

    @staticmethod
    def calculate_average_queue_size(queue_size_samples: Deque[QueueSample]) -> float:
        if not queue_size_samples:
            return 0

        total_queue_size = sum(queue_size for _, queue_size in queue_size_samples)
        return total_queue_size / len(queue_size_samples)
