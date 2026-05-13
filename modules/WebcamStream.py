import sys
import threading
import time
from typing import Any, Iterable, List, Optional, Tuple

import cv2

from modules.ImageUrlCapture import ImageUrlCapture
from modules.WebcamBuffer import WebcamBuffer
from config.config import MAX_BUFFER_SIZE
from config.stream_config import load_webcam_streams
from config.types import Frame


class WebcamStream:
    _highgui_lock = threading.Lock()

    def __init__(
        self,
        url: str,
        *,
        window_name: Optional[str] = None,
        max_buffer_size: int = MAX_BUFFER_SIZE,
    ) -> None:
        self.url = url
        self.window_name = window_name or url
        self.buffer = WebcamBuffer(max_buffer_size)
        self.stop_event = threading.Event()
        self.capture: Optional[Any] = None
        self.reader_thread: Optional[threading.Thread] = None
        self.display_thread: Optional[threading.Thread] = None
        self.last_error: Optional[BaseException] = None

    @classmethod
    def from_storage(
        cls,
        town: str,
        location: str,
        resolution: str,
        *,
        window_name: Optional[str] = None,
        max_buffer_size: int = MAX_BUFFER_SIZE,
    ) -> "WebcamStream":
        stream_url = load_webcam_streams()[town][location][resolution]
        generic_window_name = f"{town} - {location} ({resolution})"
        return cls(
            stream_url,
            window_name=window_name or generic_window_name,
            max_buffer_size=max_buffer_size,
        )

    def __enter__(self) -> "WebcamStream":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self.capture is not None and self.capture.isOpened()

    def open(self, url: Optional[str] = None) -> "WebcamStream":
        if url is not None:
            self.url = url

        self.close()
        self.capture = self.open_stream(self.url)
        self.stop_event.clear()
        self.buffer.clear()

        self.reader_thread = threading.Thread(
            target=self._read_frames_into_buffer,
            daemon=True,
        )
        self.reader_thread.start()
        return self

    def close(self) -> None:
        self.stop_event.set()

        if self.reader_thread is not None:
            self.reader_thread.join(timeout=2)
            self.reader_thread = None

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        self.buffer.clear()

    def read(self) -> Optional[Frame]:
        if not self.is_open:
            self.open()

        return self.buffer.read()

    def show(self) -> None:
        window_created = False

        if not self.is_open:
            self.open()

        try:
            with self._highgui_lock:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            window_created = True

            while not self.stop_event.is_set():
                current_frame, wait_time_ms = self._next_display_frame()

                with self._highgui_lock:
                    if current_frame is not None:
                        cv2.imshow(self.window_name, current_frame)

                    key = cv2.waitKey(wait_time_ms) & 0xFF

                if key == ord("q"):
                    self.stop_event.set()
        finally:
            self.close()
            if window_created:
                with self._highgui_lock:
                    cv2.destroyWindow(self.window_name)
            if threading.current_thread() is self.display_thread:
                self.display_thread = None

    @classmethod
    def show_many(
        cls,
        streams: Iterable["WebcamStream"],
        *,
        duration_seconds: Optional[float] = None,
    ) -> None:
        active_streams = list(streams)
        created_windows: List["WebcamStream"] = []
        end_time = time.monotonic() + duration_seconds if duration_seconds is not None else None

        try:
            for stream in active_streams:
                if not stream.is_open:
                    stream.open()

                cv2.namedWindow(stream.window_name, cv2.WINDOW_NORMAL)
                created_windows.append(stream)

            while active_streams and not any(stream.stop_event.is_set() for stream in active_streams):
                wait_times_ms: List[int] = []

                for stream in active_streams[:]:
                    if cls._window_was_closed(stream.window_name):
                        stream.stop_event.set()
                        active_streams.remove(stream)
                        stream.close()
                        continue

                    current_frame, wait_time_ms = stream._next_display_frame()
                    wait_times_ms.append(wait_time_ms)

                    if current_frame is not None:
                        cv2.imshow(stream.window_name, current_frame)

                key = cv2.waitKey(min(wait_times_ms, default=1)) & 0xFF
                if key == ord("q"):
                    for stream in active_streams:
                        stream.stop_event.set()
                    break

                if end_time is not None and time.monotonic() >= end_time:
                    break
        finally:
            for stream in active_streams:
                stream.close()

            for stream in created_windows:
                try:
                    cv2.destroyWindow(stream.window_name)
                except cv2.error:
                    pass

    def show_async(self) -> threading.Thread:
        if self.display_thread is not None and self.display_thread.is_alive():
            return self.display_thread

        self.display_thread = threading.Thread(
            target=self._show_async,
            daemon=True,
        )
        self.display_thread.start()
        return self.display_thread

    def _show_async(self) -> None:
        try:
            self.show()
        except Exception as error:
            self.last_error = error
            print(f"Could not show webcam stream '{self.window_name}': {error}", file=sys.stderr)
            self.close()
        finally:
            if threading.current_thread() is self.display_thread:
                self.display_thread = None

    def _next_display_frame(self) -> Tuple[Optional[Frame], int]:
        iteration_start_time = time.monotonic()
        current_frame, current_queue_size = self.buffer.pop()
        target_delay_ms = self.buffer.calculate_display_delay(current_queue_size)
        iteration_runtime_ms = (time.monotonic() - iteration_start_time) * 1000
        wait_time_ms = max(1, target_delay_ms - round(iteration_runtime_ms))

        return current_frame, wait_time_ms

    def _read_frames_into_buffer(self) -> None:
        if self.capture is None:
            return

        while not self.stop_event.is_set():
            frame = self.read_frame(self.capture)
            if frame is None:
                time.sleep(0.05)
                continue

            self.buffer.append(frame)

    @staticmethod
    def is_image_url(url: str) -> bool:
        url_without_query = url.split("?", 1)[0].lower()
        return url_without_query.endswith((".jpg", ".jpeg", ".png", ".webp"))

    @staticmethod
    def open_stream(url: str) -> Any:
        if WebcamStream.is_image_url(url):
            image_capture = ImageUrlCapture(url)
            if not image_capture.isOpened():
                raise RuntimeError(f"Could not open image webcam: {url}")

            return image_capture

        video_capture = cv2.VideoCapture(url)
        if not video_capture.isOpened():
            raise RuntimeError(f"Could not open stream: {url}")

        return video_capture

    @staticmethod
    def read_frame(capture: Any) -> Optional[Frame]:
        frame_available, frame = capture.read()
        if not frame_available:
            return None

        return frame

    @staticmethod
    def _window_was_closed(window_name: str) -> bool:
        try:
            return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True
