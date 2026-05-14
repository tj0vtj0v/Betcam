import sys
import threading
import time
import ctypes
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import cv2

from modules.DetectionAnnotator import DetectionAnnotator
from modules.DebugPlotter import DebugPlotter
from modules.ImageUrlCapture import ImageUrlCapture
from modules.ObjectDetector import TrackedObjectDetector
from modules.Snapshotter import Snapshotter
from modules.WebcamBuffer import WebcamBuffer
from config.config import MAX_BUFFER_SIZE, MIN_DISPLAY_DELAY_MS
from config.stream_config import load_webcam_streams
from config.types import RawFrame, StreamFrame


class WebcamStream:
    _highgui_lock = threading.Lock()

    def __init__(
        self,
        url: str,
        *,
        window_name: Optional[str] = None,
        max_buffer_size: int = MAX_BUFFER_SIZE,
        detector: Optional[TrackedObjectDetector] = None,
        annotator: Optional[DetectionAnnotator] = None,
        debug_plotter: Optional[DebugPlotter] = None,
        snapshotter: Optional[Snapshotter] = None,
        town: Optional[str] = None,
        location: Optional[str] = None,
    ) -> None:
        self.url = url
        self.window_name = window_name or url
        self.buffer = WebcamBuffer(max_buffer_size)
        self.detector = detector
        self.annotator = annotator
        self.debug_plotter = debug_plotter
        self.snapshotter = snapshotter
        self.town = town
        self.location = location
        self.stop_event = threading.Event()
        self.capture: Optional[Any] = None
        self.reader_thread: Optional[threading.Thread] = None
        self.display_thread: Optional[threading.Thread] = None
        self.last_error: Optional[BaseException] = None
        self.display_started_at: Optional[float] = None
        self.latest_stream_frame: Optional[StreamFrame] = None
        self.latest_stream_frame_lock = threading.Lock()
        self.last_displayed_at: Optional[float] = None
        self.display_fps_ema: Optional[float] = None

    @classmethod
    def from_storage(
        cls,
        town: str,
        location: str,
        resolution: str,
        *,
        window_name: Optional[str] = None,
        max_buffer_size: int = MAX_BUFFER_SIZE,
        detector: Optional[TrackedObjectDetector] = None,
        annotator: Optional[DetectionAnnotator] = None,
        debug_plotter: Optional[DebugPlotter] = None,
        snapshotter: Optional[Snapshotter] = None,
    ) -> "WebcamStream":
        stream_url = load_webcam_streams()[town][location][resolution]
        generic_window_name = f"{town} - {location} ({resolution})"
        return cls(
            stream_url,
            window_name=window_name or generic_window_name,
            max_buffer_size=max_buffer_size,
            detector=detector,
            annotator=annotator,
            debug_plotter=debug_plotter,
            snapshotter=snapshotter,
            town=town,
            location=location,
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
        self.last_displayed_at = None
        self.display_fps_ema = None
        if self.detector is not None:
            self.detector.reset()

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
        self._clear_latest_stream_frame()
        self._close_debug_plot()
        self.last_displayed_at = None
        self.display_fps_ema = None

    def read(self) -> Optional[StreamFrame]:
        if not self.is_open:
            self.open()

        return self.buffer.read()

    def snapshot(self, *, annotated: bool = True) -> Optional[Path]:
        if self.snapshotter is None:
            raise RuntimeError("Snapshots are disabled for this stream.")

        stream_frame = self._latest_stream_frame()
        if stream_frame is None:
            raise RuntimeError("No frame available for snapshot yet.")

        return self.snapshotter.save(
            stream_frame,
            town=self.town,
            location=self.location,
            annotated=annotated,
        )

    def show(self) -> None:
        window_created = False
        window_sized = False

        if not self.is_open:
            self.open()

        try:
            self.display_started_at = time.monotonic()
            self._ensure_debug_plot()
            with self._highgui_lock:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            window_created = True

            while not self.stop_event.is_set():
                current_frame, wait_time_ms, fps = self._next_display_frame()

                with self._highgui_lock:
                    if current_frame is not None:
                        self._draw_fps_overlay(current_frame, fps)
                        if not window_sized:
                            self._size_window_for_frame(self.window_name, current_frame)
                            window_sized = True
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
        cls._isolate_stream_contexts(active_streams)
        end_time = time.monotonic() + duration_seconds if duration_seconds is not None else None
        created_windows: List["WebcamStream"] = []
        sized_windows = set()

        try:
            for stream in active_streams:
                if not stream.is_open:
                    stream.open()
                stream.display_started_at = time.monotonic()
                stream._ensure_debug_plot()
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

                    current_frame, wait_time_ms, fps = stream._next_display_frame()
                    wait_times_ms.append(wait_time_ms)

                    if current_frame is not None:
                        cls._draw_fps_overlay(current_frame, fps)
                        if stream.window_name not in sized_windows:
                            cls._size_window_for_frame(stream.window_name, current_frame)
                            sized_windows.add(stream.window_name)
                        cv2.imshow(stream.window_name, current_frame)

                key = cv2.waitKey(min(wait_times_ms, default=1)) & 0xFF
                if key == ord("q"):
                    for stream in active_streams:
                        stream.stop_event.set()
                    break
                if end_time is not None and time.monotonic() >= end_time:
                    for stream in active_streams:
                        stream.stop_event.set()
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

    @staticmethod
    def _isolate_stream_contexts(streams: Iterable["WebcamStream"]) -> None:
        detector_ids = set()
        annotator_ids = set()

        for stream in streams:
            if stream.detector is not None:
                detector_id = id(stream.detector)
                if detector_id in detector_ids:
                    stream.detector = stream.detector.clone()
                else:
                    detector_ids.add(detector_id)

            if stream.annotator is not None:
                annotator_id = id(stream.annotator)
                if annotator_id in annotator_ids:
                    stream.annotator = stream.annotator.clone()
                else:
                    annotator_ids.add(annotator_id)

    def _next_display_frame(self) -> Tuple[Optional[RawFrame], int, float]:
        iteration_start_time = time.monotonic()
        current_stream_frame, current_queue_size = self.buffer.pop()
        if current_stream_frame is None:
            target_delay_ms = max(1, MIN_DISPLAY_DELAY_MS // 2)
            average_queue_size = 0.0
            fps = self.display_fps_ema or 0.0
            self._update_debug_plot(
                queue_size=current_queue_size,
                average_queue_size=average_queue_size,
                delay_ms=target_delay_ms,
            )
            return None, target_delay_ms, fps

        target_delay_ms, average_queue_size = self.buffer.calculate_display_delay_stats(current_queue_size)
        iteration_runtime_ms = (time.monotonic() - iteration_start_time) * 1000
        wait_time_ms = max(1, target_delay_ms - round(iteration_runtime_ms))
        fps = self._record_display_timing()
        current_frame = current_stream_frame.annotated_frame
        self._update_debug_plot(
            queue_size=current_queue_size,
            average_queue_size=average_queue_size,
            delay_ms=wait_time_ms,
        )

        return current_frame, wait_time_ms, fps

    def _record_display_timing(self) -> float:
        current_time = time.monotonic()
        if self.last_displayed_at is None:
            self.last_displayed_at = current_time
            self.display_fps_ema = 0.0
            return 0.0

        frame_interval_seconds = current_time - self.last_displayed_at
        self.last_displayed_at = current_time
        if frame_interval_seconds <= 0:
            return self.display_fps_ema or 0.0

        instantaneous_fps = 1.0 / frame_interval_seconds
        if self.display_fps_ema is None or self.display_fps_ema <= 0:
            self.display_fps_ema = instantaneous_fps
        else:
            self.display_fps_ema = (0.2 * instantaneous_fps) + (0.8 * self.display_fps_ema)

        return self.display_fps_ema

    def _read_frames_into_buffer(self) -> None:
        if self.capture is None:
            return

        while not self.stop_event.is_set():
            frame = self.read_frame(self.capture)
            if frame is None:
                time.sleep(0.05)
                continue

            stream_frame = self._build_stream_frame(frame)
            self._store_latest_stream_frame(stream_frame)
            self._snapshot_current_frame(stream_frame)
            self.buffer.append(stream_frame)

    def _build_stream_frame(self, frame: RawFrame) -> StreamFrame:
        timestamp = time.monotonic()
        detections = ()
        active_trails = ()
        annotated_frame = frame

        if self.detector is not None:
            detections = self.detector.detect(frame, timestamp=timestamp)
            active_trails = self.detector.active_trails(timestamp=timestamp)
            if self.annotator is not None:
                annotated_frame = self.annotator.annotate(frame, detections, active_trails)

        return StreamFrame(
            raw_frame=frame,
            annotated_frame=annotated_frame,
            detections=detections,
            active_trails=active_trails,
            timestamp=timestamp,
        )

    def _ensure_debug_plot(self) -> None:
        if self.debug_plotter is None:
            return

        self.debug_plotter.open(
            window_name=f"{self.window_name} Debug",
            maximum_buffer_size=self.buffer.frames.maxlen or MAX_BUFFER_SIZE,
        )

    def _update_debug_plot(self, *, queue_size: int, average_queue_size: float, delay_ms: int) -> None:
        if self.debug_plotter is None or self.display_started_at is None:
            return

        elapsed_time_seconds = time.monotonic() - self.display_started_at
        self.debug_plotter.update(
            elapsed_time_seconds=elapsed_time_seconds,
            queue_size=queue_size,
            average_queue_size=average_queue_size,
            delay_ms=delay_ms,
        )

    def _close_debug_plot(self) -> None:
        if self.debug_plotter is None:
            return

        self.debug_plotter.close()
        self.display_started_at = None

    def _store_latest_stream_frame(self, stream_frame: StreamFrame) -> None:
        with self.latest_stream_frame_lock:
            self.latest_stream_frame = stream_frame

    def _latest_stream_frame(self) -> Optional[StreamFrame]:
        with self.latest_stream_frame_lock:
            return self.latest_stream_frame

    def _clear_latest_stream_frame(self) -> None:
        with self.latest_stream_frame_lock:
            self.latest_stream_frame = None

    def _snapshot_current_frame(self, stream_frame: StreamFrame) -> None:
        if self.snapshotter is None:
            return

        try:
            self.snapshotter.save(
                stream_frame,
                town=self.town,
                location=self.location,
                annotated=False,
            )
        except Exception as error:
            self.last_error = error
            print(f"Could not save snapshot for webcam stream '{self.window_name}': {error}", file=sys.stderr)

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
    def read_frame(capture: Any) -> Optional[RawFrame]:
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

    @classmethod
    def _size_window_for_frame(cls, window_name: str, frame: RawFrame) -> None:
        frame_height, frame_width = frame.shape[:2]
        screen_size = cls._screen_size()

        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        if screen_size is None:
            cv2.resizeWindow(window_name, frame_width, frame_height)
            return

        screen_width, screen_height = screen_size
        if frame_width >= screen_width or frame_height >= screen_height:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            return

        cv2.resizeWindow(window_name, frame_width, frame_height)

    @staticmethod
    def _screen_size() -> Optional[Tuple[int, int]]:
        try:
            user32 = ctypes.windll.user32
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        except (AttributeError, OSError):
            return None

    @staticmethod
    def _draw_fps_overlay(frame: RawFrame, fps: float) -> None:
        label = f"{fps:.1f} FPS"
        _, frame_width = frame.shape[:2]
        origin_x = round(frame_width * 0.6)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        origin_y = 2 + text_size[1]

        cv2.putText(frame, label, (origin_x, origin_y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, label, (origin_x, origin_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
