from __future__ import annotations

import ctypes
import math
import multiprocessing
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2

from config.config import MAX_BUFFER_SIZE, MAX_STREAM_FPS, MIN_DISPLAY_DELAY_MS
from config.stream_config import load_webcam_streams
from config.types import RawFrame
from modules.DebugPlotter import DebugPlotter
from modules.FrameProcessor import FrameProcessor, processor_specs, reconstruct_processors
from modules.ImageUrlCapture import ImageUrlCapture
from modules.WebcamBuffer import WebcamBuffer


@dataclass(frozen=True)
class _DisplayOutput:
    pipeline_index: int
    display_name: str
    image: RawFrame


@dataclass(frozen=True)
class _FramePacket:
    raw_image: RawFrame
    final_image: RawFrame
    display_outputs: Tuple[_DisplayOutput, ...]


class WebcamStream:
    _highgui_lock = threading.Lock()

    def __init__(
        self,
        url: str,
        *,
        window_name: Optional[str] = None,
        max_buffer_size: int = MAX_BUFFER_SIZE,
        debug_plotter: Optional[DebugPlotter] = None,
        town: Optional[str] = None,
        location: Optional[str] = None,
    ) -> None:
        self.url = url
        self.buffer = WebcamBuffer(max_buffer_size)
        self.base_window_name = window_name or url
        self.window_name = self.base_window_name
        self.debug_plotter = debug_plotter
        self.town = town
        self.location = location
        self.processors: List[FrameProcessor] = []
        self.stop_event = threading.Event()
        self.capture: Optional[Any] = None
        self.reader_thread: Optional[threading.Thread] = None
        self.display_thread: Optional[threading.Thread] = None
        self.last_error: Optional[BaseException] = None
        self.display_started_at: Optional[float] = None
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
        debug_plotter: Optional[DebugPlotter] = None,
    ) -> "WebcamStream":
        url = load_webcam_streams()[town][location][resolution]
        return cls(
            url,
            window_name=window_name or f"{town} - {location} ({resolution})",
            max_buffer_size=max_buffer_size,
            debug_plotter=debug_plotter,
            town=town,
            location=location,
        )

    def append_processor(self, *processors: FrameProcessor) -> "WebcamStream":
        if self.is_open:
            raise RuntimeError("Processors cannot be appended while the stream is open.")
        for processor in processors:
            if not isinstance(processor, FrameProcessor):
                raise TypeError("append_processor accepts only FrameProcessor instances.")
        self.processors.extend(processors)
        return self

    def __enter__(self) -> "WebcamStream":
        return self.open()

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
        for processor in self.processors:
            processor.reset()
        self.reader_thread = threading.Thread(target=self._read_frames_into_buffer, daemon=True)
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
        self._close_debug_plot()
        self.last_displayed_at = None
        self.display_fps_ema = None

    def read(self) -> Optional[RawFrame]:
        if not self.is_open:
            self.open()
        packet = self.buffer.read()
        return None if packet is None else packet.final_image

    def _build_frame_packet(self, raw_image: RawFrame) -> _FramePacket:
        current_image = raw_image
        display_outputs = []
        for index, processor in enumerate(self.processors):
            result = processor(current_image)
            if result is None:
                raise TypeError(
                    f"Processor at index {index} ({type(processor).__name__}) returned None."
                )
            if processor.show_result:
                display_outputs.append(_DisplayOutput(index, processor.display_name(), result.copy()))
            if processor.return_result:
                current_image = result
        return _FramePacket(raw_image, current_image, tuple(display_outputs))

    def _read_frames_into_buffer(self) -> None:
        if self.capture is None:
            return
        minimum_frame_interval = 1.0 / MAX_STREAM_FPS
        while not self.stop_event.is_set():
            iteration_started_at = time.monotonic()
            frame = self.read_frame(self.capture)
            if frame is None:
                self.stop_event.wait(0.05)
                continue
            try:
                self.buffer.append(self._build_frame_packet(frame))
            except Exception as error:
                self.last_error = error
                self.stop_event.set()
                print(f"Could not process webcam stream '{self.window_name}': {error}", file=sys.stderr)
                continue
            remaining_interval = minimum_frame_interval - (
                time.monotonic() - iteration_started_at
            )
            if remaining_interval > 0:
                self.stop_event.wait(remaining_interval)

    def show(self) -> None:
        created_windows: List[str] = []
        sized_windows = set()
        if not self.is_open:
            self.open()
        processor_windows = {
            index: self._processor_window_name(index, processor)
            for index, processor in enumerate(self.processors)
            if processor.show_result
        }
        try:
            self.display_started_at = time.monotonic()
            self._ensure_debug_plot()
            with self._highgui_lock:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                created_windows.append(self.window_name)
                for name in processor_windows.values():
                    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                    created_windows.append(name)
            while not self.stop_event.is_set():
                packet, wait_time_ms, fps = self._next_display_frame()
                with self._highgui_lock:
                    if packet is not None:
                        base_image = packet.final_image.copy()
                        self._draw_overlay(base_image, fps)
                        self._show_image(self.window_name, base_image, sized_windows)
                        for output in packet.display_outputs:
                            self._show_image(
                                processor_windows[output.pipeline_index], output.image, sized_windows
                            )
                    key = cv2.waitKey(wait_time_ms) & 0xFF
                if key == ord("q"):
                    self.stop_event.set()
        finally:
            self.close()
            with self._highgui_lock:
                for name in created_windows:
                    try:
                        cv2.destroyWindow(name)
                    except cv2.error:
                        pass
            if threading.current_thread() is self.display_thread:
                self.display_thread = None

    def _show_image(self, name: str, image: RawFrame, sized_windows: set) -> None:
        if name not in sized_windows:
            self._size_window_for_frame(name, image)
            sized_windows.add(name)
        cv2.imshow(name, image)

    def _processor_window_name(self, index: int, processor: FrameProcessor) -> str:
        return f"{self.window_name} [{index + 1}] {processor.display_name()}"

    def show_async(self) -> threading.Thread:
        if self.display_thread is not None and self.display_thread.is_alive():
            return self.display_thread
        self.display_thread = threading.Thread(target=self._show_async, daemon=True)
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

    @classmethod
    def show_many(
        cls, streams: Iterable["WebcamStream"], *, duration_seconds: Optional[float] = None
    ) -> None:
        stream_specs = [stream._to_process_spec() for stream in streams]
        if not stream_specs:
            return
        context = multiprocessing.get_context("spawn")
        children: List[multiprocessing.Process] = []
        try:
            for spec in stream_specs:
                child = context.Process(
                    target=_run_isolated_stream_process,
                    args=(spec, duration_seconds),
                    daemon=False,
                )
                child.start()
                children.append(child)
            while children:
                active = []
                for child in children:
                    child.join(timeout=0.1)
                    if child.is_alive():
                        active.append(child)
                children = active
        except KeyboardInterrupt:
            pass
        finally:
            for child in children:
                if child.is_alive():
                    child.terminate()
            for child in children:
                child.join(timeout=2)

    def _to_process_spec(self) -> Dict[str, Any]:
        debug_spec = None
        if self.debug_plotter is not None:
            debug_spec = {
                "maximum_display_delay_ms": self.debug_plotter.maximum_display_delay_ms,
                "queue_sample_window_seconds": self.debug_plotter.queue_sample_window_seconds,
            }
        return {
            "url": self.url,
            "window_name": self.base_window_name,
            "max_buffer_size": self.buffer.frames.maxlen or MAX_BUFFER_SIZE,
            "town": self.town,
            "location": self.location,
            "debug_plotter": debug_spec,
            "processors": processor_specs(self.processors),
        }

    def _next_display_frame(self) -> Tuple[Optional[_FramePacket], int, float]:
        started = time.monotonic()
        packet, queue_size = self.buffer.pop()
        if packet is None:
            delay = max(1, MIN_DISPLAY_DELAY_MS // 2)
            self._update_debug_plot(queue_size=queue_size, average_queue_size=0.0, delay_ms=delay)
            return None, delay, self.display_fps_ema or 0.0
        delay, average = self.buffer.calculate_display_delay_stats(queue_size)
        minimum_frame_delay_ms = math.ceil(1000 / MAX_STREAM_FPS)
        wait = max(
            minimum_frame_delay_ms,
            delay - round((time.monotonic() - started) * 1000),
        )
        fps = self._record_display_timing()
        self._update_debug_plot(queue_size=queue_size, average_queue_size=average, delay_ms=wait)
        return packet, wait, fps

    def _record_display_timing(self) -> float:
        now = time.monotonic()
        if self.last_displayed_at is None:
            self.last_displayed_at = now
            self.display_fps_ema = 0.0
            return 0.0
        interval = now - self.last_displayed_at
        self.last_displayed_at = now
        if interval <= 0:
            return self.display_fps_ema or 0.0
        instantaneous = 1.0 / interval
        self.display_fps_ema = (
            instantaneous
            if not self.display_fps_ema
            else 0.2 * instantaneous + 0.8 * self.display_fps_ema
        )
        return self.display_fps_ema

    def _ensure_debug_plot(self) -> None:
        if self.debug_plotter is not None:
            self.debug_plotter.open(
                window_name=f"{self.window_name} Debug",
                maximum_buffer_size=self.buffer.frames.maxlen or MAX_BUFFER_SIZE,
            )

    def _update_debug_plot(self, *, queue_size: int, average_queue_size: float, delay_ms: int) -> None:
        if self.debug_plotter is not None and self.display_started_at is not None:
            self.debug_plotter.update(
                elapsed_time_seconds=time.monotonic() - self.display_started_at,
                queue_size=queue_size,
                average_queue_size=average_queue_size,
                delay_ms=delay_ms,
            )

    def _close_debug_plot(self) -> None:
        if self.debug_plotter is not None:
            self.debug_plotter.close()
        self.display_started_at = None

    @staticmethod
    def is_image_url(url: str) -> bool:
        return url.split("?", 1)[0].lower().endswith((".jpg", ".jpeg", ".png", ".webp"))

    @staticmethod
    def open_stream(url: str) -> Any:
        if WebcamStream.is_image_url(url):
            capture = ImageUrlCapture(url)
        else:
            capture = cv2.VideoCapture(url)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open stream: {url}")
        return capture

    @staticmethod
    def read_frame(capture: Any) -> Optional[RawFrame]:
        available, frame = capture.read()
        return frame if available else None

    @classmethod
    def _size_window_for_frame(cls, window_name: str, frame: RawFrame) -> None:
        height, width = frame.shape[:2]
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        screen = cls._screen_size()
        if screen is not None and (width >= screen[0] or height >= screen[1]):
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(window_name, width, height)

    @staticmethod
    def _screen_size() -> Optional[Tuple[int, int]]:
        try:
            user32 = ctypes.windll.user32
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        except (AttributeError, OSError):
            return None

    @staticmethod
    def _draw_overlay(frame: RawFrame, fps: float) -> None:
        label = f"{fps:.1f} FPS"
        origin = (round(frame.shape[1] * 0.6), 16)
        cv2.putText(frame, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _run_isolated_stream_process(spec: Dict[str, Any], duration_seconds: Optional[float]) -> None:
    debug = DebugPlotter(**spec["debug_plotter"]) if spec.get("debug_plotter") else None
    processors = reconstruct_processors(spec.get("processors", ()))
    stream = WebcamStream(
        spec["url"],
        window_name=spec.get("window_name"),
        max_buffer_size=spec.get("max_buffer_size", MAX_BUFFER_SIZE),
        debug_plotter=debug,
        town=spec.get("town"),
        location=spec.get("location"),
    ).append_processor(*processors)
    timer: Optional[threading.Timer] = None
    try:
        if duration_seconds is not None:
            timer = threading.Timer(duration_seconds, stream.stop_event.set)
            timer.daemon = True
            timer.start()
        stream.show()
    finally:
        if timer is not None:
            timer.cancel()
