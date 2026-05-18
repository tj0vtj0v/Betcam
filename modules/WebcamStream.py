import sys
import threading
import time
import ctypes
import multiprocessing
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2

from modules.DetectionAnnotator import DetectionAnnotator
from modules.DebugPlotter import DebugPlotter
from modules.ImageFilter import DifferenceFilter, StreamFilter
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
        filter: Optional[StreamFilter] = None,
        debug_plotter: Optional[DebugPlotter] = None,
        snapshotter: Optional[Snapshotter] = None,
        town: Optional[str] = None,
        location: Optional[str] = None,
    ) -> None:
        self.url = url
        self.buffer = WebcamBuffer(max_buffer_size)
        self.detector = detector
        self.annotator = annotator
        self.filter = filter
        self.base_window_name = window_name or url
        self.window_name = self._format_window_name(self.base_window_name)
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
        filter: Optional[StreamFilter] = None,
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
            filter=filter,
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
        if self.filter is not None:
            self.filter.reset()

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
        auxiliary_window_created = False
        window_sized = False
        auxiliary_window_sized = False

        if not self.is_open:
            self.open()

        try:
            self.display_started_at = time.monotonic()
            self._ensure_debug_plot()
            with self._highgui_lock:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            window_created = True

            while not self.stop_event.is_set():
                current_stream_frame, wait_time_ms, fps = self._next_display_frame()

                with self._highgui_lock:
                    if current_stream_frame is not None:
                        current_frame = current_stream_frame.annotated_frame
                        self._draw_overlay(current_frame, fps)
                        if not window_sized:
                            self._size_window_for_frame(self.window_name, current_frame)
                            window_sized = True
                        cv2.imshow(self.window_name, current_frame)

                        auxiliary_frame = current_stream_frame.auxiliary_frame
                        if auxiliary_frame is not None:
                            auxiliary_window_name = self._auxiliary_window_name()
                            if not auxiliary_window_created:
                                cv2.namedWindow(auxiliary_window_name, cv2.WINDOW_NORMAL)
                                auxiliary_window_created = True
                            if not auxiliary_window_sized:
                                self._size_window_for_frame(auxiliary_window_name, auxiliary_frame)
                                auxiliary_window_sized = True
                            cv2.imshow(auxiliary_window_name, auxiliary_frame)

                    key = cv2.waitKey(wait_time_ms) & 0xFF

                if key == ord("q"):
                    self.stop_event.set()
        finally:
            self.close()
            if window_created:
                with self._highgui_lock:
                    try:
                        cv2.destroyWindow(self.window_name)
                    except cv2.error:
                        pass
                    if auxiliary_window_created:
                        try:
                            cv2.destroyWindow(self._auxiliary_window_name())
                        except cv2.error:
                            pass
            if threading.current_thread() is self.display_thread:
                self.display_thread = None

    @classmethod
    def show_many(
        cls,
        streams: Iterable["WebcamStream"],
        *,
        duration_seconds: Optional[float] = None,
    ) -> None:
        stream_specs = [stream._to_process_spec() for stream in streams]
        if not stream_specs:
            return

        process_context = multiprocessing.get_context("spawn")
        child_processes: List[multiprocessing.Process] = []

        try:
            for stream_spec in stream_specs:
                child_process = process_context.Process(
                    target=_run_isolated_stream_process,
                    args=(stream_spec, duration_seconds),
                    daemon=False,
                )
                child_process.start()
                child_processes.append(child_process)

            while child_processes:
                active_processes = []

                for child_process in child_processes:
                    child_process.join(timeout=0.1)
                    if child_process.is_alive():
                        active_processes.append(child_process)

                child_processes = active_processes
        except KeyboardInterrupt:
            pass
        finally:
            for child_process in child_processes:
                if child_process.is_alive():
                    child_process.terminate()
            for child_process in child_processes:
                child_process.join(timeout=2)

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

    def _to_process_spec(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "window_name": self.base_window_name,
            "max_buffer_size": self.buffer.frames.maxlen or MAX_BUFFER_SIZE,
            "town": self.town,
            "location": self.location,
            "detector": self._detector_process_spec(),
            "annotator": self._annotator_process_spec(),
            "filter": self._filter_process_spec(),
            "debug_plotter": self._debug_plotter_process_spec(),
            "snapshotter": self._snapshotter_process_spec(),
        }

    def _detector_process_spec(self) -> Optional[Dict[str, Any]]:
        if self.detector is None:
            return None

        return {
            "model_path": self.detector.model_path,
            "inference_device": self.detector.inference_device,
            "model_input_size": self.detector.model_input_size,
            "detection_confidence_threshold": self.detector.detection_confidence_threshold,
            "tracker_config_path": self.detector.tracker_config_path,
            "detection_start_delay_seconds": self.detector.detection_start_delay_seconds,
            "track_history_timeout_seconds": self.detector.track_history_timeout_seconds,
            "max_track_history_points": self.detector.max_track_history_points,
            "max_inference_frame_dimension": self.detector.max_inference_frame_dimension,
            "fixed_inference_frame_budget_ms": self.detector.fixed_inference_frame_budget_ms,
            "max_frames_to_skip_after_inference": self.detector.max_frames_to_skip_after_inference,
            "inference_time_ema_alpha": self.detector.inference_time_ema_alpha,
            "same_class_iou_suppression_threshold": self.detector.same_class_iou_suppression_threshold,
            "allowed_class_names": tuple(self.detector.allowed_class_names),
        }

    def _annotator_process_spec(self) -> Optional[Dict[str, Any]]:
        if self.annotator is None:
            return None

        return {
            "show_labels": self.annotator.show_labels,
            "label_font_scale": self.annotator.label_font_scale,
            "trail_thickness": self.annotator.trail_thickness,
            "trail_point_fade_frames": self.annotator.trail_point_fade_frames,
        }

    def _filter_process_spec(self) -> Optional[Dict[str, Any]]:
        if self.filter is None:
            return None

        if isinstance(self.filter, DifferenceFilter):
            return {
                "type": "difference",
                "use_for_inference": self.filter.use_for_inference,
                "nth_last_image": self.filter.nth_last_image,
                "grayscale_output": self.filter.grayscale_output,
            }

        raise TypeError(f"Unsupported filter type: {type(self.filter).__name__}")

    def _debug_plotter_process_spec(self) -> Optional[Dict[str, Any]]:
        if self.debug_plotter is None:
            return None

        return {
            "maximum_display_delay_ms": self.debug_plotter.maximum_display_delay_ms,
            "queue_sample_window_seconds": self.debug_plotter.queue_sample_window_seconds,
        }

    def _snapshotter_process_spec(self) -> Optional[Dict[str, Any]]:
        if self.snapshotter is None:
            return None

        return {
            "base_directory": str(self.snapshotter.base_directory),
            "save_every_n_frames": self.snapshotter.save_every_n_frames,
            "scale_factor": self.snapshotter.scale_factor,
        }

    @staticmethod
    def _isolate_stream_contexts(streams: Iterable["WebcamStream"]) -> None:
        detector_ids = set()
        annotator_ids = set()
        filter_ids = set()

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

            if stream.filter is not None:
                filter_id = id(stream.filter)
                if filter_id in filter_ids:
                    stream.filter = stream.filter.clone()
                else:
                    filter_ids.add(filter_id)

    def _next_display_frame(self) -> Tuple[Optional[StreamFrame], int, float]:
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

        return current_stream_frame, wait_time_ms, fps

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
        filtered_frame = None
        detector_frame = frame
        annotated_frame = frame
        auxiliary_frame = None

        if self.filter is not None:
            filtered_frame = self.filter.apply(frame)
            if self.filter.uses_for_inference:
                detector_frame = filtered_frame
                annotated_frame = filtered_frame
            elif self.filter.shows_auxiliary_window:
                auxiliary_frame = filtered_frame

        if self.detector is not None:
            detections = self.detector.detect(detector_frame, timestamp=timestamp)
            active_trails = self.detector.active_trails(timestamp=timestamp)
            if self.annotator is not None:
                annotated_frame = self.annotator.annotate(annotated_frame, detections, active_trails)

        return StreamFrame(
            raw_frame=frame,
            filtered_frame=filtered_frame,
            annotated_frame=annotated_frame,
            auxiliary_frame=auxiliary_frame,
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

    def _draw_overlay(self, frame: RawFrame, fps: float) -> None:
        fps_label = f"{fps:.1f} FPS"
        model_label = None
        if self.detector is not None:
            model_label = Path(self.detector.model_path).name

        _, frame_width = frame.shape[:2]
        origin_x = round(frame_width * 0.6)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fps_font_scale = 0.5
        fps_thickness = 1
        fps_text_size, _ = cv2.getTextSize(fps_label, font, fps_font_scale, fps_thickness)
        fps_origin_y = 2 + fps_text_size[1]

        cv2.putText(
            frame,
            fps_label,
            (origin_x, fps_origin_y),
            font,
            fps_font_scale,
            (0, 0, 0),
            fps_thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            fps_label,
            (origin_x, fps_origin_y),
            font,
            fps_font_scale,
            (255, 255, 255),
            fps_thickness,
            cv2.LINE_AA,
        )

        if model_label is None:
            return

        model_font_scale = 0.5
        model_thickness = 1
        model_origin_y = fps_origin_y + 8 + cv2.getTextSize(model_label, font, model_font_scale, model_thickness)[0][1]
        cv2.putText(
            frame,
            model_label,
            (origin_x, model_origin_y),
            font,
            model_font_scale,
            (0, 0, 0),
            model_thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            model_label,
            (origin_x, model_origin_y),
            font,
            model_font_scale,
            (255, 255, 255),
            model_thickness,
            cv2.LINE_AA,
        )

    def _auxiliary_window_name(self) -> str:
        return f"{self.window_name} Filter"

    def _format_window_name(self, base_window_name: str) -> str:
        if self.filter is None:
            return base_window_name

        return f"{base_window_name} {self.filter.display_name()}"


def _run_isolated_stream_process(stream_spec: Dict[str, Any], duration_seconds: Optional[float]) -> None:
    detector = _build_detector_from_process_spec(stream_spec.get("detector"))
    annotator = _build_annotator_from_process_spec(stream_spec.get("annotator"))
    image_filter = _build_filter_from_process_spec(stream_spec.get("filter"))
    debug_plotter = _build_debug_plotter_from_process_spec(stream_spec.get("debug_plotter"))
    snapshotter = _build_snapshotter_from_process_spec(stream_spec.get("snapshotter"))
    stream = WebcamStream(
        stream_spec["url"],
        window_name=stream_spec.get("window_name"),
        max_buffer_size=stream_spec.get("max_buffer_size", MAX_BUFFER_SIZE),
        detector=detector,
        annotator=annotator,
        filter=image_filter,
        debug_plotter=debug_plotter,
        snapshotter=snapshotter,
        town=stream_spec.get("town"),
        location=stream_spec.get("location"),
    )
    stop_timer: Optional[threading.Timer] = None

    try:
        if duration_seconds is not None:
            stop_timer = threading.Timer(duration_seconds, stream.stop_event.set)
            stop_timer.daemon = True
            stop_timer.start()
        stream.show()
    finally:
        if stop_timer is not None:
            stop_timer.cancel()


def _build_detector_from_process_spec(detector_spec: Optional[Dict[str, Any]]) -> Optional[TrackedObjectDetector]:
    if detector_spec is None:
        return None

    return TrackedObjectDetector(**detector_spec)


def _build_annotator_from_process_spec(annotator_spec: Optional[Dict[str, Any]]) -> Optional[DetectionAnnotator]:
    if annotator_spec is None:
        return None

    return DetectionAnnotator(**annotator_spec)


def _build_debug_plotter_from_process_spec(debug_plotter_spec: Optional[Dict[str, Any]]) -> Optional[DebugPlotter]:
    if debug_plotter_spec is None:
        return None

    return DebugPlotter(**debug_plotter_spec)


def _build_filter_from_process_spec(filter_spec: Optional[Dict[str, Any]]) -> Optional[StreamFilter]:
    if filter_spec is None:
        return None

    filter_type = filter_spec.get("type")
    filter_parameters = {key: value for key, value in filter_spec.items() if key != "type"}
    if filter_type == "difference":
        return DifferenceFilter(**filter_parameters)

    raise ValueError(f"Unsupported filter process spec type: {filter_type}")


def _build_snapshotter_from_process_spec(snapshotter_spec: Optional[Dict[str, Any]]) -> Optional[Snapshotter]:
    if snapshotter_spec is None:
        return None

    return Snapshotter(**snapshotter_spec)
