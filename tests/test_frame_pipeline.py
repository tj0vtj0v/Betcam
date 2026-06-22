import inspect
import pickle
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from config.config import MAX_STREAM_FPS
from config.types import DetectionResult
from modules import DetectionAnnotator, FrameProcessor, TrackedObjectDetector, WebcamStream
from modules.FrameProcessor import reconstruct_processors


class AddProcessor(FrameProcessor):
    def __init__(self, amount=1, *, show_result=False, return_result=True):
        super().__init__(show_result=show_result, return_result=return_result)
        self.amount = amount
        self.calls = 0
        self.resets = 0

    def __call__(self, image):
        self.calls += 1
        result = image.copy()
        result += self.amount
        return result

    def reset(self):
        self.resets += 1

    def construction_settings(self):
        return {"amount": self.amount}


class FakeCapture:
    def __init__(self):
        self.open = True

    def isOpened(self):
        return self.open

    def read(self):
        return False, None

    def release(self):
        self.open = False


class FramePipelineTests(unittest.TestCase):
    def test_order_flags_and_independent_display_copies(self):
        raw = np.zeros((2, 2, 3), dtype=np.uint8)
        first = AddProcessor(1, show_result=True, return_result=False)
        second = AddProcessor(2, return_result=True)
        third = AddProcessor(4, show_result=True, return_result=True)
        fourth = AddProcessor(8, show_result=False, return_result=False)
        stream = WebcamStream("unused").append_processor(first, second, third, fourth)

        packet = stream._build_frame_packet(raw)

        self.assertEqual([first.calls, second.calls, third.calls, fourth.calls], [1, 1, 1, 1])
        self.assertTrue(np.all(packet.final_image == 6))
        self.assertTrue(np.all(packet.display_outputs[0].image == 1))
        packet.final_image[:] = 99
        self.assertTrue(np.all(packet.display_outputs[0].image == 1))

    def test_append_validation_and_open_reset(self):
        processor = AddProcessor()
        stream = WebcamStream("unused").append_processor(processor)
        with self.assertRaises(TypeError):
            stream.append_processor(object())
        with patch.object(WebcamStream, "open_stream", return_value=FakeCapture()):
            stream.open()
            self.assertEqual(processor.resets, 1)
            with self.assertRaises(RuntimeError):
                stream.append_processor(AddProcessor())
            stream.close()

    def test_read_returns_final_image(self):
        stream = WebcamStream("unused").append_processor(AddProcessor(3))
        stream.capture = FakeCapture()
        stream.buffer.append(
            stream._build_frame_packet(np.zeros((2, 2, 3), dtype=np.uint8))
        )
        self.assertTrue(np.all(stream.read() == 3))
        stream.close()

    def test_display_rate_is_capped(self):
        stream = WebcamStream("unused")
        stream.buffer.append(
            stream._build_frame_packet(np.zeros((2, 2, 3), dtype=np.uint8))
        )
        _packet, wait_time_ms, _fps = stream._next_display_frame()
        self.assertGreaterEqual(wait_time_ms, 1000 / MAX_STREAM_FPS)

    def test_specs_preserve_flags_settings_order_and_shared_reference(self):
        shared = DetectionResult()
        detector = TrackedObjectDetector(shared, return_result=False)
        annotator = DetectionAnnotator(shared, show_result=True)
        specs = pickle.loads(
            pickle.dumps([detector.construction_spec(), annotator.construction_spec()])
        )

        rebuilt = reconstruct_processors(specs)

        self.assertIs(rebuilt[0].detection_result, rebuilt[1].detection_result)
        self.assertFalse(rebuilt[0].return_result)
        self.assertTrue(rebuilt[1].show_result)

    def test_detector_and_annotator_exchange_results_by_reference(self):
        shared = DetectionResult()
        detector = TrackedObjectDetector(shared)
        annotator = DetectionAnnotator(shared)
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        detections = ("detection",)
        trails = ("trail",)
        completed = threading.Event()

        def active_trails(*, timestamp=None):
            completed.set()
            return trails

        with (
            patch.object(detector, "detect", return_value=detections),
            patch.object(detector, "active_trails", side_effect=active_trails),
        ):
            self.assertIs(detector(image), image)
            self.assertTrue(completed.wait(timeout=1))
            deadline = time.monotonic() + 1
            while shared.detections != detections and time.monotonic() < deadline:
                time.sleep(0.001)

        self.assertIs(detector.detection_result, annotator.detection_result)
        self.assertEqual(shared.detections, detections)
        self.assertEqual(shared.active_trails, trails)
        with patch.object(annotator, "annotate", return_value=image.copy()) as annotate:
            annotator(image)
            annotate.assert_called_once_with(image, detections, trails)

    def test_slow_inference_does_not_block_images(self):
        detector = TrackedObjectDetector(DetectionResult())
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        inference_started = threading.Event()
        release_inference = threading.Event()

        def slow_detect(_image, *, timestamp=None):
            inference_started.set()
            release_inference.wait(timeout=1)
            return ()

        with (
            patch.object(detector, "detect", side_effect=slow_detect),
            patch.object(detector, "active_trails", return_value=()),
        ):
            started = time.monotonic()
            self.assertIs(detector(image), image)
            self.assertLess(time.monotonic() - started, 0.1)
            self.assertTrue(inference_started.wait(timeout=1))
            started = time.monotonic()
            self.assertIs(detector(image), image)
            self.assertLess(time.monotonic() - started, 0.1)
            release_inference.set()
            deadline = time.monotonic() + 1
            while detector._inference_in_progress and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertFalse(detector._inference_in_progress)

    def test_inference_cadence_uses_budget_without_a_maximum_cap(self):
        detector = TrackedObjectDetector(
            DetectionResult(),
            fixed_inference_frame_budget_ms=25,
            inference_time_ema_alpha=1.0,
        )
        detector._frames_passed_during_inference = 3
        detector._update_inference_cadence(0.25)
        self.assertEqual(detector._frames_until_next_inference, 6)

    def test_legacy_constructor_slots_are_absent(self):
        removed = {"detector", "annotator", "filter", "snapshotter"}
        self.assertTrue(removed.isdisjoint(inspect.signature(WebcamStream).parameters))
        self.assertTrue(
            removed.isdisjoint(inspect.signature(WebcamStream.from_storage).parameters)
        )

    def test_adaptive_frame_skip_configuration_is_absent(self):
        removed = {"max_frames_to_skip_after_inference"}
        self.assertTrue(
            removed.isdisjoint(inspect.signature(TrackedObjectDetector).parameters)
        )

    def test_duplicate_processor_types_get_stable_unique_windows(self):
        stream = WebcamStream("unused", window_name="Base").append_processor(
            AddProcessor(show_result=True), AddProcessor(show_result=True)
        )
        stream.capture = FakeCapture()
        stream.buffer.append(
            stream._build_frame_packet(np.zeros((2, 2, 3), dtype=np.uint8))
        )
        names = []

        def wait_key(_delay):
            stream.stop_event.set()
            return ord("q")

        with (
            patch("cv2.namedWindow", side_effect=lambda name, _flags: names.append(name)),
            patch("cv2.imshow"),
            patch("cv2.waitKey", side_effect=wait_key),
            patch("cv2.destroyWindow"),
            patch.object(WebcamStream, "_size_window_for_frame"),
        ):
            stream.show()

        self.assertEqual(names, ["Base", "Base [1] AddProcessor", "Base [2] AddProcessor"])

    def test_base_window_displays_final_pipeline_image(self):
        stream = WebcamStream("unused", window_name="Base").append_processor(
            AddProcessor(7)
        )
        stream.capture = FakeCapture()
        stream.buffer.append(
            stream._build_frame_packet(np.zeros((2, 2, 3), dtype=np.uint8))
        )
        displayed_images = []

        def wait_key(_delay):
            stream.stop_event.set()
            return ord("q")

        with (
            patch("cv2.namedWindow"),
            patch(
                "cv2.imshow",
                side_effect=lambda _name, image: displayed_images.append(image.copy()),
            ),
            patch("cv2.waitKey", side_effect=wait_key),
            patch("cv2.destroyWindow"),
            patch.object(WebcamStream, "_size_window_for_frame"),
            patch.object(WebcamStream, "_draw_overlay"),
        ):
            stream.show()

        self.assertEqual(len(displayed_images), 1)
        self.assertTrue(np.all(displayed_images[0] == 7))


if __name__ == "__main__":
    unittest.main()
