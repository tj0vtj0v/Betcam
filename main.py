from modules import DetectionAnnotator, DetectionResult, TrackedObjectDetector, WebcamStream


if __name__ == "__main__":
    detection_result = DetectionResult()
    detector = TrackedObjectDetector(
        detection_result,
        model_path="yolo26s_960_finetune.pt",
        model_input_size=1280,
        detection_confidence_threshold=0.2,
        track_history_timeout_seconds=5.0,
        max_inference_frame_dimension=960,
        fixed_inference_frame_budget_ms=25,
    )
    annotator = DetectionAnnotator(
        detection_result,
        show_labels=False,
        trail_point_fade_frames=100,
    )
    stream = WebcamStream.from_storage(
        "Plattling", "Bahnhof", "1280x960"
    ).append_processor(detector, annotator)
    WebcamStream.show_many([stream])
