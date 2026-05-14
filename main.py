from modules import WebcamStream, TrackedObjectDetector, DetectionAnnotator, Snapshotter

if __name__ == "__main__":
    snapshotter = Snapshotter(
        save_every_n_frames=1000
    )
    detector = TrackedObjectDetector(
        model_path="yolo26s_960_finetune.pt",
        model_input_size=1280,
        detection_confidence_threshold=0.2,
        track_history_timeout_seconds=5.0,
        max_inference_frame_dimension=960,
        fixed_inference_frame_budget_ms=25,
        max_frames_to_skip_after_inference=10
    )
    annotator = DetectionAnnotator(
        show_labels=False,
        trail_point_fade_frames=100
    )

    streams = [
                  WebcamStream.from_storage(
                      "Deggendorf",
                      location,
                      "1280x960",
                      # snapshotter=snapshotter,
                      detector=detector,
                      annotator=annotator,
                  )
                  for location in [
            # "Alt-Hafen",
            # "Graflinger Tal",
            # "Schaching",
            # "Hafen",
            # "Vorstadt",
            # "Stadtmitte",
            # "Luitpoldplatz",
            "Oberer Stadtplatz",
        ]
              ] + [
                  WebcamStream.from_storage(
                      town,
                      "Bahnhof",
                      "1280x960",
                      # snapshotter=snapshotter,
                      # detector=detector,
                      # annotator=annotator,
                  )
                  for town in [
            # "Plattling",
            # "Straubing"
        ]
              ]
    WebcamStream.show_many(streams)
