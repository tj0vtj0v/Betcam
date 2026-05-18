from modules import WebcamStream, TrackedObjectDetector, DetectionAnnotator, Snapshotter, DifferenceFilter

if __name__ == "__main__":
    snapshotter = Snapshotter(
        save_every_n_frames=100
    )
    default_detector = TrackedObjectDetector(  # best
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
    default_filter = DifferenceFilter(
        mode="display_only",
        nth_last_image=1,
        grayscale_output=True,
    )

    streams = [
                  WebcamStream.from_storage(
                      "Deggendorf",
                      location,
                      "1280x960",
                      # snapshotter=snapshotter,
                      filter=default_filter,
                      detector=default_detector,
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
            # "Oberer Stadtplatz",
        ]
              ] + [
                  WebcamStream.from_storage(
                      town,
                      "Bahnhof",
                      "1280x960",
                      snapshotter=snapshotter,
                      # detector=default_detector,
                      # annotator=annotator,
                  )
                  for town in [
            # "Plattling",
            # "Straubing"
        ]
              ] + [
                  WebcamStream.from_storage(
                      "Deggendorf",
                      "Stadtmitte",
                      "1280x960",
                      filter=difference_filter,
                  )
                  for difference_filter in [
            DifferenceFilter(
                mode="display_only",
                nth_last_image=1,
                grayscale_output=True,
            ),
            DifferenceFilter(
                mode="display_only",
                nth_last_image=2,
                grayscale_output=True,
            ),
            DifferenceFilter(
                mode="display_only",
                nth_last_image=5,
                grayscale_output=True,
            ),
            DifferenceFilter(
                mode="display_only",
                nth_last_image=8,
                grayscale_output=True,
            ),
            DifferenceFilter(
                mode="display_only",
                nth_last_image=10,
                grayscale_output=True,
            ),
            DifferenceFilter(
                mode="display_only",
                nth_last_image=15,
                grayscale_output=True,
            ),
        ]
              ] + [
                  WebcamStream.from_storage(
                      "Deggendorf",
                      "Luitpoldplatz",
                      "1280x960",
                      # filter=difference_filter,
                      detector=detector,
                      annotator=annotator,
                  )
                  for detector in [
            # TrackedObjectDetector( # not consistent
            #     model_path="yolo26n_640_finetune.pt",
            #     model_input_size=1280,
            #     detection_confidence_threshold=0.2,
            #     track_history_timeout_seconds=5.0,
            #     max_inference_frame_dimension=640,
            #     fixed_inference_frame_budget_ms=25,
            #     max_frames_to_skip_after_inference=10
            # ),
            # TrackedObjectDetector( # good
            #     model_path="yolo26n_960_finetune.pt",
            #     model_input_size=1280,
            #     detection_confidence_threshold=0.2,
            #     track_history_timeout_seconds=5.0,
            #     max_inference_frame_dimension=960,
            #     fixed_inference_frame_budget_ms=25,
            #     max_frames_to_skip_after_inference=10
            # ),
            # TrackedObjectDetector( # bad on people
            #     model_path="yolo26s_640_finetune.pt",
            #     model_input_size=1280,
            #     detection_confidence_threshold=0.2,
            #     track_history_timeout_seconds=5.0,
            #     max_inference_frame_dimension=640,
            #     fixed_inference_frame_budget_ms=25,
            #     max_frames_to_skip_after_inference=10
            # ),
            # TrackedObjectDetector(  # best
            #     model_path="yolo26s_960_finetune.pt",
            #     model_input_size=1280,
            #     detection_confidence_threshold=0.2,
            #     track_history_timeout_seconds=5.0,
            #     max_inference_frame_dimension=960,
            #     fixed_inference_frame_budget_ms=25,
            #     max_frames_to_skip_after_inference=10
            # ),
            # TrackedObjectDetector( # too slow
            #     model_path="yolo26m_960_finetune.pt",
            #     model_input_size=1280,
            #     detection_confidence_threshold=0.2,
            #     track_history_timeout_seconds=5.0,
            #     max_inference_frame_dimension=960,
            #     fixed_inference_frame_budget_ms=25,
            #     max_frames_to_skip_after_inference=10
            # ),
            # TrackedObjectDetector( # holy shit
            #     model_path="yolo26m_1280_finetune.pt",
            #     model_input_size=1280,
            #     detection_confidence_threshold=0.2,
            #     track_history_timeout_seconds=5.0,
            #     max_inference_frame_dimension=1280,
            #     fixed_inference_frame_budget_ms=25,
            #     max_frames_to_skip_after_inference=10
            # )
        ]
              ]
    WebcamStream.show_many(streams)
