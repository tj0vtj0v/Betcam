from ultralytics import YOLO
import threading
import time


class OnDemandDetector:
    def __init__(self, model_path: str = "yolo26s.pt", imgsz: int = 1280):
        self.model = YOLO(model_path)
        self.imgsz = imgsz

        self.latest_frame = None
        self.frame_lock = threading.Lock()

        self.enabled = False
        self.running = False
        self.worker = None

    def push_frame(self, frame):
        with self.frame_lock:
            self.latest_frame = frame

    def start(self):
        self.running = True
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def stop(self):
        self.running = False
        if self.worker:
            self.worker.join()

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def _get_latest_frame(self):
        with self.frame_lock:
            return self.latest_frame

    def _loop(self):
        while self.running:
            if not self.enabled:
                time.sleep(0.01)
                continue

            frame = self._get_latest_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            results = self.model.track(
                frame,
                imgsz=self.imgsz,
                classes=[0, 2],
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
            )

            self.handle_results(results)

    def handle_results(self, results):
        result = results[0]

        if result.boxes is None:
            return

        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            track_id = None
            if box.id is not None:
                track_id = int(box.id[0])

            print({
                "track_id": track_id,
                "class": cls,
                "confidence": conf,
                "xyxy": xyxy,
            })