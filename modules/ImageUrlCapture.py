import time
from typing import Optional, Tuple
from urllib.request import Request, urlopen

import cv2
import numpy as np

from config.types import RawFrame


class ImageUrlCapture:
    def __init__(self, url: str, *, refresh_seconds: float = 5) -> None:
        self.url = url
        self.refresh_seconds = refresh_seconds
        self._last_fetch_time = 0.0
        self._latest_frame: Optional[RawFrame] = None
        self._is_open = True

    def isOpened(self) -> bool:
        return self._is_open

    def read(self) -> Tuple[bool, Optional[RawFrame]]:
        if not self._is_open:
            return False, None

        current_time = time.monotonic()
        seconds_until_refresh = self.refresh_seconds - (current_time - self._last_fetch_time)
        if self._latest_frame is not None and seconds_until_refresh > 0:
            time.sleep(min(seconds_until_refresh, 0.1))
            return False, None

        if not self._fetch_frame():
            return False, None

        return self._latest_frame is not None, self._latest_frame

    def release(self) -> None:
        self._is_open = False
        self._latest_frame = None

    def _fetch_frame(self) -> bool:
        request = Request(self.url, headers={"User-Agent": "Mozilla/5.0"})

        try:
            with urlopen(request, timeout=10) as response:
                image_bytes = response.read()
        except OSError:
            return False

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None:
            return False

        self._latest_frame = frame
        self._last_fetch_time = time.monotonic()
        return True
