from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict

import cv2
import numpy as np

from config.types import RawFrame
from modules.FrameProcessor import FrameProcessor


class DifferenceFilter(FrameProcessor):
    def __init__(
        self,
        *,
        nth_last_image: int = 1,
        grayscale_output: bool = False,
        show_result: bool = False,
        return_result: bool = True,
    ) -> None:
        super().__init__(show_result=show_result, return_result=return_result)
        if nth_last_image <= 0:
            raise ValueError("nth_last_image must be greater than 0.")
        self.nth_last_image = nth_last_image
        self.grayscale_output = grayscale_output
        self._history: Deque[RawFrame] = deque(maxlen=nth_last_image)

    def __call__(self, image: RawFrame) -> RawFrame:
        reference = self._history[0] if len(self._history) == self.nth_last_image else image
        result = self._difference_frame(reference, image)
        self._history.append(image.copy())
        return result

    def reset(self) -> None:
        self._history.clear()

    def display_name(self) -> str:
        return f"{type(self).__name__} n={self.nth_last_image}"

    def construction_settings(self) -> Dict[str, Any]:
        return {
            "nth_last_image": self.nth_last_image,
            "grayscale_output": self.grayscale_output,
        }

    def _difference_frame(self, old_frame: RawFrame, new_frame: RawFrame) -> RawFrame:
        if self.grayscale_output:
            old_source = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            new_source = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        else:
            old_source, new_source = old_frame, new_frame
        difference = 128.0 + ((old_source.astype(np.float32) - new_source.astype(np.float32)) / 2.0)
        difference = np.clip(difference, 0, 255).astype(np.uint8)
        return cv2.cvtColor(difference, cv2.COLOR_GRAY2BGR) if self.grayscale_output else difference
