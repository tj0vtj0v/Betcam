from __future__ import annotations

from collections import deque
from typing import Deque, Literal

import cv2
import numpy as np

from config.types import RawFrame

FilterMode = Literal["display_only", "inference"]


class StreamFilter:
    def __init__(self, *, mode: FilterMode = "display_only") -> None:
        if mode not in {"display_only", "inference"}:
            raise ValueError("mode must be either 'display_only' or 'inference'.")
        self.mode = mode

    @property
    def shows_auxiliary_window(self) -> bool:
        return self.mode == "display_only"

    @property
    def uses_for_inference(self) -> bool:
        return self.mode == "inference"

    def apply(self, frame: RawFrame) -> RawFrame:
        raise NotImplementedError

    def reset(self) -> None:
        return None

    def clone(self) -> "StreamFilter":
        raise NotImplementedError

    def display_name(self) -> str:
        return type(self).__name__


class DifferenceFilter(StreamFilter):
    def __init__(
        self,
        *,
        mode: FilterMode = "display_only",
        nth_last_image: int = 1,
        grayscale_output: bool = False,
    ) -> None:
        super().__init__(mode=mode)
        if nth_last_image <= 0:
            raise ValueError("nth_last_image must be greater than 0.")

        self.nth_last_image = nth_last_image
        self.grayscale_output = grayscale_output
        self._history: Deque[RawFrame] = deque(maxlen=nth_last_image)

    def apply(self, frame: RawFrame) -> RawFrame:
        reference_frame = self._history[0] if len(self._history) == self.nth_last_image else frame
        filtered_frame = self._difference_frame(reference_frame, frame)
        self._history.append(frame.copy())
        return filtered_frame

    def reset(self) -> None:
        self._history.clear()

    def clone(self) -> "DifferenceFilter":
        return DifferenceFilter(
            mode=self.mode,
            nth_last_image=self.nth_last_image,
            grayscale_output=self.grayscale_output,
        )

    def display_name(self) -> str:
        return f"{type(self).__name__} n={self.nth_last_image}"

    def _difference_frame(self, old_frame: RawFrame, new_frame: RawFrame) -> RawFrame:
        if self.grayscale_output:
            old_source = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            new_source = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        else:
            old_source = old_frame
            new_source = new_frame

        difference = 128.0 + ((old_source.astype(np.float32) - new_source.astype(np.float32)) / 2.0)
        difference = np.clip(difference, 0, 255).astype(np.uint8)

        if self.grayscale_output:
            return cv2.cvtColor(difference, cv2.COLOR_GRAY2BGR)

        return difference
