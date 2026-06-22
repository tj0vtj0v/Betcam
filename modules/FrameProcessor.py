from __future__ import annotations

import importlib
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List

from config.types import RawFrame


class FrameProcessor(ABC):
    """A single, configurable stage in a :class:`WebcamStream` pipeline."""

    show_result: bool = False
    return_result: bool = True

    def __init__(self, *, show_result: bool = False, return_result: bool = True) -> None:
        self.show_result = bool(show_result)
        self.return_result = bool(return_result)

    @abstractmethod
    def __call__(self, image: RawFrame) -> RawFrame:
        """Process *image* without modifying it in place."""

    def reset(self) -> None:
        pass

    def display_name(self) -> str:
        return type(self).__name__

    def construction_settings(self) -> Dict[str, Any]:
        """Return pickle-safe keyword arguments needed to reconstruct this stage."""
        return {}

    def construction_spec(self) -> Dict[str, Any]:
        processor_type = type(self)
        if "<locals>" in processor_type.__qualname__:
            raise TypeError(
                f"{processor_type.__name__} must be a top-level importable class for show_many()."
            )
        settings = self.construction_settings()
        module = importlib.import_module(processor_type.__module__)
        resolved: Any = module
        for component in processor_type.__qualname__.split("."):
            resolved = getattr(resolved, component)
        if resolved is not processor_type:
            raise TypeError("qualified class name does not resolve to this processor class")
        pickle.dumps(settings)
        return {
            "module": processor_type.__module__,
            "qualname": processor_type.__qualname__,
            "settings": settings,
            "show_result": self.show_result,
            "return_result": self.return_result,
        }


def processor_specs(processors: Iterable[FrameProcessor]) -> List[Dict[str, Any]]:
    specs = []
    for index, processor in enumerate(processors):
        try:
            specs.append(processor.construction_spec())
        except Exception as error:
            raise TypeError(
                f"Invalid processor specification at index {index} "
                f"({type(processor).__name__}): {error}"
            ) from error
    return specs


def reconstruct_processors(specs: Iterable[Dict[str, Any]]) -> List[FrameProcessor]:
    processors: List[FrameProcessor] = []
    for index, spec in enumerate(specs):
        class_name = f"{spec.get('module', '?')}.{spec.get('qualname', '?')}"
        try:
            module = importlib.import_module(spec["module"])
            processor_type: Any = module
            for component in spec["qualname"].split("."):
                processor_type = getattr(processor_type, component)
            if not isinstance(processor_type, type) or not issubclass(processor_type, FrameProcessor):
                raise TypeError("resolved class is not a FrameProcessor")
            settings = dict(spec.get("settings", {}))
            settings["show_result"] = bool(spec.get("show_result", False))
            settings["return_result"] = bool(spec.get("return_result", True))
            processors.append(processor_type(**settings))
        except Exception as error:
            raise TypeError(
                f"Could not reconstruct processor at index {index} ({class_name}): {error}"
            ) from error
    return processors
