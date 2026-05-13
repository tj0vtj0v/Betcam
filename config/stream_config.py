import json
from functools import lru_cache
from pathlib import Path
from typing import Dict


WebcamStreamsConfig = Dict[str, Dict[str, Dict[str, str]]]


@lru_cache(maxsize=1)
def load_webcam_streams() -> WebcamStreamsConfig:
    config_path = Path(__file__).with_name("config") / "webcam_streams.json"

    with config_path.open(encoding="utf-8") as config_file:
        return json.load(config_file)
