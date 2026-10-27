'Load the four detector modules and the slip/collision verifiers through their package names.'

from aha_publish import paths

import importlib
import os


# When set, a live-detector detection is confirmed with the detector's VLM
# verifier automatically (no y/N prompt, no pause), and a clear VLM "no
# failure" verdict retracts the detection. YES or UNKNOWN keeps it.
VLM_AUTO = os.getenv(
    "AHA_DETECTOR_VLM_AUTO", "").strip().lower() in ("1", "true", "yes", "on")


class DetectorBundle:
    def __init__(self, name):
        self.name = name
        self.detector = importlib.import_module(f"aha_publish.detectors.{name}.detector")
        self.vlm = (
            importlib.import_module(f"aha_publish.detectors.{name}.vlm_confirm")
            if name in {"slip", "collision"} else None
        )


_CACHE = {}


def load_detector_bundle(name):
    if name not in _CACHE:
        _CACHE[name] = DetectorBundle(name)
    return _CACHE[name]
