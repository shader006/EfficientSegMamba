"""Models package for BraTS 2023 experiments inside `all/`."""

from pathlib import Path
import sys

_MODEL_DIR = str(Path(__file__).resolve().parent)
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

from .segmamba import SegMamba  # noqa: E402

__all__ = ["SegMamba"]
