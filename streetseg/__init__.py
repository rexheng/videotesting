"""Video orchestration for the scout drone.

Takes a forward camera frame (Tello) plus an optional body-camera frame,
segments the street, and returns a short spoken cue for the AirPods.
"""

from .pipeline import Orchestrator
from .types import Box, Cue, FrameResult

__all__ = ["Box", "Cue", "FrameResult", "Orchestrator"]
