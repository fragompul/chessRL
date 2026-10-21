"""Playing on a real board: camera -> board geometry -> move tracking -> PGN."""

from .board_detect import find_board
from .estimator import DetectionEstimator, OccupancyEstimator
from .geometry import BoardCalibration
from .tracker import MoveTracker, TrackerEvent

__all__ = ["BoardCalibration", "DetectionEstimator", "MoveTracker", "OccupancyEstimator",
           "TrackerEvent", "find_board"]
