"""Playing infrastructure: matches, clocks, interactive sessions."""

from .clock import PRESETS, Clock, TimeControl
from .match import GameResult, play_game
from .session import Ending, GameSession

__all__ = ["PRESETS", "Clock", "Ending", "GameResult", "GameSession", "TimeControl", "play_game"]
