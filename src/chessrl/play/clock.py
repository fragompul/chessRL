"""Chess clocks and time controls."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from ..engine.constants import BLACK, WHITE


@dataclass(frozen=True)
class TimeControl:
    """Fischer time control: `base` seconds each plus `increment` seconds per move."""

    base: float
    increment: float = 0.0

    @property
    def name(self) -> str:
        return f"{self.base / 60:g}+{self.increment:g}"

    @property
    def category(self) -> str:
        """Lichess-style category from the estimated duration (base + 40 * increment)."""
        est = self.base + 40 * self.increment
        if est < 180:
            return "bullet"
        if est < 480:
            return "blitz"
        if est < 1500:
            return "rapid"
        return "classical"

    @classmethod
    def parse(cls, text: str) -> TimeControl:
        """'5+3' -> 5 minutes + 3 s increment; '10' -> 10 minutes, no increment."""
        base, _, inc = text.partition("+")
        return cls(float(base) * 60, float(inc or 0))


PRESETS = {
    "bullet": TimeControl.parse("1+0"),
    "blitz": TimeControl.parse("3+2"),
    "blitz5": TimeControl.parse("5+0"),
    "rapid": TimeControl.parse("10+0"),
    "rapid15": TimeControl.parse("15+10"),
    "classical": TimeControl.parse("30+0"),
}


class Clock:
    """Two-sided clock. `now` is injectable so tests (and simulations) control time."""

    def __init__(self, tc: TimeControl, now: Callable[[], float] = time.monotonic):
        self.tc = tc
        self._now = now
        self._left = {WHITE: tc.base, BLACK: tc.base}
        self._running: int | None = None
        self._since = 0.0

    def start(self, color: int) -> None:
        self._running, self._since = color, self._now()

    def remaining(self, color: int) -> float:
        left = self._left[color]
        if self._running == color:
            left -= self._now() - self._since
        return max(0.0, left)

    def flagged(self, color: int) -> bool:
        return self.remaining(color) <= 0.0

    def press(self, color: int) -> float:
        """`color` has just moved: stop its clock, add the increment, start the opponent's.

        Returns the time the move took.
        """
        spent = self._now() - self._since if self._running == color else 0.0
        self._left[color] = max(0.0, self._left[color] - spent) + self.tc.increment
        self.start(1 - color)
        return spent

    def stop(self) -> None:
        if self._running is not None:
            self._left[self._running] = self.remaining(self._running)
        self._running = None

    def restore(self, left: dict[int, float]) -> None:
        """Set remaining times (used by takebacks)."""
        self._left = dict(left)
        if self._running is not None:
            self._since = self._now()

    def snapshot(self) -> dict[int, float]:
        return {WHITE: self.remaining(WHITE), BLACK: self.remaining(BLACK)}
