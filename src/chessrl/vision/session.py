"""Play on a real board: frames in, a recorded and annotated game out.

A `VisionSession` watches a camera (or video file), finds the board, tracks every move, keeps
the game as PGN, optionally annotates it live, and — when an agent is attached — suggests (or
plays, for the side it controls) the next move so the human can carry it out on the board.
"""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..agents.base import SearchLimits
from ..engine import STARTING_FEN, WHITE, Board, Game, Move
from .board_detect import find_board
from .estimator import Estimator, OccupancyEstimator
from .geometry import BoardCalibration
from .tracker import MoveTracker, TrackerEvent


@dataclass
class Suggestion:
    move: Move
    san: str
    for_color: int


def say(text: str) -> None:
    """Speak a short phrase (macOS `say`); silently does nothing elsewhere."""
    if sys.platform == "darwin":
        try:
            subprocess.Popen(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass


class VisionSession:
    def __init__(
        self,
        *,
        calibration: BoardCalibration | None = None,
        estimator: Estimator | None = None,
        start_fen: str = STARTING_FEN,
        white_side: int | None = None,
        agent=None,
        ai_color: int | None = None,
        annotator=None,
        on_move: Callable[[TrackerEvent, Board], None] | None = None,
        on_suggestion: Callable[[Suggestion], None] | None = None,
        on_unexplained: Callable[[TrackerEvent], None] | None = None,
        headers: dict[str, str] | None = None,
        think_seconds: float = 1.0,
    ):
        self.calibration = calibration
        self.estimator = estimator
        self.start_fen = start_fen
        self.white_side = white_side
        self.agent = agent
        self.ai_color = ai_color
        self.annotator = annotator
        self.on_move, self.on_suggestion, self.on_unexplained = on_move, on_suggestion, on_unexplained
        self.think_seconds = think_seconds
        self.board = Board(start_fen)
        self.game = Game(headers={"Event": "Real-board game", "Site": "chessRL vision",
                                  "Date": dt.date.today().strftime("%Y.%m.%d"), **(headers or {})},
                         start_fen=start_fen)
        self.tracker: MoveTracker | None = None
        self.suggestion: Suggestion | None = None
        self.events: list[TrackerEvent] = []
        self.frames_seen = 0
        self._calib_tries = 0

    # --------------------------------------------------------------- frames
    def process_frame(self, frame: np.ndarray) -> TrackerEvent | None:
        self.frames_seen += 1
        if self.tracker is None and not self._try_calibrate(frame):
            return None
        ev = self.tracker.update(frame)
        if ev is None:
            return None
        self.events.append(ev)
        if ev.kind == "unexplained":
            if self.on_unexplained:
                self.on_unexplained(ev)
            return ev
        self._register(ev)
        return ev

    def _try_calibrate(self, frame: np.ndarray) -> bool:
        if self.calibration is None:
            self._calib_tries += 1
            if self._calib_tries > 60:
                raise RuntimeError("could not find the board; pass --corners or a calibration file")
            ws = self.white_side if self.white_side is not None else (None if self.start_fen == STARTING_FEN else 0)
            self.calibration = find_board(frame, white_side=ws)
            if self.calibration is None:
                return False
        if self.estimator is None:
            self.estimator = OccupancyEstimator()
        self.tracker = MoveTracker(self.board, self.estimator, self.calibration)
        self.tracker.start(frame)
        self._suggest()
        return True

    def _register(self, ev: TrackerEvent) -> None:
        for ply, move in enumerate(ev.moves, start=len(self.game.moves)):
            self.game.moves.append(move)
            if self.annotator is not None:
                before = Board(self.start_fen)
                for m in self.game.moves[:ply]:
                    before.push(m)
                try:
                    from ..analysis.annotate import comment_for

                    note = self.annotator.annotate(before, move, ply)
                    text = comment_for(note)
                    if text:
                        self.game.comments[ply] = text
                except Exception:  # noqa: BLE001 - annotation must never break recording
                    pass
        out = self.board.outcome()
        if out is not None:
            self.game.result = out.result
            self.game.headers["Termination"] = out.termination.value
        if self.on_move:
            self.on_move(ev, self.board)
        if out is None:
            self._suggest()

    def _suggest(self) -> None:
        self.suggestion = None
        if self.agent is None or self.board.outcome() is not None:
            return
        if self.ai_color is not None and self.board.turn != self.ai_color:
            return
        move = self.agent.select_move(self.board.copy(), SearchLimits(time=self.think_seconds))
        self.suggestion = Suggestion(move, self.board.san(move), self.board.turn)
        if self.on_suggestion:
            self.on_suggestion(self.suggestion)

    # --------------------------------------------------------------- output
    def pgn(self) -> str:
        return self.game.to_pgn()

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Frame with the board outline, last move and the AI's suggestion drawn on it."""
        img = frame.copy()
        cal = self.calibration
        if cal is None:
            return img
        cv2.polylines(img, [cal.corners.astype(np.int32)], True, (0, 200, 255), 2, cv2.LINE_AA)
        if self.game.moves:
            m = self.game.moves[-1]
            for sq in (m.from_sq, m.to_sq):
                cv2.polylines(img, [cal.square_polygon(sq).astype(np.int32)], True, (60, 220, 60), 3, cv2.LINE_AA)
        if self.suggestion is not None:
            a = np.array(cal.square_center(self.suggestion.move.from_sq))
            b = np.array(cal.square_center(self.suggestion.move.to_sq))
            cv2.arrowedLine(img, tuple(a.astype(int)), tuple(b.astype(int)), (255, 120, 40), 4, cv2.LINE_AA, tipLength=0.2)
            cv2.putText(img, f"AI: {self.suggestion.san}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 120, 40), 2, cv2.LINE_AA)
        side = "White" if self.board.turn == WHITE else "Black"
        cv2.putText(img, f"{len(self.game.moves)} plies - {side} to move", (16, img.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    def run(
        self,
        source: str | int,
        *,
        max_frames: int | None = None,
        out_dir: str | Path | None = None,
        save_video: bool = False,
        show: bool = False,
        speak: bool = False,
        log: Callable[[str], None] = print,
    ) -> Game:
        """Process a video file / camera index to the end (or `max_frames`)."""
        src = int(source) if isinstance(source, str) and source.isdigit() else source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video source {source!r}")
        out = Path(out_dir) if out_dir else None
        if out:
            out.mkdir(parents=True, exist_ok=True)
        writer = None
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        old_move, old_sug = self.on_move, self.on_suggestion

        def moved(ev, board):
            log(f"move {len(self.game.moves)}: {' '.join(ev.san)}  (confidence {ev.confidence:.1f})"
                + ("  [promotion assumed to queen]" if ev.promotion_assumed else ""))
            if old_move:
                old_move(ev, board)

        def suggested(s):
            log(f"  suggestion for {'White' if s.for_color == WHITE else 'Black'}: {s.san}")
            if speak:
                say(s.san.replace("O-O-O", "castle queenside").replace("O-O", "castle kingside"))
            if old_sug:
                old_sug(s)

        self.on_move, self.on_suggestion = moved, suggested
        try:
            n = 0
            while max_frames is None or n < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                n += 1
                self.process_frame(frame)
                if save_video and out:
                    img = self.draw_overlay(frame)
                    if writer is None:
                        h, w = img.shape[:2]
                        writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    writer.write(img)
                if show:
                    try:
                        cv2.imshow("chessRL vision", self.draw_overlay(frame))
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except cv2.error:
                        show = False
                        log("(no GUI support in this OpenCV build; continuing without a window)")
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            self.on_move, self.on_suggestion = old_move, old_sug
            if out:
                (out / "game.pgn").write_text(self.pgn())
                if self.calibration is not None:
                    self.calibration.save(out / "calibration.json")
        return self.game


__all__ = ["Suggestion", "VisionSession", "say", "time"]
