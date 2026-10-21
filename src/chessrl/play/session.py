"""An interactive game between a human and an AI: clocks, resign, draws, takebacks, hints.

The same object backs the terminal UI and the web UI. It is thread-safe: the web server
runs the AI's (possibly slow, deliberately delayed) move on a background thread.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..agents.base import Agent, SearchLimits
from ..agents.evaluation import evaluate
from ..engine import BLACK, STARTING_FEN, WHITE, Board, Game, Move, Termination
from .clock import Clock, TimeControl


@dataclass
class Ending:
    result: str  # "1-0" | "0-1" | "1/2-1/2"
    reason: str


class GameSession:
    def __init__(
        self,
        ai: Agent | None,
        human_color: int | None = WHITE,
        time_control: TimeControl | None = None,
        *,
        think_time: Callable[[Board, float | None, float], float] | None = None,
        realtime: bool = True,
        start_fen: str = STARTING_FEN,
        allow_takebacks: bool = True,
        hint_agent: Agent | None = None,
        headers: dict[str, str] | None = None,
        annotator=None,
        now: Callable[[], float] = time.monotonic,
    ):
        """`human_color=None` means the AI plays both sides (watch mode)."""
        self.ai = ai
        self.human_color = human_color
        self.board = Board(start_fen)
        self.start_fen = start_fen
        self.moves: list[Move] = []
        self.comments: dict[int, str] = {}
        self.headers = dict(headers or {})
        self.allow_takebacks = allow_takebacks
        self.hint_agent = hint_agent
        self.think_time = think_time
        self.realtime = realtime
        self.ending: Ending | None = None
        self.draw_offered_by: int | None = None
        self.clock = Clock(time_control, now) if time_control else None
        self._clock_snaps: list[dict[int, float]] = []
        self._lock = threading.RLock()
        self._now = now
        self.annotator = annotator
        self.annotations: dict[int, object] = {}
        self._pool: ThreadPoolExecutor | None = None
        self._pending: list = []
        if self.clock:
            self.clock.start(self.board.turn)

    # ----------------------------------------------------------- queries
    @property
    def over(self) -> bool:
        return self.ending is not None

    def is_ai_turn(self) -> bool:
        if self.over or self.ai is None:
            return False
        return self.human_color is None or self.board.turn != self.human_color

    def time_left(self, color: int) -> float | None:
        return self.clock.remaining(color) if self.clock else None

    # ------------------------------------------------------------ moves
    def play_human(self, move: Move | str) -> Move:
        with self._lock:
            self._check_flag()
            if self.over:
                raise ValueError("game is over")
            if self.human_color is not None and self.board.turn != self.human_color:
                raise ValueError("not your turn")
            if isinstance(move, str):
                move = self._parse(move)
            self._apply(move)
            return move

    def play_ai(self) -> Move:
        """Let the AI move now (blocking). Includes its human-like thinking delay if `realtime`."""
        with self._lock:
            if not self.is_ai_turn():
                raise ValueError("it is not the AI's turn")
            board = self.board.copy()
            color = board.turn
            limits = None
            if self.clock:
                limits = SearchLimits(time=max(0.05, min(self.clock.remaining(color) / 20, 10.0)))
        started = self._now()
        move = self.ai.select_move(board, limits)
        spent = self._now() - started
        delay = 0.0
        think = self.think_time or getattr(self.ai, "think_seconds", None)
        if think:
            left = self.time_left(color)
            inc = self.clock.tc.increment if self.clock else 0.0
            delay = max(0.0, think(board, left, inc) - spent)
            if left is not None:
                delay = min(delay, max(0.0, left - spent - 0.05))
        if self.realtime and delay > 0:
            time.sleep(delay)
        with self._lock:
            self._check_flag()
            if self.over or board.fen() != self.board.fen():
                raise ValueError("position changed while the AI was thinking")
            self._apply(move)
            return move

    def _parse(self, text: str) -> Move:
        try:
            return self.board.parse_uci(text)
        except ValueError:
            return self.board.parse_san(text)

    def _apply(self, move: Move) -> None:
        if move not in self.board.legal_moves():
            raise ValueError(f"illegal move {move}")
        color = self.board.turn
        if self.clock:
            self._clock_snaps.append(self.clock.snapshot())
            self.clock.press(color)
        self.draw_offered_by = None
        if self.annotator is not None:
            self._submit_annotation(self.board.copy(), move, len(self.moves))
        self.moves.append(move)
        self.board.push(move)
        outcome = self.board.outcome()
        if outcome:
            self._end(outcome.result, outcome.termination.value)

    # ------------------------------------------------------- live annotation
    def _submit_annotation(self, board_before: Board, move: Move, ply: int) -> None:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="annotator")

        def work() -> None:
            from ..analysis.annotate import comment_for

            result = self.annotator.annotate(board_before, move, ply)
            with self._lock:
                if ply < len(self.moves) and self.moves[ply] == move:  # not taken back meanwhile
                    self.annotations[ply] = result
                    text = comment_for(result)
                    if text:
                        self.comments[ply] = text

        self._pending.append(self._pool.submit(work))

    def wait_annotations(self, timeout: float | None = None) -> None:
        for f in list(self._pending):
            f.result(timeout)
        self._pending.clear()

    def _end(self, result: str, reason: str) -> None:
        self.ending = Ending(result, reason)
        if self.clock:
            self.clock.stop()

    def _check_flag(self) -> None:
        if self.over or not self.clock:
            return
        color = self.board.turn
        if self.clock.flagged(color):
            winner = 1 - color
            if self._has_mating_material(winner):
                self._end("1-0" if winner == WHITE else "0-1", "timeout")
            else:
                self._end("1/2-1/2", "timeout vs insufficient material")

    def _has_mating_material(self, color: int) -> bool:
        sign = 1 if color == WHITE else -1
        pieces = [abs(p) for p in self.board.squares if p * sign > 0 and abs(p) != 6]
        return any(t in (1, 4, 5) for t in pieces) or len(pieces) >= 2

    def check_time(self) -> None:
        with self._lock:
            self._check_flag()

    # ----------------------------------------------------- game options
    def resign(self, color: int) -> None:
        with self._lock:
            if self.over:
                return
            self._end("0-1" if color == WHITE else "1-0", "resignation")

    def offer_draw(self, color: int) -> bool:
        """Offer a draw; the AI accepts when it judges its own position as not better."""
        with self._lock:
            if self.over:
                return False
            if self.ai is not None and color == self.human_color:
                ai_color = 1 - color
                static = evaluate(self.board) * (1 if self.board.turn == ai_color else -1)
                if static <= 30 and len(self.moves) >= 20:
                    self._end("1/2-1/2", "agreement")
                    return True
                return False
            self.draw_offered_by = color
            return False

    def accept_draw(self, color: int) -> bool:
        with self._lock:
            if self.draw_offered_by is not None and self.draw_offered_by != color and not self.over:
                self._end("1/2-1/2", "agreement")
                return True
            return False

    def claim_draw(self) -> bool:
        """Claim a draw by threefold repetition or the fifty-move rule if it applies."""
        with self._lock:
            if self.over:
                return False
            if self.board.repetition_count() >= 3 or self.board.halfmove >= 100:
                self._end("1/2-1/2", "claimed draw")
                return True
            return False

    def takeback(self) -> int:
        """Undo the human's last move (and the AI reply, if any). Returns plies undone."""
        with self._lock:
            if not self.allow_takebacks or not self.moves or self.human_color is None:
                return 0
            undone = 0
            # Walk back until it is the human's turn again.
            while self.moves:
                self.board.pop()
                self.moves.pop()
                snap = self._clock_snaps.pop() if self._clock_snaps else None
                self.comments.pop(len(self.moves), None)
                self.annotations.pop(len(self.moves), None)
                undone += 1
                if self.board.turn == self.human_color:
                    if snap and self.clock:
                        self.clock.restore(snap)
                    break
            if self.clock:
                self.clock.start(self.board.turn)
            self.ending = None
            self.draw_offered_by = None
            return undone

    def hint(self, depth: int = 3) -> Move | None:
        with self._lock:
            if self.over:
                return None
            agent = self.hint_agent
            board = self.board.copy()
        if agent is None:
            from ..agents.minimax import AlphaBetaAgent

            agent = AlphaBetaAgent(depth=depth, seed=0)
        return agent.select_move(board, SearchLimits(time=2.0))

    # ------------------------------------------------------------ export
    def to_game(self) -> Game:
        with self._lock:
            headers = dict(self.headers)
            if self.clock:
                headers.setdefault("TimeControl", f"{int(self.clock.tc.base)}+{self.clock.tc.increment:g}")
            if self.ending:
                headers["Termination"] = self.ending.reason
            return Game(
                headers=headers,
                moves=list(self.moves),
                result=self.ending.result if self.ending else "*",
                start_fen=self.start_fen,
                comments=dict(self.comments),
            )

    def status_text(self) -> str:
        if self.ending:
            return f"{self.ending.result} ({self.ending.reason})"
        side = "White" if self.board.turn == WHITE else "Black"
        return f"{side} to move" + (" - check!" if self.board.is_check() else "")


__all__ = ["Ending", "GameSession", "Termination", "BLACK"]
