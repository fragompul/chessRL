"""Self-play data generation: many games in parallel, one batched network call per round.

Each game owns a `Search` tree. Every round, each active game contributes up to
`leaves_per_round` leaves; all leaves go through the network together. This keeps the
accelerator busy even though every individual game is sequential.

Optional ingredients (all off by default):
  * endgame curriculum: some games start from random KRK / KQK / KPK ... positions (RL-2)
  * league play: some games pit the current network against an older snapshot, recording
    only the current network's moves (RL-4)
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..engine import STARTING_FEN, Board, Game, Move
from ..engine.constants import BLACK, WHITE
from .buffer import K
from .curriculum import curriculum_start
from .encoding import encode_compact
from .evaluator import NetEvaluator
from .mcts import Request, Search, SearchConfig


@dataclass
class SelfPlayConfig:
    parallel: int = 16
    search: SearchConfig = field(default_factory=SearchConfig)
    temp_moves: int = 30  # sample moves ~ visit counts for this many plies, then play the best
    max_plies: int = 300
    endgames: list[str] | None = None
    endgame_prob: float = 0.0
    league_prob: float = 0.0
    adjudicate_material: float | None = None  # pawns; None = truncated games are draws


@dataclass
class SelfPlayGame:
    game: Game
    result: str
    termination: str
    pieces: np.ndarray  # [n,64] int8
    meta: np.ndarray  # [n,7] int16
    pol_idx: np.ndarray  # [n,K] int16
    pol_p: np.ndarray  # [n,K] float16
    z: np.ndarray  # [n] float32 (outcome for the player to move in each recorded position)
    opponent: str = "self"

    @property
    def plies(self) -> int:
        return len(self.game.moves)


class _Slot:
    """State of one game in flight."""

    def __init__(self, board: Board, start_fen: str, evals: dict, record: dict, shared: bool,
                 opponent: str, cfg: SelfPlayConfig, rng: np.random.Generator):
        self.board = board
        self.start_fen = start_fen
        self.evals = evals
        self.record = record
        self.shared = shared
        self.opponent = opponent
        self.cfg = cfg
        self.rng = rng
        self.moves: list[Move] = []
        self.rows: list[tuple] = []  # (pieces, meta, idx, p, mover)
        self.search = Search(board, cfg.search, rng)
        self.noise = False

    def prepare_move(self) -> None:
        """Start a fresh search for the side to move (reusing the tree when self-playing)."""
        if not self.shared:
            self.search = Search(self.board, self.cfg.search, self.rng)
        self.noise = False


def _material(board: Board) -> float:
    vals = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9}
    return sum((1 if p > 0 else -1) * vals.get(abs(p), 0) for p in board.squares)


class SelfPlayEngine:
    def __init__(
        self,
        cfg: SelfPlayConfig,
        current: NetEvaluator,
        opponents: list[tuple[str, NetEvaluator]] | None = None,
        seed: int | None = None,
    ):
        self.cfg = cfg
        self.current = current
        self.opponents = opponents or []
        self.rng = np.random.default_rng(seed)
        self.pyrng = random.Random(seed)
        self.stats = {"sims": 0, "nn_evals": 0, "seconds": 0.0}

    # ------------------------------------------------------------ game setup
    def _new_slot(self) -> _Slot:
        cfg = self.cfg
        if cfg.endgames and self.pyrng.random() < cfg.endgame_prob:
            board = curriculum_start(cfg.endgames, self.pyrng)
        else:
            board = Board(STARTING_FEN)
        start_fen = board.fen()
        if self.opponents and self.pyrng.random() < cfg.league_prob:
            name, ev = self.pyrng.choice(self.opponents)
            learner = self.pyrng.choice([WHITE, BLACK])
            evals = {learner: self.current, 1 - learner: ev}
            record = {learner: True, 1 - learner: False}
            return _Slot(board, start_fen, evals, record, False, name, cfg, self.rng)
        evals = {WHITE: self.current, BLACK: self.current}
        return _Slot(board, start_fen, evals, {WHITE: True, BLACK: True}, True, "self", cfg, self.rng)

    # ---------------------------------------------------------------- driver
    def run(
        self,
        n_games: int,
        should_stop: Callable[[], bool] = lambda: False,
        on_game: Callable[[SelfPlayGame], None] | None = None,
    ) -> list[SelfPlayGame]:
        cfg = self.cfg
        started = 0
        active: list[_Slot] = []
        finished: list[SelfPlayGame] = []
        t0 = time.perf_counter()

        def spawn() -> None:
            nonlocal started
            while started < n_games and len(active) < cfg.parallel:
                slot = self._new_slot()
                slot.prepare_move()
                active.append(slot)
                started += 1

        spawn()
        while active:
            if should_stop():
                break
            pending: list[tuple[_Slot, Request]] = []
            for s in active:
                self._gather(s, pending)
            self._evaluate(pending)
            for s in list(active):
                done = self._maybe_move(s)
                if done is not None:
                    active.remove(s)
                    finished.append(done)
                    if on_game:
                        on_game(done)
            spawn()
        self.stats["seconds"] += time.perf_counter() - t0
        return finished

    # --------------------------------------------------------------- stepping
    def _gather(self, s: _Slot, pending: list[tuple[_Slot, Request]]) -> None:
        search = s.search
        req = search.root_request()
        if req is not None:
            pending.append((s, req))
            return
        if not search.root_ready:
            return
        if not s.noise:
            search.add_root_noise()
            s.noise = True
        need = self.cfg.search.sims - search.root_visits
        want = min(self.cfg.search.leaves_per_round, max(need, 0))
        got = tries = 0
        while got < want and tries < 2 * want + 2:
            tries += 1
            r = search.select()
            if r is None:
                break
            got += 1
            if r != "done":
                pending.append((s, r))
        self.stats["sims"] += got

    def _evaluate(self, pending: list[tuple[_Slot, Request]]) -> None:
        if not pending:
            return
        groups: dict[int, list[int]] = {}
        for i, (s, _r) in enumerate(pending):
            groups.setdefault(id(s.evals[s.board.turn]), []).append(i)
        for ids in groups.values():
            ev = pending[ids[0]][0].evals[pending[ids[0]][0].board.turn]
            pieces = np.stack([pending[i][1].pieces for i in ids])
            meta = np.stack([pending[i][1].meta for i in ids])
            logits, values = ev(pieces, meta)
            self.stats["nn_evals"] += len(ids)
            for row, i in enumerate(ids):
                s, r = pending[i]
                s.search.apply(r, logits[row], float(values[row]))

    def _maybe_move(self, s: _Slot) -> SelfPlayGame | None:
        search = s.search
        if not search.root_ready or search.root_visits < self.cfg.search.sims:
            return None
        board = s.board
        mover = board.turn
        moves, actions, visits = search.policy(temperature=1.0)
        if s.record[mover]:
            pieces, meta = encode_compact(board)
            order = np.argsort(-visits)[:K]
            p = visits[order]
            p = p / p.sum()
            idx = np.full(K, -1, dtype=np.int16)
            probs = np.zeros(K, dtype=np.float16)
            idx[: len(order)] = actions[order]
            probs[: len(order)] = p
            s.rows.append((pieces, meta, idx, probs, mover))
        if len(s.moves) < self.cfg.temp_moves:
            choice = int(self.rng.choice(len(moves), p=visits))
        else:
            choice = int(np.argmax(visits))
        move = moves[choice]
        s.moves.append(move)
        board.push(move)
        if s.shared:
            search.advance(move)
        outcome = board.outcome()
        if outcome is not None:
            return self._finish(s, outcome.result, outcome.termination.value)
        if len(s.moves) >= self.cfg.max_plies:
            result, why = "1/2-1/2", "move limit"
            thr = self.cfg.adjudicate_material
            if thr is not None and abs(_material(board)) >= thr:
                result, why = ("1-0" if _material(board) > 0 else "0-1"), "material adjudication"
            return self._finish(s, result, why)
        s.prepare_move()
        return None

    def _finish(self, s: _Slot, result: str, why: str) -> SelfPlayGame:
        z_white = {"1-0": 1.0, "0-1": -1.0}.get(result, 0.0)
        n = len(s.rows)
        pieces = np.zeros((n, 64), dtype=np.int8)
        meta = np.zeros((n, 7), dtype=np.int16)
        pol_idx = np.full((n, K), -1, dtype=np.int16)
        pol_p = np.zeros((n, K), dtype=np.float16)
        z = np.zeros(n, dtype=np.float32)
        for i, (pc, mt, idx, pr, mover) in enumerate(s.rows):
            pieces[i], meta[i], pol_idx[i], pol_p[i] = pc, mt, idx, pr
            z[i] = z_white if mover == WHITE else -z_white
        game = Game(
            headers={"White": "net" if s.evals[WHITE] is self.current else s.opponent,
                     "Black": "net" if s.evals[BLACK] is self.current else s.opponent,
                     "Termination": why},
            moves=list(s.moves), result=result, start_fen=s.start_fen,
        )
        return SelfPlayGame(game, result, why, pieces, meta, pol_idx, pol_p, z, s.opponent)
