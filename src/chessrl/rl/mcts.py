"""Monte-Carlo tree search guided by a policy/value network (PUCT, AlphaZero-style).

The search is written as an explicit state machine so that many games can share one batched
network call:

    request = search.select()      # walks down the tree, returns a leaf needing evaluation
    ...evaluate a batch of requests with the network...
    search.apply(request, logits_row, value)

Features: virtual loss (several leaves per tree per batch), first-play-urgency, Dirichlet
root noise, tree reuse between moves.

Value convention: every value is from the point of view of the player to move at that node.
`Node.W[i]` accumulates the value of edge `i` for the player to move at `node`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..engine import Board, Move
from .actions import legal_actions
from .encoding import encode_compact


@dataclass
class SearchConfig:
    sims: int = 100
    c_puct: float = 1.5
    fpu_reduction: float = 0.2  # unvisited children start this far below the parent's value
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    virtual_loss: float = 1.0
    leaves_per_round: int = 1


class Node:
    __slots__ = ("moves", "actions", "P_raw", "P", "N", "W", "children", "v", "terminal", "pending")

    def __init__(self) -> None:
        self.moves: list[Move] | None = None  # None until expanded
        self.actions: np.ndarray | None = None
        self.P_raw: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.N: np.ndarray | None = None
        self.W: np.ndarray | None = None
        self.children: list[Node | None] | None = None
        self.v = 0.0
        self.terminal: float | None = None
        self.pending = False

    @property
    def expanded(self) -> bool:
        return self.moves is not None


def terminal_value(board: Board, moves: list[Move], repetitions: int = 2) -> float | None:
    """Value for the side to move if the position is terminal, else None.

    Inside the tree a position seen `repetitions` times counts as a draw (2 prunes cycles
    early); at the root the real game rule (3) applies.
    """
    if not moves:
        return -1.0 if board.is_check() else 0.0
    if (board.halfmove >= 100 or board.is_insufficient_material()
            or board.repetition_count() >= repetitions):
        return 0.0
    return None


@dataclass
class Request:
    """A leaf awaiting network evaluation."""

    node: Node
    path: list[tuple[Node, int]]
    pieces: np.ndarray
    meta: np.ndarray
    moves: list[Move]
    actions: np.ndarray
    is_root: bool = False
    result: tuple = field(default=(), repr=False)


class Search:
    def __init__(self, board: Board, cfg: SearchConfig, rng: np.random.Generator | None = None):
        self.board = board  # mutated during selection, always restored before returning
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self.root = Node()

    # ----------------------------------------------------------------- stats
    @property
    def root_visits(self) -> int:
        return int(self.root.N.sum()) if self.root.expanded else 0

    @property
    def root_ready(self) -> bool:
        return self.root.expanded

    # ----------------------------------------------------------- root set-up
    def root_request(self) -> Request | None:
        """Request to evaluate the root, or None if it is already expanded / terminal."""
        root = self.root
        if root.expanded or root.pending:
            return None
        moves, actions = legal_actions(self.board)
        term = terminal_value(self.board, moves, repetitions=3)
        if term is not None:
            self._expand_terminal(root, term)
            return None
        root.pending = True
        pieces, meta = encode_compact(self.board)
        return Request(root, [], pieces, meta, moves, actions, is_root=True)

    @staticmethod
    def _expand_terminal(node: Node, value: float) -> None:
        node.terminal = value
        node.v = value
        node.moves = []
        node.actions = np.zeros(0, dtype=np.int64)
        for name in ("P_raw", "P", "N", "W"):
            setattr(node, name, np.zeros(0, dtype=np.float32))
        node.children = []

    def add_root_noise(self) -> None:
        root = self.root
        if not root.expanded or not len(root.moves):
            return
        eps = self.cfg.dirichlet_eps
        noise = self.rng.dirichlet([self.cfg.dirichlet_alpha] * len(root.moves)).astype(np.float32)
        root.P = (1 - eps) * root.P_raw + eps * noise

    def reset_root_priors(self) -> None:
        if self.root.expanded and len(self.root.moves):
            self.root.P = self.root.P_raw.copy()

    # ------------------------------------------------------------- selection
    def _pick(self, node: Node, is_root: bool) -> int:
        n = node.N
        total = float(n.sum())
        fpu = 0.0 if is_root else self.cfg.fpu_reduction
        q = np.where(n > 0, node.W / np.maximum(n, 1e-9), node.v - fpu)
        u = self.cfg.c_puct * node.P * np.sqrt(max(total, 1.0)) / (1.0 + n)
        return int(np.argmax(q + u))

    def select(self) -> Request | None | str:
        """Descend to a leaf.

        Returns a `Request` if the leaf needs the network; "done" if the leaf was terminal and
        has been backed up; None if the descent hit a leaf that is already queued (try later).
        """
        node = self.root
        if not node.expanded:
            return None
        path: list[tuple[Node, int]] = []
        board = self.board
        vl = self.cfg.virtual_loss
        result: Request | None | str = None
        while True:
            if node.terminal is not None:
                self._backup(path, node.terminal)
                result = "done"
                break
            i = self._pick(node, node is self.root)
            node.N[i] += vl
            node.W[i] -= vl
            path.append((node, i))
            board.push(node.moves[i])
            child = node.children[i]
            if child is None:
                child = node.children[i] = Node()
            node = child
            if node.expanded:
                continue
            if node.pending:
                self._undo_virtual(path)
                result = None
                break
            moves, actions = legal_actions(board)
            term = terminal_value(board, moves)
            if term is not None:
                self._expand_terminal(node, term)
                self._backup(path, term)
                result = "done"
                break
            node.pending = True
            pieces, meta = encode_compact(board)
            result = Request(node, path, pieces, meta, moves, actions)
            break
        for _ in path:
            board.pop()
        return result

    def _undo_virtual(self, path: list[tuple[Node, int]]) -> None:
        vl = self.cfg.virtual_loss
        for node, i in path:
            node.N[i] -= vl
            node.W[i] += vl

    # ---------------------------------------------------------------- update
    def apply(self, req: Request, logits_row: np.ndarray, value: float) -> None:
        """Expand `req.node` with network output and back the value up."""
        node = req.node
        logits = logits_row[req.actions].astype(np.float64)
        logits -= logits.max()
        p = np.exp(logits)
        p = (p / p.sum()).astype(np.float32)
        node.moves, node.actions = req.moves, req.actions
        node.P_raw, node.P = p, p.copy()
        node.N = np.zeros(len(p), dtype=np.float32)
        node.W = np.zeros(len(p), dtype=np.float32)
        node.children = [None] * len(p)
        node.v = float(value)
        node.pending = False
        if not req.is_root:
            self._backup(req.path, float(value))

    def _backup(self, path: list[tuple[Node, int]], leaf_value: float) -> None:
        """`leaf_value` is for the player to move at the leaf; edges alternate in sign."""
        vl = self.cfg.virtual_loss
        v = -leaf_value
        for node, i in reversed(path):
            node.N[i] += 1.0 - vl
            node.W[i] += v + vl
            v = -v

    # ---------------------------------------------------------------- results
    def policy(self, temperature: float = 1.0) -> tuple[list[Move], np.ndarray, np.ndarray]:
        """(moves, action indices, probabilities) from the root visit counts."""
        root = self.root
        n = root.N.astype(np.float64)
        if temperature <= 1e-3:
            probs = np.zeros_like(n)
            probs[int(np.argmax(n))] = 1.0
        else:
            w = n ** (1.0 / temperature)
            probs = w / w.sum() if w.sum() > 0 else np.full_like(n, 1.0 / len(n))
        return root.moves, root.actions, probs

    def q_values(self) -> np.ndarray:
        """Mean action value of each root move for the side to move (unvisited = root value)."""
        root = self.root
        return np.where(root.N > 0, root.W / np.maximum(root.N, 1e-9), root.v)

    def advance(self, move: Move) -> None:
        """Re-root the tree at the child reached by `move` (the board is advanced by the caller)."""
        root = self.root
        child = None
        if root.expanded:
            for i, m in enumerate(root.moves):
                if m == move:
                    child = root.children[i]
                    break
        self.root = child if child is not None else Node()
        # A child can be "terminal" only by the in-tree repetition shortcut; as a root the real
        # game rules apply, so rebuild it. Also never carry a stale pending flag.
        if self.root.pending or self.root.terminal is not None:
            self.root = Node()


def run_search(
    search: Search,
    evaluator,
    sims: int,
    *,
    leaves_per_round: int = 8,
    deadline: float | None = None,
    noise: bool = False,
) -> None:
    """Run a single tree to `sims` root visits (or the deadline), evaluating leaves in batches."""
    import time

    req = search.root_request()
    if req is not None:
        logits, values = evaluator(req.pieces[None], req.meta[None])
        search.apply(req, logits[0], float(values[0]))
    if not search.root_ready or search.root.terminal is not None:
        return
    if noise:
        search.add_root_noise()
    while search.root_visits < sims:
        if deadline is not None and time.perf_counter() > deadline:
            break
        want = min(leaves_per_round, sims - search.root_visits)
        reqs, got, tries = [], 0, 0
        while got < want and tries < 2 * want + 2:
            tries += 1
            r = search.select()
            if r is None:
                break
            got += 1
            if r != "done":
                reqs.append(r)
        if not got:
            break
        if reqs:
            logits, values = evaluator(np.stack([r.pieces for r in reqs]),
                                       np.stack([r.meta for r in reqs]))
            for row, r in enumerate(reqs):
                search.apply(r, logits[row], float(values[row]))
