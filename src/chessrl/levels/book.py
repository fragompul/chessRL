"""A small opening book (also used to name openings, UI-9).

Lines are written in SAN and validated by the test-suite. The book maps a *position* (so
transpositions work) to the moves that were played from it, weighted by how many lines use them.
"""

from __future__ import annotations

import random
from collections import defaultdict

from ..engine import Board, Move

# (ECO-ish code, name, moves)
LINES: list[tuple[str, str, str]] = [
    ("C84", "Ruy Lopez: Closed", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O"),
    ("C65", "Ruy Lopez: Berlin Defense", "e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Nxe4 d4 Nd6 Bxc6 dxc6 dxe5 Nf5"),
    ("C80", "Ruy Lopez: Open", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Nxe4 d4 b5 Bb3 d5 dxe5 Be6"),
    ("C54", "Italian Game: Giuoco Pianissimo", "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6 O-O O-O"),
    ("C57", "Italian Game: Two Knights Defense", "e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 d5 exd5 Na5 Bb5+ c6"),
    ("C45", "Scotch Game", "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6 e5 Qe7 Qe2 Nd5"),
    ("C42", "Petrov's Defense", "e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5 Bd3 Nc6 O-O Be7"),
    ("C26", "Vienna Game", "e4 e5 Nc3 Nf6 Bc4 Nxe4 Qh5 Nd6 Bb3 Be7 Nf3 Nc6"),
    ("C36", "King's Gambit Accepted", "e4 e5 f4 exf4 Nf3 g5 h4 g4 Ng5 h6 Nxf7 Kxf7"),
    ("C48", "Four Knights: Spanish", "e4 e5 Nf3 Nc6 Nc3 Nf6 Bb5 Bb4 O-O O-O d3 d6"),
    ("C23", "Bishop's Opening", "e4 e5 Bc4 Nf6 d3 c6 Nf3 d5 Bb3 Bd6 O-O O-O"),
    ("B90", "Sicilian: Najdorf", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6"),
    ("B33", "Sicilian: Sveshnikov", "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Ndb5 d6 Bg5 a6"),
    ("B42", "Sicilian: Kan", "e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Bd3 Nf6 O-O Qc7 Qe2 d6"),
    ("B22", "Sicilian: Alapin", "e4 c5 c3 d5 exd5 Qxd5 d4 Nf6 Nf3 e6 Be2 Nc6"),
    ("B25", "Sicilian: Closed", "e4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 d3 d6 Be3 e6"),
    ("C11", "French: Classical", "e4 e6 d4 d5 Nc3 Nf6 Bg5 Be7 e5 Nfd7 Bxe7 Qxe7 f4 O-O"),
    ("C02", "French: Advance", "e4 e6 d4 d5 e5 c5 c3 Nc6 Nf3 Qb6 a3 Nh6"),
    ("C05", "French: Tarrasch", "e4 e6 d4 d5 Nd2 Nf6 e5 Nfd7 Bd3 c5 c3 Nc6 Ne2 cxd4"),
    ("B18", "Caro-Kann: Classical", "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7"),
    ("B12", "Caro-Kann: Advance", "e4 c6 d4 d5 e5 Bf5 Nf3 e6 Be2 c5 Be3 Nd7"),
    ("B01", "Scandinavian Defense", "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 c6 Bc4 Bf5"),
    ("B09", "Pirc: Austrian Attack", "e4 d6 d4 Nf6 Nc3 g6 f4 Bg7 Nf3 O-O Bd3 Na6"),
    ("B06", "Modern Defense", "e4 g6 d4 Bg7 Nc3 d6 Nf3 Nf6 Be2 O-O O-O c6"),
    ("B03", "Alekhine's Defense", "e4 Nf6 e5 Nd5 d4 d6 Nf3 Bg4 Be2 e6 O-O Be7"),
    ("D37", "Queen's Gambit Declined", "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 h6 Bh4 b6"),
    ("D27", "Queen's Gambit Accepted", "d4 d5 c4 dxc4 Nf3 Nf6 e3 e6 Bxc4 c5 O-O a6"),
    ("D45", "Semi-Slav", "d4 d5 c4 c6 Nf3 Nf6 Nc3 e6 e3 Nbd7 Bd3 dxc4 Bxc4 b5"),
    ("E70", "King's Indian Defense", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6"),
    ("E40", "Nimzo-Indian Defense", "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5 Nf3 c5 O-O Nc6"),
    ("E15", "Queen's Indian Defense", "d4 Nf6 c4 e6 Nf3 b6 g3 Bb7 Bg2 Be7 O-O O-O Nc3 Ne4"),
    ("A60", "Benoni Defense", "d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 d6 e4 g6 Nf3 Bg7"),
    ("D85", "Grünfeld Defense", "d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Nf3 c5"),
    ("A87", "Dutch: Leningrad", "d4 f5 g3 Nf6 Bg2 e6 Nf3 Be7 O-O O-O c4 d6"),
    ("A28", "English: Four Knights", "c4 e5 Nc3 Nf6 Nf3 Nc6 g3 d5 cxd5 Nxd5 Bg2 Nb6"),
    ("A30", "English: Symmetrical", "c4 c5 Nf3 Nf6 Nc3 e6 g3 b6 Bg2 Bb7 O-O Be7"),
    ("A07", "Réti Opening", "Nf3 d5 g3 Nf6 Bg2 e6 O-O Be7 d3 O-O Nbd2 c5"),
    ("D02", "London System", "d4 d5 Bf4 Nf6 e3 e6 Nf3 Bd6 Bg3 O-O Bd3 c5 c3 Nc6"),
    ("A45", "Trompowsky Attack", "d4 Nf6 Bg5 Ne4 Bf4 c5 f3 Qa5+ c3 Nf6 d5 Qb6"),
    ("D05", "Colle System", "d4 d5 Nf3 Nf6 e3 e6 Bd3 c5 b3 Nc6 O-O Bd6 Bb2 O-O"),
    ("A01", "Nimzo-Larsen Attack", "b3 e5 Bb2 Nc6 e3 d5 Bb5 Bd6 f4 Qh4+ g3 Qe7"),
    ("A02", "Bird's Opening", "f4 d5 Nf3 Nf6 e3 g6 Be2 Bg7 O-O O-O d3 c5"),
    ("A00", "King's Fianchetto", "g3 d5 Bg2 Nf6 Nf3 e6 O-O Be7 d3 O-O Nbd2 c5"),
    ("B20", "Sicilian Defense", "e4 c5 Nf3 d6 Bb5+ Bd7 Bxd7+ Qxd7 c4 Nc6"),
    ("D10", "Slav Defense", "d4 d5 c4 c6 Nc3 Nf6 Nf3 dxc4 a4 Bf5 e3 e6 Bxc4 Bb4"),
]


def _key(board: Board) -> str:
    return " ".join(board.fen().split()[:3])


class OpeningBook:
    def __init__(self, lines: list[tuple[str, str, str]] = LINES):
        self._moves: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._names: dict[str, tuple[str, str, int]] = {}  # position -> (eco, name, ply)
        for eco, name, text in lines:
            board = Board()
            for ply, san in enumerate(text.split()):
                move = board.parse_san(san)
                self._moves[_key(board)][move.uci()] += 1
                board.push(move)
                self._names[_key(board)] = (eco, name, ply + 1)

    def __len__(self) -> int:
        return len(self._moves)

    def choose(self, board: Board, rng: random.Random, max_plies: int | None = None) -> Move | None:
        """A book move for this position, or None when out of book (or beyond `max_plies`)."""
        if max_plies is not None and len(board._undo) >= max_plies:
            return None
        options = self._moves.get(_key(board))
        if not options:
            return None
        ucis, weights = zip(*options.items())
        move = board.parse_uci(rng.choices(ucis, weights)[0])
        return move

    def name_at(self, board: Board) -> tuple[str, str] | None:
        """(ECO, name) if this exact position is a book position."""
        hit = self._names.get(_key(board))
        return (hit[0], hit[1]) if hit else None

    def identify(self, board: Board) -> tuple[str, str] | None:
        """(ECO, opening name) of the deepest book position reached by this game, if any."""
        replay = Board()
        best = None
        for item in board._undo:
            replay.push(item[0])
            hit = self._names.get(_key(replay))
            if hit:
                best = hit
        return (best[0], best[1]) if best else None

    def identify_moves(self, moves: list[Move]) -> tuple[str, str] | None:
        b = Board()
        for m in moves:
            b.push(m)
        return self.identify(b)


_DEFAULT: OpeningBook | None = None


def default_book() -> OpeningBook:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = OpeningBook()
    return _DEFAULT
