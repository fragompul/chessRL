"""The chess board: state, move generation, make/unmake, game outcome, FEN and SAN.

Design goal for this first version is *correctness and clarity*; it is validated against
perft counts and differentially against python-chess. Speed work (bitboards / numba / native
core) is tracked in docs/ROADMAP.md and must keep these tests green.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .constants import (
    A1,
    A8,
    BISHOP,
    BLACK,
    C1,
    C8,
    CASTLE_BK,
    CASTLE_BQ,
    CASTLE_KEEP,
    CASTLE_WK,
    CASTLE_WQ,
    D1,
    D8,
    DIAG_RAYS,
    E1,
    E8,
    F1,
    F8,
    FILE_NAMES,
    G1,
    G8,
    H1,
    H8,
    KING,
    KING_TARGETS,
    KNIGHT,
    KNIGHT_TARGETS,
    LETTER_TO_TYPE,
    ORTHO_RAYS,
    PAWN,
    PAWN_ATTACKS,
    PIECE_LETTERS,
    QUEEN,
    RANK_NAMES,
    ROOK,
    STARTING_FEN,
    WHITE,
    ZOBRIST_CASTLING,
    ZOBRIST_PIECE,
    ZOBRIST_TURN,
    file_of,
    parse_square,
    rank_of,
    square_name,
)
from .move import Move

_PROMOTIONS = (QUEEN, ROOK, BISHOP, KNIGHT)


class Termination(Enum):
    CHECKMATE = "checkmate"
    STALEMATE = "stalemate"
    INSUFFICIENT_MATERIAL = "insufficient material"
    FIFTY_MOVES = "fifty-move rule"
    THREEFOLD_REPETITION = "threefold repetition"


@dataclass(frozen=True)
class Outcome:
    termination: Termination
    winner: int | None  # WHITE, BLACK, or None for a draw

    @property
    def result(self) -> str:
        return "1/2-1/2" if self.winner is None else ("1-0" if self.winner == WHITE else "0-1")


class Board:
    """Mutable chess position with a move stack (push / pop)."""

    def __init__(self, fen: str = STARTING_FEN):
        self.set_fen(fen)

    # ------------------------------------------------------------------ FEN
    def set_fen(self, fen: str) -> None:
        parts = fen.split()
        if len(parts) == 4:
            parts += ["0", "1"]
        if len(parts) != 6:
            raise ValueError(f"FEN needs 4 or 6 fields: {fen!r}")
        placement, turn, castling, ep, halfmove, fullmove = parts

        rows = placement.split("/")
        if len(rows) != 8:
            raise ValueError(f"FEN placement needs 8 ranks: {fen!r}")
        squares = [0] * 64
        for i, row in enumerate(rows):
            rank, file = 7 - i, 0
            for ch in row:
                if ch.isdigit():
                    file += int(ch)
                else:
                    ptype = LETTER_TO_TYPE.get(ch.upper())
                    if ptype is None or file > 7:
                        raise ValueError(f"bad FEN placement: {fen!r}")
                    squares[rank * 8 + file] = ptype if ch.isupper() else -ptype
                    file += 1
            if file != 8:
                raise ValueError(f"FEN rank does not have 8 files: {row!r}")

        kings = {WHITE: [], BLACK: []}
        for sq, p in enumerate(squares):
            if abs(p) == KING:
                kings[WHITE if p > 0 else BLACK].append(sq)
        if len(kings[WHITE]) != 1 or len(kings[BLACK]) != 1:
            raise ValueError("each side needs exactly one king")

        if turn not in ("w", "b"):
            raise ValueError(f"bad side to move: {turn!r}")
        rights = 0
        if castling != "-":
            for ch in castling:
                bit = {"K": CASTLE_WK, "Q": CASTLE_WQ, "k": CASTLE_BK, "q": CASTLE_BQ}.get(ch)
                if bit is None:
                    raise ValueError(f"bad castling field: {castling!r}")
                rights |= bit

        self.squares = squares
        self.king_sq = [kings[WHITE][0], kings[BLACK][0]]
        self.turn = WHITE if turn == "w" else BLACK
        self.castling = rights
        self.ep_square: int | None = None if ep == "-" else parse_square(ep)
        self.halfmove = int(halfmove)
        self.fullmove = int(fullmove)
        self.hash = self._compute_hash()
        self._undo: list[tuple] = []
        self._keys: list[tuple[int, int | None]] = [self._position_key()]

    def fen(self) -> str:
        rows = []
        for rank in range(7, -1, -1):
            row, empty = "", 0
            for file in range(8):
                p = self.squares[rank * 8 + file]
                if p == 0:
                    empty += 1
                    continue
                if empty:
                    row, empty = row + str(empty), 0
                letter = PIECE_LETTERS[abs(p)]
                row += letter if p > 0 else letter.lower()
            rows.append(row + (str(empty) if empty else ""))
        rights = "".join(
            c
            for bit, c in ((CASTLE_WK, "K"), (CASTLE_WQ, "Q"), (CASTLE_BK, "k"), (CASTLE_BQ, "q"))
            if self.castling & bit
        )
        ep = square_name(self.ep_square) if self.ep_square is not None else "-"
        return (
            f"{'/'.join(rows)} {'w' if self.turn == WHITE else 'b'} "
            f"{rights or '-'} {ep} {self.halfmove} {self.fullmove}"
        )

    def copy(self) -> Board:
        b = Board.__new__(Board)
        b.squares = self.squares[:]
        b.king_sq = self.king_sq[:]
        b.turn = self.turn
        b.castling = self.castling
        b.ep_square = self.ep_square
        b.halfmove = self.halfmove
        b.fullmove = self.fullmove
        b.hash = self.hash
        b._undo = self._undo[:]
        b._keys = self._keys[:]
        return b

    def __repr__(self) -> str:
        return f"Board({self.fen()!r})"

    def __str__(self) -> str:
        glyphs = {0: "."}
        for t, letter in PIECE_LETTERS.items():
            glyphs[t], glyphs[-t] = letter, letter.lower()
        lines = []
        for rank in range(7, -1, -1):
            cells = " ".join(glyphs[self.squares[rank * 8 + f]] for f in range(8))
            lines.append(f"{rank + 1}  {cells}")
        lines.append("   " + " ".join(FILE_NAMES))
        return "\n".join(lines)

    # ----------------------------------------------------------------- hash
    def _compute_hash(self) -> int:
        h = ZOBRIST_CASTLING[self.castling]
        for sq, p in enumerate(self.squares):
            if p:
                h ^= ZOBRIST_PIECE[p + 6][sq]
        if self.turn == BLACK:
            h ^= ZOBRIST_TURN
        return h

    def _position_key(self) -> tuple[int, int | None]:
        """Identity of a position for repetition: hash plus the e.p. square if capturable."""
        ep = self.ep_square
        if ep is not None:
            sign = 1 if self.turn == WHITE else -1
            attackers = PAWN_ATTACKS[1 - self.turn][ep]
            if not any(self.squares[s] == sign * PAWN for s in attackers):
                ep = None
        return (self.hash, ep)

    # -------------------------------------------------------------- attacks
    def is_attacked(self, sq: int, by: int) -> bool:
        """Is `sq` attacked by any piece of colour `by`?"""
        sqs = self.squares
        sign = 1 if by == WHITE else -1
        for s in PAWN_ATTACKS[1 - by][sq]:
            if sqs[s] == sign * PAWN:
                return True
        for s in KNIGHT_TARGETS[sq]:
            if sqs[s] == sign * KNIGHT:
                return True
        for s in KING_TARGETS[sq]:
            if sqs[s] == sign * KING:
                return True
        for ray in DIAG_RAYS[sq]:
            for s in ray:
                v = sqs[s]
                if v:
                    if v == sign * BISHOP or v == sign * QUEEN:
                        return True
                    break
        for ray in ORTHO_RAYS[sq]:
            for s in ray:
                v = sqs[s]
                if v:
                    if v == sign * ROOK or v == sign * QUEEN:
                        return True
                    break
        return False

    def is_check(self) -> bool:
        return self.is_attacked(self.king_sq[self.turn], 1 - self.turn)

    # ------------------------------------------------------------ move gen
    def pseudo_legal_moves(self) -> list[Move]:
        moves: list[Move] = []
        sqs = self.squares
        us = self.turn
        sign = 1 if us == WHITE else -1
        for frm in range(64):
            t = sqs[frm] * sign
            if t <= 0:
                continue
            if t == PAWN:
                self._pawn_moves(frm, us, sign, moves)
            elif t == KNIGHT:
                for to in KNIGHT_TARGETS[frm]:
                    if sqs[to] * sign <= 0:
                        moves.append(Move(frm, to))
            elif t == KING:
                for to in KING_TARGETS[frm]:
                    if sqs[to] * sign <= 0:
                        moves.append(Move(frm, to))
                self._castling_moves(frm, us, moves)
            else:
                rays = ()
                if t in (BISHOP, QUEEN):
                    rays += DIAG_RAYS[frm]
                if t in (ROOK, QUEEN):
                    rays += ORTHO_RAYS[frm]
                for ray in rays:
                    for to in ray:
                        v = sqs[to] * sign
                        if v > 0:
                            break
                        moves.append(Move(frm, to))
                        if v < 0:
                            break
        return moves

    def _pawn_moves(self, frm: int, us: int, sign: int, moves: list[Move]) -> None:
        sqs = self.squares
        fwd = 8 * sign
        start_rank, promo_rank = (1, 7) if us == WHITE else (6, 0)
        to = frm + fwd
        if sqs[to] == 0:
            if rank_of(to) == promo_rank:
                moves.extend(Move(frm, to, p) for p in _PROMOTIONS)
            else:
                moves.append(Move(frm, to))
                if rank_of(frm) == start_rank and sqs[to + fwd] == 0:
                    moves.append(Move(frm, to + fwd))
        for to in PAWN_ATTACKS[us][frm]:
            target = sqs[to] * sign
            if target < 0:
                if rank_of(to) == promo_rank:
                    moves.extend(Move(frm, to, p) for p in _PROMOTIONS)
                else:
                    moves.append(Move(frm, to))
            elif target == 0 and to == self.ep_square:
                moves.append(Move(frm, to))

    def _castling_moves(self, frm: int, us: int, moves: list[Move]) -> None:
        sqs = self.squares
        if us == WHITE:
            home, ks, qs, rights = E1, CASTLE_WK, CASTLE_WQ, self.castling
            k_rook, q_rook, rook = H1, A1, ROOK
            k_empty, q_empty = (F1, G1), (D1, C1, 1)
            k_safe, q_safe = (F1, G1), (D1, C1)
            k_to, q_to = G1, C1
        else:
            home, ks, qs, rights = E8, CASTLE_BK, CASTLE_BQ, self.castling
            k_rook, q_rook, rook = H8, A8, -ROOK
            k_empty, q_empty = (F8, G8), (D8, C8, 57)
            k_safe, q_safe = (F8, G8), (D8, C8)
            k_to, q_to = G8, C8
        if frm != home or not (rights & (ks | qs)):
            return
        them = 1 - us
        if self.is_attacked(home, them):
            return
        if rights & ks and sqs[k_rook] == rook and all(sqs[s] == 0 for s in k_empty):
            if not any(self.is_attacked(s, them) for s in k_safe):
                moves.append(Move(home, k_to))
        if rights & qs and sqs[q_rook] == rook and all(sqs[s] == 0 for s in q_empty):
            if not any(self.is_attacked(s, them) for s in q_safe):
                moves.append(Move(home, q_to))

    def legal_moves(self) -> list[Move]:
        us, them = self.turn, 1 - self.turn
        legal = []
        for m in self.pseudo_legal_moves():
            self.push(m)
            if not self.is_attacked(self.king_sq[us], them):
                legal.append(m)
            self.pop()
        return legal

    def is_legal(self, move: Move) -> bool:
        return move in self.legal_moves()

    # ----------------------------------------------------- make / unmake
    def push(self, move: Move) -> None:
        sqs = self.squares
        frm, to, promo = move
        piece = sqs[frm]
        ptype = abs(piece)
        color = WHITE if piece > 0 else BLACK
        captured = sqs[to]
        cap_sq = to
        h = self.hash ^ ZOBRIST_CASTLING[self.castling]

        if ptype == PAWN and to == self.ep_square and captured == 0 and file_of(frm) != file_of(to):
            cap_sq = to - 8 if color == WHITE else to + 8
            captured = sqs[cap_sq]

        self._undo.append(
            (move, piece, captured, cap_sq, self.castling, self.ep_square, self.halfmove, self.hash)
        )

        if captured:
            sqs[cap_sq] = 0
            h ^= ZOBRIST_PIECE[captured + 6][cap_sq]

        sqs[frm] = 0
        h ^= ZOBRIST_PIECE[piece + 6][frm]
        placed = (promo if color == WHITE else -promo) if promo else piece
        sqs[to] = placed
        h ^= ZOBRIST_PIECE[placed + 6][to]

        if ptype == KING:
            self.king_sq[color] = to
            if abs(to - frm) == 2:  # castling: also move the rook
                r_from, r_to = (to + 1, to - 1) if to > frm else (to - 2, to + 1)
                rook = sqs[r_from]
                sqs[r_from], sqs[r_to] = 0, rook
                h ^= ZOBRIST_PIECE[rook + 6][r_from] ^ ZOBRIST_PIECE[rook + 6][r_to]

        self.castling &= CASTLE_KEEP[frm] & CASTLE_KEEP[to]
        h ^= ZOBRIST_CASTLING[self.castling] ^ ZOBRIST_TURN

        self.ep_square = (frm + to) // 2 if ptype == PAWN and abs(to - frm) == 16 else None
        self.halfmove = 0 if (ptype == PAWN or captured) else self.halfmove + 1
        if color == BLACK:
            self.fullmove += 1
        self.turn = 1 - color
        self.hash = h
        self._keys.append(self._position_key())

    def pop(self) -> Move:
        move, piece, captured, cap_sq, castling, ep, halfmove, h = self._undo.pop()
        self._keys.pop()
        sqs = self.squares
        frm, to, _ = move
        color = WHITE if piece > 0 else BLACK

        sqs[to] = 0
        sqs[frm] = piece
        if captured:
            sqs[cap_sq] = captured
        if abs(piece) == KING:
            self.king_sq[color] = frm
            if abs(to - frm) == 2:
                r_from, r_to = (to + 1, to - 1) if to > frm else (to - 2, to + 1)
                sqs[r_from], sqs[r_to] = sqs[r_to], 0

        self.castling, self.ep_square, self.halfmove, self.hash = castling, ep, halfmove, h
        if color == BLACK:
            self.fullmove -= 1
        self.turn = color
        return move

    # --------------------------------------------------------------- perft
    def perft(self, depth: int) -> int:
        """Count leaf nodes of the legal-move tree: the standard movegen correctness test."""
        if depth == 0:
            return 1
        moves = self.legal_moves()
        if depth == 1:
            return len(moves)
        total = 0
        for m in moves:
            self.push(m)
            total += self.perft(depth - 1)
            self.pop()
        return total

    # ------------------------------------------------------------- outcome
    def is_checkmate(self) -> bool:
        return self.is_check() and not self.legal_moves()

    def is_stalemate(self) -> bool:
        return not self.is_check() and not self.legal_moves()

    def is_insufficient_material(self) -> bool:
        minors = []
        for sq, p in enumerate(self.squares):
            t = abs(p)
            if t in (PAWN, ROOK, QUEEN):
                return False
            if t in (KNIGHT, BISHOP):
                minors.append((t, (file_of(sq) + rank_of(sq)) % 2))
        if len(minors) <= 1:
            return True
        # Only bishops, all on the same colour of square: no mate is possible.
        return all(t == BISHOP for t, _ in minors) and len({c for _, c in minors}) == 1

    def repetition_count(self) -> int:
        """How many times the current position has occurred (including now)."""
        key = self._keys[-1]
        window = self._keys[-(self.halfmove + 1) :]
        return sum(1 for k in window if k == key)

    def outcome(self) -> Outcome | None:
        """Automatic game result, or None if the game goes on.

        The fifty-move rule and threefold repetition are applied automatically (not merely
        claimable): that is what self-play training needs.
        """
        if not self.legal_moves():
            if self.is_check():
                return Outcome(Termination.CHECKMATE, 1 - self.turn)
            return Outcome(Termination.STALEMATE, None)
        if self.is_insufficient_material():
            return Outcome(Termination.INSUFFICIENT_MATERIAL, None)
        if self.halfmove >= 100:
            return Outcome(Termination.FIFTY_MOVES, None)
        if self.repetition_count() >= 3:
            return Outcome(Termination.THREEFOLD_REPETITION, None)
        return None

    def is_game_over(self) -> bool:
        return self.outcome() is not None

    # ----------------------------------------------------------- notation
    def parse_uci(self, text: str) -> Move:
        move = Move.from_uci(text)
        if move not in self.legal_moves():
            raise ValueError(f"illegal move {text!r} in {self.fen()}")
        return move

    def san(self, move: Move) -> str:
        """Standard algebraic notation for a legal move in the current position."""
        frm, to, promo = move
        piece = self.squares[frm]
        ptype = abs(piece)
        if ptype == KING and abs(to - frm) == 2:
            text = "O-O" if to > frm else "O-O-O"
        else:
            capture = self.squares[to] != 0 or (ptype == PAWN and file_of(frm) != file_of(to))
            if ptype == PAWN:
                text = (FILE_NAMES[file_of(frm)] if capture else "") + ("x" if capture else "")
                text += square_name(to)
                if promo:
                    text += "=" + PIECE_LETTERS[promo]
            else:
                others = [
                    m
                    for m in self.legal_moves()
                    if m.to_sq == to and m.from_sq != frm and abs(self.squares[m.from_sq]) == ptype
                ]
                dis = ""
                if others:
                    if all(file_of(m.from_sq) != file_of(frm) for m in others):
                        dis = FILE_NAMES[file_of(frm)]
                    elif all(rank_of(m.from_sq) != rank_of(frm) for m in others):
                        dis = RANK_NAMES[rank_of(frm)]
                    else:
                        dis = square_name(frm)
                text = PIECE_LETTERS[ptype] + dis + ("x" if capture else "") + square_name(to)
        self.push(move)
        if self.is_check():
            text += "#" if not self.legal_moves() else "+"
        self.pop()
        return text

    def parse_san(self, text: str) -> Move:
        def norm(s: str) -> str:
            return s.replace("0", "O").rstrip("+#!?").replace("e.p.", "").strip()

        wanted = norm(text)
        legal = self.legal_moves()
        for m in legal:
            if norm(self.san(m)) == wanted:
                return m
        loose = wanted.replace("x", "").replace("=", "")
        matches = [m for m in legal if norm(self.san(m)).replace("x", "").replace("=", "") == loose]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"illegal or ambiguous SAN {text!r} in {self.fen()}")
