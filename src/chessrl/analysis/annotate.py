"""Move-by-move game analysis and automatic annotation.

For every position the scorer ranks all legal moves (side-to-move centipawns). From that one
call we know the best move, the value of the played move, and therefore how much was lost.
Classification follows win-probability loss, as on Lichess:

    best / excellent (<2%) / good (<5%) / inaccuracy (<10%) / mistake (<20%) / blunder (>=20%)
    plus "great" (the only good move) and "brilliant" (a best-move piece sacrifice)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..agents.base import SearchLimits
from ..agents.minimax import MATE
from ..engine import STARTING_FEN, WHITE, Board, Game, Move
from ..engine.constants import (
    BISHOP,
    DIAG_RAYS,
    KING,
    KING_TARGETS,
    KNIGHT,
    KNIGHT_TARGETS,
    ORTHO_RAYS,
    PAWN,
    PAWN_ATTACKS,
    QUEEN,
    ROOK,
)
from ..levels.book import default_book
from .winprob import game_accuracy, move_accuracy, win_percent

SYMBOLS = {
    "brilliant": "!!", "great": "!", "best": "", "excellent": "", "good": "",
    "inaccuracy": "?!", "mistake": "?", "blunder": "??", "book": "",
}
_VALUE = {PAWN: 1, KNIGHT: 3, BISHOP: 3, ROOK: 5, QUEEN: 9, KING: 100}
CP_MATE = 10_000


def to_cp(score: float) -> float:
    """Alpha-beta mate scores -> a large centipawn value that keeps the sign."""
    if abs(score) > MATE - 1000:
        return CP_MATE if score > 0 else -CP_MATE
    return score


def mate_in(score: float) -> int | None:
    """Full moves to mate if `score` is a mate score (positive = mating), else None."""
    if abs(score) > MATE - 1000:
        plies = MATE - int(abs(score))
        return (plies + 1) // 2
    return None


def eval_text(cp_white: float, mate: int | None = None) -> str:
    if mate is not None:
        return f"{'+' if cp_white > 0 else '-'}M{mate}"
    return f"{cp_white / 100:+.2f}"


@dataclass
class MoveAnalysis:
    ply: int
    mover: int
    san: str
    uci: str
    best_san: str
    best_uci: str
    eval_before: float  # white POV, centipawns, value of the position with best play
    eval_after: float  # white POV, after the move actually played
    cp_loss: float  # mover POV
    win_before: float  # mover POV win%
    win_after: float
    accuracy: float
    label: str
    book: str | None = None  # opening name while still in book
    mate_after: int | None = None

    @property
    def symbol(self) -> str:
        return SYMBOLS.get(self.label, "")

    @property
    def win_loss(self) -> float:
        return max(0.0, self.win_before - self.win_after)


@dataclass
class GameAnalysis:
    moves: list[MoveAnalysis] = field(default_factory=list)
    accuracy_white: float = 100.0
    accuracy_black: float = 100.0
    opening: tuple[str, str] | None = None
    final_eval: float = 0.0

    def counts(self, color: int) -> dict[str, int]:
        out: dict[str, int] = {}
        for m in self.moves:
            if m.mover == color:
                out[m.label] = out.get(m.label, 0) + 1
        return out

    def eval_series(self) -> list[float]:
        """White-POV evaluation after each ply (index 0 = start position)."""
        start = self.moves[0].eval_before if self.moves else 0.0
        return [start] + [m.eval_after for m in self.moves]

    def critical_moments(self, threshold: float = 10.0, limit: int = 5) -> list[MoveAnalysis]:
        bad = [m for m in self.moves if m.win_loss >= threshold]
        return sorted(bad, key=lambda m: -m.win_loss)[:limit]

    def to_dict(self) -> dict:
        return {
            "accuracy": {"white": round(self.accuracy_white, 1), "black": round(self.accuracy_black, 1)},
            "opening": list(self.opening) if self.opening else None,
            "counts": {"white": self.counts(WHITE), "black": self.counts(1 - WHITE)},
            "evals": [round(e, 1) for e in self.eval_series()],
            "moves": [{**asdict(m), "symbol": m.symbol} for m in self.moves],
            "critical": [m.ply for m in self.critical_moments()],
        }


# --------------------------------------------------------------------------- sacrifice test
def _lowest_attacker(board: Board, sq: int, color: int) -> int | None:
    """Value of the cheapest piece of `color` attacking `sq` (None if unattacked)."""
    sqs = board.squares
    sign = 1 if color == WHITE else -1
    best: int | None = None

    def consider(t: int) -> None:
        nonlocal best
        if best is None or _VALUE[t] < best:
            best = _VALUE[t]

    if any(sqs[s] == sign * PAWN for s in PAWN_ATTACKS[1 - color][sq]):
        consider(PAWN)
    if any(sqs[s] == sign * KNIGHT for s in KNIGHT_TARGETS[sq]):
        consider(KNIGHT)
    for rays, slider in ((DIAG_RAYS[sq], BISHOP), (ORTHO_RAYS[sq], ROOK)):
        for ray in rays:
            for s in ray:
                v = sqs[s] * sign
                if v:
                    if v == slider:
                        consider(slider)
                    elif v == QUEEN:
                        consider(QUEEN)
                    break
    if any(sqs[s] == sign * KING for s in KING_TARGETS[sq]):
        consider(KING)
    return best


def _defended(board: Board, sq: int, color: int) -> bool:
    return board.is_attacked(sq, color)


def is_sacrifice(board_before: Board, move: Move) -> bool:
    """Does the move put a piece where it can be won for less than it is worth?"""
    sqs = board_before.squares
    piece = abs(sqs[move.from_sq])
    if piece in (PAWN, KING) or move.promotion:
        return False
    gained = _VALUE.get(abs(sqs[move.to_sq]), 0)
    them = 1 - board_before.turn
    board_before.push(move)
    try:
        attacker = _lowest_attacker(board_before, move.to_sq, them)
        defended = _defended(board_before, move.to_sq, 1 - them)
    finally:
        board_before.pop()
    if attacker is None:
        return False
    loses = _VALUE[piece] - gained
    return loses >= 2 and (not defended or attacker < _VALUE[piece] - 1)


# ----------------------------------------------------------------------------- the analyser
def classify(win_loss: float, is_best: bool, only_move: bool, sacrifice: bool, win_before: float,
             win_after: float) -> str:
    if is_best and sacrifice and win_before < 85 and win_after >= 45:
        return "brilliant"
    if is_best and only_move and win_before < 90:
        return "great"
    if is_best:
        return "best"
    if win_loss < 2:
        return "excellent"
    if win_loss < 5:
        return "good"
    if win_loss < 10:
        return "inaccuracy"
    if win_loss < 20:
        return "mistake"
    return "blunder"


def analyse_position(scorer, board: Board, move: Move, limits: SearchLimits | None = None,
                     ply: int = 0, in_book: str | None = None) -> MoveAnalysis:
    """Analyse one played move given the position before it."""
    scores = scorer.score_moves(board, limits)
    by_move = dict(scores)
    best_move, best_score = scores[0]
    played = by_move.get(move, min(s for _, s in scores))
    mover = board.turn
    sign = 1 if mover == WHITE else -1
    win_b, win_a = win_percent(to_cp(best_score)), win_percent(to_cp(played))
    second = scores[1][1] if len(scores) > 1 else -CP_MATE
    only_move = len(scores) > 1 and win_b - win_percent(to_cp(second)) >= 15
    is_best = move == best_move or abs(played - best_score) < 1e-9
    label = classify(max(0.0, win_b - win_a), is_best, only_move,
                     is_best and is_sacrifice(board.copy(), move), win_b, win_a)
    if in_book:
        label = "book"
    return MoveAnalysis(
        ply=ply, mover=mover, san=board.san(move), uci=move.uci(), best_san=board.san(best_move),
        best_uci=best_move.uci(), eval_before=sign * to_cp(best_score), eval_after=sign * to_cp(played),
        cp_loss=max(0.0, to_cp(best_score) - to_cp(played)), win_before=win_b, win_after=win_a,
        accuracy=move_accuracy(win_b, win_a), label=label, book=in_book,
        mate_after=mate_in(played) if abs(played) > MATE - 1000 else None,
    )


def analyse_game(game: Game, scorer, limits: SearchLimits | None = None, use_book: bool = True,
                 progress=None) -> GameAnalysis:
    """Full analysis of a finished (or partial) game."""
    book = default_book() if use_book and game.start_fen == STARTING_FEN else None
    board = Board(game.start_fen)
    out = GameAnalysis()
    in_book = book is not None
    for ply, move in enumerate(game.moves):
        opening = None
        if in_book:
            board.push(move)
            hit = book.name_at(board)
            board.pop()
            if hit:
                opening = hit[1]
            else:
                in_book = False
        ma = analyse_position(scorer, board, move, limits, ply, opening)
        out.moves.append(ma)
        board.push(move)
        if progress:
            progress(ply + 1, len(game.moves))
    accs = {WHITE: [], 1 - WHITE: []}
    for m in out.moves:
        if m.label != "book":
            accs[m.mover].append(m.accuracy)
    out.accuracy_white, out.accuracy_black = game_accuracy(accs[WHITE]), game_accuracy(accs[1 - WHITE])
    out.opening = book.identify(board) if book is not None else None
    end = board.outcome()
    if end is not None:
        out.final_eval = CP_MATE * (0 if end.winner is None else (1 if end.winner == WHITE else -1))
    elif out.moves:
        out.final_eval = out.moves[-1].eval_after
    return out


def comment_for(m: MoveAnalysis) -> str | None:
    """Human-readable PGN comment for notable moves (None for unremarkable ones)."""
    if m.label == "book" and m.ply >= 1 and m.book:
        return f"Book: {m.book}"
    if m.label in ("brilliant", "great"):
        return f"{m.symbol} {m.label.capitalize()} move ({eval_text(m.eval_after, m.mate_after)})."
    if m.label in ("inaccuracy", "mistake", "blunder"):
        return (f"{m.symbol} {m.label.capitalize()} ({eval_text(m.eval_before)} -> "
                f"{eval_text(m.eval_after, m.mate_after)}). Best was {m.best_san}.")
    return None


def annotate_game(game: Game, analysis: GameAnalysis, include_book: bool = False) -> Game:
    """Write the analysis into the game: comments on notable moves and opening headers."""
    for m in analysis.moves:
        c = comment_for(m)
        if c and (include_book or m.label != "book"):
            game.comments[m.ply] = c
    if analysis.opening:
        game.headers["ECO"], game.headers["Opening"] = analysis.opening
    game.headers["AccuracyWhite"] = f"{analysis.accuracy_white:.1f}"
    game.headers["AccuracyBlack"] = f"{analysis.accuracy_black:.1f}"
    return game


def eval_graph_svg(analysis: GameAnalysis, width: int = 640, height: int = 220, dark: bool = False) -> str:
    """White-POV evaluation after every ply, as a chart (win-probability scaled, 0-100%)."""
    from ..eval.plots import line_chart_svg

    pts = [(i, win_percent(e)) for i, e in enumerate(analysis.eval_series())]
    return line_chart_svg({"White win %": pts}, title="Evaluation", xlabel="ply", ylabel="White win %",
                          width=width, height=height, dark=dark)
