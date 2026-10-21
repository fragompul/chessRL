"""PGN reading and writing.

`Game.comments` maps a ply index (0 = White's first move) to a free-text comment, which is
where the automatic game annotator will write its notes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .board import Board
from .constants import STARTING_FEN
from .move import Move

_SEVEN_TAG_ROSTER = ("Event", "Site", "Date", "Round", "White", "Black", "Result")
_RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}
_HEADER_RE = re.compile(r'^\[(\w+)\s+"((?:[^"\\]|\\.)*)"\]\s*$')
_MOVE_NUMBER_RE = re.compile(r"^\d+\.+$")


@dataclass
class Game:
    headers: dict[str, str] = field(default_factory=dict)
    moves: list[Move] = field(default_factory=list)
    result: str = "*"
    start_fen: str = STARTING_FEN
    comments: dict[int, str] = field(default_factory=dict)

    def board_at(self, ply: int | None = None) -> Board:
        """The position after `ply` half-moves (default: the final position)."""
        board = Board(self.start_fen)
        for m in self.moves[: len(self.moves) if ply is None else ply]:
            board.push(m)
        return board

    def to_pgn(self) -> str:
        headers = {
            "Event": "?",
            "Site": "?",
            "Date": "????.??.??",
            "Round": "?",
            "White": "?",
            "Black": "?",
        }
        headers.update(self.headers)
        headers["Result"] = self.result
        if self.start_fen != STARTING_FEN:
            headers["SetUp"], headers["FEN"] = "1", self.start_fen
        ordered = [k for k in _SEVEN_TAG_ROSTER if k in headers]
        ordered += [k for k in headers if k not in _SEVEN_TAG_ROSTER]
        lines = [f'[{k} "{headers[k].replace(chr(34), chr(92) + chr(34))}"]' for k in ordered]

        board = Board(self.start_fen)
        tokens: list[str] = []
        for ply, move in enumerate(self.moves):
            if board.turn == 0:
                tokens.append(f"{board.fullmove}.")
            elif ply == 0:
                tokens.append(f"{board.fullmove}...")
            tokens.append(board.san(move))
            if ply in self.comments:
                tokens.append("{" + self.comments[ply].replace("}", ")") + "}")
                if board.turn == 0 and ply + 1 < len(self.moves):
                    tokens.append(f"{board.fullmove}...")
            board.push(move)
        tokens.append(self.result)

        body, line = [], ""
        for tok in tokens:
            if line and len(line) + 1 + len(tok) > 80:
                body.append(line)
                line = tok
            else:
                line = f"{line} {tok}" if line else tok
        body.append(line)
        return "\n".join(lines) + "\n\n" + "\n".join(body) + "\n"


def _tokenize(movetext: str) -> list[str]:
    """Split movetext into tokens, dropping variations, NAGs and ';' comments; keeping {...}."""
    tokens: list[str] = []
    i, n, depth = 0, len(movetext), 0
    while i < n:
        ch = movetext[i]
        if ch == "{":
            j = movetext.find("}", i)
            j = n if j == -1 else j
            if depth == 0:
                tokens.append(movetext[i : j + 1])
            i = j + 1
        elif ch == ";":
            j = movetext.find("\n", i)
            i = n if j == -1 else j + 1
        elif ch == "(":
            depth, i = depth + 1, i + 1
        elif ch == ")":
            depth, i = max(0, depth - 1), i + 1
        elif ch.isspace():
            i += 1
        else:
            j = i
            while j < n and not movetext[j].isspace() and movetext[j] not in "{;()":
                j += 1
            if depth == 0 and not movetext[i:j].startswith("$"):
                tokens.append(movetext[i:j])
            i = j
    return tokens


def parse_game(text: str) -> Game:
    headers: dict[str, str] = {}
    movetext_lines: list[str] = []
    for line in text.splitlines():
        m = _HEADER_RE.match(line.strip())
        if m and not movetext_lines:
            headers[m.group(1)] = m.group(2).replace('\\"', '"')
        else:
            movetext_lines.append(line)

    start_fen = headers.get("FEN", STARTING_FEN) if headers.get("SetUp") == "1" else STARTING_FEN
    game = Game(headers=headers, start_fen=start_fen, result=headers.get("Result", "*"))
    board = Board(start_fen)
    for tok in _tokenize("\n".join(movetext_lines)):
        if tok.startswith("{"):
            if game.moves:
                game.comments[len(game.moves) - 1] = tok[1:-1].strip()
        elif tok in _RESULTS:
            game.result = tok
        elif _MOVE_NUMBER_RE.match(tok):
            continue
        else:
            move = board.parse_san(tok)
            game.moves.append(move)
            board.push(move)
    return game


def read_pgn(text: str) -> list[Game]:
    """Parse every game in a PGN string."""
    chunks = re.split(r"(?m)^(?=\[Event )", text)
    return [parse_game(c) for c in chunks if c.strip()]
