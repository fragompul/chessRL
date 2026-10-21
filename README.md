# chessRL

A chess engine written **from scratch** and a reinforcement-learning agent that learns to play
it **by self-play**. The long-term goal is a complete chess product: selectable Elo levels with
human-like timing and mistakes, a web interface, automatic game annotation, and the ability to
play on a **real board** through a camera (YOLO-based vision).

> Status: **Phase 1 complete** (rules engine). See the [roadmap](docs/ROADMAP.md).

## What exists today

`chessrl.engine` — a full, tested rules engine:

- legal move generation, make/unmake, castling, en passant, promotion
- checkmate, stalemate, insufficient material, fifty-move rule, threefold repetition
- FEN, UCI, SAN (with disambiguation) and PGN read/write (with per-move comments)
- Zobrist hashing
- validated with **perft** on six reference positions (up to 4.08 M nodes) and a
  **differential test against python-chess** over random games

```python
from chessrl.engine import Board

board = Board()
board.push(board.parse_san("e4"))
print(board)                       # ASCII board
print([m.uci() for m in board.legal_moves()][:5])
print(board.outcome())             # None while the game is running
```

## Install & test

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                    # full suite (~20 s, includes depth-4 perft)
pytest -m "not slow"      # quick
ruff check .
```

## Project docs

- [Roadmap](docs/ROADMAP.md) — phases and status
- [Ideas backlog](docs/IDEAS.md) — every idea, tracked until built
- [Decision log](docs/DECISIONS.md) — why things are the way they are

## Author

Francisco Javier Gómez Pulido — AI Lead & Data Scientist. MIT licensed.
