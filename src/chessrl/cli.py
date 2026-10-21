"""Command-line interface: `python -m chessrl <command>`."""

from __future__ import annotations

import argparse
import sys

from .engine import BLACK, WHITE, Board

_GLYPHS = {
    1: "♙", 2: "♘", 3: "♗", 4: "♖", 5: "♕", 6: "♔",
    -1: "♟", -2: "♞", -3: "♝", -4: "♜", -5: "♛", -6: "♚",
}


def render(board: Board, perspective: int = WHITE) -> str:
    ranks = range(7, -1, -1) if perspective == WHITE else range(8)
    files = range(8) if perspective == WHITE else range(7, -1, -1)
    lines = []
    for r in ranks:
        cells = [_GLYPHS.get(board.squares[r * 8 + f], "·") for f in files]
        lines.append(f"{r + 1}  " + " ".join(cells))
    letters = "abcdefgh" if perspective == WHITE else "hgfedcba"
    lines.append("   " + " ".join(letters))
    return "\n".join(lines)


def cmd_play(args: argparse.Namespace) -> int:
    from .agents.registry import make_agent
    from .play import GameSession, TimeControl

    color = {"white": WHITE, "black": BLACK}.get(args.color)
    if args.color == "random":
        import random

        color = random.choice([WHITE, BLACK])
    ai = make_agent(args.opponent, seed=args.seed)
    tc = TimeControl.parse(args.tc) if args.tc else None
    session = GameSession(ai, color, tc, realtime=not args.fast,
                          headers={"White": "You" if color == WHITE else ai.name,
                                   "Black": "You" if color == BLACK else ai.name})
    print(f"You are {'White' if color == WHITE else 'Black'} vs {ai.name}. "
          "Commands: move (SAN or UCI), undo, hint, resign, draw, pgn, fen, quit")
    while not session.over:
        print("\n" + render(session.board, color))
        if session.clock:
            print(f"⏱  White {session.time_left(WHITE):6.1f}s   Black {session.time_left(BLACK):6.1f}s")
        print(session.status_text())
        if session.is_ai_turn():
            move = session.play_ai()
            print(f"{ai.name} plays {move.uci()}")
            continue
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if text in ("quit", "exit"):
            return 0
        if text == "undo":
            print(f"undid {session.takeback()} plies")
        elif text == "hint":
            h = session.hint()
            print("hint:", session.board.san(h) if h else "-")
        elif text == "resign":
            session.resign(color)
        elif text == "draw":
            print("draw accepted" if session.offer_draw(color) else "draw declined")
        elif text == "pgn":
            print(session.to_game().to_pgn())
        elif text == "fen":
            print(session.board.fen())
        else:
            try:
                session.play_human(text)
            except ValueError as e:
                print("!", e)
    print("\n" + render(session.board, color))
    print("Game over:", session.status_text())
    print(session.to_game().to_pgn())
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .web.server import serve

    serve(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chessrl", description="chessRL command line")
    sub = p.add_subparsers(dest="command", required=True)

    play = sub.add_parser("play", help="play against an agent in the terminal")
    play.add_argument("--opponent", default="ab2", help="random | greedy | abN | level:<elo> | ckpt:<path>")
    play.add_argument("--color", choices=["white", "black", "random"], default="white")
    play.add_argument("--tc", default=None, help="time control like 5+3 (minutes+increment)")
    play.add_argument("--seed", type=int, default=None)
    play.add_argument("--fast", action="store_true", help="no artificial thinking delay")
    play.set_defaults(func=cmd_play)

    serve_p = sub.add_parser("serve", help="start the web interface")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--no-browser", action="store_true")
    serve_p.set_defaults(func=cmd_serve)

    from . import cli_ext

    cli_ext.register(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
