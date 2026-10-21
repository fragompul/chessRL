"""Speak the UCI protocol so any chessRL agent can play in Arena, Cute Chess, lichess-bot, ...

    python -m chessrl uci --agent level:1400          # humanised level
    python -m chessrl uci --agent ckpt:runs/main/ckpt/last_weights.pt@200
    python -m chessrl uci --agent level:Club --realtime   # sleep like a human would
"""

from __future__ import annotations

import sys
import time
from typing import TextIO

from .agents.base import SearchLimits
from .engine import STARTING_FEN, Board


def _limits(params: dict[str, int], board: Board) -> SearchLimits:
    if "movetime" in params:
        return SearchLimits(time=params["movetime"] / 1000.0)
    if "nodes" in params:
        return SearchLimits(nodes=params["nodes"])
    if "depth" in params:
        return SearchLimits(depth=params["depth"])
    mine = "wtime" if board.turn == 0 else "btime"
    inc = params.get("winc" if board.turn == 0 else "binc", 0)
    if mine in params:
        left = params[mine] / 1000.0
        return SearchLimits(time=max(0.05, min(left / 30.0 + inc / 1000.0 * 0.8, left * 0.25)))
    return SearchLimits(time=1.0)


def parse_position(tokens: list[str]) -> Board:
    """`position startpos|fen <fen> [moves m1 m2 ...]` -> Board."""
    i = 1
    if tokens[i] == "startpos":
        board, i = Board(STARTING_FEN), i + 1
    elif tokens[i] == "fen":
        j = i + 1
        end = tokens.index("moves") if "moves" in tokens else len(tokens)
        board, i = Board(" ".join(tokens[j:end])), end
    else:
        raise ValueError(f"bad position command: {tokens}")
    if i < len(tokens) and tokens[i] == "moves":
        for uci in tokens[i + 1 :]:
            board.push(board.parse_uci(uci))
    return board


def run_uci(agent, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout, realtime: bool = False,
            name: str = "chessRL") -> None:
    board = Board()

    def out(line: str) -> None:
        stdout.write(line + "\n")
        stdout.flush()

    for raw in stdin:
        tokens = raw.split()
        if not tokens:
            continue
        cmd = tokens[0]
        if cmd == "uci":
            out(f"id name {name} ({getattr(agent, 'name', 'agent')})")
            out("id author Francisco Javier Gómez Pulido")
            out("uciok")
        elif cmd == "isready":
            out("readyok")
        elif cmd == "ucinewgame":
            board = Board()
        elif cmd == "position":
            try:
                board = parse_position(tokens)
            except ValueError as e:
                out(f"info string {e}")
        elif cmd == "go":
            params: dict[str, int] = {}
            it = iter(tokens[1:])
            for key in it:
                if key in ("wtime", "btime", "winc", "binc", "movetime", "depth", "nodes", "movestogo"):
                    try:
                        params[key] = int(next(it))
                    except (StopIteration, ValueError):
                        break
            if not board.legal_moves():
                out("bestmove 0000")
                continue
            t0 = time.perf_counter()
            move = agent.select_move(board.copy(), _limits(params, board))
            if realtime and hasattr(agent, "think_seconds"):
                mine = params.get("wtime" if board.turn == 0 else "btime")
                inc = params.get("winc" if board.turn == 0 else "binc", 0) / 1000.0
                wait = agent.think_seconds(board, None if mine is None else mine / 1000.0, inc)
                time.sleep(max(0.0, wait - (time.perf_counter() - t0)))
            out(f"bestmove {move.uci()}")
        elif cmd == "quit":
            break
        # `stop`, `ponderhit`, `setoption`, `debug`: accepted and ignored (searches are synchronous)
