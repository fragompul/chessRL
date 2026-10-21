"""Drive any UCI engine (Stockfish, Lc0, ...) as an `Agent`.

Used to calibrate our Elo against Stockfish with `UCI_LimitStrength` / `UCI_Elo`, and by the
tests with a tiny fake engine.
"""

from __future__ import annotations

import shutil
import subprocess
import threading

from ..agents.base import SearchLimits
from ..engine import Board, Move


class UCIError(RuntimeError):
    pass


class UCIEngine:
    def __init__(self, command: str | list[str], options: dict[str, str | int | bool] | None = None,
                 startup_timeout: float = 10.0):
        cmd = [command] if isinstance(command, str) else list(command)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._send("uci")
        self.name = cmd[0]
        for line in self._read_until("uciok", startup_timeout):
            if line.startswith("id name "):
                self.name = line[8:].strip()
        for k, v in (options or {}).items():
            self.set_option(k, v)
        self.ready()

    # ----------------------------------------------------------------- io
    def _send(self, line: str) -> None:
        if self.proc.poll() is not None:
            raise UCIError("engine process has exited")
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read_until(self, token: str, timeout: float) -> list[str]:
        lines: list[str] = []
        result: dict = {}

        def reader() -> None:
            for line in self.proc.stdout:
                line = line.strip()
                lines.append(line)
                if line.startswith(token):
                    result["ok"] = True
                    return

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        t.join(timeout)
        if not result.get("ok"):
            raise UCIError(f"timeout waiting for {token!r}; got {lines[-3:]}")
        return lines

    # ------------------------------------------------------------ commands
    def set_option(self, name: str, value) -> None:
        if isinstance(value, bool):
            value = "true" if value else "false"
        self._send(f"setoption name {name} value {value}")

    def ready(self) -> None:
        self._send("isready")
        self._read_until("readyok", 10.0)

    def new_game(self) -> None:
        self._send("ucinewgame")
        self.ready()

    def best_move(self, board: Board, limits: SearchLimits | None = None) -> Move:
        limits = limits or SearchLimits(time=0.1)
        self._send(f"position fen {board.fen()}")
        if limits.nodes:
            go = f"go nodes {limits.nodes}"
        elif limits.depth:
            go = f"go depth {limits.depth}"
        else:
            go = f"go movetime {int((limits.time or 0.1) * 1000)}"
        self._send(go)
        wait = (limits.time or 5.0) + 30.0
        lines = self._read_until("bestmove", wait)
        parts = lines[-1].split()
        if len(parts) < 2 or parts[1] in ("(none)", "0000"):
            raise UCIError(f"engine returned no move: {lines[-1]!r}")
        return board.parse_uci(parts[1])

    def close(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=2)
        except Exception:  # noqa: BLE001
            self.proc.kill()

    def __enter__(self) -> UCIEngine:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class UCIAgent:
    """`Agent` adapter around a UCI engine."""

    def __init__(self, command: str | list[str], options: dict | None = None,
                 movetime: float = 0.05, name: str | None = None):
        self.engine = UCIEngine(command, options)
        self.movetime = movetime
        self.name = name or self.engine.name

    def select_move(self, board: Board, limits: SearchLimits | None = None) -> Move:
        limits = limits or SearchLimits(time=self.movetime)
        return self.engine.best_move(board, limits)

    def close(self) -> None:
        self.engine.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


def find_stockfish() -> str | None:
    return shutil.which("stockfish")


def stockfish_agent(elo: int | None = None, movetime: float = 0.05, path: str | None = None,
                    threads: int = 1) -> UCIAgent:
    """Stockfish limited to `elo` (UCI_Elo range 1320-3190), or full strength if None."""
    exe = path or find_stockfish()
    if not exe:
        raise FileNotFoundError("stockfish not found (try `brew install stockfish`)")
    opts: dict = {"Threads": threads}
    name = "stockfish"
    if elo is not None:
        opts.update({"UCI_LimitStrength": True, "UCI_Elo": int(elo)})
        name = f"stockfish-{elo}"
    return UCIAgent(exe, opts, movetime=movetime, name=name)
