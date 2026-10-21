"""Tiny JSON API + static page, built on the standard library only (no web framework)."""

from __future__ import annotations

import json
import threading
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import parse_qs, urlparse

from ..agents.registry import list_opponents, make_agent
from ..analysis import analyse_game, annotate_game, eval_graph_svg
from ..analysis.live import LiveAnnotator
from ..engine import BLACK, WHITE, Board, Move
from ..levels.book import default_book
from ..play import GameSession, TimeControl


class Hub:
    """Holds the live sessions. One per server; tests build their own."""

    def __init__(self) -> None:
        self.sessions: dict[str, GameSession] = {}
        self.thinking: dict[str, bool] = {}
        self.errors: dict[str, str] = {}
        self.analyses: dict[str, dict] = {}
        self.lock = threading.Lock()

    # ------------------------------------------------------------- lifecycle
    def new_game(self, params: dict) -> str:
        import random

        color = params.get("color", "white")
        if color == "random":
            color = random.choice(["white", "black"])
        human = {"white": WHITE, "black": BLACK, "none": None}[color]
        seed = params.get("seed")
        ai = make_agent(params.get("opponent", "level:Club"), seed=seed)
        tc = TimeControl.parse(params["tc"]) if params.get("tc") else None
        headers = {
            "White": "You" if human == WHITE else ai.name,
            "Black": "You" if human == BLACK else ai.name,
            "Event": "chessRL web game",
        }
        session = GameSession(
            ai, human, tc,
            realtime=bool(params.get("realtime", True)),
            allow_takebacks=bool(params.get("takebacks", True)),
            headers=headers,
            annotator=LiveAnnotator() if params.get("live_annotation") else None,
        )
        sid = uuid.uuid4().hex[:12]
        with self.lock:
            self.sessions[sid] = session
        self._maybe_ai(sid)
        return sid

    def get(self, sid: str) -> GameSession:
        with self.lock:
            if sid not in self.sessions:
                raise KeyError(sid)
            return self.sessions[sid]

    # ------------------------------------------------------------------- AI
    def _maybe_ai(self, sid: str) -> None:
        session = self.get(sid)
        session.check_time()
        if not session.is_ai_turn():
            return
        with self.lock:
            if self.thinking.get(sid):
                return
            self.thinking[sid] = True

        def run() -> None:
            try:
                session.play_ai()
            except Exception as e:  # noqa: BLE001 - surfaced to the client
                if "position changed" not in str(e):
                    self.errors[sid] = str(e)
            finally:
                with self.lock:
                    self.thinking[sid] = False
            if session.human_color is None and not session.over:
                self._maybe_ai(sid)

        threading.Thread(target=run, daemon=True).start()

    # ---------------------------------------------------------------- state
    def state(self, sid: str) -> dict:
        s = self.get(sid)
        s.check_time()
        board = s.board
        legal: dict[str, list[str]] = {}
        for m in board.legal_moves():
            legal.setdefault(m.uci()[:2], []).append(m.uci())
        sans, replay = [], Board(s.start_fen)
        for m in s.moves:
            sans.append(replay.san(m))
            replay.push(m)
        return {
            "sid": sid,
            "fen": board.fen(),
            "turn": "white" if board.turn == WHITE else "black",
            "human": None if s.human_color is None else ("white" if s.human_color == WHITE else "black"),
            "legal": {} if s.over else legal,
            "last": s.moves[-1].uci() if s.moves else None,
            "check": board.is_check(),
            "king_squares": {"white": board.king_sq[WHITE], "black": board.king_sq[BLACK]},
            "moves": sans,
            "comments": {str(k): v for k, v in s.comments.items()},
            "clock": None if not s.clock else {
                "white": s.time_left(WHITE), "black": s.time_left(BLACK)},
            "status": s.status_text(),
            "over": s.over,
            "ending": None if not s.ending else {"result": s.ending.result, "reason": s.ending.reason},
            "thinking": bool(self.thinking.get(sid)),
            "draw_offered": s.draw_offered_by is not None,
            "error": self.errors.pop(sid, None),
            "opponent": s.ai.name if s.ai else None,
            "opening": _opening(board),
            "annotations": {
                str(p): {"label": a.label, "symbol": a.symbol, "best": a.best_san,
                         "eval": a.eval_after, "accuracy": round(a.accuracy, 1)}
                for p, a in list(s.annotations.items())
            },
            "live_eval": (lambda last: None if last is None else last.eval_after)(
                s.annotations[max(s.annotations)] if s.annotations else None),
        }

    # ------------------------------------------------------------- analysis
    def start_analysis(self, sid: str, depth: int = 3) -> dict:
        s = self.get(sid)
        job = self.analyses.get(sid)
        if job and job["status"] == "running":
            return job
        game = s.to_game()
        job = {"status": "running", "progress": 0.0, "result": None, "plies": len(game.moves)}
        self.analyses[sid] = job

        def run() -> None:
            from ..agents.base import SearchLimits
            from ..agents.minimax import AlphaBetaAgent

            try:
                scorer = AlphaBetaAgent(depth=depth, seed=0, name="analysis")
                a = analyse_game(game, scorer, SearchLimits(time=1.5),
                                 progress=lambda i, n: job.update(progress=i / max(n, 1)))
                annotate_game(game, a)
                job["result"] = {**a.to_dict(), "svg": eval_graph_svg(a, dark=True), "pgn": game.to_pgn()}
                job["status"] = "done"
            except Exception as e:  # noqa: BLE001
                job["status"], job["error"] = "error", str(e)

        threading.Thread(target=run, daemon=True).start()
        return job

    # -------------------------------------------------------------- actions
    def move(self, sid: str, uci: str) -> dict:
        self.get(sid).play_human(uci)
        self._maybe_ai(sid)
        return self.state(sid)

    def action(self, sid: str, name: str) -> dict:
        s = self.get(sid)
        color = s.human_color if s.human_color is not None else WHITE
        extra: dict = {}
        if name == "resign":
            s.resign(color)
        elif name == "draw":
            extra["accepted"] = s.offer_draw(color)
        elif name == "claim":
            extra["accepted"] = s.claim_draw()
        elif name == "takeback":
            extra["undone"] = s.takeback()
        elif name == "hint":
            h = s.hint()
            extra["hint"] = h.uci() if h else None
            extra["hint_san"] = s.board.san(h) if h else None
        else:
            raise ValueError(f"unknown action {name!r}")
        out = self.state(sid)
        out.update(extra)
        self._maybe_ai(sid)
        return out


def _opening(board) -> list[str] | None:
    hit = default_book().identify(board)
    return list(hit) if hit else None


def make_handler(hub: Hub):
    static = resources.files("chessrl.web").joinpath("static")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # silence request logging
            pass

        def _json(self, payload, status=HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _text(self, text: str, ctype: str, status=HTTPStatus.OK) -> None:
            body = text.encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def _route(self, method: str) -> None:
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            path = url.path
            try:
                if method == "GET" and path in ("/", "/index.html"):
                    return self._text(static.joinpath("index.html").read_text(), "text/html; charset=utf-8")
                if method == "GET" and path == "/api/opponents":
                    return self._json(list_opponents())
                if method == "POST" and path == "/api/new":
                    return self._json(hub.state(hub.new_game(self._body())))
                sid = q.get("sid", "")
                if method == "GET" and path == "/api/state":
                    return self._json(hub.state(sid))
                if method == "GET" and path == "/api/pgn":
                    return self._text(hub.get(sid).to_game().to_pgn(), "application/x-chess-pgn")
                if method == "POST" and path == "/api/move":
                    return self._json(hub.move(sid, self._body()["uci"]))
                if method == "POST" and path == "/api/analysis/start":
                    job = hub.start_analysis(sid)
                    return self._json({k: v for k, v in job.items() if k != "result"})
                if method == "GET" and path == "/api/analysis":
                    job = hub.analyses.get(sid)
                    return self._json(job or {"status": "none"})
                if method == "POST" and path.startswith("/api/action/"):
                    return self._json(hub.action(sid, path.rsplit("/", 1)[1]))
                for name, fn in EXTRA_ROUTES.items():
                    if path == name:
                        return self._json(fn(hub, sid, q, self._body() if method == "POST" else {}))
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except KeyError:
                self._json({"error": "unknown session"}, HTTPStatus.NOT_FOUND)
            except (ValueError, json.JSONDecodeError) as e:
                self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)

        def do_GET(self) -> None:
            self._route("GET")

        def do_POST(self) -> None:
            self._route("POST")

    return Handler


# Later phases register extra JSON endpoints here: path -> fn(hub, sid, query, body) -> dict
EXTRA_ROUTES: dict = {}


def create_server(host: str = "127.0.0.1", port: int = 8000, hub: Hub | None = None) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(hub or Hub()))


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    server = create_server(host, port)
    url = f"http://{host}:{server.server_address[1]}"
    print(f"chessRL web UI on {url}  (Ctrl-C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


__all__ = ["Hub", "create_server", "serve", "Move"]
