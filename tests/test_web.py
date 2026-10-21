import json
import threading
import time
import urllib.request

import pytest

from chessrl.web.server import Hub, create_server


@pytest.fixture()
def base():
    server = create_server("127.0.0.1", 0, Hub())
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def call(base, path, body=None):
    req = urllib.request.Request(
        base + path, data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            raw = r.read().decode()
            return json.loads(raw) if r.headers["Content-Type"] == "application/json" else raw
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode())


def test_index_and_opponents(base):
    assert "<title>chessRL</title>" in call(base, "/")
    assert any(o["id"] == "ab2" for o in call(base, "/api/opponents"))


def test_game_flow(base):
    s = call(base, "/api/new", {"color": "white", "opponent": "ab1", "realtime": False, "seed": 0})
    sid = s["sid"]
    assert s["human"] == "white" and "e2" in s["legal"]
    assert "error" in call(base, f"/api/move?sid={sid}", {"uci": "e2e5"})
    s = call(base, f"/api/move?sid={sid}", {"uci": "e2e4"})
    for _ in range(50):  # wait for the AI reply
        s = call(base, f"/api/state?sid={sid}")
        if len(s["moves"]) == 2:
            break
        time.sleep(0.05)
    assert len(s["moves"]) == 2 and s["turn"] == "white"
    h = call(base, f"/api/action/hint?sid={sid}", {})
    assert h["hint"]
    assert call(base, f"/api/action/takeback?sid={sid}", {})["moves"] == []
    pgn = call(base, f"/api/pgn?sid={sid}")
    assert "[White" in pgn
    r = call(base, f"/api/action/resign?sid={sid}", {})
    assert r["over"] and r["ending"]["result"] == "0-1"


def test_black_human_ai_opens(base):
    sid = call(base, "/api/new", {"color": "black", "opponent": "ab1", "realtime": False})["sid"]
    for _ in range(50):
        s = call(base, f"/api/state?sid={sid}")
        if s["moves"]:
            break
        time.sleep(0.05)
    assert len(s["moves"]) == 1 and s["turn"] == "black"


def test_unknown_session(base):
    assert call(base, "/api/state?sid=nope")["error"] == "unknown session"


def test_live_annotation_and_analysis_endpoints(base):
    sid = call(base, "/api/new", {"color": "white", "opponent": "ab1", "realtime": False,
                                  "seed": 0, "live_annotation": True})["sid"]
    call(base, f"/api/move?sid={sid}", {"uci": "e2e4"})
    for _ in range(100):
        s = call(base, f"/api/state?sid={sid}")
        if len(s["moves"]) == 2 and s["annotations"]:
            break
        time.sleep(0.1)
    assert s["opening"] is not None or s["moves"]
    assert "0" in s["annotations"]
    call(base, f"/api/analysis/start?sid={sid}", {})
    for _ in range(200):
        job = call(base, f"/api/analysis?sid={sid}")
        if job["status"] != "running":
            break
        time.sleep(0.1)
    assert job["status"] == "done" and "accuracy" in job["result"] and job["result"]["svg"].startswith("<svg")


def test_insight_endpoint(base):
    sid = call(base, "/api/new", {"color": "white", "opponent": "ab2", "realtime": False, "seed": 0})["sid"]
    ins = call(base, f"/api/insight?sid={sid}")
    assert 1 <= len(ins["moves"]) <= 5 and ins["turn"] == "white"
    shares = [m["share"] for m in ins["moves"]]
    assert shares == sorted(shares, reverse=True) and 0 < sum(shares) <= 1.0001
    assert {"uci", "san", "cp"} <= set(ins["moves"][0])
