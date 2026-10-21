"""Fixtures to check that docs/demo/chessrl.js matches the Python encoding bit for bit.

    python scripts/make_js_fixtures.py          # writes docs/demo/test/fixtures.json
    cd docs/demo/test && npm ci && node test.js
"""

import hashlib
import json
import random
from pathlib import Path

from chessrl.engine import Board
from chessrl.rl.actions import legal_actions
from chessrl.rl.encoding import encode_board

rng = random.Random(2024)
cases = []
while len(cases) < 60:
    board = Board()
    moves = []
    for _ in range(rng.randint(0, 120)):
        legal = board.legal_moves()
        if not legal or board.outcome() is not None:
            break
        m = rng.choice(legal)
        moves.append(m.uci())
        board.push(m)
    if board.outcome() is not None or not board.legal_moves():
        continue
    planes = encode_board(board)
    legal, idx = legal_actions(board)
    cases.append({
        "moves": moves,
        "fen": board.fen(),
        "repeated": bool(board.repetition_count() >= 2),
        "planes_sha256": hashlib.sha256(planes.astype("float32").tobytes()).hexdigest(),
        "legal": {m.uci(): int(i) for m, i in zip(legal, idx)},
    })
out = Path(__file__).parent.parent / "docs/demo/test/fixtures.json"
out.write_text(json.dumps(cases))
print(f"wrote {len(cases)} fixtures to {out}")
