"""Rate the classical baselines (random, greedy, alpha-beta depth 1-3) on the Stockfish UCI_Elo
scale and write src/chessrl/agents/baseline_elo.json.

Usage: python scripts/calibrate_baselines.py [--games 6] [--movetime 0.05]
The numbers are on the scale "Stockfish UCI_Elo at <movetime> ms/move" (see docs/EVALUATION.md).
"""

import argparse
import itertools
import json
import time
from pathlib import Path

from chessrl.agents.registry import make_agent
from chessrl.eval.arena import play_match
from chessrl.eval.elo import estimate_ratings
from chessrl.eval.uci import stockfish_agent

ap = argparse.ArgumentParser()
ap.add_argument("--games", type=int, default=6)
ap.add_argument("--movetime", type=float, default=0.05)
ap.add_argument("--sf-elos", default="1320,1500")
ap.add_argument("--agents", default="random,greedy,ab1,ab2,ab3")
ap.add_argument("--out", default=str(Path(__file__).parent.parent / "src/chessrl/agents/baseline_elo.json"))
args = ap.parse_args()

agents = {n: make_agent(n, seed=0) for n in args.agents.split(",")}
sf_elos = [int(e) for e in args.sf_elos.split(",")]
results, detail = [], {}
t0 = time.time()


def record(a, b, m):
    s = m.stats
    results.extend([(a, b, 1.0)] * s.wins + [(a, b, 0.5)] * s.draws + [(a, b, 0.0)] * s.losses)
    detail[f"{a} vs {b}"] = [s.wins, s.draws, s.losses]
    print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)


for a, b in itertools.combinations(agents.values(), 2):
    record(a.name, b.name, play_match(a, b, args.games, max_plies=200))
for elo in sf_elos:
    sf = stockfish_agent(elo, args.movetime)
    for a in agents.values():
        if a.name in ("random",):
            continue
        record(a.name, sf.name, play_match(a, sf, args.games, max_plies=200))
    sf.close()

anchors = {f"stockfish-{e}": float(e) for e in sf_elos}
ratings = estimate_ratings(results, anchors)
out = {
    "scale": f"Stockfish UCI_Elo anchors {sf_elos}, {args.movetime * 1000:.0f} ms/move, "
             f"{args.games} games per pairing, random 4-ply openings",
    "ratings": {k: round(v) for k, v in ratings.items() if not k.startswith("stockfish")},
    "results": detail,
}
Path(args.out).write_text(json.dumps(out, indent=2))
print(json.dumps(out["ratings"], indent=2))
