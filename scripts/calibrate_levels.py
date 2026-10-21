"""Measure the real strength of the humanised levels and store it in levels/levels.json.

Each level plays a few games against anchor agents whose rating is known
(src/chessrl/agents/baseline_elo.json, produced by scripts/calibrate_baselines.py); ratings
are then estimated by maximum likelihood with the anchors fixed. Restartable: levels that
already have a measurement are skipped unless --force.

    python scripts/calibrate_levels.py --games 4
    python scripts/calibrate_levels.py --games 8 --levels Club,Intermediate --force
"""

import argparse
import json
import time
from pathlib import Path

from chessrl.agents.registry import make_agent
from chessrl.eval.arena import play_match
from chessrl.eval.elo import estimate_ratings
from chessrl.levels.levels import LEVELS_FILE, default_levels, make_level_agent, save_levels

ap = argparse.ArgumentParser()
ap.add_argument("--games", type=int, default=4, help="games per (level, anchor) pairing")
ap.add_argument("--anchors", default="ab1,ab2,ab3")
ap.add_argument("--levels", default=None, help="comma list of level names (default: all)")
ap.add_argument("--force", action="store_true")
ap.add_argument("--max-plies", type=int, default=200)
args = ap.parse_args()

baseline = json.loads((Path(__file__).parent.parent / "src/chessrl/agents/baseline_elo.json").read_text())
anchor_elo = {k: float(v) for k, v in baseline["ratings"].items()}
anchors = [make_agent(a, seed=1) for a in args.anchors.split(",")]
levels = default_levels()
wanted = set(args.levels.split(",")) if args.levels else None
t0 = time.time()

for lv in levels:
    if (wanted and lv.name not in wanted) or (lv.measured_elo is not None and not args.force):
        continue
    agent = make_level_agent(lv, seed=0)
    results = []
    for anchor in anchors:
        m = play_match(agent, anchor, args.games, max_plies=args.max_plies, seed=7)
        s = m.stats
        results += [(agent.name, anchor.name, 1.0)] * s.wins + [(agent.name, anchor.name, 0.5)] * s.draws \
            + [(agent.name, anchor.name, 0.0)] * s.losses
        print(f"[{time.time() - t0:5.0f}s] {lv.name} vs {anchor.name}: {s}", flush=True)
    rating = estimate_ratings(results, {a.name: anchor_elo[a.name] for a in anchors})[agent.name]
    lv.measured_elo = round(rating)
    save_levels(levels)  # partial progress survives an interruption
    print(f"==> {lv.name}: nominal {lv.elo:.0f}, measured {lv.measured_elo}", flush=True)
print("written", LEVELS_FILE)
