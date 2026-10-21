"""Fit each level's `error_scale` so its average centipawn loss matches the target for its Elo.

    python scripts/tune_levels.py --positions 80

Cheap (no games): rank moves once with the base engine and a reference engine, then simulate
the level's choices. Writes src/chessrl/levels/levels.json; follow with calibrate_levels.py to
measure the resulting Elo with games.
"""

import argparse
import time

from chessrl.levels.levels import default_levels, save_levels
from chessrl.levels.tune import fit_error_scale, prepare, sample_positions, target_acpl

ap = argparse.ArgumentParser()
ap.add_argument("--positions", type=int, default=80)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ref-depth", type=int, default=3)
ap.add_argument("--seconds", type=float, default=2.0, help="cap per search when preparing positions")
args = ap.parse_args()

levels = default_levels()
depths = sorted({lv.depth for lv in levels})
t0 = time.time()
print(f"preparing {args.positions} positions (base depths {depths}, ref depth {args.ref_depth})…", flush=True)
positions = prepare(sample_positions(args.positions, args.seed), depths, args.ref_depth, args.seconds,
                    progress=lambda i, n: print(f"\r  {i}/{n} ({time.time() - t0:.0f}s)", end="", flush=True))
print()
for lv in levels:
    scale, got = fit_error_scale(positions, lv.depth, lv.elo, lv.style)
    lv.error_scale, lv.acpl = round(scale, 3), round(got, 1)
    print(f"{lv.name:18s} elo {lv.elo:.0f} depth {lv.depth}: error_scale {scale:.2f}  "
          f"ACPL {got:.0f} (target {target_acpl(lv.elo):.0f})", flush=True)
save_levels(levels)
print("saved levels.json")
