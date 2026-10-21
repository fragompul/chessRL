# Evaluation & Elo

## Measuring strength

| Tool | What it does |
|------|--------------|
| `chessrl arena --a ckpt:runs/main/ckpt/iter_000100.pt@100 --b ab3 --games 20` | head-to-head match, alternating colours, shared random openings, `+W =D -L`, Elo difference with 95 % CI |
| `chessrl elo --agent ckpt:… --games 10` | rates an agent on the **Stockfish `UCI_Elo` scale** (1320–2300 anchors by default) |
| `chessrl ladder --run runs/main --anchors random=0,ab1=…,ab2=…` | rates every snapshot of a run against anchor agents → `elo.json` + `elo.svg` (Elo vs training iteration) |
| `chessrl analyse game.pgn` | per-move accuracy / classification of a game |

Agent specs everywhere: `random`, `greedy`, `abN`, `ckpt:<path>[@sims]`, `sf[:elo]`, `level:<name|elo>`.

Stockfish is found on `PATH` (`brew install stockfish`).

## How ratings are computed

* Match score → Elo difference via the logistic model; the CI uses the trinomial variance.
* Many pairwise results → maximum-likelihood ratings (`eval.elo.estimate_ratings`), with some
  players **anchored** to known ratings (Stockfish levels). A weak prior keeps shutouts finite.
* `sprt_status` gives an SPRT decision for "is net A at least +X Elo better than B", usable as an
  early-stopping rule in matches.

## Caveats (please read)

* **An Elo number only means something relative to its anchors.** Ours are on the *Stockfish
  `UCI_Elo` scale at a short fixed move time*. Stockfish's own calibration targets long time
  controls, so at 50 ms/move it plays weaker than nominal; treat results as a consistent
  *relative* scale, not as FIDE/lichess/chess.com Elo.
* Below Stockfish's minimum (1320) the estimate is an extrapolation chained through our own
  weaker agents.
* The fixed set of random openings (4 plies) gives variety but is not a balanced opening suite.
* Python-engine search is slow, so deep anchors take a long time; calibration scripts are
  restartable and write their results to JSON.

## Calibrating levels

`scripts/calibrate_baselines.py` rates the baseline engines; `scripts/calibrate_levels.py` plays
every humanised level against Stockfish and stores the *measured* Elo in
`src/chessrl/levels/levels.json`, which the UIs display instead of the nominal target.

## Status of the level calibration (2026-10-22)

* **Error tuning (done):** every level's `error_scale` is fitted so its average centipawn loss
  matches the target for its nominal Elo (ACPL within 1-2 cp of target on 70 positions).
* **Elo by games (inconclusive, not published):** a first pass with 4 games per anchor produced
  non-monotonic values (e.g. Advanced 721, Expert 923) — pure noise plus saturated anchors. Until a
  larger run (≥ 20 games per anchor, idle machine) is done, the UIs show the *nominal* Elo, which
  should be read as "targets the typical error profile of that rating", not as a measured rating.
