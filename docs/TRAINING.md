# Training guide

Training is designed for **many short sessions** — each one can be stopped at any time and
continued later, on this Mac or on Colab.

## Start / continue

```bash
# first session: creates runs/main from the "mac" preset (≈ 20 iterations)
python -m chessrl train --run runs/main --preset mac --iterations 20

# any later session: just point at the same run, it resumes automatically
python -m chessrl train --run runs/main --iterations 20
python -m chessrl train --run runs/main --max-minutes 45      # time-boxed
python -m chessrl status --run runs/main
```

Override any hyper-parameter (stored in `runs/main/config.json` afterwards):

```bash
python -m chessrl train --run runs/main --set selfplay.search.sims=200 --set batch_size=512
```

Presets: `smoke` (seconds, for tests), `mac` (64×6 net, 100 sims, M-series laptop),
`colab` (128×10 net, 200 sims, one GPU).

## What is saved (and where)

| File | Content |
|------|---------|
| `config.json` | all hyper-parameters |
| `ckpt/latest.pt` | network, optimiser, counters, RNG state → exact resume |
| `ckpt/iter_NNNNNN.pt` | weights-only snapshots every `snapshot_every` iterations |
| `ckpt/last_weights.pt` | weights at the moment the session ended |
| `buffer.npz` | replay buffer (every `buffer_save_every` iterations and on exit) |
| `metrics.jsonl` | one JSON line per iteration (losses, W/D/L, sims/s, eval score…) |
| `games/iter_*.pgn` | sample self-play games |

Stopping with Ctrl-C (or SIGTERM, or `--max-minutes`) finishes the current step, saves, and
exits. A second Ctrl-C aborts immediately (you lose at most the unsaved iteration).

## Curriculum

Early on, self-play from the opening rarely produces decisive games. With the `mac`/`colab`
presets, a share of games start from random endgames (KQK, KRK, KPK, KBBK, KRRK, KQKR, KRKP,
KRPKR); the share decays linearly to a small floor (`endgame_decay_iters`).

## Colab

Open `notebooks/colab_training.ipynb`, set a GPU runtime, run all cells. The run folder lives
on Google Drive so sessions survive disconnects. Download `ckpt/iter_*.pt` files to play locally:

```bash
python -m chessrl play --opponent ckpt:path/to/iter_000100.pt
```

## Tuning notes

* `replay_ratio` controls how many times each position is reused (default 4).
* `selfplay.search.sims` is the biggest lever on data quality; `parallel` × `leaves_per_round`
  controls the network batch size.
* Watch `eval_score` (MCTS net vs alpha-beta depth `eval_depth`) in `chessrl status`.
