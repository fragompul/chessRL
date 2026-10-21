# chessRL — working notes

From-scratch chess engine + self-play RL agent (+ levels, analysis, UIs, real-board vision).
Public GitHub repo, English everywhere (chat with the owner is in Spanish).

## Process
- Plan: `docs/ROADMAP.md`. **Every new idea goes into `docs/IDEAS.md` first**; design choices go
  into `docs/DECISIONS.md`. Keep them updated as part of any change.
- Python 3.12 venv at `.venv`. Before committing: `.venv/bin/ruff check .` and
  `.venv/bin/pytest -q -m "not slow"` (full suite incl. depth-4 perft: `pytest -q`).
- `python-chess` is a **test oracle only** — never import it from `src/`.
- Any engine change/optimisation must keep perft + differential tests green.
- Do **not** launch real training runs unless the owner asks (smoke/unit tests are fine).
- Hardware: one Apple M5 Mac (16 GB, MPS); Google Colab GPU for long runs.

## Map
- `src/chessrl/engine` rules engine · `agents` baselines/alpha-beta/MCTS agent · `rl` encoding,
  actions, env, network, MCTS, self-play, buffer, trainer (`train.py`), export
- `eval` arena/Elo/Stockfish-UCI/plots · `levels` book, think-time, human-like agent, tuning
- `play` clocks/session/match · `web` stdlib server + `static/index.html` · `analysis` accuracy &
  annotation · `vision` camera pipeline + synthetic renderer · `uci_server.py` · `cli.py`/`cli_ext.py`
- `docs/demo` browser demo (JS port verified by `docs/demo/test`), `scripts/` calibration & tools

## Gotchas
- Squares: a1 = 0 … h8 = 63; pieces are signed ints (+white/−black).
- The in-tree repetition shortcut (2-fold) must never leave a terminal node as a search *root*
  (see D-010 and the regression test).
- Vision tracker constants (`estimator.TEMPERATURE`, `tracker.min_margin`) were tuned on synthetic
  data with `scripts/tune_vision.py`; re-run it after touching the estimator.
- No `sleep` chains in tooling; background long jobs (calibration, CI watch) and read results later.
