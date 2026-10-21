# chessRL — working notes

From-scratch chess engine + self-play RL agent. Public GitHub repo, English everywhere.

- Plan lives in `docs/ROADMAP.md`; **every new idea goes into `docs/IDEAS.md`** first; choices
  go into `docs/DECISIONS.md`. Keep them updated as part of any change.
- Engine is in `src/chessrl/engine`. `python-chess` is a test oracle only — never import it
  from `src/`.
- Python 3.12 venv at `.venv`. Run `.venv/bin/pytest` (use `-m "not slow"` for a quick pass) and
  `.venv/bin/ruff check .` before committing.
- Any engine change/optimisation must keep perft + differential tests green.
- Hardware: single Apple M5 Mac (16 GB, MPS); Google Colab GPU for long runs.
