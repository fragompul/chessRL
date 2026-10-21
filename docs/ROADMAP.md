# Roadmap

Status legend: ✅ done · 🚧 in progress · ⬜ planned. Every idea that comes up goes into
[IDEAS.md](IDEAS.md) first and is promoted here when it gets scheduled.

## Phase 0 — Foundations ✅
- [x] Repo, `src/` layout, `pyproject.toml`, ruff + pytest, CI, MIT licence
- [x] Docs scaffolding: roadmap, ideas backlog, decision log

## Phase 1 — Chess engine from scratch ✅
- [x] Board, pieces, legal move generation, make/unmake
- [x] Castling, en passant, promotion, check / checkmate / stalemate
- [x] Draws: insufficient material, fifty-move rule, threefold repetition
- [x] FEN, UCI moves, SAN (with disambiguation), PGN read/write with per-move comments
- [x] Zobrist hashing
- [x] Validation: perft on 6 reference positions (+ depth-4 slow tests) and a differential
      test against python-chess over random games (legal moves, FEN, SAN, status, hashes)

## Phase 2 — Playable game ✅
- [x] Baseline agents: random, greedy-material, minimax/alpha-beta (also useful as Elo anchors)
- [x] CLI play (human vs agent, agent vs agent)
- [x] Web UI (board, drag & drop, promotion picker, clocks, move list, flip, PGN/FEN copy)

## Phase 3 — RL environment ✅
- [x] Board → tensor encoding (piece planes + castling / ep / repetition / move-count planes)
- [x] Action space (AlphaZero-style 8×8×73 or compact from-to-promo) + legal-move mask
- [x] Gymnasium-compatible `ChessEnv`
- [x] Engine speed pass: profiled — self-play reaches ~8k simulations/s on an M5 with a 64×6 net, so the Python engine is *not* the bottleneck; ENG-1/ENG-2 stay in the backlog

## Phase 4 — Learning by self-play ✅ (code complete; no training run launched yet)
- [x] Policy/value ResNet (small first, sized for an Apple M-series Mac, MPS)
- [x] MCTS with PUCT, Dirichlet noise, temperature schedule
- [x] Self-play workers, replay buffer, training loop, checkpointing
- [x] Curriculum on endgames (KRK, KPK, KQK) before full games
- [x] Experiment tracking: JSONL metrics + `chessrl status`, JSON configs with presets (TensorBoard/W&B left as an idea, TRK-1)
- [x] Colab notebook for longer GPU runs

## Phase 5 — Evaluation & Elo ⬜
- [ ] Match runner, SPRT-style significance, Elo estimation (BayesElo / Ordo-like)
- [ ] Calibration against Stockfish with `UCI_LimitStrength` / `UCI_Elo`
- [ ] Checkpoint-vs-checkpoint ladder and Elo-vs-training-steps curve

## Phase 6 — Difficulty levels & human-likeness ⬜
- [ ] Levels by checkpoint, MCTS simulations and temperature; each calibrated to a target Elo
- [ ] Human-like errors (more mistakes in complex positions / low time)
- [ ] Realistic thinking time model (depends on Elo, position complexity, clock)
- [ ] Opening book scaled by level

## Phase 7 — Full game experience ⬜
- [ ] Clocks: blitz / rapid / classical, increment, time odds
- [ ] Game options: colour choice, takebacks, hints, resign / draw offers
- [ ] Automatic annotation while playing (eval swings, !? / ?? tags, comments in PGN)
- [ ] Post-game analysis: accuracy %, eval graph, critical moments

## Phase 8 — Real-board vision ⬜
- [ ] Board detection + homography (4 corners), square grid
- [ ] Piece detection (YOLO, 12 classes), synthetic-data generation for training
- [ ] Position tracking via legality: infer the played move from frame diffs + legal moves
- [ ] Live recording, auto PGN, AI suggests / plays replies (TTS optional)

## Phase 9 — Showcase ⬜
- [ ] ONNX export + in-browser demo (play the AI with no server)
- [ ] Interpretability views (value heatmaps, policy arrows)
- [ ] Write-up / blog post / technical report
