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

## Phase 2 — Playable game ⬜
- [ ] Baseline agents: random, greedy-material, minimax/alpha-beta (also useful as Elo anchors)
- [ ] CLI play (human vs agent, agent vs agent)
- [ ] Web UI (board, drag & drop, move list, flip board, FEN/PGN import/export)

## Phase 3 — RL environment ⬜
- [ ] Board → tensor encoding (piece planes + castling / ep / repetition / move-count planes)
- [ ] Action space (AlphaZero-style 8×8×73 or compact from-to-promo) + legal-move mask
- [ ] Gymnasium-compatible `ChessEnv`
- [ ] Engine speed pass if profiling says it blocks self-play (see IDEAS: ENG-1, ENG-2)

## Phase 4 — Learning by self-play ⬜
- [ ] Policy/value ResNet (small first, sized for an Apple M-series Mac, MPS)
- [ ] MCTS with PUCT, Dirichlet noise, temperature schedule
- [ ] Self-play workers, replay buffer, training loop, checkpointing
- [ ] Curriculum on endgames (KRK, KPK, KQK) before full games
- [ ] Experiment tracking (TensorBoard / W&B), reproducible configs
- [ ] Colab notebook for longer GPU runs

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
