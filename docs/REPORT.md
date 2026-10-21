# chessRL — technical report (draft)

> Status: the system is complete; the results sections below are placeholders until a training
> run has been completed. Everything else describes what is implemented and tested.

## 1. Goals

Build, from scratch and in the open, a complete chess stack: a rules engine, a reinforcement-learning
agent that learns by self-play, calibrated difficulty levels with human-like timing and mistakes,
a playable interface, automatic annotation, and the ability to play on a physical board through a
camera.

## 2. The engine (`chessrl.engine`)

Mailbox board (64 signed ints) with make/unmake, pin-aware legal move generation (only king moves,
check evasions, pinned pieces and en passant are verified by actually making the move), Zobrist
hashing, FEN/UCI/SAN/PGN. **Validation**: perft on six reference positions up to 4.08 M nodes, and
a differential test against `python-chess` over random games (legal moves, FEN, SAN, status, hashes
at every ply). `python-chess` is a test dependency only.

## 3. Learning

* **Representation.** 20 planes from the mover's perspective (board mirrored for Black), 4672-action
  space (64 from-squares × 73 move types), validated as a bijection on legal moves.
* **Search.** PUCT MCTS with first-play urgency, Dirichlet root noise, virtual loss, tree reuse; leaf
  evaluation is batched across many parallel games.
* **Data.** Compact replay buffer (~340 bytes/position), left-right mirror augmentation when nobody
  can castle.
* **Curriculum.** A share of games starts from random KQK/KRK/KPK/KBBK/KRRK/KQKR/KRKP/KRPKR positions;
  the share decays during training.
* **Operation.** Session-based training with atomic checkpoints (full resume incl. RNG), weights-only
  snapshots for levels/league/Elo ladders; ~8 k simulations/s on an Apple M5 with a 64×6 net.

## 4. Evaluation

Arena matches with shared random openings and colour swap, maximum-likelihood ratings with fixed
anchors, SPRT, calibration against Stockfish `UCI_Elo` (caveats in EVALUATION.md).

## 5. Levels and human-likeness

Elo → agent settings: search depth / simulations, softmax temperature over centipawn scores,
error rate that grows with position complexity and low clock, opening-book depth, and a think-time
model (forced/book moves are instant, sharp positions take longer, clock-aware). Calibrated levels
are stored in `levels.json`.

## 6. Analysis

Lichess-style win-probability accuracy per move, classification (best … blunder, plus great and
brilliant sacrifices), critical moments, PGN annotation, live annotation while playing.

## 7. Vision

Camera → board detection → homography → still-frame detection → legal-move inference. See VISION.md.

## 8. Results (to fill in)

* Elo vs training iteration (`elo.svg` from `chessrl ladder`)
* Ablations: net size, simulations/move, curriculum on/off, league on/off (RL-7)
* Calibration table of the difficulty levels
* Vision accuracy on real recordings

## 9. Limitations & future work

See IDEAS.md: native/bitboard engine core, Gumbel search, auxiliary heads, learned human-like model
(Maia-style), UCI server, Lichess bot, edge deployment of the vision stack.
