# Ideas backlog

Single place where every idea lives so nothing gets lost. Add new ones at the bottom of the
right section with the next free ID. Status: `idea` → `planned` (in ROADMAP) → `done`/`dropped`.

## Engine
| ID | Idea | Status |
|----|------|--------|
| ENG-1 | Speed: maintain piece lists, cheaper legality (pin/check masks) instead of push/pop filtering | idea |
| ENG-2 | Speed: bitboards + numba, or a Rust/C++ core with Python bindings; must keep perft + differential tests green | idea |
| ENG-3 | Chess960 / Fischer random support | idea |
| ENG-4 | UCI protocol front-end so the agent plugs into Arena / Cute Chess / lichess-bot | done |
| ENG-5 | Endgame tablebase probing (Syzygy) as optional evaluation / adjudication | idea |

## Learning
| ID | Idea | Status |
|----|------|--------|
| RL-1 | AlphaZero-style self-play with MCTS (PUCT) and policy/value ResNet | done |
| RL-2 | Endgame curriculum (KRK → KPK → KQK → …) to bootstrap learning cheaply | done |
| RL-3 | Optional supervised warm-start on human games (documented as a deviation from "pure RL") | idea |
| RL-4 | Population / league training: keep past checkpoints as opponents, avoid forgetting | done (opt-in `league_prob_start`) |
| RL-5 | Gumbel AlphaZero / fewer simulations per move for faster iteration | idea |
| RL-6 | Mixed-precision + `torch.compile` on MPS; large runs on Colab GPU | idea |
| RL-7 | Ablations as a first-class deliverable (net size, sims/move, buffer size, curriculum on/off) | idea |

| RL-8 | Left-right mirror augmentation when nobody can castle | done |
| RL-9 | Material-based adjudication of move-limit games (off by default: not "pure" RL) | done (opt-in) |
| TRK-1 | TensorBoard / Weights & Biases logging on top of the JSONL metrics | idea |
| RL-10 | Resignation threshold with verification games to shorten self-play | idea |
| RL-11 | Search improvements: transposition reuse across moves, Gumbel root selection, batched virtual loss tuning | idea |
| RL-12 | Auxiliary heads (material / game phase / moves-left) to speed up learning | idea |

## Strength levels & human-likeness
| ID | Idea | Status |
|----|------|--------|
| LVL-1 | Levels from historical checkpoints + MCTS sims + temperature, calibrated to Elo | done |
| LVL-2 | Human-like errors: error probability grows with position complexity and low clock (Maia-style learned model still open) | done (rule-based) |
| LVL-3 | Think-time model: instant for forced moves/book, longer in sharp positions, scaled by Elo | done |
| LVL-4 | Personality / style presets (aggressive, solid done; endgame specialist, "trappy" open) | partly done |
| LVL-5 | Adaptive difficulty: adjust to the user's recent results | done (`AdaptiveElo`, UI hook open) |
| LVL-6 | Opening book scaled by level (strong levels deeper, weak levels leave book early) | done |

## Evaluation
| ID | Idea | Status |
|----|------|--------|
| EVAL-1 | Elo calibration vs Stockfish `UCI_Elo`; publish Elo-vs-training-steps curve | done |
| EVAL-2 | Tactics test suites (e.g. Lichess puzzle database) as a quick strength probe | idea |
| EVAL-3 | Reproducible benchmark script + results table in README | idea |

## Product / UI
| ID | Idea | Status |
|----|------|--------|
| UI-1 | Web board with drag & drop, move list, flip, PGN/FEN import/export | done |
| UI-2 | Clocks and time controls, increments | done |
| UI-3 | Post-game analysis: accuracy %, eval graph, move classification (brilliant … blunder) | done |
| UI-4 | Automatic annotation written into PGN comments during play | done |
| UI-5 | Interpretability view: value heatmap, policy arrows, attention maps (ties to `see-inside-ml`) | done |
| UI-6 | Voice: announce moves with TTS; voice input for moves | idea |
| UI-7 | In-browser play via ONNX Runtime Web (no server) | done |
| UI-8 | Lichess bot account running the agent at a chosen level | documented (untested without an account); see INTEGRATIONS.md |
| UI-9 | Opening explorer / "which opening am I in" tagging (ECO) | done |

## Vision (real board)
| ID | Idea | Status |
|----|------|--------|
| VIS-1 | Board detection + homography from 4 corners, robust to camera angle/lighting | done |
| VIS-2 | YOLO piece detector (12 classes), reuse learnings from `edge-multi-object-tracking` | done (wrapper + dataset + train script; weights not trained yet) |
| VIS-3 | Synthetic training data (rendered boards/pieces, randomised lighting) to avoid manual labelling | done |
| VIS-4 | Legality-constrained tracking: only accept moves that are legal from the previous state | done |
| VIS-5 | Hand/occlusion handling: decide the move when the hand leaves the board | idea |
| VIS-6 | Record video + auto-annotated PGN, replay synced with video | done |
| VIS-7 | Edge deployment (ONNX / INT8) for Raspberry Pi / phone | idea |
| VIS-8 | Robot arm or LED board for the AI's physical moves | idea |

## Project / portfolio
| ID | Idea | Status |
|----|------|--------|
| PRJ-1 | Technical report / blog post: from rules engine to self-play agent | draft done, results pending |
| PRJ-2 | Pin the repo, add GIFs of play and the Elo curve to README | idea |
| PRJ-3 | Model card for released checkpoints (data, strength, limits) | template done |

## Added during implementation
| ID | Idea | Status |
|----|------|--------|
| NEW-1 | Re-sync the vision tracker from a single frame when tracking is lost (piece-level estimator) | idea |
| NEW-2 | Tree search that shares evaluations across transpositions (MCTS DAG) | idea |
| NEW-3 | Opening trainer: drill a repertoire against the book/AI and score deviations | idea |
| NEW-4 | Puzzle mode generated from the analyser's critical moments in the user's own games | idea |
| NEW-5 | Mobile app wrapper (web UI + camera) reusing the ONNX model | idea |
