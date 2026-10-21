# Ideas backlog

Single place where every idea lives so nothing gets lost. Add new ones at the bottom of the
right section with the next free ID. Status: `idea` → `planned` (in ROADMAP) → `done`/`dropped`.

## Engine
| ID | Idea | Status |
|----|------|--------|
| ENG-1 | Speed: maintain piece lists, cheaper legality (pin/check masks) instead of push/pop filtering | idea |
| ENG-2 | Speed: bitboards + numba, or a Rust/C++ core with Python bindings; must keep perft + differential tests green | idea |
| ENG-3 | Chess960 / Fischer random support | idea |
| ENG-4 | UCI protocol front-end so the agent plugs into Arena / Cute Chess / lichess-bot | idea |
| ENG-5 | Endgame tablebase probing (Syzygy) as optional evaluation / adjudication | idea |

## Learning
| ID | Idea | Status |
|----|------|--------|
| RL-1 | AlphaZero-style self-play with MCTS (PUCT) and policy/value ResNet | planned |
| RL-2 | Endgame curriculum (KRK → KPK → KQK → …) to bootstrap learning cheaply | planned |
| RL-3 | Optional supervised warm-start on human games (documented as a deviation from "pure RL") | idea |
| RL-4 | Population / league training: keep past checkpoints as opponents, avoid forgetting | idea |
| RL-5 | Gumbel AlphaZero / fewer simulations per move for faster iteration | idea |
| RL-6 | Mixed-precision + `torch.compile` on MPS; large runs on Colab GPU | idea |
| RL-7 | Ablations as a first-class deliverable (net size, sims/move, buffer size, curriculum on/off) | idea |

## Strength levels & human-likeness
| ID | Idea | Status |
|----|------|--------|
| LVL-1 | Levels from historical checkpoints + MCTS sims + temperature, calibrated to Elo | planned |
| LVL-2 | Maia-style human-like play: error probability grows with position complexity and low clock | idea |
| LVL-3 | Think-time model: instant for forced moves/book, longer in sharp positions, scaled by Elo | planned |
| LVL-4 | Personality / style presets (aggressive, solid, endgame specialist, "trappy") | idea |
| LVL-5 | Adaptive difficulty: adjust to the user's recent results | idea |
| LVL-6 | Opening book scaled by level (strong levels deeper, weak levels leave book early) | planned |

## Evaluation
| ID | Idea | Status |
|----|------|--------|
| EVAL-1 | Elo calibration vs Stockfish `UCI_Elo`; publish Elo-vs-training-steps curve | planned |
| EVAL-2 | Tactics test suites (e.g. Lichess puzzle database) as a quick strength probe | idea |
| EVAL-3 | Reproducible benchmark script + results table in README | idea |

## Product / UI
| ID | Idea | Status |
|----|------|--------|
| UI-1 | Web board with drag & drop, move list, flip, PGN/FEN import/export | planned |
| UI-2 | Clocks and time controls, increments | planned |
| UI-3 | Post-game analysis: accuracy %, eval graph, move classification (brilliant … blunder) | planned |
| UI-4 | Automatic annotation written into PGN comments during play | planned |
| UI-5 | Interpretability view: value heatmap, policy arrows, attention maps (ties to `see-inside-ml`) | idea |
| UI-6 | Voice: announce moves with TTS; voice input for moves | idea |
| UI-7 | In-browser play via ONNX Runtime Web (no server) | idea |
| UI-8 | Lichess bot account running the agent at a chosen level | idea |
| UI-9 | Opening explorer / "which opening am I in" tagging (ECO) | idea |

## Vision (real board)
| ID | Idea | Status |
|----|------|--------|
| VIS-1 | Board detection + homography from 4 corners, robust to camera angle/lighting | planned |
| VIS-2 | YOLO piece detector (12 classes), reuse learnings from `edge-multi-object-tracking` | planned |
| VIS-3 | Synthetic training data (rendered boards/pieces, randomised lighting) to avoid manual labelling | idea |
| VIS-4 | Legality-constrained tracking: only accept moves that are legal from the previous state | planned |
| VIS-5 | Hand/occlusion handling: decide the move when the hand leaves the board | idea |
| VIS-6 | Record video + auto-annotated PGN, replay synced with video | planned |
| VIS-7 | Edge deployment (ONNX / INT8) for Raspberry Pi / phone | idea |
| VIS-8 | Robot arm or LED board for the AI's physical moves | idea |

## Project / portfolio
| ID | Idea | Status |
|----|------|--------|
| PRJ-1 | Technical report / blog post: from rules engine to self-play agent | idea |
| PRJ-2 | Pin the repo, add GIFs of play and the Elo curve to README | idea |
| PRJ-3 | Model card for released checkpoints (data, strength, limits) | idea |
