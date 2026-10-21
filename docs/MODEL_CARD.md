# Model card — chessRL network (template)

Fill this in when publishing a checkpoint (PRJ-3).

| | |
|---|---|
| **Architecture** | AlphaZero-style ResNet, `channels × blocks` = _TBD_, policy head 73×64 (4672 actions), value head (tanh) |
| **Parameters** | _TBD_ |
| **Input** | 20×8×8 planes from the side to move's perspective (see `rl/encoding.py`) |
| **Training** | self-play (MCTS, _N_ sims/move), _N_ games, endgame curriculum _yes/no_, hardware _TBD_ |
| **Run config** | `config.json` of the run (committed alongside the weights) |
| **Strength** | Elo on the Stockfish `UCI_Elo` scale at _t_ ms/move: _TBD_ ± _TBD_ (see EVALUATION.md caveats) |
| **Intended use** | playing against humans at selectable levels, research/education |
| **Limitations** | trained from self-play only (no human games); no opening theory beyond what self-play found; Elo is relative to the calibration scale, not FIDE |
| **Files** | `iter_NNNNNN.pt` (PyTorch), `model.onnx` (browser/edge) |
