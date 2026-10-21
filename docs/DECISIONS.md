# Decision log

Short ADR-style record of choices and why, newest last.

## D-001 — Engine written from scratch; python-chess only as a test oracle
The rules engine is our own (the point of the project). `python-chess` is a dev dependency used
exclusively by tests (differential testing), never imported from `src/`.

## D-002 — Python first, optimise later
Clarity and correctness first. Speed work (ENG-1/ENG-2) happens only once profiling of
self-play shows the engine is the bottleneck, and must keep perft + differential tests green.

## D-003 — Board representation
Mailbox: 64-int list, signed pieces (+white / −black), a1 = 0 … h8 = 63. Legal moves are
generated pseudo-legally then filtered by making the move and testing king safety.

## D-004 — Automatic draws
Fifty-move rule (100 half-moves) and threefold repetition end the game automatically in
`Board.outcome()`, because self-play needs unambiguous terminal states. The e.p. square
counts for repetition only when an e.p. capture is geometrically possible.

## D-005 — FEN en passant field
`Board.fen()` always writes the e.p. square after a double pawn push (strict FEN standard),
unlike python-chess's default, which writes it only when a legal capture exists. The
differential test compares with `en_passant="fen"`.

## D-006 — Language and licence
All code, docs and comments in English. MIT licence. Repo is public from day one.

## D-007 — Learning algorithm and data format
AlphaZero-style: ResNet policy/value net (73-plane x 64-square action space, 4672 actions),
PUCT MCTS with FPU, Dirichlet root noise, virtual loss and tree reuse. All positions are encoded
from the side-to-move's perspective (board mirrored for Black). The replay buffer stores positions
compactly (64 int8 + 7 int16 + sparse top-64 policy, ~340 B) and expands them to planes per batch.

## D-008 — Many games, one network call
Self-play runs N games in parallel and sends one batched call per round, which keeps the
accelerator busy. Measured: ~8k simulations/s on an Apple M5 (MPS), 64x6 net.

## D-009 — Training is session-based and always resumable
`latest.pt` (net + optimiser + counters + RNG) and `buffer.npz` are written atomically after
each iteration; Ctrl-C/SIGTERM and `--max-minutes` stop at a safe point. Weights-only snapshots
(`iter_*.pt`) feed levels, league play and Elo ladders. Configuration lives in the run dir.

## D-010 — In-tree vs real repetition
Inside the search tree a position seen twice scores as a draw (prunes cycles); at the root and
in the game itself the real threefold rule applies. A re-rooted tree is rebuilt if its new root
was marked terminal only by the in-tree shortcut (found by a stalled-self-play regression test).

## D-011 — Legal move generation avoids make/unmake where possible
`legal_moves` only verifies by making the move when it can expose the king (king moves, check
evasions, pinned pieces, en passant). Doubled alpha-beta speed (20k → 36k nodes/s) with perft and
the differential test as the safety net.

## D-012 — Elo scale
Ratings are maximum-likelihood estimates with fixed anchors on the Stockfish `UCI_Elo` scale at a
short fixed move time. We call it "Elo" but document it as a relative scale (see EVALUATION.md).
Below Stockfish's floor (1320) values are chained through our own weaker agents.

## D-013 — Human-likeness is tuned on error, not on games
Elo by games is slow, so each level's error parameters are fitted to a target average centipawn
loss (`levels/tune.py`, cheap, cached move rankings) and Elo is then *measured* with games. The
first calibration showed the untuned levels playing far below their nominal Elo (e.g. "Club"
1200 → 721), which is why the tuning step exists.

## D-014 — Vision: explain the picture with a legal move
The tracker never recognises the board from scratch: the state is known, so it scores every legal
move (and every legal move pair) by how well its predicted board matches the camera. Frames are
judged only when still. Appearance is learnt on-line from the known position (no trained model
needed); a YOLO detector plugs in through the same interface. Per-square appearance uses an
8×6 luminance patch relative to the square's own border ring (robust to lighting) and a Gaussian
per exact (square colour, piece) class with an Occam penalty for poorly known classes.

## D-015 — Browser demo is a verified port, not a re-implementation by eye
`docs/demo/chessrl.js` mirrors the Python encoding and action space; a Node test replays 60 random
games with chess.js and requires identical planes (SHA-256) and action indices.
