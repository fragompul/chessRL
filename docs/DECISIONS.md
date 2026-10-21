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
