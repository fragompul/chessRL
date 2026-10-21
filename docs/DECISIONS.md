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
