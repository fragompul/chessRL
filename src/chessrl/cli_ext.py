"""CLI subcommands for training, evaluation, vision, etc. (registered into the main parser)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _cmd_train(args: argparse.Namespace) -> int:
    from .rl.config import TrainConfig, apply_overrides, preset
    from .rl.train import train

    run = Path(args.run)
    cfg_path = run / "config.json"
    if cfg_path.exists() and not args.preset:
        cfg = TrainConfig.load(cfg_path)
    else:
        cfg = preset(args.preset or "mac")
        if args.fresh and run.exists():
            raise SystemExit(f"{run} already exists; pick another --run or delete it")
    cfg = apply_overrides(cfg, args.set or [])
    cfg.name = run.name
    summary = train(run, cfg, iterations=args.iterations, max_minutes=args.max_minutes)
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("rng")}, indent=2, default=str))
    return 0


def register(sub: argparse._SubParsersAction) -> None:  # noqa: SLF001
    t = sub.add_parser("train", help="run / resume self-play training (checkpointed, resumable)")
    t.add_argument("--run", required=True, help="run directory, e.g. runs/main")
    t.add_argument("--preset", choices=["smoke", "mac", "colab"],
                   help="start a new run from a preset (ignored when the run already has a config)")
    t.add_argument("--iterations", type=int, default=None, help="iterations to run in THIS session")
    t.add_argument("--max-minutes", type=float, default=None, help="stop after this wall-clock time")
    t.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override config, e.g. --set selfplay.search.sims=200")
    t.add_argument("--fresh", action="store_true", help="refuse to reuse an existing run dir")
    t.set_defaults(func=_cmd_train)
    register_status(sub)
    register_eval(sub)
    register_analysis(sub)


def _cmd_status(args: argparse.Namespace) -> int:
    from .rl.train import RunPaths

    paths = RunPaths(Path(args.run))
    if not paths.metrics.exists():
        print(f"no metrics yet in {paths.root}")
        return 1
    rows = [json.loads(line) for line in paths.metrics.read_text().splitlines() if line.strip()]
    last = rows[-1]
    games = sum(r["games"] for r in rows)
    secs = sum(r.get("iter_seconds", 0) for r in rows)
    print(f"run {paths.root}: {len(rows)} iterations logged, {games} self-play games, {secs / 3600:.2f} h")
    print("last:", json.dumps(last))
    evals = [(r["iteration"], r["eval_score"]) for r in rows if "eval_score" in r]
    if evals:
        print("eval vs baseline (iteration: score):", ", ".join(f"{i}: {s:.2f}" for i, s in evals[-8:]))
    snaps = paths.snapshots()
    print(f"{len(snaps)} snapshots; latest checkpoint: {'yes' if paths.latest.exists() else 'no'}")
    return 0


def register_status(sub: argparse._SubParsersAction) -> None:  # noqa: SLF001
    s = sub.add_parser("status", help="summarise a training run")
    s.add_argument("--run", required=True)
    s.set_defaults(func=_cmd_status)


def _agent(spec: str, seed: int = 0):
    from .agents.registry import make_agent

    return make_agent(spec, seed=seed)


def _cmd_arena(args: argparse.Namespace) -> int:
    from .eval.arena import play_match

    a, b = _agent(args.a), _agent(args.b, seed=1)
    m = play_match(a, b, args.games, opening_plies=args.opening_plies, max_plies=args.max_plies,
                   on_game=lambda i, r: print(f"game {i + 1}: {r.result} ({r.termination}, {r.plies} plies)"))
    print(m)
    if args.pgn:
        m.save_pgn(args.pgn)
        print("PGN saved to", args.pgn)
    return 0


def _cmd_elo(args: argparse.Namespace) -> int:
    from .eval.calibrate import DEFAULT_SF_ELOS, calibrate_vs_stockfish

    elos = tuple(int(e) for e in args.stockfish_elos.split(",")) if args.stockfish_elos else DEFAULT_SF_ELOS
    cal = calibrate_vs_stockfish(_agent(args.agent), elos, games=args.games, movetime=args.movetime)
    print(cal)
    print("note:", cal.note)
    return 0


def _cmd_ladder(args: argparse.Namespace) -> int:
    from .agents.neural import MCTSAgent
    from .eval.calibrate import elo_ladder
    from .rl.train import RunPaths

    paths = RunPaths(Path(args.run))
    snaps = paths.snapshots()[:: max(1, args.every)]
    if not snaps:
        raise SystemExit("no snapshots found in the run (ckpt/iter_*.pt)")
    anchors = []
    for spec in args.anchors.split(","):
        name, _, elo = spec.partition("=")
        agent = _agent(name)
        anchors.append((agent, float(elo or 0)))
    agents = []
    for p in snaps:
        it = int(p.stem.split("_")[1])
        ag = MCTSAgent.from_checkpoint(str(p), sims=args.sims, name=f"net-{it}", seed=0)
        agents.append((it, ag))
    rows = elo_ladder(agents, anchors, games=args.games, out_dir=paths.root,
                      opening_plies=args.opening_plies, max_plies=args.max_plies)
    for r in rows:
        print(f"iteration {r['iteration']:>6}: Elo {r['elo']:.0f}")
    print(f"written {paths.root / 'elo.json'} and elo.svg")
    return 0


def register_eval(sub: argparse._SubParsersAction) -> None:  # noqa: SLF001
    a = sub.add_parser("arena", help="match between two agents (random|greedy|abN|ckpt:path|sf:elo)")
    a.add_argument("--a", required=True)
    a.add_argument("--b", required=True)
    a.add_argument("--games", type=int, default=10)
    a.add_argument("--opening-plies", type=int, default=4)
    a.add_argument("--max-plies", type=int, default=300)
    a.add_argument("--pgn", default=None)
    a.set_defaults(func=_cmd_arena)

    e = sub.add_parser("elo", help="estimate an agent's Elo against Stockfish UCI_Elo levels")
    e.add_argument("--agent", required=True)
    e.add_argument("--games", type=int, default=10)
    e.add_argument("--movetime", type=float, default=0.05)
    e.add_argument("--stockfish-elos", default=None, help="comma list, default 1320,1500,...,2300")
    e.set_defaults(func=_cmd_elo)

    ld = sub.add_parser("ladder", help="rate a run's snapshots against anchor agents -> elo.json/svg")
    ld.add_argument("--run", required=True)
    ld.add_argument("--anchors", default="random=0,ab1=600,ab2=900",
                    help="name=elo list (replace with calibrated values)")
    ld.add_argument("--every", type=int, default=1, help="use every Nth snapshot")
    ld.add_argument("--games", type=int, default=8)
    ld.add_argument("--sims", type=int, default=100)
    ld.add_argument("--opening-plies", type=int, default=4)
    ld.add_argument("--max-plies", type=int, default=250)
    ld.set_defaults(func=_cmd_ladder)


def _cmd_analyse(args: argparse.Namespace) -> int:
    from .agents.base import SearchLimits
    from .agents.minimax import AlphaBetaAgent
    from .analysis import analyse_game, annotate_game, eval_graph_svg
    from .engine import read_pgn

    games = read_pgn(Path(args.pgn).read_text())
    scorer = AlphaBetaAgent(depth=args.depth, seed=0, name="analysis")
    out = []
    for n, game in enumerate(games, 1):
        def show(i: int, total: int, n: int = n) -> None:
            print(f"\r game {n}: ply {i}/{total}", end="", flush=True)

        a = analyse_game(game, scorer, SearchLimits(time=args.seconds), progress=show)
        print()
        annotate_game(game, a)
        w, b = game.headers.get("White", "?"), game.headers.get("Black", "?")
        print(f"{w} vs {b}  ({game.result})  opening: {' · '.join(a.opening) if a.opening else '-'}")
        print(f"  accuracy  White {a.accuracy_white:.1f}%   Black {a.accuracy_black:.1f}%")
        for color, name in ((0, "White"), (1, "Black")):
            print(f"  {name}: " + ", ".join(f"{v} {k}" for k, v in a.counts(color).items() if k != "book"))
        for m in a.critical_moments():
            print(f"  critical: {m.ply // 2 + 1}{'.' if m.ply % 2 == 0 else '...'} {m.san}{m.symbol}"
                  f"  (best {m.best_san})")
        if args.svg:
            Path(args.svg if len(games) == 1 else f"{args.svg}.{n}.svg").write_text(eval_graph_svg(a))
        out.append(game.to_pgn())
    if args.out:
        Path(args.out).write_text("\n".join(out))
        print("annotated PGN written to", args.out)
    return 0


def register_analysis(sub: argparse._SubParsersAction) -> None:  # noqa: SLF001
    p = sub.add_parser("analyse", help="analyse and annotate the games in a PGN file")
    p.add_argument("pgn")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--seconds", type=float, default=1.5, help="max seconds per move")
    p.add_argument("--out", default=None, help="write the annotated PGN here")
    p.add_argument("--svg", default=None, help="write the evaluation graph here")
    p.set_defaults(func=_cmd_analyse)
