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
    register_vision(sub)
    register_export(sub)


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


def _parse_corners(text: str):
    import numpy as np

    vals = [float(v) for v in text.replace(";", ",").split(",")]
    if len(vals) != 8:
        raise SystemExit("--corners needs 8 numbers: a1x,a1y,h1x,h1y,h8x,h8y,a8x,a8y (image pixels)")
    return np.array(vals, dtype="float32").reshape(4, 2)


def _cmd_vision(args: argparse.Namespace) -> int:
    from .agents.registry import make_agent
    from .vision.estimator import DetectionEstimator
    from .vision.geometry import BoardCalibration
    from .vision.session import VisionSession

    calibration = None
    if args.calibration:
        calibration = BoardCalibration.load(args.calibration)
    elif args.corners:
        calibration = BoardCalibration(_parse_corners(args.corners))
    white_side = {"bottom": 0, "left": 1, "top": 2, "right": 3}.get(args.white_side)
    agent = make_agent(args.opponent) if args.opponent else None
    ai_color = {"white": 0, "black": 1}.get(args.ai_color)
    annotator = None
    if args.annotate:
        from .analysis.live import LiveAnnotator

        annotator = LiveAnnotator(depth=3, seconds_per_move=1.0)
    estimator = None
    if args.yolo:
        from .vision.detectors import YoloPieceDetector

        if calibration is None:
            raise SystemExit("--yolo needs --calibration/--corners (the board must be located first)")
        estimator = DetectionEstimator(YoloPieceDetector(args.yolo), calibration)
    session = VisionSession(calibration=calibration, estimator=estimator, white_side=white_side,
                            agent=agent, ai_color=ai_color, annotator=annotator,
                            think_seconds=args.think, headers={"White": args.white, "Black": args.black})
    game = session.run(args.source, out_dir=args.out, save_video=args.save_video, show=args.show,
                       speak=args.speak, max_frames=args.max_frames)
    print(f"\n{len(game.moves)} moves recorded; result {game.result}")
    print(session.pgn())
    return 0


def _cmd_vision_demo(args: argparse.Namespace) -> int:
    """Render a synthetic video of a game, then run the whole camera pipeline on it."""
    import numpy as np

    from .engine import Board, Game, read_pgn
    from .vision.session import VisionSession
    from .vision.synth import BoardStyle, CameraSpec, game_frames, write_video

    out = Path(args.out)
    if args.pgn:
        game = read_pgn(Path(args.pgn).read_text())[0]
    else:
        board, game = Board(), Game()
        for san in args.moves.split():
            m = board.parse_san(san)
            game.moves.append(m)
            board.push(m)
    rng = np.random.default_rng(args.seed)
    spec = CameraSpec.random(rng, any_rotation=args.rotate)
    frames = game_frames(game, spec, BoardStyle.random(rng), idle_frames=args.idle, seed=args.seed)
    video = write_video(frames, out / "synthetic.mp4")
    print("synthetic video:", video)
    session = VisionSession(agent=None)
    seen = session.run(str(video), out_dir=out / "result", save_video=True, log=print)
    ok = [m.uci() for m in seen.moves] == [m.uci() for m in game.moves]
    print(f"\nrecognised {len(seen.moves)}/{len(game.moves)} moves — {'all correct' if ok else 'MISMATCH'}")
    return 0 if ok else 1


def _cmd_vision_dataset(args: argparse.Namespace) -> int:
    from .vision.dataset import make_yolo_dataset

    yaml = make_yolo_dataset(args.out, args.images, args.seed, any_rotation=args.rotate,
                             progress=lambda i, n: print(f"\r{i}/{n}", end="", flush=True))
    print(f"\ndataset ready: {yaml}")
    return 0


def register_vision(sub: argparse._SubParsersAction) -> None:  # noqa: SLF001
    v = sub.add_parser("vision", help="track a real game from a camera / video, record PGN, suggest moves")
    v.add_argument("--source", required=True, help="camera index (0, 1...) or a video file")
    v.add_argument("--calibration", help="calibration JSON saved by a previous run")
    v.add_argument("--corners", help="a1x,a1y,h1x,h1y,h8x,h8y,a8x,a8y in image pixels (skips auto-detection)")
    v.add_argument("--white-side", choices=["bottom", "left", "top", "right"],
                   help="where White sits in the image (default: detected from the start position)")
    v.add_argument("--opponent", help="agent that suggests/plays moves, e.g. level:Club or ckpt:path")
    v.add_argument("--ai-color", choices=["white", "black"], help="only suggest for this colour")
    v.add_argument("--yolo", help="YOLO weights for piece-level recognition (needs ultralytics)")
    v.add_argument("--annotate", action="store_true", help="live annotation into the PGN comments")
    v.add_argument("--out", default="recordings/latest")
    v.add_argument("--save-video", action="store_true")
    v.add_argument("--show", action="store_true")
    v.add_argument("--speak", action="store_true", help="read suggestions aloud (macOS `say`)")
    v.add_argument("--think", type=float, default=1.0, help="seconds the agent may think per suggestion")
    v.add_argument("--max-frames", type=int)
    v.add_argument("--white", default="White")
    v.add_argument("--black", default="Black")
    v.set_defaults(func=_cmd_vision)

    d = sub.add_parser("vision-demo", help="end-to-end demo on a synthetic video (no camera needed)")
    d.add_argument("--out", default="data/vision_demo")
    d.add_argument("--moves", default="e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6 O-O Bg4 h3 h5 hxg4 hxg4")
    d.add_argument("--pgn")
    d.add_argument("--seed", type=int, default=3)
    d.add_argument("--idle", type=int, default=5)
    d.add_argument("--rotate", action="store_true", help="camera on any side of the board")
    d.set_defaults(func=_cmd_vision_demo)

    s = sub.add_parser("vision-dataset", help="generate a synthetic YOLO dataset of piece photos")
    s.add_argument("--out", default="data/yolo")
    s.add_argument("--images", type=int, default=500)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--rotate", action="store_true")
    s.set_defaults(func=_cmd_vision_dataset)


def _cmd_export(args: argparse.Namespace) -> int:
    from .rl.checkpoint import load_net
    from .rl.export import export_onnx, verify_onnx

    net = load_net(args.checkpoint)
    path = export_onnx(net, args.out)
    err = verify_onnx(net, path)
    print(f"exported {path} ({path.stat().st_size / 1e6:.2f} MB), max |torch - onnx| = {err:.2e}")
    return 0


def register_export(sub: argparse._SubParsersAction) -> None:  # noqa: SLF001
    e = sub.add_parser("export", help="export a checkpoint to ONNX")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--out", default="model.onnx")
    e.set_defaults(func=_cmd_export)
