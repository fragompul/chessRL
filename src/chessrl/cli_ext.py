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
