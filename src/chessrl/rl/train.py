"""The training loop: self-play -> replay buffer -> gradient steps -> checkpoint, repeated.

Designed to be run in many short sessions ("tandas"). Everything needed to continue is
persisted in the run directory after every iteration, atomically, so you can stop at any
moment (Ctrl-C, closing the laptop, Colab disconnecting) and resume later:

    runs/<name>/
      config.json            hyper-parameters of the run
      ckpt/latest.pt         network + optimiser + counters + RNG state (full resume)
      ckpt/iter_000010.pt    weights-only snapshots (levels, league, Elo ladder)
      buffer.npz             replay buffer
      metrics.jsonl          one JSON line per iteration
      games/iter_*.pgn       a few self-play games per iteration
"""

from __future__ import annotations

import json
import random
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..engine import Game
from .buffer import ReplayBuffer
from .checkpoint import load_full, load_net, save_full, save_weights
from .config import TrainConfig
from .evaluator import NetEvaluator
from .network import ChessNet, get_device
from .selfplay import SelfPlayEngine, SelfPlayGame


@dataclass
class RunPaths:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config.json"

    @property
    def ckpt_dir(self) -> Path:
        return self.root / "ckpt"

    @property
    def latest(self) -> Path:
        return self.ckpt_dir / "latest.pt"

    @property
    def buffer(self) -> Path:
        return self.root / "buffer.npz"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.jsonl"

    @property
    def games_dir(self) -> Path:
        return self.root / "games"

    def snapshot(self, iteration: int) -> Path:
        return self.ckpt_dir / f"iter_{iteration:06d}.pt"

    def snapshots(self) -> list[Path]:
        return sorted(self.ckpt_dir.glob("iter_*.pt"))


def policy_value_loss(net: ChessNet, batch, device, value_weight: float = 1.0):
    planes, idx, probs, z = (t.to(device) for t in batch)
    logits, value = net(planes)
    logp = F.log_softmax(logits, dim=1)
    valid = (idx >= 0).float()
    picked = logp.gather(1, idx.clamp(min=0))
    policy_loss = -(probs * picked * valid).sum(dim=1).mean()
    value_loss = F.mse_loss(value, z)
    return policy_loss + value_weight * value_loss, policy_loss.detach(), value_loss.detach()


class Trainer:
    """Holds network, optimiser and buffer; one `run_iteration` = one self-play + train cycle."""

    def __init__(self, run_dir: str | Path, cfg: TrainConfig, log=print):
        self.paths = RunPaths(Path(run_dir))
        self.cfg = cfg
        self.log = log
        self.device = get_device(cfg.device)
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.pyrng = random.Random(cfg.seed)
        self.net = ChessNet(cfg.net).to(self.device)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr_at(0), weight_decay=cfg.weight_decay)
        self.buffer = ReplayBuffer(cfg.buffer_capacity)
        self.state = {"iteration": 0, "games": 0, "positions": 0, "steps": 0, "elapsed": 0.0,
                      "last_eval": None}
        self._league_cache: dict[Path, NetEvaluator] = {}

    # ------------------------------------------------------------ persistence
    def resume(self) -> bool:
        if not self.paths.latest.exists():
            return False
        blob = load_full(self.paths.latest)
        self.net.load_state_dict(blob["state_dict"])
        self.opt.load_state_dict(blob["optimizer"])
        self.state.update(blob["state"])
        if "np_rng" in blob["state"]:
            self.rng.bit_generator.state = blob["state"]["np_rng"]
        if "py_rng" in blob["state"]:
            self.pyrng.setstate(blob["state"]["py_rng"])
        if self.paths.buffer.exists():
            self.buffer = ReplayBuffer.load(self.paths.buffer, self.cfg.buffer_capacity)
        self.log(f"resumed at iteration {self.state['iteration']} "
                 f"({self.state['games']} games, buffer {len(self.buffer)})")
        return True

    def save(self, save_buffer: bool = True) -> None:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        if save_buffer:
            self.buffer.save(self.paths.buffer)
        self.state["np_rng"] = self.rng.bit_generator.state
        self.state["py_rng"] = self.pyrng.getstate()
        save_full(self.paths.latest, self.net, self.opt, self.state)

    # ------------------------------------------------------------- one cycle
    def run_iteration(self, should_stop=lambda: False) -> dict:
        cfg, st = self.cfg, self.state
        it = st["iteration"]
        t0 = time.perf_counter()
        cfg.selfplay.endgame_prob = cfg.endgame_prob_at(it)
        for g in self.opt.param_groups:
            g["lr"] = cfg.lr_at(it)

        opponents = self._league_opponents()
        engine = SelfPlayEngine(
            _with_league(cfg, bool(opponents)), NetEvaluator(self.net, self.device),
            opponents=opponents, seed=int(self.pyrng.random() * 2**31),
        )
        games = engine.run(cfg.games_per_iter, should_stop=should_stop)
        new_positions = 0
        for g in games:
            self.buffer.add(g.pieces, g.meta, g.pol_idx, g.pol_p, g.z)
            new_positions += len(g.z)
        st["games"] += len(games)
        st["positions"] += new_positions
        t_self = time.perf_counter() - t0

        metrics = {"iteration": it, "games": len(games), "new_positions": new_positions,
                   "buffer": len(self.buffer), "lr": cfg.lr_at(it),
                   "endgame_prob": cfg.selfplay.endgame_prob, "selfplay_seconds": round(t_self, 2),
                   "sims_per_sec": round(engine.stats["sims"] / max(t_self, 1e-9), 1),
                   **_game_stats(games)}
        complete = len(games) == cfg.games_per_iter
        if complete and len(self.buffer) >= cfg.min_buffer:
            metrics.update(self._train_steps(new_positions))
        self._write_pgn(it, games)
        if complete:
            st["iteration"] = it + 1
        st["elapsed"] += time.perf_counter() - t0
        metrics["iter_seconds"] = round(time.perf_counter() - t0, 2)
        return metrics

    def _train_steps(self, new_positions: int) -> dict:
        cfg = self.cfg
        steps = int(np.clip(round(new_positions * cfg.replay_ratio / cfg.batch_size),
                            cfg.min_steps_per_iter, cfg.max_steps_per_iter))
        self.net.train()
        tot = pl = vl = ent = 0.0
        t0 = time.perf_counter()
        for _ in range(steps):
            batch = self.buffer.sample(cfg.batch_size, self.rng)
            loss, p_l, v_l = policy_value_loss(self.net, batch, self.device, cfg.value_loss_weight)
            probs = batch[2]
            ent += float(-(probs * probs.clamp(min=1e-12).log()).sum(dim=1).mean())
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
            self.opt.step()
            tot += float(loss.detach())
            pl += float(p_l)
            vl += float(v_l)
        self.net.eval()
        self.state["steps"] += steps
        return {"steps": steps, "loss": round(tot / steps, 4), "policy_loss": round(pl / steps, 4),
                "value_loss": round(vl / steps, 4), "target_entropy": round(ent / steps, 4),
                "train_seconds": round(time.perf_counter() - t0, 2)}

    def _league_opponents(self) -> list[tuple[str, NetEvaluator]]:
        if self.cfg.league_prob_start <= 0:
            return []
        out = []
        for path in self.paths.snapshots()[-self.cfg.league_pool:]:
            if path not in self._league_cache:
                self._league_cache[path] = NetEvaluator(load_net(path, self.device), self.device)
            out.append((path.stem, self._league_cache[path]))
        return out

    def _write_pgn(self, it: int, games: list[SelfPlayGame]) -> None:
        if not games or not self.cfg.pgn_games_per_iter:
            return
        self.paths.games_dir.mkdir(parents=True, exist_ok=True)
        text = "\n".join(_pgn(g.game, f"self-play it{it}") for g in games[: self.cfg.pgn_games_per_iter])
        (self.paths.games_dir / f"iter_{it:06d}.pgn").write_text(text)

    # -------------------------------------------------------------- eval
    def evaluate(self) -> dict:
        """Score the current network (MCTS) against baselines of increasing strength."""
        from ..agents import AlphaBetaAgent, MCTSAgent
        from ..agents.registry import make_agent
        from ..play import play_game

        cfg = self.cfg
        agent = MCTSAgent(NetEvaluator(self.net, self.device), sims=cfg.eval_sims, name="net", seed=0)
        opponents = [make_agent(n, seed=0) for n in cfg.eval_baselines]
        opponents.append(AlphaBetaAgent(depth=cfg.eval_depth, seed=0))
        out: dict = {"eval_games": cfg.eval_games}
        for k, base in enumerate(opponents):
            score = 0.0
            for i in range(cfg.eval_games):
                white, black = (agent, base) if i % 2 == 0 else (base, agent)
                score += play_game(white, black, max_plies=160).score_for(white is agent)
            out[f"eval_{base.name}"] = score / max(cfg.eval_games, 1)
            if k == len(opponents) - 1:  # keep the historic keys for the strongest baseline
                out["eval_vs"], out["eval_score"] = base.name, out[f"eval_{base.name}"]
        return out


def _with_league(cfg: TrainConfig, has_opponents: bool):
    cfg.selfplay.league_prob = cfg.league_prob_start if has_opponents else 0.0
    return cfg.selfplay


def _game_stats(games: list[SelfPlayGame]) -> dict:
    if not games:
        return {}
    n = len(games)
    return {
        "white_wins": sum(g.result == "1-0" for g in games),
        "black_wins": sum(g.result == "0-1" for g in games),
        "draws": sum(g.result == "1/2-1/2" for g in games),
        "avg_plies": round(sum(g.plies for g in games) / n, 1),
    }


def _pgn(game: Game, event: str) -> str:
    game.headers.setdefault("Event", event)
    return game.to_pgn()


def train(
    run_dir: str | Path,
    cfg: TrainConfig,
    iterations: int | None = None,
    max_minutes: float | None = None,
    log=print,
) -> dict:
    """Run (or continue) training for up to `iterations` more iterations / `max_minutes`.

    Stops cleanly on Ctrl-C / SIGTERM, always leaving a resumable checkpoint.
    """
    paths = RunPaths(Path(run_dir))
    paths.root.mkdir(parents=True, exist_ok=True)
    cfg.save(paths.config)
    trainer = Trainer(run_dir, cfg, log)
    trainer.resume()

    stop = {"flag": False}

    def request_stop(signum, _frame):
        if stop["flag"]:
            raise KeyboardInterrupt
        stop["flag"] = True
        log("stop requested: finishing the current step, then saving…")

    old = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    started = time.perf_counter()
    deadline = started + max_minutes * 60 if max_minutes else None

    def should_stop() -> bool:
        return stop["flag"] or (deadline is not None and time.perf_counter() > deadline)

    done = 0
    try:
        while not should_stop() and (iterations is None or done < iterations):
            m = trainer.run_iteration(should_stop)
            complete = trainer.state["iteration"] > m["iteration"]
            if complete:
                done += 1
                if cfg.snapshot_every and trainer.state["iteration"] % cfg.snapshot_every == 0:
                    save_weights(paths.snapshot(trainer.state["iteration"]), trainer.net,
                                 {"iteration": trainer.state["iteration"],
                                  "games": trainer.state["games"]})
                if cfg.eval_every and trainer.state["iteration"] % cfg.eval_every == 0 \
                        and not should_stop():
                    m.update(trainer.evaluate())
                    trainer.state["last_eval"] = m["eval_score"]
            trainer.save(save_buffer=(trainer.state["iteration"] % max(cfg.buffer_save_every, 1) == 0)
                         or should_stop())
            with open(paths.metrics, "a") as f:
                f.write(json.dumps(m) + "\n")
            log(_fmt(m))
    finally:
        for s, h in old.items():
            signal.signal(s, h)
        trainer.save()
        save_weights(paths.ckpt_dir / "last_weights.pt", trainer.net,
                     {"iteration": trainer.state["iteration"]})
    return {"iterations_done": done, **trainer.state,
            "seconds": round(time.perf_counter() - started, 1)}


def _fmt(m: dict) -> str:
    parts = [f"it {m['iteration']:>4}", f"games {m['games']:>3}", f"pos {m['new_positions']:>5}",
             f"buf {m['buffer']:>6}"]
    if "loss" in m:
        parts.append(f"loss {m['loss']:.3f} (p {m['policy_loss']:.3f} v {m['value_loss']:.3f})")
    parts.append(f"W/D/B {m.get('white_wins', 0)}/{m.get('draws', 0)}/{m.get('black_wins', 0)}")
    parts.append(f"{m['iter_seconds']:.0f}s")
    if "eval_score" in m:
        extra = " ".join(f"{k[5:]}={v:.2f}" for k, v in m.items() if k.startswith("eval_") and k not in
                         ("eval_score", "eval_vs", "eval_games"))
        parts.append(f"eval {extra}")
    return " | ".join(parts)
