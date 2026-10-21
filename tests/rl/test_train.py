import json

import numpy as np
import torch

from chessrl.rl.checkpoint import load_full, load_net
from chessrl.rl.config import TrainConfig, apply_overrides, preset
from chessrl.rl.train import RunPaths, train


def quiet(*_a, **_k):
    pass


def test_config_roundtrip_and_overrides(tmp_path):
    cfg = preset("mac")
    cfg.save(tmp_path / "c.json")
    again = TrainConfig.load(tmp_path / "c.json")
    assert again.to_dict() == cfg.to_dict()
    apply_overrides(again, ["selfplay.search.sims=7", "net.blocks=2", "lr_schedule=[[0,0.01]]"])
    assert again.selfplay.search.sims == 7 and again.net.blocks == 2 and again.lr_at(500) == 0.01
    try:
        apply_overrides(again, ["nope=1"])
        raise AssertionError
    except KeyError:
        pass
    assert preset("mac").endgame_prob_at(0) == 0.6
    assert abs(preset("mac").endgame_prob_at(150) - 0.05) < 1e-9


def test_training_in_sessions_with_resume(tmp_path):
    run = tmp_path / "run"
    cfg = preset("smoke")
    s1 = train(run, cfg, iterations=2, log=quiet)
    paths = RunPaths(run)
    assert s1["iterations_done"] == 2 and s1["iteration"] == 2
    assert paths.latest.exists() and paths.buffer.exists() and paths.metrics.exists()
    assert paths.snapshot(1).exists() and paths.snapshot(2).exists()
    blob = load_full(paths.latest)
    w_before = {k: v.clone() for k, v in blob["state_dict"].items()}

    # Second session continues from the checkpoint instead of starting over.
    s2 = train(run, preset("smoke"), iterations=2, log=quiet)
    assert s2["iteration"] == 4 and s2["games"] == 16
    lines = [json.loads(line) for line in paths.metrics.read_text().splitlines()]
    assert [m["iteration"] for m in lines] == [0, 1, 2, 3]
    assert any("loss" in m for m in lines)  # gradient steps happened
    assert any("eval_score" in m for m in lines)
    net = load_net(paths.snapshot(4))
    assert net(torch.zeros(1, 20, 8, 8))[0].shape == (1, 4672)
    changed = any(not torch.equal(w_before[k], v) for k, v in load_full(paths.latest)["state_dict"].items())
    assert changed
    assert (paths.games_dir / "iter_000000.pgn").exists()


def test_time_budget_stops_cleanly_and_leaves_resumable_state(tmp_path):
    run = tmp_path / "run"
    cfg = preset("smoke")
    cfg.games_per_iter = 8
    out = train(run, cfg, max_minutes=0.0001, log=quiet)  # ~6 ms: stops almost immediately
    assert out["iterations_done"] <= 1
    assert RunPaths(run).latest.exists()
    out2 = train(run, cfg, iterations=1, log=quiet)  # can still continue afterwards
    assert out2["iterations_done"] == 1


def test_rng_state_is_restored(tmp_path):
    run = tmp_path / "run"
    train(run, preset("smoke"), iterations=1, log=quiet)
    st = load_full(RunPaths(run).latest)["state"]
    assert "np_rng" in st and "py_rng" in st
    g = np.random.default_rng()
    g.bit_generator.state = st["np_rng"]  # must be a valid bit-generator state
