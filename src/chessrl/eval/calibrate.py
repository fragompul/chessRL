"""Strength measurement: calibrate against Stockfish levels and build Elo-vs-training ladders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..agents.base import Agent
from .arena import MatchResult, play_match
from .elo import MatchStats, estimate_ratings
from .plots import line_chart_svg
from .uci import stockfish_agent

DEFAULT_SF_ELOS = (1320, 1500, 1700, 1900, 2100, 2300)


@dataclass
class Calibration:
    agent: str
    rating: float
    low: float
    high: float
    table: dict[str, MatchStats]
    note: str = ""

    def __str__(self) -> str:
        rows = "\n".join(f"  vs {k}: {v}" for k, v in self.table.items())
        return f"{self.agent}: Elo ~{self.rating:.0f} [{self.low:.0f}, {self.high:.0f}]\n{rows}"


def calibrate_vs_stockfish(
    agent: Agent,
    elos: tuple[int, ...] = DEFAULT_SF_ELOS,
    games: int = 10,
    movetime: float = 0.05,
    stockfish_path: str | None = None,
    **match_kwargs,
) -> Calibration:
    """Estimate `agent`'s Elo on the Stockfish `UCI_Elo` scale.

    Caveat: Stockfish's UCI_Elo is calibrated for long time controls; with short `movetime`
    it plays weaker than nominal, so treat the result as a *relative* anchor.
    """
    results, table = [], {}
    for elo in elos:
        sf = stockfish_agent(elo, movetime, stockfish_path)
        try:
            m = play_match(agent, sf, games, **match_kwargs)
        finally:
            sf.close()
        table[sf.name] = m.stats
        s = m.stats
        results += [(agent.name, sf.name, 1.0)] * s.wins + [(agent.name, sf.name, 0.5)] * s.draws \
            + [(agent.name, sf.name, 0.0)] * s.losses
    anchors = {f"stockfish-{e}": float(e) for e in elos}
    rating = estimate_ratings(results, anchors)[agent.name]
    # Interval: propagate the match-level uncertainty of the most informative opponent.
    informative = min(table.items(), key=lambda kv: abs(kv[1].score - 0.5))
    lo, hi = informative[1].elo_interval()
    base = float(informative[0].split("-")[1])
    return Calibration(agent.name, rating, base + lo, base + hi, table,
                       note=f"Stockfish UCI_Elo scale at {movetime * 1000:.0f} ms/move")


def elo_ladder(
    snapshots: list[tuple[int, Agent]],
    anchors: list[tuple[Agent, float]],
    games: int = 8,
    out_dir: str | Path | None = None,
    **match_kwargs,
) -> list[dict]:
    """Rate training snapshots against fixed anchor agents (with known ratings).

    `snapshots` = [(iteration, agent)], `anchors` = [(agent, elo)]. Returns one dict per
    snapshot and optionally writes `elo.json` / `elo.svg` into `out_dir`.
    """
    anchor_ratings = {a.name: e for a, e in anchors}
    rows = []
    for it, agent in snapshots:
        results = []
        for anchor, _ in anchors:
            m: MatchResult = play_match(agent, anchor, games, **match_kwargs)
            s = m.stats
            results += [(agent.name, anchor.name, 1.0)] * s.wins \
                + [(agent.name, anchor.name, 0.5)] * s.draws + [(agent.name, anchor.name, 0.0)] * s.losses
        r = estimate_ratings(results, anchor_ratings)[agent.name]
        rows.append({"iteration": it, "agent": agent.name, "elo": round(r, 1)})
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "elo.json").write_text(json.dumps(rows, indent=2))
        svg = line_chart_svg({"net": [(r["iteration"], r["elo"]) for r in rows]},
                             title="Elo vs training iteration", xlabel="iteration", ylabel="Elo")
        (out / "elo.svg").write_text(svg)
    return rows
