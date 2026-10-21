# Integrations

## Any UCI GUI / tournament manager (Arena, Cute Chess, Lucas Chess, …)

```bash
python -m chessrl uci --agent level:1400            # a humanised level
python -m chessrl uci --agent ab3                   # classical alpha-beta, depth 3
python -m chessrl uci --agent ckpt:runs/main/ckpt/last_weights.pt@200   # trained net, 200 simulations
python -m chessrl uci --agent level:Club --realtime # sleeps like a human (needs clocks from the GUI)
```

Add it as a UCI engine whose command is the line above (use the full path to the virtualenv's
`python`). `go movetime/nodes/depth/wtime/btime/winc/binc` are honoured; the search is synchronous,
so `stop` is accepted but ignored.

Quick check without a GUI:

```bash
printf 'uci\nisready\nposition startpos moves e2e4\ngo movetime 500\nquit\n' | python -m chessrl uci --agent ab2
```

## Lichess bot

[lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) drives any UCI engine. After creating
a BOT account and token (see its README), point its `engine` section at the command above, e.g.

```yaml
engine:
  dir: "/path/to/chessRL"
  name: "run_uci.sh"          # a one-line script: exec /path/to/.venv/bin/python -m chessrl uci --agent level:1600 --realtime
  protocol: "uci"
```

(Not tested against Lichess itself: that requires a real account/token.)

## Cutechess tournaments for strength measurement

```bash
cutechess-cli -engine name=net cmd="python -m chessrl uci --agent ckpt:…@100" proto=uci \
              -engine name=sf1500 cmd=stockfish option.UCI_LimitStrength=true option.UCI_Elo=1500 proto=uci \
              -each tc=10+0.1 -games 100 -openings file=book.pgn format=pgn
```

## Browser

Export a checkpoint and drop it into the demo page: [docs/demo](demo/index.html)
(`python -m chessrl export --checkpoint … --out docs/demo/model.onnx`, then serve `docs/demo`
with any static server or GitHub Pages).
