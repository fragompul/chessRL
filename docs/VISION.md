# Playing on a real board (camera)

![Vision demo: the tracker follows a game on a synthetic photo; green = last move, blue = AI suggestion](img/vision-demo.jpg)

`chessrl vision` watches a camera (or a video file), follows every move you play on a physical
board, records the game as PGN, optionally annotates it live, and — if you attach an agent —
suggests (or "plays", for the side it controls) the next move so you can carry it out by hand.

```bash
pip install -e ".[vision]"

# no camera? run the whole pipeline on a synthetic video of a game (camera on any side of the board)
python -m chessrl vision-demo --out data/vision_demo --rotate

# a recording or a live camera
python -m chessrl vision --source game.mp4 --out recordings/g1 --save-video
python -m chessrl vision --source 0 --opponent level:Club --ai-color black --annotate --speak
```

The first frame must show the **starting position** (so the system can learn what your pieces look
like and which side is White). Mid-game starts work too with `--white-side` and a FEN-aware session.
Outputs in `--out`: `game.pgn`, `calibration.json`, optionally `annotated.mp4`.

## How it works

1. **Find the board** (`vision/board_detect.py`). `findChessboardCornersSB` on the inner 7×7 grid
   when it works; otherwise board-shaped quadrilateral candidates (convex hulls of edge contours)
   refined by a coarse-to-fine search that aligns the 8×8 grid lines with the image gradients
   (≈ 3 px accuracy on synthetic photos). `--corners` bypasses detection entirely.
2. **Rectify** with a homography so every frame is a flat 512×512 board, White at the bottom.
   Orientation (which side White sits on) comes from piece brightness plus checker parity.
3. **Wait for a still picture.** A frame is judged only after several frames with (almost) no
   changed pixels, so the moving hand and the lifted piece never confuse the tracker.
4. **Explain the picture with a legal move** (`vision/tracker.py`). The game state is known, so
   the question is *which legal move happened*. For every legal move we know exactly which squares
   change; each hypothesis is scored by how well its predicted board matches what the camera sees.
   Illegal moves can never be produced. If one move does not explain it, two plies are tried
   (a fast player); otherwise an `unexplained` event is raised for the user to resolve.
5. **What the camera sees** comes from an estimator (`vision/estimator.py`):
   * `OccupancyEstimator` — no trained model. Learns the look of each (square colour, piece)
     combination from the known starting position and keeps adapting after every confirmed move.
   * `DetectionEstimator` — per-square beliefs from a YOLO piece detector (`--yolo weights.pt`).
     Also distinguishes promotion pieces; the appearance estimator assumes a queen when unsure.
6. **Session** (`vision/session.py`): PGN, optional live annotation, AI suggestion arrows,
   spoken suggestions (macOS `say`).

## Training a YOLO piece detector

```bash
python -m chessrl vision-dataset --out data/yolo --images 4000 --rotate   # synthetic photos + labels
python scripts/train_yolo.py --data data/yolo/data.yaml --epochs 60       # needs `pip install ultralytics`
python -m chessrl vision --source 0 --calibration cal.json --yolo runs/detect/train/weights/best.pt
```

The synthetic generator (`vision/synth.py`) renders flat boards with piece silhouettes that lean
like real tall pieces, then applies random perspective, roll, lighting gradients, blur and noise
over a cluttered background, with a moving "hand" occluding squares in videos. It gives
unlimited labelled data, but **synthetic-only training will not transfer perfectly**: mix in real
photos of your own board (a few hundred are enough to fine-tune) before trusting it in play.

## Honest limitations

* Everything here is validated on **synthetic** footage; real-camera behaviour (glare, shadows,
  similar piece/square colours, hands resting on the board) still needs real-world tuning.
* The appearance estimator separates white from black pieces by silhouette/brightness; very
  low-contrast sets need the YOLO estimator.
* A hand resting still over the board for longer than the settle window can be mis-read; the
  `unexplained`/`undo`/`force_move` hooks exist so a human can correct the record.
* The headless OpenCV build has no windows: `--show` requires `opencv-python` instead of
  `opencv-python-headless`.

## Ideas on the list (docs/IDEAS.md)

Hand/occlusion model (VIS-5), edge deployment with ONNX/INT8 on a Raspberry Pi or phone (VIS-7),
robot arm or LED board to make the AI's moves physical (VIS-8), board-state re-sync from a single
frame when tracking is lost.
