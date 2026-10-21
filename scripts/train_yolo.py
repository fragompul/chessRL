"""Train a YOLO piece detector on the synthetic dataset (and, ideally, real photos too).

    python -m chessrl vision-dataset --out data/yolo --images 4000
    python scripts/train_yolo.py --data data/yolo/data.yaml --epochs 60

Requires `pip install ultralytics`. Not run by the test-suite. Training on Colab's GPU is the
sensible choice; copy `runs/detect/.../weights/best.pt` back and use it with
`python -m chessrl vision --yolo best.pt ...`.
"""

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--model", default="yolov8n.pt", help="starting weights (n/s/m sizes)")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--imgsz", type=int, default=960)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--device", default=None)
args = ap.parse_args()

from ultralytics import YOLO  # noqa: E402

model = YOLO(args.model)
model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
            degrees=0, flipud=0.0, fliplr=0.0, mosaic=0.5)  # chess pieces: no flips (colours/rank matter)
print("best weights:", model.trainer.best)
