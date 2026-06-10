from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


"""训练 YOLO 模型的入口脚本。
"""


def main() -> None:
    """Run with `./.venv/bin/python scripts/train_yolo.py`."""
    # 这是整个大作业里“从数据到模型”的核心一步。
    parser = argparse.ArgumentParser(description="Train YOLOv8-seg for red marker tracking")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset.yaml", help="Dataset config path")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="Pretrained model name or path")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="mps", help="Training device, e.g. mps/cpu")
    parser.add_argument("--project", default="marker_tracking", help="Training output project folder")
    parser.add_argument("--name", default="run_1", help="Run name")
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )
    print(results)
    print(f"best.pt: {ROOT / args.project / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
