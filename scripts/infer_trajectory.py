from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run with `./.venv/bin/python scripts/infer_trajectory.py`."""
    parser = argparse.ArgumentParser(description="Run YOLO inference on a video and export trajectory centers")
    parser.add_argument("--weights", type=Path, default=ROOT / "marker_tracking" / "run_1" / "weights" / "best.pt", help="Model weights")
    parser.add_argument("--video", type=Path, required=True, help="Input video path")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "trajectory.csv", help="CSV output path")
    parser.add_argument("--device", default="mps", help="Inference device, e.g. mps/cpu")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--source-zero-y", type=float, default=0.0, help="Reference y0 in pixels for later physical conversion")
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {args.video}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame_index", "x_min", "y_min", "x_max", "y_max", "x_center", "y_center", "confidence"])

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, device=args.device, conf=args.conf, verbose=False)
            if results:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    best_index = int(boxes.conf.argmax().item())
                    best_box = boxes[best_index]
                    x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().tolist()
                    x_center = (x1 + x2) / 2.0
                    y_center = (y1 + y2) / 2.0
                    confidence = float(best_box.conf[0].item())
                    writer.writerow([frame_index, x1, y1, x2, y2, x_center, y_center, confidence])
                else:
                    writer.writerow([frame_index, "", "", "", "", "", "", ""])
            else:
                writer.writerow([frame_index, "", "", "", "", "", "", ""])

            frame_index += 1

    cap.release()
    print(f"轨迹已保存到: {args.output}")


if __name__ == "__main__":
    main()
