from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


"""把单段视频的检测结果导出成表格、标注视频和位移曲线。
"""


def main() -> None:
    """Run with `./.venv/bin/python scripts/export_tracking_results.py`."""
    # 这个脚本同时服务于“看效果”和“留数据”两个目标。
    parser = argparse.ArgumentParser(description='Export annotated video and displacement curve from one video')
    parser.add_argument('--weights', type=Path, default=ROOT / 'marker_tracking' / 'run_1' / 'weights' / 'best.pt', help='Model weights')
    parser.add_argument('--video', type=Path, required=True, help='Input video path')
    parser.add_argument('--device', default='mps', help='Inference device')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--y0', type=float, required=True, help='Reference y0 in pixels')
    parser.add_argument('--k', type=float, required=True, help='Scale factor in mm/pixel')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'tracking_demo', help='Output directory')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / 'trajectory.csv'
    video_path = args.output_dir / 'annotated.mp4'
    plot_path = args.output_dir / 'displacement_curve.png'

    model = YOLO(str(args.weights))
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f'无法打开视频: {args.video}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    rows: list[dict[str, float]] = []
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['frame_index', 'time_s', 'x_min', 'y_min', 'x_max', 'y_max', 'x_center', 'y_center', 'y_real_mm'])

        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(frame, device=args.device, conf=args.conf, verbose=False)
            annotated = frame.copy()
            x_center = y_center = y_real = None

            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                # 选置信度最高的框，避免一帧多目标时结果混乱。
                boxes = results[0].boxes
                best_index = int(boxes.conf.argmax().item())
                best_box = boxes[best_index]
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().tolist()
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                y_real = (y_center - args.y0) * args.k

                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(annotated, (int(round(x_center)), int(round(y_center))), 4, (0, 0, 255), -1)
                cv2.putText(
                    annotated,
                    f'y={y_real:.2f} mm',
                    (int(x1), max(20, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                csv_writer.writerow([
                    frame_index,
                    frame_index / fps,
                    x1,
                    y1,
                    x2,
                    y2,
                    x_center,
                    y_center,
                    y_real,
                ])
                rows.append({'frame_index': frame_index, 'time_s': frame_index / fps, 'y_real_mm': y_real})
            else:
                csv_writer.writerow([frame_index, frame_index / fps, '', '', '', '', '', '', ''])

            writer.write(annotated)
            frame_index += 1

    cap.release()
    writer.release()

    if rows:
        times = [row['time_s'] for row in rows]
        displacements = [row['y_real_mm'] for row in rows]
        plt.figure(figsize=(10, 5))
        plt.plot(times, displacements, linewidth=1.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement (mm)')
        plt.title('Displacement-Time Curve')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()

    print(f'annotated video: {video_path}')
    print(f'trajectory csv: {csv_path}')
    print(f'displacement plot: {plot_path}')


if __name__ == '__main__':
    main()
