from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def find_red_blob(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 70, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 70, 50])
    upper2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 20:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, x + w, y + h, x + w / 2.0, y + h / 2.0


def process_video(video_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'trajectory.csv'
    video_out = output_dir / 'annotated.mp4'
    plot_path = output_dir / 'displacement_curve.png'

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'无法打开视频: {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    rows: list[tuple[float, float]] = []
    first_y = None

    with csv_path.open('w', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['frame_index', 'time_s', 'x_min', 'y_min', 'x_max', 'y_max', 'x_center', 'y_center', 'dy_px'])

        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = find_red_blob(frame)
            annotated = frame.copy()
            if result is not None:
                x1, y1, x2, y2, cx, cy = result
                if first_y is None:
                    first_y = cy
                dy = cy - first_y
                rows.append((frame_index / fps, dy))

                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(annotated, (int(round(cx)), int(round(cy))), 5, (0, 0, 255), -1)
                cv2.putText(
                    annotated,
                    f'dy={dy:.1f}px',
                    (int(x1), max(20, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                csv_writer.writerow([frame_index, frame_index / fps, x1, y1, x2, y2, cx, cy, dy])
            else:
                csv_writer.writerow([frame_index, frame_index / fps, '', '', '', '', '', '', ''])

            writer.write(annotated)
            frame_index += 1

    cap.release()
    writer.release()

    if rows:
        times = [t for t, _ in rows]
        displacements = [dy for _, dy in rows]
        plt.figure(figsize=(10, 5))
        plt.plot(times, displacements, linewidth=1.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement (px)')
        plt.title(f'Displacement-Time Curve: {video_path.stem}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()

    print(f'[{video_path.name}] {video_out}')
    print(f'[{video_path.name}] {csv_path}')
    print(f'[{video_path.name}] {plot_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Batch run color tracking demo on multiple videos')
    parser.add_argument('--video-root', type=Path, default=ROOT / 'videos' / '4.1', help='Video folder')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs' / '4.1_smallest5', help='Output folder')
    args = parser.parse_args()

    videos = sorted([p for p in args.video_root.glob('*.mp4')], key=lambda p: (p.stat().st_size, p.name))[:5]
    for video in videos:
        process_video(video, args.output_root / video.stem)


if __name__ == '__main__':
    main()
