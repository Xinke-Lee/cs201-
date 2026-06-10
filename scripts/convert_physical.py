from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


"""把像素坐标换算成物理坐标的小脚本。
"""


def main() -> None:
    """Run with `./.venv/bin/python scripts/convert_physical.py`."""
    # 这里的核心思想是：先读轨迹，再按比例尺换算成毫米。
    parser = argparse.ArgumentParser(description="Convert pixel trajectory to physical coordinates")
    parser.add_argument("--input", type=Path, default=ROOT / "outputs" / "trajectory.csv", help="Trajectory CSV file")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "trajectory_physical.csv", help="Converted CSV output")
    parser.add_argument("--k", type=float, required=True, help="Scale factor in mm/pixel")
    parser.add_argument("--y0", type=float, required=True, help="Pixel y reference zero")
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError("--k must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", newline="", encoding="utf-8") as input_file, args.output.open("w", newline="", encoding="utf-8") as output_file:
        reader = csv.DictReader(input_file)
        fieldnames = ["frame_index", "x_center_px", "y_center_px", "x_center_mm", "y_center_mm"]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            x_center_text = row.get("x_center", "")
            y_center_text = row.get("y_center", "")
            if not x_center_text or not y_center_text:
                continue

            x_center = float(x_center_text)
            y_center = float(y_center_text)
            y_real = (y_center - args.y0) * args.k
            x_real = x_center * args.k

            writer.writerow({
                "frame_index": row.get("frame_index", ""),
                "x_center_px": f"{x_center:.6f}",
                "y_center_px": f"{y_center:.6f}",
                "x_center_mm": f"{x_real:.6f}",
                "y_center_mm": f"{y_real:.6f}",
            })

    print(f"已输出物理坐标: {args.output}")


if __name__ == "__main__":
    main()
