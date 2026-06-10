from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


"""清洗 YOLO 标签文件的小脚本。
"""


@dataclass
class LabelReport:
    images: int = 0
    copied: int = 0
    missing: int = 0
    invalid: int = 0


def iter_images(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def clean_label_file(source_path: Path) -> tuple[list[str], list[str]]:
    # 逐行检查标签内容，保留合法样本并记录问题行。
    cleaned_lines: list[str] = []
    issues: list[str] = []

    for line_number, raw_line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            issues.append(f"{source_path}:{line_number} 字段数量不足")
            continue

        try:
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            issues.append(f"{source_path}:{line_number} 包含非数值字段")
            continue

        if class_id < 0:
            issues.append(f"{source_path}:{line_number} 类别 id 不能为负数")
            continue

        if len(coords) == 4:
            cx, cy, width, height = coords
            if not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height)):
                issues.append(f"{source_path}:{line_number} 归一化坐标必须在 [0, 1] 范围内")
                continue
            if width <= 0.0 or height <= 0.0:
                issues.append(f"{source_path}:{line_number} 宽高必须大于 0")
                continue
            cleaned_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")
            continue

        if len(coords) < 6 or len(coords) % 2 != 0:
            issues.append(f"{source_path}:{line_number} 多边形坐标数量必须为偶数且不少于 6")
            continue

        if any(value < 0.0 or value > 1.0 for value in coords):
            issues.append(f"{source_path}:{line_number} 归一化坐标必须在 [0, 1] 范围内")
            continue

        cleaned_lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in coords))

    return cleaned_lines, issues


def process_split(
    split_name: str,
    images_root: Path,
    labels_root: Path,
    source_root: Path,
    allow_empty: bool,
    strict: bool,
) -> LabelReport:
    # 按 train/val 两个子集批量处理，模拟真实训练前的数据整理流程。
    report = LabelReport()

    image_dir = images_root / split_name
    if not image_dir.exists():
        raise FileNotFoundError(f"找不到图片目录: {image_dir}")

    output_dir = labels_root / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in iter_images(image_dir):
        report.images += 1
        source_label = source_root / split_name / f"{image_path.stem}.txt"
        if not source_label.exists():
            source_label = image_path.with_suffix(".txt")

        output_label = output_dir / f"{image_path.stem}.txt"

        if not source_label.exists():
            report.missing += 1
            if allow_empty:
                output_label.write_text("", encoding="utf-8")
            continue

        cleaned_lines, issues = clean_label_file(source_label)
        if issues:
            report.invalid += 1
            for issue in issues:
                print(f"[invalid] {issue}")
            if strict:
                continue

        output_label.write_text("\n".join(cleaned_lines) + ("\n" if cleaned_lines else ""), encoding="utf-8")
        report.copied += 1

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗并归档 LabelImg 导出的 YOLO 标签文件")
    parser.add_argument("--images-root", type=Path, default=Path("dataset/images"), help="图片根目录")
    parser.add_argument("--labels-root", type=Path, default=Path("dataset/labels"), help="输出标签根目录")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("dataset/images"),
        help="原始标签根目录，默认是图片所在目录",
    )
    parser.add_argument("--allow-empty-labels", action="store_true", help="缺失标签时生成空白 txt")
    parser.add_argument("--strict", action="store_true", help="遇到无效标签时跳过写入该文件")
    args = parser.parse_args()

    total = LabelReport()
    for split_name in ("train", "val"):
        report = process_split(
            split_name=split_name,
            images_root=args.images_root,
            labels_root=args.labels_root,
            source_root=args.source_root,
            allow_empty=args.allow_empty_labels,
            strict=args.strict,
        )
        total.images += report.images
        total.copied += report.copied
        total.missing += report.missing
        total.invalid += report.invalid
        print(
            f"[{split_name}] images={report.images}, copied={report.copied}, missing={report.missing}, invalid={report.invalid}"
        )

    print(
        f"完成: images={total.images}, copied={total.copied}, missing={total.missing}, invalid={total.invalid}"
    )


if __name__ == "__main__":
    main()
