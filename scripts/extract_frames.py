from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


"""从视频里抽帧，生成 YOLO 训练图片集。
"""


def iter_videos(video_root: Path) -> list[Path]:
    return sorted(
        path for path in video_root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def choose_split(video_path: Path, val_ratio: float) -> str:
    # 用文件名哈希决定 train/val，保证同一个视频始终落到同一集合里。
    digest = hashlib.md5(str(video_path).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


def extract_frames(
    video_path: Path,
    output_dir: Path,
    interval: int,
    max_frames: int | None,
) -> tuple[int, int]:
    # 按固定间隔抽帧，兼顾覆盖度和数据量。
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    saved_count = 0
    frame_index = 0
    stem = video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_index % interval == 0:
            if max_frames is not None and saved_count >= max_frames:
                break
            output_path = output_dir / f"{stem}_frame{frame_index:06d}.jpg"
            cv2.imwrite(str(output_path), frame)
            saved_count += 1

        frame_index += 1

    capture.release()
    return frame_index, saved_count


def main() -> None:
    parser = argparse.ArgumentParser(description="从视频中按固定间隔抽帧，构建 YOLO 数据集图片目录")
    parser.add_argument("--video-root", type=Path, default=Path("videos"), help="视频根目录")
    parser.add_argument("--output-root", type=Path, default=Path("dataset/images"), help="输出图片根目录")
    parser.add_argument("--interval", type=int, default=30, help="每隔多少帧保存一张图")
    parser.add_argument("--max-frames-per-video", type=int, default=12, help="每个视频最多保存多少张图")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="按视频级别分配到验证集的比例")
    args = parser.parse_args()

    if args.interval <= 0:
        raise ValueError("--interval 必须大于 0")
    if args.max_frames_per_video <= 0:
        raise ValueError("--max-frames-per-video 必须大于 0")
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio 必须在 0 和 1 之间")

    videos = iter_videos(args.video_root)
    if not videos:
        raise FileNotFoundError(f"在 {args.video_root} 下没有找到视频文件")

    train_dir = args.output_root / "train"
    val_dir = args.output_root / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    total_videos = 0
    total_frames = 0
    total_saved = 0

    for video_path in videos:
        split = choose_split(video_path, args.val_ratio)
        output_dir = train_dir if split == "train" else val_dir
        frame_count, saved_count = extract_frames(
            video_path,
            output_dir,
            args.interval,
            args.max_frames_per_video,
        )
        total_videos += 1
        total_frames += frame_count
        total_saved += saved_count
        print(f"[{split}] {video_path} -> {saved_count}/{frame_count} 帧")

    print(f"完成: {total_videos} 个视频, {total_frames} 帧, 已保存 {total_saved} 张图片")


if __name__ == "__main__":
    main()
