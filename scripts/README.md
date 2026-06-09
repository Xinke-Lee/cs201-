# scripts 目录说明

建议统一用项目根目录下的 `./.venv` 运行所有脚本：

```bash
./.venv/bin/python scripts/train_yolo.py --device mps
./.venv/bin/python scripts/infer_trajectory.py --video videos/4.1/0.mp4
./.venv/bin/python scripts/convert_physical.py --k 0.5 --y0 320
./.venv/bin/python scripts/export_tracking_results.py --video videos/4.1/0.mp4 --y0 320 --k 0.5
```

## 脚本用途

- `train_yolo.py`：训练模型。
- `infer_trajectory.py`：逐帧推理并导出轨迹。
- `convert_physical.py`：像素坐标转物理坐标。
- `export_tracking_results.py`：输出标注视频和位移曲线。
- `export_opencv_preview.py`：导出单帧 OpenCV 预览图。
- `run_color_tracking_demo.py`：颜色分割备用演示，不是主流程必需。
- `extract_frames.py`：从视频抽帧做数据集。
- `clean_yolo_labels.py`：清洗并归档标签。
- `browser_annotator.py`：矩形标注网页工具。
- `browser_polygon_annotator.py`：多边形标注网页工具。

## 可删除项

如果你后续不再手工标注，可以考虑删除：

- `browser_annotator.py`
- `browser_polygon_annotator.py`

如果你只想保留 YOLO 主流程，也可以删除：

- `run_color_tracking_demo.py`

如果数据集已经完全整理好，也可以删除：

- `clean_yolo_labels.py`
- `extract_frames.py`
