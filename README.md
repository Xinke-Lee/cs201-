# 红色标识物追踪项目

本项目基于 YOLO 完成红色标识物的检测、轨迹提取和像素到物理量转换。整体流程分为数据准备、模型训练、视频推理和物理量换算四步。

## 1. 项目搭建顺序

### 1.1 视频抽帧与数据集构建

先从原视频中抽取单帧图像，构建训练数据集。针对单一目标追踪，通常抽取 100 到 300 帧具有代表性的图像即可。

- 使用 OpenCV 按固定间隔读取视频并保存为 `.jpg` 图像。
- 优先保留包含红色标识物不同状态的图片，例如静止、运动模糊、不同形变和不同背景位置。
- 在项目根目录下建立 YOLO 所需的数据集目录：

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

对应脚本：`scripts/extract_frames.py`

### 1.2 图像标注与标签整理

接着对抽帧图片进行标注，生成 YOLO 格式的标签文件。

- 打开标注工具，对红色标识物框选或多边形标注。
- 将类别统一命名为 `red_marker`。
- 标注完成后，把标签文件按训练集/验证集划分到 `labels/train/` 和 `labels/val/`。
- 如果标签文件格式需要清洗或重新归档，可以使用脚本批量整理。

对应脚本：
- `scripts/browser_annotator.py`：矩形框标注网页工具
- `scripts/browser_polygon_annotator.py`：多边形标注网页工具
- `scripts/clean_yolo_labels.py`：标签清洗与归档

### 1.3 配置训练环境与模型训练

在项目根目录创建 `dataset.yaml`，定义数据路径和类别：

```yaml
path: ./dataset
train: images/train
val: images/val
nc: 1
names: ['red_marker']
```

训练脚本会调用 Ultralytics YOLO，并在 Mac 上显式指定 `mps` 设备以加速训练。

```python
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    device='mps',
    batch=16,
    project='marker_tracking',
    name='run_1'
)
```

训练完成后，最佳权重通常位于：

```text
marker_tracking/run_1/weights/best.pt
```

对应脚本：`scripts/train_yolo.py`

### 1.4 视频推理与轨迹坐标提取

训练完成后，使用最佳权重对完整视频逐帧推理，提取红色标识物的边界框和中心点。

YOLO 默认输出边界框坐标为：

$$[x_{min}, y_{min}, x_{max}, y_{max}]$$

中心点计算公式为：

$$x_{center} = \frac{x_{min} + x_{max}}{2}$$

$$y_{center} = \frac{y_{min} + y_{max}}{2}$$

脚本会把每帧的边界框和中心点坐标保存成 CSV，方便后续绘图和分析。

对应脚本：`scripts/infer_trajectory.py`

### 1.5 像素坐标到真实物理量的转换

为了计算位移、速度或加速度，需要把像素坐标换算成物理坐标。

- 在画面中找到直尺上的两个已知刻度。
- 读取这两个刻度之间的像素距离 `D_pixel`。
- 结合真实距离 `D_real`，得到比例系数：

$$k = \frac{D_{real}}{D_{pixel}}$$

其中 `k` 的单位是 `mm/pixel`。

若以某一条参考线或某一点 `y_0` 作为零点，则第 `i` 帧的真实纵向位置为：

$$Y_{real} = (y_{center} - y_0) \times k$$

对应脚本：`scripts/convert_physical.py`

### 1.6 输出效果视频与位移时间曲线

如果你想直接看到处理结果，可以把检测框、中心点和位移曲线一起导出。

对应脚本：`scripts/export_tracking_results.py`

## 2. 项目结构与各模块作用

```text
.
├── dataset.yaml                # YOLO 数据集配置
├── dataset/                    # 数据集目录
├── outputs/                    # 推理结果、曲线图、预览图输出目录
├── requirements.txt            # Python 依赖
├── README.md                   # 项目说明
├── scripts/
│   ├── train_yolo.py           # 训练 YOLO 模型
│   ├── infer_trajectory.py     # 视频推理，导出轨迹 CSV
│   ├── convert_physical.py     # 像素坐标转物理坐标
│   ├── export_tracking_results.py  # 导出效果视频和位移曲线
│   ├── export_opencv_preview.py    # 导出单帧 OpenCV 预览图
│   ├── batch_color_tracking_demo.py # 批量颜色跟踪演示
│   ├── extract_frames.py       # 视频抽帧构建数据集
│   ├── clean_yolo_labels.py    # 标签清洗与归档
│   ├── browser_annotator.py    # 矩形框标注工具
│   └── browser_polygon_annotator.py # 多边形标注工具
└── videos/                     # 原始视频
```

### 模块说明

- `dataset.yaml`：训练时的数据集配置文件。
- `scripts/train_yolo.py`：训练入口，负责生成 `marker_tracking/run_1/weights/best.pt`。
- `scripts/infer_trajectory.py`：对视频逐帧检测，输出轨迹中心点坐标。
- `scripts/convert_physical.py`：把像素轨迹换算成真实物理量。
- `scripts/export_tracking_results.py`：生成带检测框的效果视频和位移时间曲线。
- `scripts/export_opencv_preview.py`：导出单帧 OpenCV 预览图，便于检查原图和边缘。
- `scripts/batch_color_tracking_demo.py`：对 `videos/4.1` 中最小的 5 个视频做批量颜色跟踪演示，属于备用结果生成流程。
- `scripts/extract_frames.py`：从视频中抽帧。
- `scripts/clean_yolo_labels.py`：清理和整理标签格式。
- `scripts/browser_annotator.py` / `scripts/browser_polygon_annotator.py`：网页标注工具。

## 3. 使用方法

### 3.1 安装依赖

建议统一使用项目根目录下的虚拟环境：

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

如果还需要生成曲线图或预览图，也要确保 `matplotlib` 已安装：

```bash
./.venv/bin/python -m pip install matplotlib
```

### 3.2 训练模型

```bash
./.venv/bin/python scripts/train_yolo.py --device mps
```

可选参数：

- `--epochs 100`
- `--imgsz 640`
- `--batch 16`
- `--name run_1`

### 3.3 视频推理

```bash
./.venv/bin/python scripts/infer_trajectory.py \
  --video source_video.mp4 \
  --weights marker_tracking/run_1/weights/best.pt \
  --device mps
```

输出默认保存在 `outputs/trajectory.csv`。

### 3.4 生成效果视频和位移曲线

```bash
./.venv/bin/python scripts/export_tracking_results.py \
  --video source_video.mp4 \
  --y0 320 \
  --k 0.5
```

输出目录中会包含：
- `annotated.mp4`
- `trajectory.csv`
- `displacement_curve.png`

### 3.5 像素转物理坐标

```bash
./.venv/bin/python scripts/convert_physical.py --k 0.5 --y0 320
```

### 3.6 导出 OpenCV 预览图

```bash
./.venv/bin/python scripts/export_opencv_preview.py \
  --video videos/4.1/0.mp4 \
  --frame-index 0 \
  --output outputs/opencv_preview.jpg
```

### 3.7 批量处理 `videos/4.1` 中最小的 5 个视频

```bash
./.venv/bin/python scripts/batch_color_tracking_demo.py
```

结果会输出到：`outputs/4.1_smallest5/`

## 关于运动模糊

红色标识物在剧烈运动时**通常会出现一定程度的边缘模糊**，尤其在曝光时间较长或运动速度较快的帧中更明显。这个现象会影响标注边界和检测稳定性，因此建议在标注时尽量包住模糊区域，并在训练时保留包含模糊的样本。
