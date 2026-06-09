from __future__ import annotations

import argparse
import html
import json
import os
import pathlib
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(images_root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        path for path in images_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def labels_root_for_image(image_path: pathlib.Path, images_root: pathlib.Path, labels_root: pathlib.Path) -> pathlib.Path:
    relative_path = image_path.relative_to(images_root)
    return labels_root / relative_path.parent / f"{image_path.stem}.txt"


def bbox_to_polygon(cx: float, cy: float, width: float, height: float) -> list[list[float]]:
    half_width = width / 2.0
    half_height = height / 2.0
    return [
        [cx - half_width, cy - half_height],
        [cx + half_width, cy - half_height],
        [cx + half_width, cy + half_height],
        [cx - half_width, cy + half_height],
    ]


def parse_annotation_line(raw_line: str) -> dict | None:
    line = raw_line.strip()
    if not line:
        return None

    parts = line.split()
    if len(parts) < 5:
        return None

    try:
        class_id = int(parts[0])
        coords = [float(value) for value in parts[1:]]
    except ValueError:
        return None

    if len(coords) == 4:
        points = bbox_to_polygon(coords[0], coords[1], coords[2], coords[3])
        return {"class_id": class_id, "points": points}

    if len(coords) < 6 or len(coords) % 2 != 0:
        return None

    points = [[coords[index], coords[index + 1]] for index in range(0, len(coords), 2)]
    if len(points) < 3:
        return None

    return {"class_id": class_id, "points": points}


def load_annotations(label_path: pathlib.Path) -> list[dict]:
    if not label_path.exists():
        return []

    annotations: list[dict] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        annotation = parse_annotation_line(raw_line)
        if annotation is not None:
            annotations.append(annotation)
    return annotations


def save_annotations(label_path: pathlib.Path, annotations: list[dict]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for annotation in annotations:
      class_id = int(annotation.get("class_id", 0))
      points = annotation.get("points", [])
      if not isinstance(points, list) or len(points) < 3:
          continue

      normalized: list[str] = [str(class_id)]
      valid_point_count = 0
      for point in points:
          if not isinstance(point, (list, tuple)) or len(point) != 2:
              continue
          try:
              x = float(point[0])
              y = float(point[1])
          except (TypeError, ValueError):
              continue
          x = max(0.0, min(1.0, x))
          y = max(0.0, min(1.0, y))
          normalized.append(f"{x:.6f}")
          normalized.append(f"{y:.6f}")
          valid_point_count += 1

      if valid_point_count >= 3:
          lines.append(" ".join(normalized))

    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def html_template(title: str) -> str:
    template = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #0b1020;
      --panel: rgba(16, 22, 42, 0.92);
      --panel-border: rgba(255, 255, 255, 0.08);
      --text: #eef2ff;
      --muted: #9aa4bf;
      --accent: #ff5b5f;
      --accent-2: #6ee7ff;
      --warning: #ffb84d;
      --success: #44d19e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(255, 91, 95, 0.2), transparent 28%),
        radial-gradient(circle at top right, rgba(110, 231, 255, 0.16), transparent 26%),
        linear-gradient(180deg, #060913, var(--bg));
      color: var(--text);
      min-height: 100vh;
    }
    header {
      display: flex;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      padding: 16px 20px;
      border-bottom: 1px solid var(--panel-border);
      background: rgba(6, 9, 19, 0.75);
      position: sticky;
      top: 0;
      backdrop-filter: blur(12px);
      z-index: 10;
    }
    header h1 {
      margin: 0;
      font-size: 18px;
      letter-spacing: 0.04em;
    }
    header .meta { color: var(--muted); font-size: 13px; }
    main {
      display: grid;
      grid-template-columns: minmax(300px, 360px) 1fr;
      gap: 16px;
      padding: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 18px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
      overflow: hidden;
    }
    .sidebar { padding: 16px; display: flex; flex-direction: column; gap: 14px; }
    .nav { display: flex; gap: 8px; flex-wrap: wrap; }
    button, .button {
      appearance: none;
      border: 0;
      border-radius: 12px;
      padding: 10px 12px;
      background: rgba(255, 255, 255, 0.08);
      color: var(--text);
      cursor: pointer;
      font-weight: 600;
    }
    button.primary { background: linear-gradient(135deg, var(--accent), #ff8a3d); }
    button.secondary { background: rgba(110, 231, 255, 0.16); }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    .stat {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.05);
    }
    .stat strong { color: var(--accent-2); }
    .image-list {
      max-height: 34vh;
      overflow: auto;
      border-radius: 14px;
      border: 1px solid var(--panel-border);
      background: rgba(255, 255, 255, 0.03);
    }
    .image-item {
      padding: 8px 10px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      cursor: pointer;
      font-size: 13px;
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .image-item.active { color: var(--text); background: rgba(255, 91, 95, 0.14); }
    .image-item.saved { color: var(--success); }
    .help { font-size: 12px; color: var(--muted); line-height: 1.7; }
    .canvas-panel {
      display: flex;
      flex-direction: column;
      min-height: calc(100vh - 112px);
    }
    .canvas-toolbar {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--panel-border);
      align-items: center;
    }
    .canvas-toolbar .path { color: var(--muted); font-size: 13px; word-break: break-all; }
    .stage {
      position: relative;
      flex: 1;
      overflow: auto;
      padding: 14px;
      display: flex;
      justify-content: center;
      align-items: flex-start;
      background:
        linear-gradient(45deg, rgba(255,255,255,0.02) 25%, transparent 25%),
        linear-gradient(-45deg, rgba(255,255,255,0.02) 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, rgba(255,255,255,0.02) 75%),
        linear-gradient(-45deg, transparent 75%, rgba(255,255,255,0.02) 75%);
      background-size: 28px 28px;
      background-position: 0 0, 0 14px, 14px -14px, -14px 0px;
    }
    .viewport { position: relative; display: inline-block; user-select: none; }
    .image-layer {
      display: block;
      max-width: none;
      image-rendering: auto;
      user-select: none;
      pointer-events: none;
    }
    .overlay {
      position: absolute;
      inset: 0;
      overflow: visible;
      touch-action: none;
      cursor: crosshair;
    }
    .polygon-shape {
      fill: rgba(255, 91, 95, 0.14);
      stroke: rgba(255, 91, 95, 0.95);
      stroke-width: 2;
      cursor: pointer;
    }
    .polygon-shape.active {
      fill: rgba(255, 184, 77, 0.14);
      stroke: rgba(255, 184, 77, 0.98);
      stroke-width: 3;
    }
    .polygon-draft {
      fill: rgba(110, 231, 255, 0.08);
      stroke: rgba(110, 231, 255, 0.98);
      stroke-width: 2;
      stroke-dasharray: 7 6;
      pointer-events: none;
    }
    .vertex {
      fill: rgba(255, 255, 255, 0.98);
      stroke: rgba(10, 14, 24, 0.8);
      stroke-width: 1.5;
      cursor: pointer;
    }
    .vertex.active {
      fill: var(--warning);
      stroke: rgba(40, 26, 0, 0.8);
      stroke-width: 2;
    }
    .draft-vertex {
      fill: var(--accent-2);
      stroke: rgba(0, 0, 0, 0.7);
      stroke-width: 1.25;
      pointer-events: none;
    }
    .rubber-line {
      fill: none;
      stroke: var(--accent-2);
      stroke-width: 2;
      stroke-dasharray: 6 5;
      pointer-events: none;
    }
    .footer { padding: 10px 16px 14px; color: var(--muted); font-size: 12px; border-top: 1px solid var(--panel-border); }
    input[type=range] { width: 160px; }
    .tiny { font-size: 12px; color: var(--muted); }
    @media (max-width: 1060px) {
      main { grid-template-columns: 1fr; }
      .canvas-panel { min-height: 70vh; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>YOLO 多边形标注器</h1>
      <div class="meta">单类：red_marker · 支持闭合、插点、拖点、删除点 · 导出 YOLO 实例分割格式</div>
    </div>
    <div class="meta" id="saveState">未保存</div>
  </header>
  <main>
    <section class="panel sidebar">
      <div class="nav">
        <button id="prevBtn">上一张</button>
        <button id="nextBtn">下一张</button>
        <button id="saveBtn" class="primary">保存 YOLO txt</button>
        <button id="newBtn" class="secondary">新建多边形</button>
        <button id="closeBtn" class="secondary">闭合当前</button>
        <button id="deletePolygonBtn" class="secondary">删选中多边形</button>
      </div>
      <div class="stat"><span>图片进度</span><strong id="progress">0 / 0</strong></div>
      <div class="stat"><span>多边形数量</span><strong id="polygonCount">0</strong></div>
      <div class="stat"><span>缩放</span><strong><input id="zoom" type="range" min="0.4" max="1.6" step="0.05" value="1" /></strong></div>
      <div class="tiny">快捷键：W 开始/继续绘制，A/D 或 ←/→ 切换图片，Backspace 删除选中顶点，Ctrl/Cmd+S 保存。左键点顶点添加点，双击或点起点闭合。</div>
      <div class="image-list" id="imageList"></div>
      <div class="help">
        <div>1. 点击 <b>新建多边形</b> 或按 <b>W</b> 开始绘制。</div>
        <div>2. 鼠标逐点点击生成顶点，线段会跟随鼠标移动。</div>
        <div>3. 闭合后可拖动顶点微调；点击边线可插入新顶点；右键顶点可删除。</div>
        <div>4. 保存会写入对应的 <code>dataset/labels/train</code> 或 <code>dataset/labels/val</code>。</div>
      </div>
    </section>
    <section class="panel canvas-panel">
      <div class="canvas-toolbar">
        <div>
          <div id="currentName" style="font-weight:700">未加载</div>
          <div class="path" id="currentPath"></div>
        </div>
        <div class="tiny" id="hint">点击新建多边形或按 W 进入绘制模式</div>
      </div>
      <div class="stage" id="stage">
        <div class="viewport" id="viewport">
          <img class="image-layer" id="image" alt="annotator image" />
          <svg class="overlay" id="overlay" xmlns="http://www.w3.org/2000/svg"></svg>
        </div>
      </div>
      <div class="footer" id="status">准备就绪。</div>
    </section>
  </main>
  <script>
    const state = {
      images: [],
      index: 0,
      imageSize: { width: 0, height: 0 },
      scale: 1,
      polygons: [],
      activePolygonIndex: -1,
      selectedVertexIndex: -1,
      draft: null,
      drawMode: false,
      dragging: null,
      dirty: false,
    };

    const els = {
      imageList: document.getElementById('imageList'),
      progress: document.getElementById('progress'),
      polygonCount: document.getElementById('polygonCount'),
      currentName: document.getElementById('currentName'),
      currentPath: document.getElementById('currentPath'),
      status: document.getElementById('status'),
      saveState: document.getElementById('saveState'),
      overlay: document.getElementById('overlay'),
      image: document.getElementById('image'),
      stage: document.getElementById('stage'),
      viewport: document.getElementById('viewport'),
      zoom: document.getElementById('zoom'),
      prevBtn: document.getElementById('prevBtn'),
      nextBtn: document.getElementById('nextBtn'),
      saveBtn: document.getElementById('saveBtn'),
      newBtn: document.getElementById('newBtn'),
      closeBtn: document.getElementById('closeBtn'),
      deletePolygonBtn: document.getElementById('deletePolygonBtn'),
      hint: document.getElementById('hint'),
    };

    function setStatus(message, warn = false) {
      els.status.textContent = message;
      els.saveState.textContent = message;
      els.saveState.style.color = warn ? 'var(--warning)' : 'var(--success)';
    }

    async function api(path, options = {}) {
      const response = await fetch(path, options);
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || response.statusText);
      }
      const contentType = response.headers.get('content-type') || '';
      if (contentType.includes('application/json')) {
        return response.json();
      }
      return response.text();
    }

    function currentImage() {
      return state.images[state.index];
    }

    function activePolygon() {
      if (state.activePolygonIndex < 0) return null;
      return state.polygons[state.activePolygonIndex] || null;
    }

    function clamp(value, minimum, maximum) {
      return Math.max(minimum, Math.min(maximum, value));
    }

    function pointDistance(a, b) {
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      return Math.hypot(dx, dy);
    }

    function pointToSegmentDistance(point, start, end) {
      const dx = end.x - start.x;
      const dy = end.y - start.y;
      const lengthSquared = dx * dx + dy * dy;
      if (lengthSquared === 0) {
        return pointDistance(point, start);
      }
      const t = clamp(((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared, 0, 1);
      const projection = { x: start.x + t * dx, y: start.y + t * dy };
      return { distance: pointDistance(point, projection), projection, t };
    }

    function pointInPolygon(point, polygon) {
      let inside = false;
      const points = polygon.points;
      for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
        const xi = points[i].x;
        const yi = points[i].y;
        const xj = points[j].x;
        const yj = points[j].y;
        const intersect = ((yi > point.y) !== (yj > point.y)) &&
          (point.x < ((xj - xi) * (point.y - yi)) / ((yj - yi) || 1e-12) + xi);
        if (intersect) inside = !inside;
      }
      return inside;
    }

    function isNearVertex(point, vertex, threshold) {
      return pointDistance(point, vertex) <= threshold;
    }

    function findVertexHit(point, threshold, preferActive = true) {
      const polygonOrder = [];
      if (preferActive && state.activePolygonIndex >= 0) {
        polygonOrder.push(state.activePolygonIndex);
      }
      for (let index = 0; index < state.polygons.length; index += 1) {
        if (index !== state.activePolygonIndex) {
          polygonOrder.push(index);
        }
      }

      for (const polygonIndex of polygonOrder) {
        const polygon = state.polygons[polygonIndex];
        for (let vertexIndex = 0; vertexIndex < polygon.points.length; vertexIndex += 1) {
          if (isNearVertex(point, polygon.points[vertexIndex], threshold)) {
            return { polygonIndex, vertexIndex };
          }
        }
      }
      return null;
    }

    function findEdgeHit(point, threshold, preferActive = true) {
      const polygonOrder = [];
      if (preferActive && state.activePolygonIndex >= 0) {
        polygonOrder.push(state.activePolygonIndex);
      }
      for (let index = 0; index < state.polygons.length; index += 1) {
        if (index !== state.activePolygonIndex) {
          polygonOrder.push(index);
        }
      }

      let best = null;
      for (const polygonIndex of polygonOrder) {
        const polygon = state.polygons[polygonIndex];
        for (let startIndex = 0; startIndex < polygon.points.length; startIndex += 1) {
          const endIndex = (startIndex + 1) % polygon.points.length;
          const start = polygon.points[startIndex];
          const end = polygon.points[endIndex];
          const result = pointToSegmentDistance(point, start, end);
          if (result.distance <= threshold && (!best || result.distance < best.distance)) {
            best = { polygonIndex, startIndex, endIndex, projection: result.projection, distance: result.distance };
          }
        }
      }
      return best;
    }

    function imagePointFromEvent(event) {
      const rect = els.overlay.getBoundingClientRect();
      const x = (event.clientX - rect.left) / state.scale;
      const y = (event.clientY - rect.top) / state.scale;
      return {
        x: clamp(x, 0, state.imageSize.width),
        y: clamp(y, 0, state.imageSize.height),
      };
    }

    function scaledPoint(point) {
      return {
        x: point.x * state.scale,
        y: point.y * state.scale,
      };
    }

    function renderList() {
      els.imageList.innerHTML = '';
      state.images.forEach((item, index) => {
        const div = document.createElement('div');
        div.className = 'image-item' + (index === state.index ? ' active' : '');
        div.textContent = item.relpath;
        div.dataset.saved = item.saved ? 'true' : 'false';
        div.addEventListener('click', () => loadImage(index));
        els.imageList.appendChild(div);
      });
      Array.from(els.imageList.children).forEach((item) => {
        item.classList.toggle('saved', item.dataset.saved === 'true');
      });
    }

    function updateProgress() {
      els.progress.textContent = `${state.index + 1} / ${state.images.length}`;
      const item = currentImage();
      if (!item) return;
      els.currentName.textContent = item.name;
      els.currentPath.textContent = item.relpath;
      els.prevBtn.disabled = state.index <= 0;
      els.nextBtn.disabled = state.index >= state.images.length - 1;
      els.polygonCount.textContent = String(state.polygons.length + (state.draft ? 1 : 0));
    }

    function refreshViewport() {
      const width = Math.round(state.imageSize.width * state.scale);
      const height = Math.round(state.imageSize.height * state.scale);
      els.image.style.width = `${width}px`;
      els.image.style.height = `${height}px`;
      els.overlay.setAttribute('viewBox', `0 0 ${width} ${height}`);
      els.overlay.setAttribute('width', width);
      els.overlay.setAttribute('height', height);
      els.viewport.style.width = `${width}px`;
      els.viewport.style.height = `${height}px`;
    }

    function polygonToPointsAttribute(points) {
      return points.map((point) => `${point.x * state.scale},${point.y * state.scale}`).join(' ');
    }

    function renderSvg() {
      const elements = [];

      state.polygons.forEach((polygon, polygonIndex) => {
        const pointsAttribute = polygonToPointsAttribute(polygon.points);
        const active = polygonIndex === state.activePolygonIndex;
        elements.push(`<polygon class="polygon-shape${active ? ' active' : ''}" data-kind="polygon" data-polygon="${polygonIndex}" points="${pointsAttribute}"></polygon>`);
        polygon.points.forEach((point, vertexIndex) => {
          const scaled = scaledPoint(point);
          elements.push(`<circle class="vertex${active && vertexIndex === state.selectedVertexIndex ? ' active' : ''}" data-kind="vertex" data-polygon="${polygonIndex}" data-vertex="${vertexIndex}" cx="${scaled.x}" cy="${scaled.y}" r="6"></circle>`);
        });
      });

      if (state.draft && state.draft.points.length > 0) {
        const draftPoints = state.draft.points.map((point) => `${point.x * state.scale},${point.y * state.scale}`).join(' ');
        elements.push(`<polyline class="polygon-draft" points="${draftPoints}"></polyline>`);
        state.draft.points.forEach((point) => {
          const scaled = scaledPoint(point);
          elements.push(`<circle class="draft-vertex" cx="${scaled.x}" cy="${scaled.y}" r="5"></circle>`);
        });
        const lastPoint = state.draft.points[state.draft.points.length - 1];
        const hoverPoint = state.draft.hover || lastPoint;
        if (lastPoint && hoverPoint) {
          const a = scaledPoint(lastPoint);
          const b = scaledPoint(hoverPoint);
          elements.push(`<line class="rubber-line" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"></line>`);
        }
      }

      els.overlay.innerHTML = elements.join('');
    }

    function renderAll() {
      if (!state.imageSize.width || !state.imageSize.height) return;
      refreshViewport();
      renderSvg();
      updateProgress();
      renderList();
      els.deletePolygonBtn.disabled = state.activePolygonIndex < 0;
      els.closeBtn.disabled = !state.draft || state.draft.points.length < 3;
    }

    function setDirty(message) {
      state.dirty = true;
      setStatus(message, true);
      renderAll();
    }

    function startDraft() {
      if (!state.draft) {
        state.draft = { points: [], hover: null };
      }
      state.drawMode = true;
      state.selectedVertexIndex = -1;
      els.hint.textContent = '依次点击顶点，双击或点起点闭合';
    }

    function finalizeDraft() {
      if (!state.draft || state.draft.points.length < 3) return false;
      state.polygons.push({ class_id: 0, points: state.draft.points.map((point) => ({ x: point.x, y: point.y })) });
      state.activePolygonIndex = state.polygons.length - 1;
      state.selectedVertexIndex = -1;
      state.draft = null;
      state.drawMode = false;
      setDirty('多边形已闭合，尚未保存');
      return true;
    }

    function addDraftPoint(point) {
      if (!state.draft) {
        state.draft = { points: [], hover: null };
      }
      state.draft.points.push({ x: point.x, y: point.y });
      state.draft.hover = { x: point.x, y: point.y };
      state.drawMode = true;
      setDirty('已添加顶点，尚未保存');
    }

    function polygonAtPoint(point) {
      for (let index = state.polygons.length - 1; index >= 0; index -= 1) {
        if (pointInPolygon(point, state.polygons[index])) {
          return index;
        }
      }
      return -1;
    }

    function insertVertexAt(polygonIndex, edgeIndex, point) {
      const polygon = state.polygons[polygonIndex];
      const insertIndex = edgeIndex + 1;
      polygon.points.splice(insertIndex, 0, { x: point.x, y: point.y });
      state.activePolygonIndex = polygonIndex;
      state.selectedVertexIndex = insertIndex;
      setDirty('已插入新顶点');
    }

    function deleteActiveVertex(polygonIndex, vertexIndex) {
      const polygon = state.polygons[polygonIndex];
      polygon.points.splice(vertexIndex, 1);
      if (polygon.points.length < 3) {
        state.polygons.splice(polygonIndex, 1);
        state.activePolygonIndex = state.polygons.length - 1;
        state.selectedVertexIndex = -1;
      } else {
        state.activePolygonIndex = polygonIndex;
        state.selectedVertexIndex = Math.min(vertexIndex, polygon.points.length - 1);
      }
      setDirty('已删除顶点');
    }

    function deleteActivePolygon() {
      if (state.activePolygonIndex < 0) return;
      state.polygons.splice(state.activePolygonIndex, 1);
      state.activePolygonIndex = state.polygons.length - 1;
      state.selectedVertexIndex = -1;
      setDirty('已删除多边形');
    }

    function normalizeAnnotationsForSave() {
      const annotations = [];
      state.polygons.forEach((polygon) => {
        if (!polygon.points || polygon.points.length < 3) return;
        annotations.push({
          class_id: 0,
          points: polygon.points.map((point) => ({
            x: point.x / state.imageSize.width,
            y: point.y / state.imageSize.height,
          })),
        });
      });
      return annotations;
    }

    async function saveCurrent(silent = true) {
      if (!currentImage()) return;

      if (state.draft && state.draft.points.length >= 3) {
        finalizeDraft();
      }

      const response = await api('/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          path: currentImage().relpath,
          annotations: normalizeAnnotationsForSave(),
        }),
      });

      state.dirty = false;
      state.images[state.index].saved = true;
      renderList();
      if (!silent) {
        setStatus(response.message || '已保存');
      } else {
        setStatus('已保存');
      }
    }

    async function loadImage(index, allowAutoSave = true) {
      if (index < 0 || index >= state.images.length) return;
      if (allowAutoSave && state.dirty) {
        await saveCurrent(false);
      }

      state.index = index;
      state.polygons = [];
      state.activePolygonIndex = -1;
      state.selectedVertexIndex = -1;
      state.draft = null;
      state.drawMode = false;
      state.dragging = null;
      state.dirty = false;

      updateProgress();
      const item = currentImage();
      setStatus(`加载中：${item.name}`, true);

      const imageUrl = `/image?path=${encodeURIComponent(item.relpath)}`;
      const image = new Image();
      image.decoding = 'async';
      image.src = imageUrl;
      await image.decode();

      state.imageSize = { width: image.naturalWidth, height: image.naturalHeight };
      els.image.src = imageUrl;
      els.image.alt = item.name;

      const payload = await api(`/annotation?path=${encodeURIComponent(item.relpath)}`);
      state.polygons = (payload.annotations || []).map((annotation) => ({
        class_id: 0,
        points: annotation.points.map((point) => ({ x: point[0], y: point[1] })),
      }));
      state.activePolygonIndex = state.polygons.length > 0 ? 0 : -1;
      state.selectedVertexIndex = -1;

      renderAll();
      setStatus(`已加载：${item.name}`);
    }

    function hitTestPolygonElement(target) {
      const polygonIndex = Number(target.dataset.polygon);
      if (Number.isNaN(polygonIndex)) return -1;
      return polygonIndex;
    }

    els.overlay.addEventListener('mousemove', (event) => {
      if (!state.imageSize.width || !state.imageSize.height) return;
      const point = imagePointFromEvent(event);
      if (state.draft) {
        state.draft.hover = point;
        renderSvg();
      }
      if (state.dragging) {
        const polygon = state.polygons[state.dragging.polygonIndex];
        if (polygon) {
          polygon.points[state.dragging.vertexIndex] = { x: point.x, y: point.y };
          renderSvg();
          state.dirty = true;
        }
      }
    });

    els.overlay.addEventListener('mousedown', (event) => {
      if (!state.imageSize.width || !state.imageSize.height) return;
      const point = imagePointFromEvent(event);
      const threshold = 8 / state.scale;

      if (event.button === 2) {
        event.preventDefault();
        const vertexHit = findVertexHit(point, threshold);
        if (vertexHit) {
          deleteActiveVertex(vertexHit.polygonIndex, vertexHit.vertexIndex);
        }
        return;
      }

      const targetKind = event.target && event.target.dataset ? event.target.dataset.kind : '';
      if (state.draft) {
        const firstPoint = state.draft.points[0];
        if (firstPoint && state.draft.points.length >= 3 && isNearVertex(point, firstPoint, threshold)) {
          finalizeDraft();
          renderAll();
          return;
        }
        addDraftPoint(point);
        renderAll();
        return;
      }

      if (targetKind === 'vertex') {
        const polygonIndex = Number(event.target.dataset.polygon);
        const vertexIndex = Number(event.target.dataset.vertex);
        if (!Number.isNaN(polygonIndex) && !Number.isNaN(vertexIndex)) {
          state.activePolygonIndex = polygonIndex;
          state.selectedVertexIndex = vertexIndex;
          state.dragging = { polygonIndex, vertexIndex };
          setDirty('拖动顶点中...');
        }
        return;
      }

      if (targetKind === 'polygon') {
        const polygonIndex = hitTestPolygonElement(event.target);
        if (polygonIndex >= 0) {
          state.activePolygonIndex = polygonIndex;
          state.selectedVertexIndex = -1;
          renderAll();
        }
        return;
      }

      const edgeHit = findEdgeHit(point, threshold);
      if (edgeHit) {
        insertVertexAt(edgeHit.polygonIndex, edgeHit.startIndex, edgeHit.projection);
        renderAll();
        return;
      }

      const polygonIndex = polygonAtPoint(point);
      if (polygonIndex >= 0) {
        state.activePolygonIndex = polygonIndex;
        state.selectedVertexIndex = -1;
        renderAll();
      }
    });

    window.addEventListener('mouseup', () => {
      if (state.dragging) {
        state.dragging = null;
        state.dirty = true;
        setStatus('顶点已调整，尚未保存', true);
      }
    });

    els.overlay.addEventListener('dblclick', (event) => {
      if (!state.draft) return;
      const point = imagePointFromEvent(event);
      const threshold = 8 / state.scale;
      if (state.draft.points.length === 0) return;
      const lastPoint = state.draft.points[state.draft.points.length - 1];
      if (!isNearVertex(point, lastPoint, threshold)) {
        state.draft.points.push(point);
      }
      finalizeDraft();
      renderAll();
    });

    els.overlay.addEventListener('contextmenu', (event) => {
      event.preventDefault();
    });

    els.newBtn.addEventListener('click', () => {
      startDraft();
      renderAll();
      setStatus('绘制模式已开启', false);
    });

    els.closeBtn.addEventListener('click', () => {
      if (finalizeDraft()) {
        renderAll();
      }
    });

    els.deletePolygonBtn.addEventListener('click', () => {
      deleteActivePolygon();
    });

    els.saveBtn.addEventListener('click', () => saveCurrent(false));
    els.prevBtn.addEventListener('click', () => loadImage(state.index - 1));
    els.nextBtn.addEventListener('click', () => loadImage(state.index + 1));

    els.zoom.addEventListener('input', () => {
      state.scale = Number(els.zoom.value);
      renderAll();
    });

    window.addEventListener('keydown', async (event) => {
      const key = event.key.toLowerCase();
      if ((event.ctrlKey || event.metaKey) && key === 's') {
        event.preventDefault();
        await saveCurrent(false);
        return;
      }
      if (key === 'a' || key === 'arrowleft') {
        event.preventDefault();
        await loadImage(Math.max(0, state.index - 1));
        return;
      }
      if (key === 'd' || key === 'arrowright') {
        event.preventDefault();
        await loadImage(Math.min(state.images.length - 1, state.index + 1));
        return;
      }
      if (key === 'backspace' || key === 'delete') {
        event.preventDefault();
        if (state.activePolygonIndex >= 0 && state.selectedVertexIndex >= 0) {
          deleteActiveVertex(state.activePolygonIndex, state.selectedVertexIndex);
        }
        return;
      }
      if (key === 'w') {
        event.preventDefault();
        startDraft();
        renderAll();
        setStatus('绘制模式已开启', false);
      }
    });

    async function bootstrap() {
      const payload = await api('/api/images');
      state.images = payload.images || [];
      updateProgress();
      renderList();
      if (!state.images.length) {
        setStatus('没有找到图片，请检查 dataset/images', true);
        return;
      }
      await loadImage(0, false);
    }

    bootstrap().catch((error) => {
      console.error(error);
      setStatus(error.message || String(error), true);
    });
  </script>
</body>
</html>"""
    return template.replace("__TITLE__", html.escape(title))


class AnnotatorHandler(SimpleHTTPRequestHandler):
    server_version = "YOLOPolygonAnnotator/1.0"

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    @property
    def app(self):
        return self.server.app  # type: ignore[attr-defined]

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except ConnectionResetError:
            return

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _safe_image_path(self, relative_path: str) -> pathlib.Path:
        relative = pathlib.Path(relative_path)
        candidate = (self.app.images_root / relative).resolve()
        if self.app.images_root not in candidate.parents and candidate != self.app.images_root:
            raise ValueError("invalid image path")
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            html_body = html_template(self.app.title).encode("utf-8")
            self._send_bytes(html_body, "text/html; charset=utf-8")
            return

        if parsed.path == "/api/images":
            images = []
            for image_path in self.app.images:
                relative = image_path.relative_to(self.app.images_root)
                label_path = labels_root_for_image(image_path, self.app.images_root, self.app.labels_root)
                images.append({
                    "name": image_path.name,
                    "relpath": str(relative).replace(os.sep, "/"),
                    "label": str(label_path),
                    "saved": label_path.exists(),
                })
            self._send_json({"images": images})
            return

        if parsed.path == "/image":
            query = urllib.parse.parse_qs(parsed.query)
            relpath = query.get("path", [""])[0]
            image_path = self._safe_image_path(relpath)
            body = image_path.read_bytes()
            suffix = image_path.suffix.lower()
            content_type = "image/jpeg"
            if suffix == ".png":
                content_type = "image/png"
            elif suffix == ".bmp":
                content_type = "image/bmp"
            elif suffix == ".webp":
                content_type = "image/webp"
            self._send_bytes(body, content_type)
            return

        if parsed.path == "/annotation":
            query = urllib.parse.parse_qs(parsed.query)
            relpath = query.get("path", [""])[0]
            image_path = self._safe_image_path(relpath)
            label_path = labels_root_for_image(image_path, self.app.images_root, self.app.labels_root)
            annotations = load_annotations(label_path)
            self._send_json({"annotations": annotations, "label_path": str(label_path)})
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/save":
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return

        payload = self._read_json()
        relpath = payload.get("path")
        annotations = payload.get("annotations", [])
        if not isinstance(relpath, str) or not isinstance(annotations, list):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid payload")
            return

        image_path = self._safe_image_path(relpath)
        label_path = labels_root_for_image(image_path, self.app.images_root, self.app.labels_root)

        cleaned: list[dict] = []
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            points = annotation.get("points", [])
            if not isinstance(points, list) or len(points) < 3:
                continue
            cleaned_points: list[list[float]] = []
            for point in points:
                if not isinstance(point, dict):
                    continue
                try:
                    x = float(point.get("x"))
                    y = float(point.get("y"))
                except (TypeError, ValueError):
                    continue
                cleaned_points.append([x, y])
            if len(cleaned_points) >= 3:
                cleaned.append({"class_id": int(annotation.get("class_id", 0)), "points": cleaned_points})

        save_annotations(label_path, cleaned)
        self._send_json({"ok": True, "message": f"已保存 {label_path}"})


class AnnotatorApp:
    def __init__(self, images_root: pathlib.Path, labels_root: pathlib.Path, title: str) -> None:
        self.images_root = images_root.resolve()
        self.labels_root = labels_root.resolve()
        self.title = title
        self.images = iter_images(self.images_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="本地浏览器版 YOLO 多边形标注器")
    parser.add_argument("--images-root", type=pathlib.Path, default=pathlib.Path("dataset/images"), help="图片根目录")
    parser.add_argument("--labels-root", type=pathlib.Path, default=pathlib.Path("dataset/labels"), help="标签根目录")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--title", default="YOLO 多边形标注器", help="页面标题")
    parser.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    app = AnnotatorApp(args.images_root, args.labels_root, args.title)

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = Server((args.host, args.port), AnnotatorHandler)
    server.app = app  # type: ignore[attr-defined]

    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(f"http://{args.host}:{args.port}", new=2)).start()

    print(f"浏览器标注器已启动: http://{args.host}:{args.port}")
    print(f"图片目录: {app.images_root}")
    print(f"标签目录: {app.labels_root}")
    print("按 Ctrl+C 停止服务。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()