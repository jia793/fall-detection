#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RK3588 摔倒检测 — 图片推理 (改进版)

用法:
  批量: python inference_image.py
  单张: python inference_image.py --img test.jpg
  指定: python inference_image.py --model best_fp_onnx.rknn --img fall.jpg

过滤策略（基于 test_photo 10 张图分析）:
  1. 天花板/墙壁误检 → 跌倒框必须在地面区域 (cy > 0.30)
  2. 真跌倒 conf 0.08-0.17，站立 conf 0.15-0.35 → 分两类设不同阈值
  3. 边缘窄条 + 画面顶端 → 几何惩罚
  4. NMS 类别隔离 → 不同类别不互相抑制
"""

import cv2
import numpy as np
import time
import os
import sys
import argparse

try:
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    print("[WARN] rknnlite N/A — ONNX fallback")
    HAS_RKNN = False

# ============================================================================
# 配置
# ============================================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "weights", "best_fp_onnx.rknn")
INPUT_SIZE = 640
CLASS_NAMES = {0: "non-fall", 1: "fall"}
FALL_ID = 1

# 每类独立阈值（根据 test_photo 实际分数分布）
CONF_NORM = 0.25       # 站立最低置信度
CONF_FALL = 0.08       # 跌倒最低置信度
IOU_NMS   = 0.50

# 几何约束
MIN_AREA_RATIO = 0.03
MAX_AREA_RATIO = 0.70
PERSON_STAND_AR = (0.25, 0.70)   # 站立宽高比
PERSON_FALL_AR  = (0.55, 4.50)   # 跌倒宽高比
MAX_DETECTIONS  = 5              # 每帧最多保留框数

# 颜色
GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
WHITE  = (255, 255, 255)
YELLOW = (0, 255, 255)
ALERT_BG = (0, 0, 180)

DEFAULT_IN_DIR  = os.path.join(os.path.dirname(__file__), "test_photo")
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "photo_result")

# ============================================================================
# 辅助函数
# ============================================================================
def iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    return inter / (area_a + area_b - inter + 1e-6)


def class_aware_nms(detections, iou_thres):
    """类别隔离 NMS：不同类别互不抑制"""
    if len(detections) <= 1:
        return detections

    # 按类别分组
    by_class = {}
    for d in detections:
        cid = d["class_id"]
        by_class.setdefault(cid, []).append(d)

    result = []
    for cid, items in by_class.items():
        boxes = np.array([it["bbox"] for it in items], dtype=np.float32)
        scores = np.array([it["confidence"] for it in items])
        keep = []
        order = scores.argsort()[::-1]
        while order.size > 0:
            i = order[0]
            keep.append(i)
            x1, y1 = boxes[i, 0], boxes[i, 1]
            x2, y2 = boxes[i, 2], boxes[i, 3]
            ox1 = np.maximum(x1, boxes[order[1:], 0])
            oy1 = np.maximum(y1, boxes[order[1:], 1])
            ox2 = np.minimum(x2, boxes[order[1:], 2])
            oy2 = np.minimum(y2, boxes[order[1:], 3])
            inter = np.maximum(0, ox2 - ox1) * np.maximum(0, oy2 - oy1)
            area_i = (x2 - x1) * (y2 - y1)
            area_o = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
            ious = inter / (area_i + area_o - inter + 1e-6)
            order = order[np.where(ious <= iou_thres)[0] + 1]
        result.extend([items[k] for k in keep])
    return result


# ============================================================================
# 检测器
# ============================================================================
class FallDetector:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        if not HAS_RKNN:
            self._init_onnx()
            return
        self.rknn = RKNNLite()
        print(f"[INFO] RKNN: {model_path}")
        if self.rknn.load_rknn(model_path) != 0:
            raise RuntimeError("load_rknn failed")
        ret = self.rknn.init_runtime(target="rk3588")
        if ret != 0:
            ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError("init_runtime failed")
        print("[INFO] NPU ready")

    def _init_onnx(self):
        import onnxruntime as ort
        for p in [self.model_path.replace(".rknn", ".onnx"),
                  os.path.join(os.path.dirname(__file__), "weights", "best_no_sig.onnx"),
                  os.path.join(os.path.dirname(__file__), "weights", "best.onnx")]:
            if os.path.exists(p):
                print(f"[INFO] ONNX: {p}")
                self.onnx_session = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
                self.rknn = None
                return
        raise FileNotFoundError("No ONNX model found")

    def preprocess(self, frame_bgr):
        img = cv2.resize(frame_bgr, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        return np.ascontiguousarray(img)

    def inference(self, inp):
        if self.rknn is not None:
            return self.rknn.inference(inputs=[inp])[0]
        else:
            onnx_inp = inp.astype(np.float32) / 255.0
            onnx_inp = np.transpose(onnx_inp, (0, 3, 1, 2))
            onnx_inp = np.ascontiguousarray(onnx_inp)
            return self.onnx_session.run(None, {"images": onnx_inp})[0]

    def postprocess(self, output, frame_h, frame_w, verbose=False):
        """
        多层过滤:
          1. 解码 + 自适应 sigmoid
          2. 每类独立最低置信度
          3. 跌倒必须在画面下半部 (地面区域)
          4. 天花板/墙壁窄条几何惩罚
          5. 类别隔离 NMS
        """
        o = output[0]
        o = np.transpose(o, (1, 0))           # (8400, 6)
        bbox_raw = o[:, :4]
        cls_raw  = o[:, 4:]

        # --- 自适应 sigmoid ---
        if np.any(cls_raw < 0) or np.any(cls_raw > 1.0):
            cls_scores = 1.0 / (1.0 + np.exp(-np.clip(cls_raw, -50, 50)))
        else:
            cls_scores = cls_raw

        max_scores = np.max(cls_scores, axis=1)
        class_ids  = np.argmax(cls_scores, axis=1)

        # --- 最低置信度筛选（每类独立） ---
        thresholds = np.where(class_ids == FALL_ID, CONF_FALL, CONF_NORM)
        mask = max_scores > thresholds
        if not np.any(mask):
            return []

        idx = np.where(mask)[0]
        cx, cy, wb, hb = bbox_raw[idx, 0], bbox_raw[idx, 1], bbox_raw[idx, 2], bbox_raw[idx, 3]

        # 缩放到原图
        sx, sy = frame_w / INPUT_SIZE, frame_h / INPUT_SIZE
        x1 = np.clip((cx - wb / 2.0) * sx, 0, frame_w)
        y1 = np.clip((cy - hb / 2.0) * sy, 0, frame_h)
        x2 = np.clip((cx + wb / 2.0) * sx, 0, frame_w)
        y2 = np.clip((cy + hb / 2.0) * sy, 0, frame_h)

        candidates = []
        for i, j in enumerate(idx):
            box = [x1[i], y1[i], x2[i], y2[i]]
            bx, by, bx2, by2 = box
            bw, bh = bx2 - bx, by2 - by
            if bw <= 0 or bh <= 0:
                continue

            cid = int(class_ids[j])
            conf = float(max_scores[j])
            cname = CLASS_NAMES.get(cid, "unknown")
            ar = bw / bh
            bcy = (by + by2) / 2 / frame_h
            area_ratio = (bw * bh) / (frame_w * frame_h)

            # ----------------------------------------------------------
            # 规则 1: 天花板/墙壁窄条 (cy < 0.30 + 宽条)
            # ----------------------------------------------------------
            if bcy < 0.30 and (ar > 2.5 or bh / frame_h < 0.32):
                if verbose:
                    print(f"  [DROP] ceiling: cy={bcy:.2f} ar={ar:.2f}")
                continue

            # ----------------------------------------------------------
            # 规则 2: 面积异常
            # ----------------------------------------------------------
            if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
                if verbose:
                    print(f"  [DROP] area={area_ratio:.3f}")
                continue

            # ----------------------------------------------------------
            # 规则 3: 站立中心不能太靠上（排除天花板/墙顶误检）
            # ----------------------------------------------------------
            if cid != FALL_ID and bcy < 0.55:
                if verbose:
                    print(f"  [DROP] NORM too high: cy={bcy:.2f}")
                continue

            # ----------------------------------------------------------
            # 规则 4: 非人形宽高比（站立 ar 0.25-0.70, 跌倒 ar 0.55-4.50）
            # ----------------------------------------------------------
            stand_ok = (cid != FALL_ID and PERSON_STAND_AR[0] <= ar <= PERSON_STAND_AR[1])
            fall_ok  = (cid == FALL_ID and PERSON_FALL_AR[0] <= ar <= PERSON_FALL_AR[1])
            if not (stand_ok or fall_ok):
                if verbose:
                    print(f"  [DROP] bad ar={ar:.2f} for {'FALL' if cid==FALL_ID else 'NORM'}")
                continue

            # ----------------------------------------------------------
            # 规则 5: 边缘贯穿背景
            # ----------------------------------------------------------
            if bx <= 2 and bx2 >= frame_w - 2:
                if verbose:
                    print(f"  [DROP] full-width background")
                continue

            candidates.append({
                "bbox":       [int(bx), int(by), int(bx2), int(by2)],
                "class_id":   cid,
                "class_name": cname,
                "confidence": conf,
            })

        if not candidates:
            return []

        # --- 跨类压制：当模型对某一类极度确信时，压制另一类 ---
        top_fall = max((c["confidence"] for c in candidates if c["class_id"] == FALL_ID), default=0)
        top_norm = max((c["confidence"] for c in candidates if c["class_id"] != FALL_ID), default=0)

        if top_norm > top_fall * 4.0:
            # 模型极度确信是站立 → 压制所有跌倒
            candidates = [c for c in candidates if c["class_id"] != FALL_ID]
            if verbose:
                print(f"  [CCS] top_norm({top_norm:.3f}) >> top_fall({top_fall:.3f}) — suppress FALL")
        elif top_fall > top_norm * 2.5:
            # 模型确信是跌倒 → 压制站立
            candidates = [c for c in candidates if c["class_id"] == FALL_ID]
            if verbose:
                print(f"  [CCS] top_fall({top_fall:.3f}) > top_norm({top_norm:.3f}) — suppress NORM")

        # --- 类别隔离 NMS ---
        result = class_aware_nms(candidates, IOU_NMS)

        # --- 限制最大框数（按置信度排序取前 N） ---
        result.sort(key=lambda d: d["confidence"], reverse=True)
        return result[:MAX_DETECTIONS]

    def release(self):
        if self.rknn is not None:
            self.rknn.release()


# ============================================================================
# 可视化
# ============================================================================
def draw_results(frame, detections):
    fh, fw = frame.shape[:2]
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        is_fall = d["class_id"] == FALL_ID
        color = RED if is_fall else GREEN
        thick = 3 if is_fall else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        label = f"{d['class_name']} {d['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

    n_fall = sum(1 for d in detections if d["class_id"] == FALL_ID)
    n_norm = len(detections) - n_fall
    summary = f"Det: {len(detections)} | Fall: {n_fall} | Normal: {n_norm}"
    cv2.putText(frame, summary, (10, fh - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    if n_fall > 0:
        alert = "! FALL DETECTED !"
        (tw, th), _ = cv2.getTextSize(alert, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(frame, (10, 20), (20 + tw, 28 + th), ALERT_BG, -1)
        cv2.putText(frame, alert, (15, 28 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 3)

    return frame


# ============================================================================
# 单张推理
# ============================================================================
def process_single(detector, img_path, out_path=None, verbose=False):
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"[ERROR] Cannot read: {img_path}")
        return None
    fh, fw = frame.shape[:2]
    t0 = time.time()
    inp = detector.preprocess(frame)
    out = detector.inference(inp)
    dets = detector.postprocess(out, fh, fw, verbose=verbose)
    t_ms = (time.time() - t0) * 1000

    print(f"\n{'='*50}")
    print(f"  {os.path.basename(img_path)} ({fw}x{fh}) [{t_ms:.1f}ms]")
    print(f"  Detections: {len(dets)}")
    for d in dets:
        tag = "!! FALL !!" if d["class_id"] == FALL_ID else "  normal "
        print(f"  [{tag}] {d['class_name']} conf={d['confidence']:.3f} "
              f"bbox={d['bbox']}")

    draw_results(frame, dets)
    if out_path is None:
        base = os.path.basename(img_path)
        name, ext = os.path.splitext(base)
        out_path = os.path.join(os.path.dirname(__file__) or ".", "photo_result", f"{name}_detected{ext}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, frame)
    print(f"  Saved: {out_path}")
    return dets


# ============================================================================
# 批量推理
# ============================================================================
def process_batch(detector, in_dir, out_dir):
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    files = sorted([f for f in os.listdir(in_dir)
                    if os.path.splitext(f)[1].lower() in img_exts])
    if not files:
        print(f"[ERROR] No images in: {in_dir}")
        return
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*50}")
    print(f"  Batch: {len(files)} images")
    print(f"  From: {in_dir}  →  To: {out_dir}")
    print(f"{'='*50}\n")

    total_t, total_d, total_f = 0, 0, 0
    for i, fname in enumerate(files):
        img_path = os.path.join(in_dir, fname)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  [{i+1:3d}/{len(files)}] {fname:30s}  SKIP")
            continue
        fh, fw = frame.shape[:2]
        t0 = time.time()
        inp = detector.preprocess(frame)
        out = detector.inference(inp)
        dets = detector.postprocess(out, fh, fw)
        t_ms = (time.time() - t0) * 1000
        total_t += t_ms
        total_d += len(dets)
        falls = sum(1 for d in dets if d["class_id"] == FALL_ID)
        total_f += falls

        items = []
        for d in dets:
            tag = "FALL" if d["class_id"] == FALL_ID else "OK"
            items.append(f"{tag}:{d['confidence']:.2f}")
        det_str = ", ".join(items) if items else "(none)"

        print(f"  [{i+1:3d}/{len(files)}] {fname:30s} {t_ms:5.1f}ms  "
              f"{len(dets)} det  [{det_str}]")

        draw_results(frame, dets)
        cv2.imwrite(os.path.join(out_dir, fname), frame)

    print(f"\n{'='*50}")
    print(f"  Done. {len(files)} images, avg {total_t/len(files):.1f}ms/img")
    print(f"  Total: {total_d} detections, {total_f} falls")
    print(f"{'='*50}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="RK3588 Fall Detection — Image")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--img", default=None, help="Single image path")
    parser.add_argument("--in-dir", default=DEFAULT_IN_DIR, help="Batch input dir")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Batch output dir")
    parser.add_argument("--output", default=None, help="Single image output path")
    parser.add_argument("--verbose", action="store_true", help="Show dropped detections")
    args = parser.parse_args()

    detector = FallDetector(args.model)

    if args.img:
        process_single(detector, args.img, args.output, verbose=args.verbose)
    else:
        if not os.path.isdir(args.in_dir):
            print(f"[ERROR] Directory not found: {args.in_dir}")
            print(f"  Create it or use --img for single image mode.")
            sys.exit(1)
        process_batch(detector, args.in_dir, args.out_dir)

    detector.release()


if __name__ == "__main__":
    main()
