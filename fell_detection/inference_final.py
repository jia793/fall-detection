#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RK3588 摔倒检测 — 最终版
多层过滤策略，精准消除背景假阳性
"""

import cv2
import numpy as np
import time
import os
import argparse
from collections import deque

try:
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    print("[WARN] rknnlite 未安装 — 将使用 ONNX (PC 调试)")
    HAS_RKNN = False

# ============================================================================
# 配置
# ============================================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "weights", "best.rknn")
INPUT_SIZE = 640
CLASS_NAMES = {0: "non-fall", 1: "fall"}
FALL_CLASS_ID = 1
ALERT_COOLDOWN = 3.0
DEBUG_N = 3                          # 前 N 帧打印调试

# 几何约束 — 人在画面中的合理范围
PERSON_MIN_AREA_RATIO  = 0.03        # 最小占画面 3%
PERSON_MAX_AREA_RATIO  = 0.70        # 最大占画面 70%
PERSON_MIN_WIDTH_RATIO = 0.04        # 宽度至少 4% 画面宽
PERSON_MIN_HEIGHT_RATIO = 0.08       # 高度至少 8% 画面高
PERSON_STAND_AR = (0.20, 0.85)       # 站立/行走: 宽高比 0.20-0.85
PERSON_FALL_AR  = (0.70, 2.80)       # 摔倒/躺下: 宽高比 0.70-2.80

# 自适应置信度
CONF_HIGH = 0.40                     # 高置信度阈值 (几何不匹配时要求)
CONF_LOW  = 0.12                     # 低置信度阈值 (几何匹配时可用)
IOU_THRES = 0.50                     # NMS 阈值

# 时序过滤
TRACK_WINDOW = 6                     # 观察窗口 (帧)
TRACK_MIN_HITS = 2                   # 最少命中次数

# 颜色
GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
WHITE  = (255, 255, 255)
YELLOW = (0, 255, 255)
ALERT_BG = (0, 0, 180)


# ============================================================================
# 几何评分器
# ============================================================================
def geometry_score(box, frame_w, frame_h):
    """
    评估检测框有多像一个人。
    返回 0.0-1.0，越高越像真人。
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return 0.0

    ar = w / h                                    # 宽高比
    area_ratio = (w * h) / (frame_w * frame_h)    # 面积占比
    width_ratio = w / frame_w
    height_ratio = h / frame_h
    cx = (x1 + x2) / 2 / frame_w                 # 中心 x (归一化)
    cy = (y1 + y2) / 2 / frame_h                 # 中心 y (归一化)

    score = 1.0

    # 1. 面积 — 太小或太大扣分
    if area_ratio < 0.02:
        score *= max(0.0, area_ratio / 0.02)
    elif area_ratio < PERSON_MIN_AREA_RATIO:
        score *= 0.3 + 0.7 * (area_ratio / PERSON_MIN_AREA_RATIO)
    elif area_ratio > PERSON_MAX_AREA_RATIO:
        score *= max(0.0, (0.9 - area_ratio) / 0.2)

    # 2. 宽高比 — 不在站立/摔倒范围扣分
    if PERSON_STAND_AR[0] <= ar <= PERSON_STAND_AR[1]:
        pass  # 站立 — 满分
    elif PERSON_FALL_AR[0] <= ar <= PERSON_FALL_AR[1]:
        pass  # 摔倒 — 满分
    elif ar < 0.15:
        score *= max(0.0, ar / 0.15)             # 太窄
    elif ar > 3.0:
        score *= max(0.0, (4.0 - ar) / 1.0)      # 太宽
    else:
        score *= 0.5                              # 灰色地带

    # 3. 边缘异常检测
    # 真人不会头顶紧贴画面顶部 + 同时占满画面全高
    if y1 <= 3 and height_ratio > 0.55:
        score *= 0.1   # 几乎确定是背景
    elif y1 <= 3 and height_ratio > 0.35:
        score *= 0.3
    # 框同时紧贴左右两侧 → 可能是整块背景
    if x1 <= 2 and x2 >= frame_w - 2:
        score *= 0.15
    # 一般边缘裁切
    if x1 <= 2:
        score *= 0.8
    if x2 >= frame_w - 2:
        score *= 0.85

    # 4. 宽度/高度比例合理性
    if width_ratio < PERSON_MIN_WIDTH_RATIO:
        score *= 0.3
    if height_ratio < PERSON_MIN_HEIGHT_RATIO:
        score *= 0.3

    # 5. 中心位置 — 人在画面顶端不合理（除非紧贴边缘）
    if cy < 0.08 and y1 > 10:
        score *= 0.5

    return max(0.0, min(1.0, score))


def iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    return inter / (area_a + area_b - inter + 1e-6)


def nms(boxes, scores, iou_thres):
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou_vals = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou_vals <= iou_thres)[0] + 1]
    return keep


def temporal_filter(history, min_hits=TRACK_MIN_HITS):
    """时序一致性过滤 — 跨帧稳定的检测才保留"""
    if len(history) < min_hits:
        return history[-1] if history else []

    all_boxes = []
    for frame_dets in history:
        for d in frame_dets:
            all_boxes.append(d)

    if not all_boxes:
        return []

    # 贪心聚类
    clusters = []
    used = [False] * len(all_boxes)
    for i in range(len(all_boxes)):
        if used[i]:
            continue
        cluster = [all_boxes[i]]
        used[i] = True
        for j in range(i + 1, len(all_boxes)):
            if used[j]:
                continue
            for c in cluster:
                if iou(all_boxes[j]["bbox"], c["bbox"]) > 0.30:
                    cluster.append(all_boxes[j])
                    used[j] = True
                    break
        clusters.append(cluster)

    # 只保留命中次数达标的
    stable = []
    for cl in clusters:
        if len(cl) >= min_hits:
            best = max(cl, key=lambda d: d["confidence"])
            stable.append(best)
    return stable


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
        print(f"[INFO] Loading RKNN: {model_path}")
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
                print(f"[INFO] ONNX fallback: {p}")
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

    def postprocess(self, output, frame_h, frame_w, debug=False):
        """
        多层过滤后处理:
        (1) 解码 + sigmoid (自动检测)
        (2) 自适应置信度 (几何匹配 → 低阈值, 不匹配 → 高阈值)
        (3) NMS
        (4) 几何评分筛除 (< 0.3 丢弃)
        (5) 限制数量
        """
        o = output[0]                           # (6, 8400)
        o = np.transpose(o, (1, 0))             # (8400, 6)
        bbox_raw = o[:, :4]
        cls_raw  = o[:, 4:]

        # --- Sigmoid (自动检测) ---
        if np.any(cls_raw < 0) or np.any(cls_raw > 1.0):
            cls_scores = 1.0 / (1.0 + np.exp(-np.clip(cls_raw, -50, 50)))
        else:
            cls_scores = cls_raw

        max_scores = np.max(cls_scores, axis=1)
        class_ids  = np.argmax(cls_scores, axis=1)

        if debug:
            print(f"  [RAW] cls range=[{cls_raw.min():.3f},{cls_raw.max():.3f}]")
            print(f"  [SIG] max_score={max_scores.max():.4f} "
                  f">0.5:{np.sum(max_scores>0.5)} >0.35:{np.sum(max_scores>0.35)} "
                  f">0.15:{np.sum(max_scores>0.15)}")

        # --- 候选筛选 (用最低阈值先收集) ---
        candidate_mask = max_scores > CONF_LOW
        if not np.any(candidate_mask):
            return []

        idx = np.where(candidate_mask)[0]

        # 坐标转换
        cx, cy, wb, hb = bbox_raw[idx, 0], bbox_raw[idx, 1], bbox_raw[idx, 2], bbox_raw[idx, 3]
        x1 = cx - wb / 2.0
        y1 = cy - hb / 2.0
        x2 = cx + wb / 2.0
        y2 = cy + hb / 2.0

        scale_x = frame_w / INPUT_SIZE
        scale_y = frame_h / INPUT_SIZE
        x1 = np.clip(x1 * scale_x, 0, frame_w)
        y1 = np.clip(y1 * scale_y, 0, frame_h)
        x2 = np.clip(x2 * scale_x, 0, frame_w)
        y2 = np.clip(y2 * scale_y, 0, frame_h)

        # --- 逐框过滤 ---
        kept_boxes = []
        kept_scores = []
        kept_classes = []
        kept_confs = []

        for i, j in enumerate(idx):
            box = [x1[i], y1[i], x2[i], y2[i]]
            cls_id = int(class_ids[j])
            conf = float(max_scores[j])
            geo = geometry_score(box, frame_w, frame_h)

            # 自适应阈值: 几何像人 → 低阈值, 几何不像 → 高阈值
            threshold = CONF_LOW + (1.0 - geo) * (CONF_HIGH - CONF_LOW)

            if conf < threshold:
                if debug and conf > 0.05:
                    print(f"  [DROP] cls={cls_id} conf={conf:.3f} geo={geo:.2f} "
                          f"thresh={threshold:.3f} box={[int(v) for v in box]}")
                continue

            kept_boxes.append(box)
            kept_scores.append(conf * geo)       # 综合评分
            kept_classes.append(cls_id)
            kept_confs.append(conf)

        if not kept_boxes:
            return []

        # --- NMS ---
        boxes_arr = np.array(kept_boxes)
        keep = nms(boxes_arr, np.array(kept_scores), IOU_THRES)

        # --- 组合结果 ---
        results = []
        for k in keep[:8]:
            results.append({
                "bbox":       [int(v) for v in kept_boxes[k]],
                "class_id":   kept_classes[k],
                "class_name": CLASS_NAMES.get(kept_classes[k], "unknown"),
                "confidence": kept_confs[k],
            })

        return results

    def release(self):
        if self.rknn is not None:
            self.rknn.release()


# ============================================================================
# 可视化
# ============================================================================
def draw_results(frame, detections, fps, alert):
    fh, fw = frame.shape[:2]
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        is_fall = d["class_id"] == FALL_CLASS_ID
        color = RED if is_fall else GREEN
        thick = 3 if is_fall else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        label = f"{d['class_name']} {d['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)

    if alert:
        text = "! FALL DETECTED !"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(frame, (10, 50), (20 + tw, 58 + th), ALERT_BG, -1)
        cv2.putText(frame, text, (15, 55 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 3)

    stats = f"Det: {len(detections)} | Fall: {sum(1 for d in detections if d['class_id']==FALL_CLASS_ID)}"
    cv2.putText(frame, stats, (10, fh - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


# ============================================================================
# 主循环
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="RK3588 Fall Detection - Final")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--video")
    parser.add_argument("--cam")
    parser.add_argument("--save")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("=" * 50)
    print("  RK3588 Fall Detection — Final")
    print("=" * 50)

    detector = FallDetector(args.model)

    # 视频源
    if args.video:
        cap = cv2.VideoCapture(args.video)
        print(f"[INFO] Video: {args.video}")
    else:
        # 自动找摄像头
        cap = None
        for idx in [11, 12, 13, 14, 15, 16, 17, 0]:
            c = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if c.isOpened():
                ret, f = c.read()
                if ret and f is not None and f.size > 0:
                    cap = c
                    print(f"[INFO] Camera: /dev/video{idx}")
                    break
                c.release()
        if cap is None:
            raise RuntimeError("No camera found")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

    writer = None
    if args.save:
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 25, (fw, fh))
        print(f"[INFO] Saving: {args.save}")

    fps_q = deque(maxlen=30)
    det_history = deque(maxlen=TRACK_WINDOW)
    last_alert = 0
    alert_on = False
    fc = 0

    print("\n[INFO] Running — press 'q' to quit\n")

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            fh, fw = frame.shape[:2]

            inp = detector.preprocess(frame)
            out = detector.inference(inp)
            debug = (fc < DEBUG_N) or args.debug
            dets = detector.postprocess(out, fh, fw, debug=debug)

            # 时序过滤
            det_history.append(dets)
            dets = temporal_filter(list(det_history))

            if debug:
                nf = sum(1 for d in dets if d["class_id"] == FALL_CLASS_ID)
                nn = len(dets) - nf
                print(f"  [Frame {fc}] final: {len(dets)} (normal:{nn} fall:{nf})")
                for d in dets:
                    print(f"    -> {d['class_name']} conf={d['confidence']:.3f} {d['bbox']}")

            # 告警
            falls = [d for d in dets if d["class_id"] == FALL_CLASS_ID]
            now = time.time()
            if falls and (now - last_alert > ALERT_COOLDOWN):
                alert_on = True
                last_alert = now
                best = max(falls, key=lambda d: d["confidence"])
                print(f"\n{'='*40}\n! FALL! conf={best['confidence']:.2%} "
                      f"frame={fc}\n{'='*40}\n")
            elif not falls:
                alert_on = False

            # 绘制
            dt = time.time() - t0
            fps_q.append(1.0 / max(dt, 0.001))
            draw_results(frame, dets, np.mean(fps_q), alert_on)

            if not args.no_show:
                cv2.imshow("Fall Detection Final", frame)
            if writer:
                writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fc += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")
    finally:
        cap.release()
        detector.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\n[INFO] Done. {fc} frames processed.")


if __name__ == "__main__":
    main()
