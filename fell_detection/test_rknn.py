#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证脚本 — 在 RK3588 上测试模型是否能正常加载和推理
运行: python test_rknn.py
"""

import cv2
import numpy as np
import time
import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

try:
    from rknnlite.api import RKNNLite
    print("[OK] rknnlite 已导入")
except ImportError:
    print("[FAIL] rknnlite 未安装！请在 RK3588 开发板上运行。")
    sys.exit(1)

MODEL_PATH = os.path.join(os.path.dirname(__file__),  "best.rknn")

# ============================================================
# Step 1: 加载模型
# ============================================================
print("\n[Step 1] 加载模型...")
rknn = RKNNLite()
ret = rknn.load_rknn(MODEL_PATH)
if ret != 0:
    print(f"[FAIL] load_rknn 失败, ret={ret}")
    sys.exit(1)
print("[OK] 模型加载成功")

# ============================================================
# Step 2: 初始化运行时
# ============================================================
print("\n[Step 2] 初始化 NPU 运行时...")
ret = rknn.init_runtime(target="rk3588")
if ret != 0:
    print(f"[WARN] target='rk3588' 失败 (ret={ret}), 尝试默认...")
    ret = rknn.init_runtime()
    if ret != 0:
        print(f"[FAIL] init_runtime 失败, ret={ret}")
        sys.exit(1)
print("[OK] NPU 运行时就绪")

# ============================================================
# Step 3: 查询模型信息
# ============================================================
print("\n[Step 3] 模型信息:")
try:
    input_info = rknn.query(input_output_query="input")
    output_info = rknn.query(input_output_query="output")
    print(f"  输入: {input_info}")
    print(f"  输出: {output_info}")
except Exception as e:
    print(f"  [WARN] query 失败: {e}")

# ============================================================
# Step 4: 创建测试输入 & 推理
# ============================================================
print("\n[Step 4] 预热推理 (3次)...")

# 创建测试 tensor: float32, NCHW
test_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
test_input = np.clip(test_input, 0, 1)  # 模拟归一化图像

for i in range(3):
    t0 = time.time()
    output = rknn.inference(inputs=[test_input])
    t = (time.time() - t0) * 1000
    print(f"  第{i+1}次: {t:.1f}ms, 输出shape={output[0].shape}, dtype={output[0].dtype}")

# ============================================================
# Step 5: 真实图片推理测试
# ============================================================
print("\n[Step 5] 真实图片推理测试...")

# 创建一个测试图片 (640x640 纯色)
test_img = np.ones((640, 640, 3), dtype=np.uint8) * 128  # 灰色图
# 也支持从文件读取
test_img_path = "test_frame.jpg"
use_real = False

# 检查是否有摄像头
cap = cv2.VideoCapture(11, cv2.CAP_V4L2)  # ELF 2 MIPI 摄像头
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        test_img = frame
        use_real = True
        print("  [INFO] 使用摄像头真实画面")
    cap.release()
else:
    # 尝试生成一个带形状的测试图
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (200, 200), (440, 440), (200, 200, 200), -1)
    cv2.circle(test_img, (320, 320), 100, (150, 150, 150), -1)
    print("  [INFO] 使用合成测试图 (无摄像头)")

# 预处理
img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (640, 640))
img_chw = img_resized.transpose(2, 0, 1)
img_float = img_chw.astype(np.float32) / 255.0
img_batch = np.expand_dims(img_float, axis=0)
img_contiguous = np.ascontiguousarray(img_batch)

print(f"  输入 tensor: shape={img_contiguous.shape}, dtype={img_contiguous.dtype}")

t0 = time.time()
output = rknn.inference(inputs=[img_contiguous])
t_infer = (time.time() - t0) * 1000

# 解析输出
out = output[0]               # (1, 6, 8400)
out_t = np.transpose(out[0], (1, 0))  # (8400, 6)
bbox = out_t[:, :4]           # cx, cy, w, h
cls_logits = out_t[:, 4:]     # class logits
cls_scores = 1.0 / (1.0 + np.exp(-cls_logits))
max_scores = np.max(cls_scores, axis=1)
max_class = np.argmax(cls_scores, axis=1)

# Top 5 detections
top_idx = np.argsort(max_scores)[-5:][::-1]
print(f"\n  推理耗时: {t_infer:.1f}ms")
print(f"  Top 5 检测结果:")
for i, idx in enumerate(top_idx):
    score = max_scores[idx]
    cls_id = max_class[idx]
    cls_name = "fall" if cls_id == 1 else "non-fall"
    bx, by, bw, bh = bbox[idx]
    print(f"    [{i}] {cls_name} | conf={score:.4f} | "
          f"bbox=({bx:.0f}, {by:.0f}, {bw:.0f}, {bh:.0f})")

# 检测到的目标数
detections = np.sum(max_scores > 0.35)
print(f"\n  置信度>0.35 的检测数: {detections}")

# ============================================================
# 完成
# ============================================================
print("\n" + "=" * 50)
print("[OK] ✅ 所有测试通过！模型可以在 RK3588 上正常运行。")
print(f"[INFO] 推理速度: ~{t_infer:.1f}ms/帧 (~{1000/t_infer:.0f} FPS)")
print("=" * 50)

rknn.release()
cv2.destroyAllWindows()
