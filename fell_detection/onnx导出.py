from ultralytics import YOLO

# 直接写完整的绝对路径，Windows下用 r"" 避免转义问题
model = YOLO(r"D:\diedao_yolov11\Real-Time-Fall-Detection-using-YOLO-main\Model\weights\best.pt")

# 导出 ONNX，保留你设置的参数
model.export(
    format="onnx",
    imgsz=640,
    opset=12,
    simplify=True
)