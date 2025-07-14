from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Use a small model for faster training
model.train(data='data.yaml', epochs=50, imgsz=640)