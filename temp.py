from ultralytics import YOLO
model = YOLO("yolov8n.pt")
print(type(model.model))
print(model.model)