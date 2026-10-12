from ultralytics import YOLO

model=YOLO(r"yolo26n.pt")
print(model.task)
print(model.names)