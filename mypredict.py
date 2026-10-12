from ultralytics import YOLO

model = YOLO(r"yolo26n.pt")
model.predict(
    source=0,
    save=False,
    show=True,
)