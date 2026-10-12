from ultralytics import YOLO

if __name__ == "__main__" :
    model=YOLO(r"yolo26n.pt")##加载模型
    model.train(
        data="staple.yaml",##例子训练的数据集coco8
        epochs=10,##训练10轮 如果不理想可能这个参数太小模型未收敛
        imgsz=640,##图片尺寸
        batch=-1,##每次投喂图片数量  用-1yolo自动适配合适的
        cache=False,##用“ram”可以提前加载和缩放但会占用内存
        workers=0,##多线程
    )
