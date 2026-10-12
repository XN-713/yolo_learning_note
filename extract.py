import os
import cv2
from tqdm import tqdm  # 用于显示进度条

# ============ 需要你修改的参数 ============

# 视频所在文件夹
VIDEO_DIR = r"D:\deeplearning\make_dataset\video"

# 图片保存文件夹
IMAGE_DIR = r"D:\deeplearning\make_dataset\image"

# 每秒提取几帧（想调整就改这个数字）
# 例如：1 表示每秒提取1帧，2 表示每秒提取2帧，0.5 表示每2秒提取1帧
FPS_EXTRACT = 5

# 图片格式
IMAGE_EXT = ".jpg"

# ==========================================


def extract_frames(video_path, save_dir, video_name, fps_extract):
    """
    从一个视频里按指定频率抽帧保存为图片
    video_path:   视频文件的完整路径
    save_dir:     图片保存的文件夹
    video_name:   视频名字（不带扩展名），用来给图片命名
    fps_extract:  每秒抽多少帧
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[错误] 打不开视频: {video_path}")
        return

    # 获取视频原始帧率（比如 30fps 表示每秒30帧）
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算：每隔多少帧提取一张图片
    # 比如视频30fps，你想每秒抽1帧，那每隔30帧抽一张
    # 想每秒抽2帧，就每隔15帧抽一张
    interval = max(1, int(round(video_fps / fps_extract)))

    # 图片序号（从1开始，最后格式化成5位数字）
    img_index = 1

    # 用 tqdm 显示进度条，total=总帧数
    pbar = tqdm(total=total_frames, desc=f"处理 {video_name}", unit="帧")

    frame_id = 0  # 当前读到第几帧
    while True:
        # 读取一帧，ret 表示是否读取成功，frame 是这一帧的图像数据
        ret, frame = cap.read()
        if not ret:
            break  # 读完了，退出循环

        # 判断当前帧是否需要保存
        # frame_id 从 0 开始，每隔 interval 帧保存一张
        if frame_id % interval == 0:
            # 生成图片文件名，例如 001_00001.jpg
            # :05d 表示用0补齐到5位数字
            img_name = f"{video_name}_{img_index:05d}{IMAGE_EXT}"
            img_path = os.path.join(save_dir, img_name)
            # 保存图片
            cv2.imwrite(img_path, frame)
            img_index += 1

        frame_id += 1
        pbar.update(1)  # 进度条前进1

    pbar.close()
    cap.release()  # 释放视频对象
    print(f"[完成] {video_name} 共提取了 {img_index - 1} 张图片")


def main():
    # 检查并创建保存图片的文件夹（不存在就自动创建）
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # 找出视频文件夹里所有的 .mp4 文件
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(".mp4")]

    if len(video_files) == 0:
        print("没有找到任何 .mp4 视频文件，请检查路径！")
        return

    print(f"共找到 {len(video_files)} 个视频文件，开始处理...\n")

    # 依次处理每个视频
    for video_file in video_files:
        # 视频完整路径
        video_path = os.path.join(VIDEO_DIR, video_file)
        # 视频名字（去掉 .mp4 后缀），例如 001.mp4 -> 001
        video_name = os.path.splitext(video_file)[0]

        # 调用抽帧函数
        extract_frames(video_path, IMAGE_DIR, video_name, FPS_EXTRACT)

    print("\n全部处理完成！")


# 如果直接运行这个文件，就执行 main()
if __name__ == "__main__":
    main()