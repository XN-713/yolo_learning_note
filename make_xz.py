import os
import random
import shutil  # 用于移动文件

# ============ 参数设置 ============

# 原始图片文件夹（从这个文件夹里挑图）
SOURCE_DIR = r"D:\deeplearning\make_dataset\image"

# 抽取出来的图片，70% 放到这里
TRAIN_DIR = r"D:\deeplearning\make_dataset\xz_dataset\images\train"

# 抽取出来的图片，30% 放到这里
VAL_DIR = r"D:\deeplearning\make_dataset\xz_dataset\images\val"

# 抽取比例：20% 的图片会被抽出来
EXTRACT_RATIO = 0.20

# 抽出来的图片里，多少比例放 TRAIN，多少放 VAL
TRAIN_RATIO = 0.70
VAL_RATIO = 0.30

# 支持的图片格式
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# 随机种子，固定后每次运行抽取结果一样；不想要固定就改成 None
RANDOM_SEED = 42

# ===================================


def main():
    # 1. 检查源文件夹是否存在
    if not os.path.exists(SOURCE_DIR):
        print(f"[错误] 源文件夹不存在: {SOURCE_DIR}")
        return

    # 2. 创建目标文件夹（不存在就自动创建）
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    # 3. 找出源文件夹里所有的图片文件
    all_images = [
        f for f in os.listdir(SOURCE_DIR)
        if f.lower().endswith(IMAGE_EXTS)
    ]

    total = len(all_images)
    if total == 0:
        print("源文件夹里没有找到任何图片！")
        return

    print(f"源文件夹共有 {total} 张图片")

    # 4. 设置随机种子（保证结果可复现）
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    # 5. 随机打乱顺序，然后取前 20%
    random.shuffle(all_images)
    extract_num = int(total * EXTRACT_RATIO)  # 要抽出来的数量
    extracted = all_images[:extract_num]

    print(f"将随机抽取 {extract_num} 张图片 (约 {EXTRACT_RATIO*100:.0f}%)")

    # 6. 把这 20% 再按 70 / 30 划分
    train_num = int(len(extracted) * TRAIN_RATIO)
    train_list = extracted[:train_num]           # 前 70% 放 TRAIN
    val_list   = extracted[train_num:]           # 剩下 30% 放 VAL

    print(f"  -> {len(train_list)} 张移到 TRAIN 文件夹")
    print(f"  -> {len(val_list)} 张移到 VAL 文件夹")
    print("开始移动文件...\n")

    # 7. 移动文件（用 shutil.move）
    moved_train = 0
    for i, filename in enumerate(train_list, 1):
        src = os.path.join(SOURCE_DIR, filename)
        dst = os.path.join(TRAIN_DIR, filename)
        shutil.move(src, dst)   # 移动（剪切）而不是复制
        moved_train += 1
        print(f"\r[TRAIN] {i}/{len(train_list)}  {filename}", end="")

    print()  # 换行

    moved_val = 0
    for i, filename in enumerate(val_list, 1):
        src = os.path.join(SOURCE_DIR, filename)
        dst = os.path.join(VAL_DIR, filename)
        shutil.move(src, dst)
        moved_val += 1
        print(f"\r[VAL]   {i}/{len(val_list)}  {filename}", end="")

    print("\n")  # 换行

    # 8. 打印总结
    print("=" * 50)
    print(f"完成！源文件夹还剩 {total - extract_num} 张图片")
    print(f"TRAIN 文件夹: {moved_train} 张 -> {TRAIN_DIR}")
    print(f"VAL   文件夹: {moved_val} 张 -> {VAL_DIR}")


if __name__ == "__main__":
    main()