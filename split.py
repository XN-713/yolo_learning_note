import os
import random
import shutil
from tqdm import tqdm   # 进度条库，需要 pip install tqdm


SUPPORTED_IMAGE_EXTS = (".jpg", ".jpeg", ".png")##支持的图片格式

def split_yolo_dataset(
    images_dir="image",           # 原始图片文件夹（里面是 .jpg）
    labels_dir="label",           # 原始标签文件夹（里面是 .txt）
    output_dir="staple",           # 划分结果保存的根目录
    train_ratio=0.8,               # 训练集比例（可调）
    val_ratio=0.2,                 # 验证集比例（可调）
    #test_ratio=0.0,                # 测试集比例（可调）
    seed=42,                       # 随机种子（固定后每次划分结果一致）
    #image_ext=".jpg",              # 图片后缀
    label_ext=".txt",              # 标签后缀
    copy_files=True,          # True=复制文件(不破坏原数据)
    overwrite=False,
):
    """
    按比例随机划分 YOLO 格式数据集（带进度条）。

    对应关系示例：
        images/abc.jpg  <-->  labels/abc.txt

    划分后目录结构：
        staple/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/
    """

    # ---------- 第 0 步：检查比例是否合法 ----------
    total_ratio = train_ratio + val_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"划分比例之和必须等于 1，当前为 {total_ratio} "
            f"(train={train_ratio}, val={val_ratio})"
        )



    #total_ratio = train_ratio + val_ratio + test_ratio
    #if abs(total_ratio - 1.0) > 1e-6:
    #    raise ValueError(
    #        f"划分比例之和必须等于 1，当前为 {total_ratio} "
    #        f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
    #    )

    #  ---------- 第 0.5 步：旧输出保护 ----------
    if os.path.exists(output_dir) and os.listdir(output_dir):
        if not overwrite:
            raise FileExistsError(
                f"输出目录已存在且不为空：{os.path.abspath(output_dir)}\n"
                f"为避免旧文件残留，请先手动删除，或者把 overwrite 设为 True。"
            )
        else:
            print(f"检测到旧输出目录，overwrite=True，先清空：{output_dir}")
            shutil.rmtree(output_dir)


    # ---------- 第 1 步：固定随机种子 ----------
    # 相同数据集 + 相同 seed => 每次划分结果一致（可复现）
    random.seed(seed)

    # ---------- 第 2 步：扫描 images 文件夹 ----------
    all_items = []  # 每个元素是 (name, ext)，比如 ("abc", ".jpg")！！！！特别注意是元组！！！

    for f in os.listdir(images_dir):
        name, ext = os.path.splitext(f) #将文件按照分隔符.分隔开赋值给元组
        if ext.lower() in SUPPORTED_IMAGE_EXTS: #看看文件是否在支持的格式内是的话加到列表里
            all_items.append((name, ext.lower()))



    print(f"共扫描到 {len(all_items)} 张图片，开始校验标签...")

    # ---------- 第 3 步：过滤缺少标签的样本（带进度条） ----------
    valid_items = []
    missing_items = []

    # tqdm 包住循环即可显示进度条
    for name,ext in tqdm(all_items, desc="校验标签", ncols=80):
        image_path = os.path.join(images_dir, name + ext)
        label_path = os.path.join(labels_dir, name + label_ext)

        if os.path.isfile(label_path):
            valid_items.append((name, ext))
        else:
            missing_items.append((name, ext))
            # 用 tqdm.write 打印，避免进度条被打乱
            tqdm.write(
                f"[缺失标签] 图片 {image_path} 缺少对应的标签文件 {label_path}"
            )

    # ---------- 第 4 步：打乱有效样本 ----------
    random.shuffle(valid_items)

    # ---------- 第 5 步：按比例切分 ----------
    n = len(valid_items)
    n_train = int(n * train_ratio)
    n_val= n - n_train


    #n_val = int(n * val_ratio)
    # 剩下的给测试集，避免因取整丢样本
    #n_test = n - n_train - n_val

    train_names = valid_items[:n_train]
    val_names = valid_items[n_train:n_train + n_val]
    #test_names = valid_items[n_train + n_val:]

    splits = {
        "train": train_names,
        "val": val_names,
        #"test": test_names,
    }

    # ---------- 第 6 步：创建目录 + 复制文件（带进度条） ----------
    for split_name, names in splits.items():
        out_images = os.path.join(output_dir, "images", split_name)
        out_labels = os.path.join(output_dir, "labels", split_name)
        os.makedirs(out_images, exist_ok=True)
        os.makedirs(out_labels, exist_ok=True)

        # 每个 split 一个进度条，显示当前正在处理哪个子集
        for name,ext in tqdm(names, desc=f"复制 {split_name}", ncols=80):
            src_image = os.path.join(images_dir, name + ext)
            src_label = os.path.join(labels_dir, name + label_ext)
            dst_image = os.path.join(out_images, name + ext)
            dst_label = os.path.join(out_labels, name + label_ext)

            if copy_files:
                # copy2 保留元数据，不会修改原文件
                shutil.copy2(src_image, dst_image)
                shutil.copy2(src_label, dst_label)
            else:
                shutil.move(src_image, dst_image)
                shutil.move(src_label, dst_label)

    # ---------- 第 7 步：打印汇总 ----------
    print("=" * 50)
    print(f"有效训练总数：{n}")
    print(f"训练集数量：{len(train_names)}")
    print(f"验证集数量：{len(val_names)}")
    #print(f"测试集数量：{len(test_names)}")
    print(f"缺失标签数量：{len(missing_items)}")
    print("=" * 50)
    print(f"划分结果已保存到：{os.path.abspath(output_dir)}")

# ---------- 直接运行 ----------
if __name__ == "__main__":
    split_yolo_dataset(
        images_dir="image",
        labels_dir="label",
        output_dir="staple",
        train_ratio=0.8,
        val_ratio=0.2,
       # test_ratio=0.0,
        seed=42,           # 换数字 -> 划分变化；不变 -> 结果永远一致
        #image_ext=".jpg",
        label_ext=".txt",
        copy_files=True,   # True = 复制（不动原文件）
        overwrite=False,   # 默认保护旧输出，不直接覆盖
    )