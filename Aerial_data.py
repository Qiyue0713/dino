# import json
# import os
# import shutil
# from tqdm import tqdm
#
# # ===================== 配置项（按你的实际路径修改） =====================
# # 文件夹路径
# IMAGES_DIR = "/data/serverF/Codes/Aerial-data/images"  # 原始5000张图片存放目录
# TRAIN_JSONL_PATH = "/data/serverF/Codes/Aerial-data/annotation/vg_train_odvg.jsonl"  # 官方提供的train标注文件
# VAL_JSONL_PATH = "/data/serverF/Codes/Aerial-data/annotation/vg_val_odvg.jsonl"  # 官方提供的val标注文件
# TEST_JSONL_PATH = "/data/serverF/Codes/Aerial-data/annotation/vg_test_odvg.jsonl"  # 官方提供的test标注文件
#
# # 目标文件夹（分类后的图片存放位置，会自动创建）
# TRAIN_IMG_DIR = "/data/serverF/Codes/Aerial-data/train"
# VAL_IMG_DIR = "/data/serverF/Codes/Aerial-data/val"
# TEST_IMG_DIR = "/data/serverF/Codes/Aerial-data/test"
#
# # jsonl里的图片名字段（你的格式是"filename"，无需修改）
# IMG_NAME_FIELD = "filename"
#
#
# # ===================== 工具函数：提取jsonl中的图片名 =====================
# def extract_img_names(jsonl_path, set_name):
#     """
#     从指定jsonl文件提取所有唯一的图片名（含后缀）
#     :param jsonl_path: 标注文件路径
#     :param set_name: 数据集名称（train/val/test），仅用于日志提示
#     :return: 去重后的图片名列表
#     """
#     img_names = set()  # 集合自动去重（单图多标注只保留一次）
#     if not os.path.exists(jsonl_path):
#         raise FileNotFoundError(f"【{set_name}】标注文件不存在：{jsonl_path}")
#
#     with open(jsonl_path, "r", encoding="utf-8") as f:
#         for line_num, line in enumerate(f, 1):
#             try:
#                 data = json.loads(line.strip())
#                 img_name = data[IMG_NAME_FIELD]  # 提取完整图片名（如xxx.jpg）
#                 img_names.add(img_name)
#             except Exception as e:
#                 print(f"⚠️  【{set_name}】第{line_num}行解析失败 | 错误：{e} | 内容：{line[:100]}...")
#                 continue
#
#     img_list = list(img_names)
#     print(f"✅ 【{set_name}】从标注文件提取到 {len(img_list)} 张唯一图片")
#     return img_list
#
#
# # ===================== 步骤1：提取所有数据集的图片名 =====================
# train_img_names = extract_img_names(TRAIN_JSONL_PATH, "train")
# val_img_names = extract_img_names(VAL_JSONL_PATH, "val")
# test_img_names = extract_img_names(TEST_JSONL_PATH, "test")
#
# # 检查数据集间是否有重复图片（正常应该无重复）
# train_val_dup = set(train_img_names) & set(val_img_names)
# train_test_dup = set(train_img_names) & set(test_img_names)
# val_test_dup = set(val_img_names) & set(test_img_names)
#
# if train_val_dup:
#     print(f"❌ 警告：train和val存在 {len(train_val_dup)} 张重复图片！示例：{list(train_val_dup)[:3]}...")
# if train_test_dup:
#     print(f"❌ 警告：train和test存在 {len(train_test_dup)} 张重复图片！示例：{list(train_test_dup)[:3]}...")
# if val_test_dup:
#     print(f"❌ 警告：val和test存在 {len(val_test_dup)} 张重复图片！示例：{list(val_test_dup)[:3]}...")
#
# # ===================== 步骤2：创建目标文件夹 =====================
# for dir_path in [TRAIN_IMG_DIR, VAL_IMG_DIR, TEST_IMG_DIR]:
#     if not os.path.exists(dir_path):
#         os.makedirs(dir_path)
#         print(f"\n📁 创建文件夹：{dir_path}")
#
# # ===================== 步骤3：批量复制图片（按标注匹配） =====================
# # 获取原始图片列表（用集合提高匹配效率）
# all_img_files = set(os.listdir(IMAGES_DIR))
# print(f"\n📸 原始images文件夹共找到 {len(all_img_files)} 张图片")
#
#
# def copy_images(img_list, target_dir, set_name):
#     """
#     按图片名列表复制图片到目标文件夹
#     :param img_list: 要复制的图片名列表
#     :param target_dir: 目标文件夹
#     :param set_name: 数据集名称（用于日志）
#     """
#     missing_count = 0
#     print(f"\n🚀 开始复制【{set_name}】图片...")
#     for img_name in tqdm(img_list):
#         if img_name in all_img_files:
#             src_path = os.path.join(IMAGES_DIR, img_name)
#             dst_path = os.path.join(target_dir, img_name)
#             shutil.copy2(src_path, dst_path)  # 保留图片元信息
#         else:
#             missing_count += 1
#             print(f"❌ 【{set_name}】缺失图片：{img_name}")
#
#     success_count = len(img_list) - missing_count
#     print(f"✅ 【{set_name}】复制完成：成功 {success_count} 张，缺失 {missing_count} 张")
#     return success_count, missing_count
#
#
# # 复制train/val/test图片
# train_success, train_missing = copy_images(train_img_names, TRAIN_IMG_DIR, "train")
# val_success, val_missing = copy_images(val_img_names, VAL_IMG_DIR, "val")
# test_success, test_missing = copy_images(test_img_names, TEST_IMG_DIR, "test")
#
# # ===================== 步骤4：最终验证 =====================
# print("\n" + "=" * 60)
# print("📊 数据划分最终结果验证")
# print("=" * 60)
# # 统计目标文件夹实际图片数
# train_final = len(os.listdir(TRAIN_IMG_DIR))
# val_final = len(os.listdir(VAL_IMG_DIR))
# test_final = len(os.listdir(TEST_IMG_DIR))
#
# print(f"【train】标注提取 {len(train_img_names)} 张 → 实际复制 {train_final} 张")
# print(f"【val】标注提取 {len(val_img_names)} 张 → 实际复制 {val_final} 张")
# print(f"【test】标注提取 {len(test_img_names)} 张 → 实际复制 {test_final} 张")
# print(f"总计复制：{train_final + val_final + test_final} 张")
# print(f"原始图片总数：{len(all_img_files)} 张")
#
# # 检查是否全部匹配
# total_extracted = len(train_img_names) + len(val_img_names) + len(test_img_names)
# if train_final + val_final + test_final == len(all_img_files) and total_extracted == len(all_img_files):
#     print("\n🎉 划分成功！所有图片均已正确分类，无遗漏/重复")
# else:
#     print("\n⚠️  警告：图片数量不匹配！可能存在：")
#     print("  1. 标注中的图片名与实际图片名不一致")
#     print("  2. 不同数据集标注存在重复图片")
#     print("  3. 原始图片有未被标注覆盖的情况")

import os
import json
import random
import math
from PIL import Image, ImageDraw

# ======================
# 配置
# ======================
ANNOTATION_FILE = "/data/serverF/Codes/Aerial-data/annotation/vg_train_odvg.jsonl"   # jsonl 标注文件
IMAGE_ROOT = "/data/serverF/Codes/Aerial-data/train"                # 图片根目录
VIS_DIR = "debug_vis"                # 可视化输出
NUM_VIS = 50                         # 随机可视化数量
EPS = 1e-3                           # 浮点容忍误差

os.makedirs(VIS_DIR, exist_ok=True)

# ======================
# 统计量
# ======================
total_images = 0
total_boxes = 0

missing_image = 0
invalid_structure = 0
invalid_bbox = 0
out_of_bounds = 0
non_positive_area = 0

valid_samples = []

# ======================
# 读取并检查
# ======================
with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue

        total_images += 1

        try:
            meta = json.loads(line)
        except json.JSONDecodeError:
            print(f"[JSON ERROR] line {line_idx}")
            invalid_structure += 1
            continue

        # ---------- filename ----------
        if "filename" not in meta:
            print(f"[STRUCT ERROR] missing filename at line {line_idx}")
            invalid_structure += 1
            continue

        img_path = os.path.join(IMAGE_ROOT, meta["filename"])
        if not os.path.exists(img_path):
            print(f"[MISSING IMAGE] {img_path}")
            missing_image += 1
            continue

        # ---------- image size ----------
        try:
            img = Image.open(img_path).convert("RGB")
            W, H = img.size
        except Exception as e:
            print(f"[IMAGE ERROR] {img_path}: {e}")
            missing_image += 1
            continue

        # ---------- grounding regions ----------
        regions = meta.get("grounding", {}).get("regions", [])
        if not isinstance(regions, list) or len(regions) == 0:
            print(f"[STRUCT ERROR] empty regions at {meta['filename']}")
            invalid_structure += 1
            continue

        valid_samples.append((meta, img, W, H))
        total_boxes += len(regions)

        for r in regions:
            bbox = r.get("bbox", None)
            if bbox is None or len(bbox) != 4:
                invalid_bbox += 1
                continue

            x1, y1, x2, y2 = bbox

            # 数值检查
            if any([
                not isinstance(v, (int, float)) or
                math.isnan(v) or
                math.isinf(v)
                for v in bbox
            ]):
                invalid_bbox += 1
                continue

            # 面积检查（xyxy 假设）
            if x2 <= x1 or y2 <= y1:
                non_positive_area += 1

            # 越界检查（允许极小误差）
            if (
                x1 < -EPS or y1 < -EPS or
                x2 > W + EPS or y2 > H + EPS
            ):
                out_of_bounds += 1

# ======================
# 打印统计结果
# ======================
print("\n========== AerialVG 数据校验结果 ==========")
print(f"Images total           : {total_images}")
print(f"Boxes total            : {total_boxes}")
print(f"Missing images         : {missing_image}")
print(f"Invalid structure      : {invalid_structure}")
print(f"Invalid bbox format    : {invalid_bbox}")
print(f"Non-positive area bbox : {non_positive_area}")
print(f"Out-of-bounds bbox     : {out_of_bounds}")
print("==========================================\n")

# ======================
# 随机可视化
# ======================
print(f"Saving {NUM_VIS} visualizations to {VIS_DIR} ...")
vis_samples = random.sample(valid_samples, min(NUM_VIS, len(valid_samples)))

for idx, (meta, img, W, H) in enumerate(vis_samples):
    draw = ImageDraw.Draw(img)

    for r in meta["grounding"]["regions"]:
        bbox = r["bbox"]
        phrase = r.get("phrase", "")

        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        if phrase:
            draw.text((x1, max(0, y1 - 10)), phrase, fill="red")

    save_path = os.path.join(
        VIS_DIR,
        f"{idx:03d}_" + os.path.basename(meta["filename"])
    )
    img.save(save_path)

print("Done.")
