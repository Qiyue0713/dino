import json
import os


def modify_coco_category_ids(json_path, save_path=None):
    """
    修改COCO格式json文件的类别ID和标注类别ID（均减1）
    :param json_path: 原始val.json文件路径
    :param save_path: 修改后的文件保存路径，默认在原始路径后加"_modified"后缀
    """
    # 1. 读取原始JSON文件
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"未找到文件：{json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 2. 修改categories字段中的id（每个id减1）
    if 'categories' in coco_data and len(coco_data['categories']) > 0:
        for category in coco_data['categories']:
            # 确保id是整数类型，避免异常
            if isinstance(category['id'], int):
                category['id'] -= 1
                print(f"类别 {category['name']} 的id已从 {category['id'] + 1} 修改为 {category['id']}")
            else:
                raise TypeError(f"类别ID必须是整数类型，当前类别：{category}")
    else:
        raise KeyError("COCO数据中缺少 'categories' 字段或该字段为空")

    # 3. 修改annotations字段中的category_id（每个category_id减1）
    if 'annotations' in coco_data and len(coco_data['annotations']) > 0:
        for annotation in coco_data['annotations']:
            if isinstance(annotation['category_id'], int):
                annotation['category_id'] -= 1
            else:
                raise TypeError(f"标注类别ID必须是整数类型，当前标注：{annotation}")
        print(f"成功修改 {len(coco_data['annotations'])} 条标注的category_id")
    else:
        raise KeyError("COCO数据中缺少 'annotations' 字段或该字段为空")

    # 4. 保存修改后的JSON文件
    if save_path is None:
        # 自动生成保存路径（原始文件名+_modified.json）
        file_dir, file_name = os.path.split(json_path)
        name_prefix, name_suffix = os.path.splitext(file_name)
        save_path = os.path.join(file_dir, f"{name_prefix}_modified{name_suffix}")

    with open(save_path, 'w', encoding='utf-8') as f:
        # ensure_ascii=False 避免中文等特殊字符乱码，indent=2 格式化输出，便于查看
        json.dump(coco_data, f, ensure_ascii=False, indent=2)

    print(f"修改完成！文件已保存至：{save_path}")
    return save_path


# ------------------- 调用示例 -------------------
if __name__ == "__main__":
    # 请替换为你的val.json文件实际路径
    original_json_path = "/data/serverF/Codes/Open-GroundingDino/my_data/annotations/val.json"  # 相对路径（同文件夹下）
    # 若需指定保存路径，可传入第二个参数，例如：
    # modified_save_path = "val_modified.json"
    # modify_coco_category_ids(original_json_path, modified_save_path)

    # 默认保存（自动添加_modified后缀）
    modify_coco_category_ids(original_json_path)