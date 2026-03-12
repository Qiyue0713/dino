# 验证char_to_token方法是否正常工作
from util.get_tokenlizer import get_tokenlizer
from models.GroundingDINO.groundingdino import create_positive_map

# 1. 获取 BertTokenizerFast 实例
tokenizer = get_tokenlizer("/data/serverF/Codes/Open-GroundingDino/bert-base-uncased")
# 新增：打印 tokenizer 的类名（核心判断语句）
print(f"当前 tokenizer 类名：{tokenizer.__class__.__name__}")

# 2. 模拟源码中的 caption 和 cat_list
cat_list = ["person", "car", "dog"]
caption = " . ".join(cat_list) + ' .'

# 3. 生成 BatchEncoding 对象（和源码一致）
tokenized = tokenizer(caption, padding="longest", return_tensors="pt")

# 4. 模拟 create_positive_map 中的调用
start_ind = caption.find(cat_list[2])  # 找到第一个类别在 caption 中的起始索引
try:
    beg_pos = tokenized.char_to_token(start_ind)
    end_pos = tokenized.char_to_token(start_ind + len(cat_list[2]) - 1)
    print(f"✅ 源码调用逻辑验证成功")
    print(f"类别 {cat_list[2]} 对应 token 起始索引：{beg_pos}，结束索引：{end_pos}")
except Exception as e:
    print(f"❌ 验证失败，错误：{e}")