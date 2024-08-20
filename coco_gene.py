import labelme2coco

# # 自定义类别映射
# category_mapping = {
#     "stair": 1,
#     # 添加其他类别
# }

# 转换为 COCO 格式
labelme2coco.convert(
    'labelme_json',
    'coco',
    category_id_start=1,
    train_split_rate=0.9
)
