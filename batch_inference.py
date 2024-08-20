#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File Name: batch_inference.py
@File Function: 
@Author: chenjunhao (chenjunhao07@baidu.com)
@Created Time: 2024/8/19
"""
import os
import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.transforms import ResizeShortestEdge
from detectron2.structures import ImageList
from detectron2.utils.visualizer import ColorMode

# 自定义类别名称
CLASS_NAMES = ["stair"]
MASK_COLOR = (253.0, 191.0, 111.0)  # 自定义掩码颜色，橙色

# 注册自定义类别元数据
MetadataCatalog.get("my_custom_dataset").thing_classes = CLASS_NAMES
MetadataCatalog.get("my_custom_dataset").thing_colors = [MASK_COLOR]
class MyVisualizer(Visualizer):
    def _jitter(self, color):
        return color

# 配置模型
cfg = get_cfg()
cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "output_3/model_final.pth"  # 训练好的模型权重
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置置信度阈值
cfg.MODEL.DEVICE = "cuda"  # 使用 GPU
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)   # 只有一个类别：楼梯
cfg.INPUT.MIN_SIZE_TEST = 480  # 设置输入图像最小尺寸
cfg.INPUT.MAX_SIZE_TEST = 640  # 设置输入图像最大尺寸

# 构建模型和加载权重
model = build_model(cfg)
model.eval()
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)


# 数据加载和预处理
def preprocess_images(image_paths, cfg):
    """
    预处理图像，将图像调整为模型所需的输入格式。
    """
    images = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
        transform = ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        img = transform.get_transform(img).apply_image(img)
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))  # HWC -> CHW
        images.append(img)
    return images


# 批量推理函数
def batch_inference(model, image_paths, cfg):
    """
    对一批图像进行推理。
    """
    images = preprocess_images(image_paths, cfg)
    images = torch.stack(images)  # 将所有图像堆叠成一个批次

    # 将批量图像输入模型
    with torch.no_grad():
        outputs = model([{"image": img} for img in images])

    return outputs


# 输入输出路径
input_root = "test_images"  # 输入文件夹路径
output_root = "test_images_output"  # 输出文件夹路径

if not os.path.exists(output_root):
    os.makedirs(output_root)

# 批量推理
batch_size = 32  # 设定批量大小
image_paths = []

for subdir, _, files in os.walk(input_root):
    for file in files:
        if file.endswith(".png"):
            image_paths.append(os.path.join(subdir, file))

# 批量处理并保存结果
for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i + batch_size]
    outputs = batch_inference(model, batch_paths, cfg)

    for j, output in enumerate(outputs):
        img = cv2.imread(batch_paths[j])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式

        # 检查推理结果是否为空
        instances = output["instances"]
        img = cv2.resize(img, (instances.image_size[1], instances.image_size[0]))
        if len(instances) == 0:
            print(f"No instances found for image: {batch_paths[j]}")
            # continue

        print(f"Detected {len(instances)} instances in image: {batch_paths[j]}")

        metadata = MetadataCatalog.get("my_custom_dataset")

        # 可视化结果
        v = MyVisualizer(img[:, :, ::-1], metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)

        out = v.draw_instance_predictions(instances.to("cpu"))

        output_dir = batch_paths[j].replace(input_root, output_root)
        output_file = output_dir
        # output_file = output_dir.split('/')[0] + "/" + output_dir.split('/')[1] + "_" + output_dir.split('/')[2]
        # if not os.path.exists(output_subdir):
        #     os.makedirs(output_subdir)

        cv2.imwrite(output_file, out.get_image()[:, :, ::-1])
        print(f"Processed and saved: {output_file}")