from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import cv2
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 确保类别数量和训练时一致
cfg.MODEL.WEIGHTS = "output/model_final.pth"  # 使用训练后的模型权重
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置推理阈值

predictor = DefaultPredictor(cfg)

# 读取图像
image_path = "test_images/img.png"
im = cv2.imread(image_path)

# 进行推理
outputs = predictor(im)

# 可视化结果
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("my_dataset"), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# 显示结果
plt.imshow(out.get_image()[:, :, ::-1])
plt.show()


# import os
# import cv2
# import torch
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.model_zoo import model_zoo
# from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.data import MetadataCatalog
# from detectron2.modeling import build_model
# from detectron2.checkpoint import DetectionCheckpointer
#
# # 自定义类别名称
# CLASS_NAMES = ["stair"]
# MASK_COLOR = (253.0, 191.0, 111.0)  # 自定义掩码颜色，橙色
#
# # 注册自定义类别元数据
# MetadataCatalog.get("my_custom_dataset").thing_classes = CLASS_NAMES
# MetadataCatalog.get("my_custom_dataset").thing_colors = [MASK_COLOR]
#
# class MyVisualizer(Visualizer):
#     def _jitter(self, color):
#         return color
#
# # 配置模型
# cfg = get_cfg()
# cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = "output/model_final.pth"  # 训练好的模型权重
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置置信度阈值
# cfg.MODEL.DEVICE = "cuda"  # 使用 GPU
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)  # 只有一个类别：楼梯
#
# # 设置输入图像的固定尺寸（宽度和高度）
# cfg.INPUT.MIN_SIZE_TEST = 480  # 固定高度
# cfg.INPUT.MAX_SIZE_TEST = 640  # 固定宽度
#
# # 构建模型和加载权重
# model = build_model(cfg)
# model.eval()
# DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
#
# # 预处理图像
# def preprocess_image(image_path, cfg):
#     """
#     预处理单张图像，将图像调整为模型所需的输入格式。
#     """
#     img = cv2.imread(image_path)
#     img_resized = cv2.resize(img, (cfg.INPUT.MAX_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST))  # 调整图像大小
#     img_tensor = torch.as_tensor(img_resized.astype("float32").transpose(2, 0, 1))  # HWC -> CHW
#     return img_resized, img_tensor
#
# # 推理函数
# def inference(model, image_path, cfg):
#     """
#     对单张图像进行推理。
#     """
#     original_image, processed_image = preprocess_image(image_path, cfg)
#
#     # 将图像输入模型
#     with torch.no_grad():
#         output = model([{"image": processed_image}])[0]
#
#     return original_image, output
#
# # 单张图像推理和保存结果
# def process_and_save_image(input_image_path, output_image_path, model, cfg):
#     original_image, output = inference(model, input_image_path, cfg)
#
#     # 检查推理结果是否为空
#     instances = output["instances"]
#     if len(instances) == 0:
#         print(f"No instances found for image: {input_image_path}")
#         return
#
#     print(f"Detected {len(instances)} instances in image: {input_image_path}")
#
#     # 设置可视化的元数据
#     metadata = MetadataCatalog.get("my_custom_dataset")
#
#     # 可视化结果，并指定掩码颜色
#     v = MyVisualizer(original_image[:, :, ::-1], metadata, instance_mode=ColorMode.SEGMENTATION)
#
#     out = v.draw_instance_predictions(instances.to("cpu"))
#
#     # 保存结果图像
#     cv2.imwrite(output_image_path, out.get_image()[:, :, ::-1])
#     print(f"Processed and saved: {output_image_path}")
#
# # 指定单张图像的输入和输出路径
# input_image_path = "stair_datasets/00877-4ok3usBNeis/color_114.png"  # 输入的单张图像路径
# output_image_path = "output_visualizations/image_output.png"  # 输出的可视化结果路径
#
# # 确保输出文件夹存在
# output_dir = os.path.dirname(output_image_path)
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# # 处理并保存单张图像
# process_and_save_image(input_image_path, output_image_path, model, cfg)