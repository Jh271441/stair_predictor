from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.model_zoo import model_zoo

register_coco_instances("stair_train", {}, "coco_2/train.json", ".")
register_coco_instances("stair_val", {}, "coco_2/val.json", ".")

# dataset_dicts = DatasetCatalog.get("stair_train")
metadata = MetadataCatalog.get("stair_train")


cfg = get_cfg()
cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("stair_train",)
cfg.DATASETS.TEST = ("stair_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "output/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 只有一个类别：楼梯


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, cfg, False)


trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)

evaluator = COCOEvaluator("stair_val", cfg, False)
val_loader = build_detection_test_loader(cfg, "stair_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))