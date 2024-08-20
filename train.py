from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.engine import DefaultTrainer, hooks
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

register_coco_instances("stair_train", {}, "coco_3/train.json", ".")
register_coco_instances("stair_val", {}, "coco_3/val.json", ".")

dataset_dicts = DatasetCatalog.get("stair_train")
metadata = MetadataCatalog.get("stair_train")

cfg = get_cfg()
cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("stair_train",)
cfg.DATASETS.TEST = ("stair_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 20000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 只有一个类别：楼梯
cfg.TEST.EVAL_PERIOD = cfg.SOLVER.MAX_ITER // 10

cfg.OUTPUT_DIR = "output_3_add5"


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, cfg, False)

    def build_hooks(self):
        hooks_list = super().build_hooks()
        hooks_list.insert(
            -1,  # 在最后一个钩子（通常是 CheckpointSaver）之前插入
            hooks.EvalHook(
                self.cfg.TEST.EVAL_PERIOD,
                lambda: inference_on_dataset(
                    self.model,
                    build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0]),
                    self.build_evaluator(self.cfg, self.cfg.DATASETS.TEST[0])
                )
            )
        )
        return hooks_list


trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
