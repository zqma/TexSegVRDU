from typing import Union
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg, CfgNode
from detectron2 import model_zoo
import os
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()


def register_dataset(name, json_path, image_root):
    class_labels = ['text', 'title', 'list', 'table', 'figure']
    register_coco_instances(
        name,
        {},
        json_path,
        image_root
    )
    MetadataCatalog.get(name).thing_classes = class_labels


def build_config(
        # model_name,
        dataset_train_name: Union[str, None],
        dataset_test_name: Union[str, None],
        proj_config,
) -> CfgNode:
    model_zoo_config_name = proj_config['detectron2']['model_zoo_config_name']
    #model_zoo_config_name = model_name
    model_checkout_dir = proj_config['detectron2']['model_checkout_dir']
    model_weights_file = model_checkout_dir + "/model_final.pth"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_zoo_config_name))
    if dataset_train_name:
        cfg.DATASETS.TRAIN = (dataset_train_name,)
    if dataset_test_name:
        cfg.DATASETS.TEST = (dataset_test_name,)
    cfg.OUTPUT_DIR = model_checkout_dir
    cfg.DATALOADER.NUM_WORKERS = 4
    if os.path.exists(model_weights_file):
        cfg.MODEL.WEIGHTS = model_weights_file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = proj_config['detectron2'].getfloat(
        'prediction_score_threshold')
    cfg.SOLVER.IMS_PER_BATCH = proj_config['detectron2'].getint(
        'im_batch_size')
    cfg.SOLVER.BASE_LR = proj_config['detectron2'].getfloat('base_lr')
    cfg.SOLVER.MAX_ITER = proj_config['detectron2'].getint('max_iter')
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = proj_config['detectron2'].getint(
        'roi_batch_size')
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    cfg.TEST.EVAL_PERIOD = 1000
    # cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    # cfg.INPUT.MIN_SIZE_TRAIN = (600, 632, 664, 696, 728, 760)
    # cfg.INPUT.MIN_SIZE_TRAIN = (1000,)  # all set to 800
    cfg.INPUT.FORMAT = "BGR"
    return cfg


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)
