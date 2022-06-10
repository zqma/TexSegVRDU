from textseg.utils import read_config
from textseg.detectron_trainer import register_dataset, build_config, MyTrainer

def train():
    proj_config = read_config()
    train_dataset_name = "docvqa_train"
    val_dataset_name = "docvqa_val"
    train_json = f"{proj_config['docvqa']['data_root']}/train/train_coco.json"
    train_img_root = f"{proj_config['docvqa']['data_root']}/train/documents"
    val_json = f"{proj_config['docvqa']['data_root']}/val/val_coco.json"
    val_img_root = f"{proj_config['docvqa']['data_root']}/val/documents"

    register_dataset(train_dataset_name, train_json, train_img_root)
    register_dataset(val_dataset_name, val_json, val_img_root)
    model_zoo_config_name = proj_config['detectron2']['model_zoo_config_name']
    model_checkout_dir = proj_config['detectron2']['model_checkout_dir']

    # Detectron config
    cfg = build_config(model_zoo_config_name, train_dataset_name, val_dataset_name, model_checkout_dir, proj_config)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    train()
