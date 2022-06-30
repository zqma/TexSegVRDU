import glob
import json
from pathlib import Path
from textseg.utils import read_config


def compute_bbx(four_coordinates):
    x = min(four_coordinates[0::2])
    y = min(four_coordinates[1::2])
    bb_w = max(four_coordinates[0::2]) - x
    bb_h = max(four_coordinates[1::2]) - y
    assert bb_h > 0
    assert bb_w > 0
    return x, y, bb_w, bb_h


def docvqa2coco(docvqa_json_folder):
    coco_json = {
        "info": {"name": "DocVQA", "format": "COCO"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{'supercategory': '', 'id': 1, 'name': 'text'},
                       {'supercategory': '', 'id': 2, 'name': 'title'},
                       {'supercategory': '', 'id': 3, 'name': 'list'},
                       {'supercategory': '', 'id': 4, 'name': 'table'},
                       {'supercategory': '', 'id': 5, 'name': 'figure'}]}

    for f in glob.glob(f"{docvqa_json_folder}/*.json"):

        vqa_json = json.load(open(f))
        assert len(vqa_json['recognitionResults']) == 1
        width = vqa_json['recognitionResults'][0]['width']
        height = vqa_json['recognitionResults'][0]['height']
        text_lines = vqa_json['recognitionResults'][0]['lines']
        im_id = len(coco_json['images'])
        image_dict = {
            "id": im_id,
            "width": width,
            "height": height,
            "file_name": Path(f).stem + ".png",

        }
        coco_json['images'].append(image_dict)
        annotations = []
        for line in text_lines:
            anno_id = len(annotations) + len(coco_json["annotations"])

            anno_dict = {
                "id": anno_id,
                "category_id": 1,
                "iscrowd": 0,
                "segmentation": [],
                "image_id": im_id,
                "area": 0,
                "bbox": compute_bbx(line['boundingBox']),
                "text": line["text"]
            }
            annotations.append(anno_dict)
        coco_json['annotations'].extend(annotations)
    return coco_json


if __name__ == "__main__":
    config = read_config()
    val_json = docvqa2coco(f"{config['docvqa']['data_root']}/val/ocr_results")
    json.dump(val_json, open(
        f"{config['docvqa']['data_root']}/val/val_coco.json", "w"))
    train_json = docvqa2coco(
        f"{config['docvqa']['data_root']}/train/ocr_results")
    json.dump(train_json, open(
        f"{config['docvqa']['data_root']}/train/train_coco.json", "w"))
