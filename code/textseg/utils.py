import configparser

import numpy as np

from config import ROOT_DIR, CONFIG_DIR
from PIL import Image
import cv2

def read_config():
    config = configparser.ConfigParser()
    config.read(CONFIG_DIR/"config.ini")
    return config


def read_im(image_path):
    pil_im = Image.open(image_path)
    if pil_im.mode != "RGB":
        pil_im = pil_im.convert("RGB")
    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
