from textseg.predictor import detect_segments, MyPredictor, cfg
from glob import glob
from textseg.utils import read_config
from detectron2.engine import DefaultPredictor 
from textseg.utils import read_im

proj_config = read_config()
DATA_ROOT = proj_config['docvqa']['data_root']

test_folder = f"{DATA_ROOT}/train/documents"
predictor = MyPredictor(cfg)

image_paths = list(glob(f"{test_folder}/*.png"))
# print()
# #predictor = DefaultPredictor(cfg)

# for im_p in image_paths:    
#     cv_im = read_im(im_p)
#     outputs = predictor(cv_im)
#     print(outputs['instances'].to("cpu"))
#     input()

detect_segments(predictor, image_paths, 2)
